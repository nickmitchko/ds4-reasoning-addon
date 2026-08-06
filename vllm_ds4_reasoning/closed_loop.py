"""Phase-1 closed-loop latent decode (in-worker, eager path, single sequence).

This is the Phase-1 de-risk vehicle from
``docs/closed_loop_latent_decode_plan.md``: prove the CoLaR closed-loop
mechanism -- one prefill, then N autoregressive *latent* decode steps whose input
embedding is ``decoder(head(prev_hidden))`` fed back from the model's own
source-layer hidden -- WITHOUT any vLLM fork/scheduler change.

How it avoids a fork edit
-------------------------
The insight (grounded in gpu_model_runner.py + scheduler.py): a decode step's KV /
position / slot_mapping bookkeeping is driven purely by the scheduler's
``num_computed_tokens`` counter, NOT by what the token *is*. So a "latent step" can
ride an ORDINARY decode step -- we only overwrite that step's INPUT EMBEDDING and
discard the (junk) token it samples. On the eager path (``enforce_eager=True``),
Python forward hooks fire on every pass, giving two seams:

* **Hook A** on ``model.model.layers[source_layer]``: stash the LAST row of the
  layer output (the source-layer hidden of the current last token) into a
  worker-process global. After the prefill pass this is the ``<think>`` anchor
  (feeds latent step 1); after each decode pass it is that step's own hidden --
  the closed-loop feedback signal.
* **Hook B** on ``model.model.embed_tokens``: for the first ``n_latent`` DECODE
  passes (embed output has exactly 1 row for a single-sequence decode), OVERWRITE
  that row with ``decoder(head(LN(prev_hidden)))``. After ``n_latent`` steps the
  hook goes inert and ordinary token decode produces the answer.

SINGLE SEQUENCE ONLY -- like ``hook_inject.set_injection``, this keys off "the
decode pass has one embed row", which only holds for batch size 1. Batched /
compiled / cudagraph closed-loop is the fork-side Workstream-R work (Phase 1
"rider" design in the plan); this module is the mechanism proof.

Everything here runs INSIDE each TP worker (dispatched via ``LLM.apply_model``),
so it uses module-level functions + worker-process globals (no closures cross the
RPC boundary) -- same discipline as ``hook_inject`` / ``vllm_forward_hook``.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Worker-process globals (one set per worker). Populated by apply_model calls and
# read/written by the hooks running in the same process.
_WORKER_STATE = None  # dict, see _setup_closed_loop


def _default_state():
    return {
        # config / weights (set at setup)
        "head": None,          # ReasoningCompressionHead on device
        "decoder": None,       # LatentDecoder on device
        "source_layer": None,  # int
        "latent_dim": None,    # int
        "hidden_size": None,   # int
        "n_latent": 0,         # HARD CAP on latent decode steps (max_latent)
        "device": None,
        "dtype": None,
        # Phase 3: learned stop. When use_stop, the latent phase ends as soon as
        # sigmoid(stop_logit(prev_hidden)) > stop_threshold (after >= min_latent
        # steps); n_latent is then just the safety cap.
        "use_stop": False,
        "stop_threshold": 0.5,
        "min_latent": 1,
        "stopped": False,      # set True once the learned stop fired
        "stop_step": None,     # step index at which it fired
        # hook handles
        "layer_handle": None,
        "embed_handle": None,
        # runtime loop state (reset per generation via reset_loop)
        "prev_hidden": None,   # (hidden,) last source-layer hidden, GPU
        "step": 0,             # decode steps taken since reset
        "prefill_seen": False,
        "trace": [],           # per-step diagnostics (small, CPU floats)
        # closed-loop TRAINING extras
        "collect_hidden": False,  # record self-generated source hiddens
        "hidden_traj": [],        # list[(hidden,) CPU float] per latent step
    }


def _layer_hook(module, args, output):
    """Hook A: stash the last row of the source-layer output.

    DeepseekV4 decoder layers return a tuple ``(hidden_states, residual, ...)``;
    ``output[0]`` is the 2D ``(num_tokens, hidden)`` residual-stream hidden (same
    tensor the capture buffer scatters). We keep the LAST row -- the current last
    token's hidden -- which is the anchor for the NEXT latent step.
    """
    st = _WORKER_STATE
    if st is None:
        return output
    hs = output[0] if isinstance(output, tuple) else output
    if not isinstance(hs, torch.Tensor) or hs.dim() != 2:
        return output
    # Detach + clone so the autograd graph / buffer reuse can't mutate it.
    st["prev_hidden"] = hs[-1].detach().clone()
    if hs.shape[0] > 1:
        st["prefill_seen"] = True
    elif st["collect_hidden"] and st["step"] <= st["n_latent"]:
        # Closed-loop training: record the self-generated source hidden of each
        # latent decode step so the driver can regress the head onto it (DAgger).
        # Kept on CPU (crosses the RPC boundary); small (hidden,) vectors.
        st["hidden_traj"].append(hs[-1].detach().to("cpu", copy=True).float())
    return output


@torch.no_grad()
def _decode_next_embed(st):
    """prev_hidden (source layer) -> injectable input embedding via head+decoder.

    Mirrors ReasoningInjector.hidden_to_injectable exactly: LayerNorm the source
    hidden, head -> mu, LayerNorm the latent, decoder -> hidden-space vector.
    """
    h = st["prev_hidden"].to(st["device"]).float()
    h = F.layer_norm(h, (st["hidden_size"],))
    mu, _ = st["head"](h.unsqueeze(0))            # (1, latent_dim)
    mu = F.layer_norm(mu, (st["latent_dim"],))
    vec = st["decoder"](mu)                        # (1, hidden)
    return vec.squeeze(0)


@torch.no_grad()
def _stop_prob(st):
    """sigmoid(stop_logit(LN(prev_hidden))) in [0,1]. The head's stop head was
    BCE-trained to fire on the group where reasoning ends (</think>). Returns 0.0
    if the head has no stop head (v2) or prev_hidden is unset."""
    if st["prev_hidden"] is None or not hasattr(st["head"], "stop_logit"):
        return 0.0
    h = st["prev_hidden"].to(st["device"]).float()
    h = F.layer_norm(h, (st["hidden_size"],))
    return float(torch.sigmoid(st["head"].stop_logit(h.unsqueeze(0))).cpu())


def _embed_hook(module, args, output):
    """Hook B: on a single-sequence DECODE pass, overwrite the input embedding
    with the decoded latent for the first ``n_latent`` steps.

    ``output`` is the embed_tokens result ``(num_tokens, hidden)``. A prefill pass
    has ``num_tokens > 1``; a single-sequence decode has exactly 1 row. We act only
    on decode rows and only while ``step < n_latent`` and we have a prev_hidden.
    """
    st = _WORKER_STATE
    if st is None:
        return output
    if not isinstance(output, torch.Tensor) or output.dim() != 2:
        return output
    # Only decode passes (one row). Prefill (many rows) passes through untouched;
    # its embeddings are the real prompt embeddings.
    if output.shape[0] != 1:
        return output
    # Hard cap (n_latent == max_latent) or no anchor yet -> resume token decode.
    if st["step"] >= st["n_latent"] or st["prev_hidden"] is None:
        return output
    # Already terminated by the learned stop on an earlier step this generation.
    if st["stopped"]:
        return output

    # Phase 3: learned termination. Evaluate the stop head on the anchor that
    # would produce THIS step's latent; if it fires (after >= min_latent steps),
    # do NOT inject -- let the backbone decode a real token, ending the latent
    # phase on the learned </think> signal. n_latent stays as the safety cap.
    p_stop = _stop_prob(st) if st["use_stop"] else 0.0
    if (st["use_stop"] and st["step"] >= st["min_latent"]
            and p_stop > st["stop_threshold"]):
        st["stopped"] = True
        st["stop_step"] = st["step"]
        st["trace"].append({
            "step": st["step"] + 1, "stop_prob": round(p_stop, 4),
            "action": "stop", "prev_hidden_norm":
                float(st["prev_hidden"].float().norm().cpu()),
        })
        return output

    vec = _decode_next_embed(st).to(device=output.device, dtype=output.dtype)
    out = output.clone()
    out[0] = vec
    st["step"] += 1
    # Cheap per-step diagnostic (norms only -- no big tensors cross RPC).
    st["trace"].append({
        "step": st["step"],
        "stop_prob": round(p_stop, 4),
        "action": "inject",
        "prev_hidden_norm": float(st["prev_hidden"].float().norm().cpu()),
        "inject_norm": float(vec.float().norm().cpu()),
    })
    return out


# --- apply_model entry points (run inside each worker) ----------------------


def setup_closed_loop(model, *, head_sd, decoder_sd, source_layer, hidden_size,
                      latent_dim, n_latent, dtype_str="float32",
                      use_stop=False, stop_threshold=0.5, min_latent=1):
    """Runs INSIDE each worker: load head/decoder into worker globals and install
    the two hooks. Idempotent -- re-registers cleanly if called again.

    Args:
        model: the backbone (from apply_model / get_model()).
        head_sd: ReasoningCompressionHead state_dict (CPU tensors).
        decoder_sd: LatentDecoder state_dict (CPU tensors).
        source_layer: capture layer index (e.g. 35).
        hidden_size, latent_dim: geometry.
        n_latent: HARD CAP on latent decode steps (safety bound / fixed-N when
            use_stop is False).
        dtype_str: compute dtype for the head/decoder ("float32").
        use_stop: Phase 3 -- end the latent phase on the learned stop head instead
            of a fixed N.
        stop_threshold: sigmoid(stop_logit) above which the latent phase ends.
        min_latent: minimum latent steps before the stop head can fire.
    """
    global _WORKER_STATE
    # Local import so the picklable applier stays tiny; the addon package is on
    # PYTHONPATH inside the worker.
    from .models import LatentDecoder, ReasoningCompressionHead

    remove_closed_loop(model)  # clear any prior hooks/state
    st = _default_state()

    device = next(model.parameters()).device
    dtype = getattr(torch, dtype_str)

    head = ReasoningCompressionHead(hidden_size=hidden_size, latent_dim=latent_dim)
    head.load_state_dict(head_sd)
    head.to(device=device, dtype=dtype).eval()

    decoder = LatentDecoder(hidden_size=hidden_size, latent_dim=latent_dim)
    decoder.load_state_dict(decoder_sd)
    decoder.to(device=device, dtype=dtype).eval()

    st.update({
        "head": head, "decoder": decoder,
        "source_layer": int(source_layer),
        "latent_dim": int(latent_dim), "hidden_size": int(hidden_size),
        "n_latent": int(n_latent), "device": device, "dtype": dtype,
        "use_stop": bool(use_stop), "stop_threshold": float(stop_threshold),
        "min_latent": int(min_latent),
    })

    layer = model.model.layers[int(source_layer)]
    st["layer_handle"] = layer.register_forward_hook(_layer_hook)
    st["embed_handle"] = model.model.embed_tokens.register_forward_hook(_embed_hook)

    _WORKER_STATE = st
    return True


def reset_loop(model, *, n_latent=None, collect_hidden=False,
               use_stop=None, stop_threshold=None, min_latent=None):
    """Runs INSIDE each worker: reset per-generation loop state before a new
    generate(). Optionally change the latent-step cap, stop config, and enable
    trajectory collection (closed-loop training)."""
    st = _WORKER_STATE
    if st is None:
        return False
    st["prev_hidden"] = None
    st["step"] = 0
    st["prefill_seen"] = False
    st["trace"] = []
    st["hidden_traj"] = []
    st["collect_hidden"] = bool(collect_hidden)
    st["stopped"] = False
    st["stop_step"] = None
    if n_latent is not None:
        st["n_latent"] = int(n_latent)
    if use_stop is not None:
        st["use_stop"] = bool(use_stop)
    if stop_threshold is not None:
        st["stop_threshold"] = float(stop_threshold)
    if min_latent is not None:
        st["min_latent"] = int(min_latent)
    return True


def fetch_hidden_traj(model):
    """Runs INSIDE each worker: return the self-generated source-hidden
    trajectory (list of CPU float tensors, one per latent step) collected during
    the last closed-loop rollout. Empty if collect_hidden was off."""
    st = _WORKER_STATE
    if st is None:
        return []
    return list(st.get("hidden_traj", []))


def refresh_weights(model, *, head_sd, decoder_sd):
    """Runs INSIDE each worker: reload the rollout head/decoder weights (used to
    periodically sync the worker's stale rollout policy to the trained head).
    No hook re-registration; just copies parameters in place."""
    st = _WORKER_STATE
    if st is None:
        return False
    st["head"].load_state_dict(head_sd)
    st["decoder"].load_state_dict(decoder_sd)
    st["head"].to(device=st["device"], dtype=st["dtype"]).eval()
    st["decoder"].to(device=st["device"], dtype=st["dtype"]).eval()
    return True


def fetch_trace(model):
    """Runs INSIDE each worker: return this worker's per-step diagnostics
    (picklable). Rank-agnostic; the driver keeps rank 0's (see note in
    vllm_forward_hook about post-all-reduce identical hidden across ranks)."""
    st = _WORKER_STATE
    if st is None:
        return {"active": False}
    return {
        "active": True,
        "n_latent": st["n_latent"],
        "steps_run": st["step"],
        "use_stop": st["use_stop"],
        "stopped": st["stopped"],
        "stop_step": st["stop_step"],
        "stop_threshold": st["stop_threshold"],
        "trace": list(st["trace"]),
    }


def remove_closed_loop(model):
    """Runs INSIDE each worker: remove both hooks and clear state."""
    global _WORKER_STATE
    st = _WORKER_STATE
    if st is not None:
        for key in ("layer_handle", "embed_handle"):
            h = st.get(key)
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass
    _WORKER_STATE = None
    return True
