"""Phase-4/5 HTTP serve integration: BATCHED, compile-safe closed-loop latent
decode over ``vllm serve``.

Wires the fork's batched latent rider (gpu_model_runner.ds4_pre_forward /
ds4_setup_latent) into the live OpenAI chat endpoint. On a chat request the
backbone prefills ONCE, runs autoregressive LATENT decode steps (each input
embedding = ``decoder(head(prev layer-35 hidden))``, fed back from the model's own
hidden via the per-req anchor store), then decodes the answer as normal tokens --
all inside the single ``engine_client.generate``, and BATCHED across concurrent
requests (each req_id gets its own latent phase).

Mechanism:
  * The plugin ``register()`` runs in the API-SERVER process (arg_utils'
    load_general_plugins) and monkeypatches
    ``OpenAIServingChat._create_chat_completion`` to (1) arm the batched rider in
    every worker once via ``AsyncLLM.collective_rpc`` -> ``ds4_setup_latent``, and
    (2) tag each request's SamplingParams (``vllm_xargs`` -> ``ds4_latent_*``
    scalars) so the fork's ``ds4_pre_forward`` drives that request's latent phase
    keyed by req_id.
  * No global lock, no per-request reset: per-req latent state lives on the runner.

REQUIREMENTS (batched compile-safe path):
  * ``--enable-prompt-embeds`` -- the rider overwrites ``inputs_embeds.gpu`` rows;
    without it the pure-token path never touches that buffer.
  * ``VLLM_DS4_REASONING_CAPTURE_LAYER=35`` -- the anchor store's feedback signal.
  * NO ``--enforce-eager`` and NO ``--max-num-seqs 1`` needed: the rider runs in
    execute_model (cudagraph-safe, batched). Streaming is supported.

Learned stop (VLLM_DS4_REASONING_STOP=1) uses the retrained head's stop signal;
otherwise a fixed ``max_latent`` cap bounds the latent phase.

Output-token floor (VLLM_DS4_REASONING_MIN_OUTPUT_TOKENS, default 2048): the
latent <think> phase consumes output tokens before the answer, so a client that
requests a small ``max_tokens`` (notably Claude Code) can have its whole budget
spent thinking, yielding an empty text block (stop_reason=length). The hook
raises each request's output budget to at least this floor PLUS the request's
latent-step cap (each latent step bills one reserved accounting token), so the
floor bounds the answer alone; override per request via the
``x-ds4-min-output-tokens`` header, or set the floor to 0 to disable.
"""

import logging
import os

logger = logging.getLogger(__name__)

# --- env knobs --------------------------------------------------------------
ENV_MAX_LATENT = "VLLM_DS4_REASONING_MAX_LATENT"   # latent-step cap (n_latent)
ENV_MIN_LATENT = "VLLM_DS4_REASONING_MIN_LATENT"
ENV_USE_STOP = "VLLM_DS4_REASONING_STOP"           # 1 -> enable learned stop
ENV_STOP_THRESH = "VLLM_DS4_REASONING_STOP_THRESHOLD"
# Output-token floor. The closed-loop latent reasoning phase consumes output
# tokens inside <think> before the answer; if the client's max_tokens is smaller
# than the reasoning needs, the budget is spent thinking and no answer is emitted
# (empty text block, stop_reason=length). Clients like Claude Code send a modest
# max_tokens that is too low for the head. We FLOOR the request's output budget to
# this value so the answer always has room. 0 disables the floor.
ENV_MIN_OUTPUT = "VLLM_DS4_REASONING_MIN_OUTPUT_TOKENS"
# Force-emit "</think>" as the sampled token on the step the latent phase ends.
# WHY (measured 2026-08-01): the stop head fires correctly -- 12/12 requests
# reported end=stop at 28-53 of 256 latent steps -- but ending the phase only
# stopped INJECTING latents. Nothing emitted the </think> transition, so the
# backbone kept generating prose while still inside <think> and the deepseek_v4
# reasoning parser put the ENTIRE answer in reasoning_content, leaving the client's
# content field empty. Across 136 scored long-context serve rows, 135 had
# empty_answer == answer_in_reasoning_only: the text was always produced and always
# stranded. Default ON -- an answer no client can read is not a useful default; set
# to 0 to restore the previous behavior for A/B measurement.
ENV_CLOSE_THINK = "VLLM_DS4_REASONING_CLOSE_THINK"
# The literal reasoning-end token. DeepSeek's think tags are plain ASCII (unlike
# the FULL-WIDTH role markers) and tokenize to a single id (verified: 128822). The
# id is resolved from the ENGINE's tokenizer at arm time rather than hardcoded.
THINK_END_TOKEN = "</think>"

# --- per-request HTTP HEADERS -----------------------------------------------
# Clients that can set headers but not custom body fields (notably Claude Code
# via ANTHROPIC_CUSTOM_HEADERS) can drive the reasoning head through these. They
# work on BOTH /v1/chat/completions and /v1/messages (the Anthropic handler
# subclasses OpenAIServingChat, so both flow through the patched
# _create_chat_completion). Header values win over the request body / env
# defaults. Names are lowercase (Starlette headers are case-insensitive).
HDR_THINKING = "x-ds4-thinking"              # 1/true/yes/on -> chat_template thinking
HDR_MAX_LATENT = "x-ds4-max-latent"          # int  -> ds4_latent_n
HDR_USE_STOP = "x-ds4-use-stop"              # 0/1  -> ds4_latent_use_stop
HDR_STOP_THRESH = "x-ds4-stop-threshold"     # float-> ds4_latent_stop_threshold
HDR_MIN_LATENT = "x-ds4-min-latent"          # int  -> ds4_latent_min_latent
HDR_MIN_OUTPUT = "x-ds4-min-output-tokens"   # int  -> floor on request max_tokens

_INSTALLED = False   # workers have the batched rider armed
_PATCHED = False     # OpenAIServingChat already monkeypatched


def _cfg_int(name, default):
    v = os.environ.get(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _cfg_float(name, default):
    v = os.environ.get(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _apply_request_headers(request, raw_request) -> None:
    """Map ``x-ds4-*`` HTTP headers onto the request (chat_template thinking +
    the flat ds4_latent_* xargs the rider reads). Best-effort; a malformed header
    value is skipped rather than failing the request. Header values take
    precedence over any body-supplied vllm_xargs / chat_template_kwargs.

    Enables clients that can only set headers (e.g. Claude Code via
    ANTHROPIC_CUSTOM_HEADERS) to fully control the reasoning head, including over
    /v1/messages where the Anthropic->OpenAI translation drops body vllm_xargs.
    """
    headers = getattr(raw_request, "headers", None)
    if not headers:
        return

    # thinking -> chat_template_kwargs
    tv = headers.get(HDR_THINKING)
    if tv is not None:
        ck = dict(getattr(request, "chat_template_kwargs", None) or {})
        ck["thinking"] = _truthy(tv)
        ck["enable_thinking"] = ck["thinking"]  # DeepSeek templates accept either
        request.chat_template_kwargs = ck

    # ds4_latent_* -> vllm_xargs (parsed to the rider's expected scalar types)
    xargs = dict(getattr(request, "vllm_xargs", None) or {})
    for hdr, key, cast in (
        (HDR_MAX_LATENT, "ds4_latent_n", int),
        (HDR_MIN_LATENT, "ds4_latent_min_latent", int),
        (HDR_STOP_THRESH, "ds4_latent_stop_threshold", float),
    ):
        hv = headers.get(hdr)
        if hv is not None:
            try:
                xargs[key] = cast(hv)
            except (TypeError, ValueError):
                logger.warning("DS4 header %s=%r not a %s; ignored.",
                               hdr, hv, cast.__name__)
    uv = headers.get(HDR_USE_STOP)
    if uv is not None:
        xargs["ds4_latent_use_stop"] = 1 if _truthy(uv) else 0
    if xargs:
        request.vllm_xargs = xargs


def _apply_output_floor(request, raw_request, floor_default,
                        latent_default) -> None:
    """Raise the request's output-token budget to at least ``floor`` so the
    latent reasoning phase can't consume the whole budget inside <think> and
    leave no room for the answer (the Claude-Code failure: a modest max_tokens
    is too low for the head, so the response is an empty text block with
    stop_reason=length). The per-request ``x-ds4-min-output-tokens`` header wins
    over the env default. A value of 0 disables the floor. We never LOWER the
    client's budget, only raise it.

    ChatCompletionRequest carries both ``max_tokens`` (deprecated) and
    ``max_completion_tokens``; vLLM derives the sampling limit from whichever is
    set, so we floor both to keep them consistent.

    Every latent step now emits one reserved accounting token
    (``DS4_LATENT_PAD_TOKEN_ID``, needed to keep the engine's token-count
    invariant true under speculative decoding), and those tokens count toward
    ``num_output_tokens`` -- so ``check_stop`` bills the latent phase against
    ``max_tokens``. The floor is therefore raised by the request's latent cap, so
    a floor of N still leaves N tokens for the ANSWER rather than N minus the
    latent-step count. Must be called AFTER ``_apply_request_headers``, which is
    what puts a per-request ``x-ds4-max-latent`` override into ``vllm_xargs``.
    """
    floor = floor_default
    headers = getattr(raw_request, "headers", None)
    if headers is not None:
        hv = headers.get(HDR_MIN_OUTPUT)
        if hv is not None:
            try:
                floor = int(hv)
            except (TypeError, ValueError):
                logger.warning("DS4 header %s=%r not an int; ignored.",
                               HDR_MIN_OUTPUT, hv)
    if floor <= 0:
        return
    latent_budget = latent_default
    xargs = getattr(request, "vllm_xargs", None) or {}
    try:
        latent_budget = int(xargs.get("ds4_latent_n", latent_default))
    except (TypeError, ValueError):
        pass
    floor += max(0, latent_budget)

    def _set(attr, val):
        try:
            setattr(request, attr, val)
        except (AttributeError, ValueError):  # frozen / validated field
            pass

    attrs = [a for a in ("max_tokens", "max_completion_tokens")
             if hasattr(request, a)]
    set_vals = {a: getattr(request, a) for a in attrs
                if getattr(request, a) is not None}
    if not set_vals:
        # No budget specified at all -> plant the floor on every field.
        for a in attrs:
            _set(a, floor)
        return
    # Only RAISE fields the client already set. Never plant the floor onto a
    # None field while another field carries a larger budget: since vLLM lets
    # max_completion_tokens win, doing so could LOWER the effective budget.
    for a, cur in set_vals.items():
        if cur < floor:
            _set(a, floor)


# --- worker-side RPC adapters (run inside each worker via collective_rpc) ----
# collective_rpc hands the callable the WORKER object (partial(fn, worker)), so
# these unwrap the model and delegate to the model-based closed_loop functions.
# Module-level + picklable (cloudpickle over the RPC boundary).

def _rpc_setup_latent(worker, *, head_bytes, decoder_bytes, **kw):
    """Arm the BATCHED latent rider on the worker's model_runner (the fork's
    ds4_setup_latent). State dicts arrive as torch.save bytes (robust across the
    stdlib-pickle engine-core RPC -- shipping tensors directly hits the
    memoryview-not-picklable error)."""
    import io
    import torch
    head_sd = torch.load(io.BytesIO(head_bytes), map_location="cpu")
    decoder_sd = torch.load(io.BytesIO(decoder_bytes), map_location="cpu")
    return worker.model_runner.ds4_setup_latent(
        head_sd, decoder_sd, **kw)


def _rpc_latent_supported(worker):
    """True iff this fork build has the batched rider (ds4_setup_latent)."""
    return hasattr(worker.model_runner, "ds4_setup_latent")


def _rpc_latent_stats(worker):
    """Per-request latent-phase diagnostics (steps/stop_step/max_p/end)."""
    mr = worker.model_runner
    return mr.ds4_fetch_latent_stats() if hasattr(mr, "ds4_fetch_latent_stats") else {}


def _rpc_set_traj_capture(worker, *, enabled=True):
    """Fix D: arm/disarm per-step self-generated trajectory capture on the
    batched rider (clears any prior buffer). No-op on builds without it."""
    mr = worker.model_runner
    if hasattr(mr, "ds4_set_traj_capture"):
        return mr.ds4_set_traj_capture(enabled)
    return False


def _rpc_latent_traj(worker):
    """Fix D: per-request self-generated source-hidden trajectory
    (req_id -> list[(hidden,) CPU float]) from the last compile-safe rollout.
    Empty on builds without it or if capture was not armed."""
    mr = worker.model_runner
    return mr.ds4_fetch_latent_traj() if hasattr(mr, "ds4_fetch_latent_traj") else {}


def _rpc_traj_supported(worker):
    """True iff this fork build exposes the Fix-D trajectory capture API."""
    return hasattr(worker.model_runner, "ds4_fetch_latent_traj")


def _rpc_set_traj_capture_sample(worker, *, enabled=True, sample=False,
                                 sample_sigma=0.0):
    """Arm trajectory capture WITH optional stochastic latent sampling
    (RAFT/best-of-N or GRPO rollout). ``sample_sigma`` > 0 forces a fixed
    exploration width so N rollouts genuinely differ. No-op on builds without the
    sampling API; degrades gracefully across older signatures."""
    mr = worker.model_runner
    if hasattr(mr, "ds4_set_traj_capture"):
        try:
            return mr.ds4_set_traj_capture(enabled, sample=sample,
                                           sample_sigma=sample_sigma)
        except TypeError:
            try:
                return mr.ds4_set_traj_capture(enabled, sample=sample)
            except TypeError:
                return mr.ds4_set_traj_capture(enabled)
    return False


def _rpc_latent_actions(worker):
    """Fix C: per-request SAMPLED latent actions z_k (req_id -> list of
    (latent_dim,) CPU float) from the last GRPO rollout. Empty otherwise."""
    mr = worker.model_runner
    return mr.ds4_fetch_latent_actions() if hasattr(
        mr, "ds4_fetch_latent_actions") else {}


def _rpc_grpo_supported(worker):
    """True iff this fork build exposes the Fix-C GRPO sampling API."""
    return hasattr(worker.model_runner, "ds4_fetch_latent_actions")


# --- serving-side orchestration ---------------------------------------------

class Ds4RiderUnavailable(RuntimeError):
    """The rider cannot be armed because the engine config forbids it.

    Distinct from an ordinary addon error: those degrade to base serving, this
    one must surface, because the request asked for latent reasoning and the
    server can never provide it in this configuration.
    """


async def _resolve_think_end_id(engine):
    """Token id of ``</think>``, from the ENGINE's own tokenizer.

    Resolved rather than hardcoded: the id is a property of the served model, and a
    wrong constant would stamp an arbitrary token into every answer. Returns None if
    it cannot be resolved as a SINGLE token (the rider then falls back to its old
    behavior of merely ceasing injection) -- a multi-token marker cannot be emitted
    in the one sampled-token slot this mechanism controls.
    """
    try:
        tok = engine.get_tokenizer()
        if hasattr(tok, "__await__"):        # older async accessor
            tok = await tok
        tid = tok.convert_tokens_to_ids(THINK_END_TOKEN)
        if tid is None or int(tid) < 0:
            ids = tok(THINK_END_TOKEN, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                logger.warning("DS4: %r is not a single token (%d ids); not "
                               "closing </think>.", THINK_END_TOKEN, len(ids))
                return None
            tid = ids[0]
        return int(tid)
    except Exception as e:  # noqa: BLE001 -- never break serving over this
        logger.warning("DS4: could not resolve the </think> token id (%s); the "
                       "latent phase will not emit it.", e)
        return None


async def _ensure_installed(engine, injector, cfg):
    """Arm the batched latent rider in every worker once (lazy, first request)."""
    global _INSTALLED
    if _INSTALLED:
        return
    import io
    import torch
    b = injector.bundle
    # Ship state dicts as torch.save BYTES: the engine-core RPC pickles args with
    # STDLIB pickle (not cloudpickle), which chokes on the memoryview backing the
    # torch.load-mmap'd bundle tensors ("cannot pickle memoryview objects").
    def _to_bytes(sd):
        buf = io.BytesIO()
        torch.save({k: v.detach().cpu() for k, v in sd.items()}, buf)
        return buf.getvalue()
    head_bytes = _to_bytes(b.reasoning_head.state_dict())
    dec_bytes = _to_bytes(b.decoder.state_dict())
    think_end_id = (await _resolve_think_end_id(engine)
                    if cfg["close_think"] else None)
    try:
        await engine.collective_rpc(_rpc_setup_latent, kwargs=dict(
            head_bytes=head_bytes, decoder_bytes=dec_bytes,
            source_layer=int(b.source_layer or 35),
            hidden_size=int(b.hidden_size), latent_dim=int(b.latent_dim),
            use_stop=cfg["use_stop"], stop_threshold=cfg["stop_threshold"],
            min_latent=cfg["min_latent"], think_end_token_id=think_end_id,
            # The bundle's ARCHITECTURE. Without it the worker builds the v1
            # classes and a v2 bundle dies on a net.4 size mismatch, so every
            # Aux Head v4 checkpoint was unservable through this path.
            arch=getattr(b, "arch", "v1"), bank_m=getattr(b, "bank_m", 1),
        ))
    except Exception as e:  # noqa: BLE001
        # ds4_setup_latent validates the ENGINE config (prompt-embeds, and the
        # rider-vs-speculative-decoding conflict). Those are unfixable
        # misconfigurations, not transient errors: the rider will never arm, so
        # every request would be silently served on the bare backbone -- or
        # worse, on a half-dead worker. Mark it fatal so the caller reports it
        # instead of degrading, and re-raise on every subsequent request.
        raise Ds4RiderUnavailable(str(e)) from e
    _INSTALLED = True
    logger.info("DS4 batched latent rider armed in workers "
                "(max_latent=%d use_stop=%s think_end_id=%s).",
                cfg["max_latent"], cfg["use_stop"], think_end_id)


def _make_patched_create(orig, injector):
    """Build the wrapped _create_chat_completion (BATCHED, compile-safe path).

    Per request we (once) arm the batched rider in the workers, then tag the
    request's SamplingParams via ``vllm_xargs`` with flat ``ds4_latent_*`` scalars
    so the fork's ds4_pre_forward drives the latent phase per req_id -- no global
    lock, no per-request reset, and concurrent/batched requests each get their own
    latent phase. Streaming works too (state is keyed by req_id, driven every
    decode step regardless of when the API generator drains)."""
    cfg = {
        "max_latent": _cfg_int(ENV_MAX_LATENT, 8),
        "min_latent": _cfg_int(ENV_MIN_LATENT, 1),
        "use_stop": os.environ.get(ENV_USE_STOP, "0") == "1",
        "stop_threshold": _cfg_float(ENV_STOP_THRESH, 0.5),
        # Floor on output tokens so the latent <think> phase can't starve the
        # answer. Default 2048 (verified 0/12 runaway at 2000 vs 1/12 at 400).
        "min_output_tokens": _cfg_int(ENV_MIN_OUTPUT, 2048),
        # Emit </think> when the latent phase ends (see ENV_CLOSE_THINK).
        "close_think": os.environ.get(ENV_CLOSE_THINK, "1") == "1",
    }

    def _tag_request(request):
        """Inject flat ds4_latent_* scalars into request.vllm_xargs (the OpenAI
        field only permits scalar values, so we can't nest a dict)."""
        xargs = dict(getattr(request, "vllm_xargs", None) or {})
        xargs.setdefault("ds4_latent_n", cfg["max_latent"])
        xargs.setdefault("ds4_latent_use_stop", 1 if cfg["use_stop"] else 0)
        xargs.setdefault("ds4_latent_stop_threshold", cfg["stop_threshold"])
        xargs.setdefault("ds4_latent_min_latent", cfg["min_latent"])
        request.vllm_xargs = xargs

    debug = os.environ.get("VLLM_DS4_REASONING_DEBUG", "0") == "1"

    async def _patched(self, request, *args, **kwargs):
        engine = self.engine_client
        raw_request = args[0] if args else kwargs.get("raw_request")
        try:
            # Header overrides FIRST: x-ds4-* headers set chat_template thinking +
            # ds4_latent_* xargs, so they win over env defaults (via _tag_request's
            # setdefault) and reach the template render (before orig()). This is
            # how Claude Code -- which sets headers but not custom body fields --
            # controls the reasoning head, including over /v1/messages.
            _apply_request_headers(request, raw_request)
            _apply_output_floor(request, raw_request, cfg["min_output_tokens"],
                                cfg["max_latent"])
            await _ensure_installed(engine, injector, cfg)
            _tag_request(request)
        except Ds4RiderUnavailable as e:
            # Engine misconfiguration -- do NOT fall through to base serving.
            # Falling through here is what let a rider+spec-decode server answer
            # with incoherent text instead of reporting the conflict.
            logger.error("DS4 latent rider unavailable: %s", e)
            raise
        except Exception as e:  # noqa: BLE001 -- never break serving on addon error
            logger.warning("DS4 latent tag/install failed (%s); serving base.", e)
        result = await orig(self, request, *args, **kwargs)
        if debug:
            # Read the rider's per-request latent diagnostics AFTER generation so
            # we can see whether the learned stop fired (end=stop) or ran to the
            # cap (end=cap), and the max stop-prob reached. Best-effort; never
            # breaks serving. Non-streaming responses only (streaming returns a
            # generator here, but the stats are still populated on the runner).
            # Use print (not logger) -- the addon logger does not propagate to the
            # API-server stdout, so logger.info is invisible in the serve log.
            try:
                stats_ranks = await engine.collective_rpc(_rpc_latent_stats)
                stats = next((s for s in stats_ranks if s), {})
                items = list(stats.items())[-4:]
                if not items:
                    print("[ds4-debug] NO latent stats (rider not armed / no "
                          "latent steps injected!)", flush=True)
                for rid, s in items:
                    print(f"[ds4-debug] req {rid} -> steps={s.get('steps')} "
                          f"stop_step={s.get('stop_step')} "
                          f"max_p={s.get('max_p', 0.0):.4f} end={s.get('end')}",
                          flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[ds4-debug] stats fetch failed: {type(e).__name__}: {e}",
                      flush=True)
        return result

    return _patched


def install_serving_hook(injector) -> bool:
    """Monkeypatch OpenAIServingChat._create_chat_completion (API-server process).

    Idempotent. Returns True if patched (or already patched), False if the
    serving class could not be imported (e.g. offline / non-serve process, where
    this is a harmless no-op).
    """
    global _PATCHED
    if _PATCHED:
        return True
    try:
        # Fork path (restructured): chat_completion/serving.py.
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
    except Exception:  # noqa: BLE001
        try:
            from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        except Exception as e:  # noqa: BLE001
            logger.debug("OpenAIServingChat not importable (%s); serving hook "
                         "not installed (fine outside the API server).", e)
            return False

    if not hasattr(OpenAIServingChat, "_create_chat_completion"):
        logger.warning("OpenAIServingChat has no _create_chat_completion; "
                       "serving hook not installed.")
        return False

    orig = OpenAIServingChat._create_chat_completion
    OpenAIServingChat._create_chat_completion = _make_patched_create(orig, injector)
    _PATCHED = True
    logger.info("DS4 closed-loop serving hook installed on OpenAIServingChat.")
    return True
