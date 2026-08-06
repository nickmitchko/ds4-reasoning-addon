"""Worker-side embedding-injection hooks (pristine-fork injection path).

DeepSeek-V4 in the SM120 fork uses hash-based MoE expert routing keyed on
``input_ids`` (``gate.tid2eid``), so every MoE layer requires token ids. vLLM's
native ``prompt_embeds`` path nulls ``input_ids`` when embeddings are supplied,
which crashes hash-MoE (``DeepSeek V4 hash MoE routing requires input_ids``).

So instead of ``prompt_embeds`` we inject by OVERWRITING the embedding-layer
output at chosen positions with a decoded latent vector, via a forward hook on
``model.model.embed_tokens``. Token ids flow normally (placeholders route fine),
and the overwrite lands at exactly the point vLLM would have used
``inputs_embeds`` -- before the hc_mult stream expansion. No fork edits.

These functions run INSIDE each TP worker (dispatched via ``LLM.apply_model``),
symmetric to the capture hooks in ``modules/vllm_forward_hook.py``. State lives
in worker-process globals. The injected tensor crosses the RPC boundary as a CPU
tensor and is moved to device inside the hook.
"""

import torch

# Worker-process globals (one set per worker).
_WORKER_INJECT_SPEC = None      # dict: {"positions": LongTensor, "embeds": Tensor}
_WORKER_INJECT_HANDLE = None


def _inject_hook(module, args, output):
    """Overwrite embedding rows at the configured positions.

    Fires only on the prefill pass (output rows span the injected positions).
    ``output`` is the embed_tokens result, shape (num_tokens, hidden); under TP
    it is already all-reduced (VocabParallelEmbedding reduces internally), so the
    same full vector is written on every rank -- a replace, not an add.
    """
    spec = _WORKER_INJECT_SPEC
    if spec is None:
        return output
    if not isinstance(output, torch.Tensor) or output.dim() != 2:
        return output

    positions = spec["positions"]
    num_tokens = output.shape[0]
    # Only act when this pass actually contains the target positions (prefill).
    if int(positions.max()) >= num_tokens:
        return output

    embeds = spec["embeds"].to(device=output.device, dtype=output.dtype)
    idx = positions.to(output.device)
    output = output.clone()
    output[idx] = embeds
    return output


def set_injection(model, positions, embeds):
    """Runs INSIDE each worker: install (or refresh) the injection hook.

    SINGLE-SEQUENCE ONLY. ``positions`` are absolute indices into the flattened
    ``(num_tokens, hidden)`` embed output; with a batched prefill those offsets
    would land in whichever sequence occupies those flat rows, not a chosen one,
    and only one spec exists per worker. The eager ``HookedVLLMEngine.generate``
    raises on batches >1, which keeps this path single-sequence. For batched
    injection use compile_safe=True with ``build_embeds_prompts_batch`` (each
    request self-describes its own embeds), not this hook.

    Args:
        positions: list[int] sequence positions to overwrite.
        embeds: CPU tensor (len(positions), hidden_size) of decoded vectors.
    """
    global _WORKER_INJECT_SPEC, _WORKER_INJECT_HANDLE
    _WORKER_INJECT_SPEC = {
        "positions": torch.as_tensor(positions, dtype=torch.long),
        "embeds": embeds,
    }
    if _WORKER_INJECT_HANDLE is None:
        embed = model.model.embed_tokens
        _WORKER_INJECT_HANDLE = embed.register_forward_hook(_inject_hook)
    return True


def clear_injection(model):
    """Runs INSIDE each worker: disable injection (leave hook installed, inert)."""
    global _WORKER_INJECT_SPEC
    _WORKER_INJECT_SPEC = None
    return True


def remove_injection(model):
    """Runs INSIDE each worker: remove the hook entirely."""
    global _WORKER_INJECT_HANDLE, _WORKER_INJECT_SPEC
    if _WORKER_INJECT_HANDLE is not None:
        try:
            _WORKER_INJECT_HANDLE.remove()
        except Exception:
            pass
    _WORKER_INJECT_HANDLE = None
    _WORKER_INJECT_SPEC = None
    return True
