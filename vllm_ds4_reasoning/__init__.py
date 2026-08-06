"""DS4 reasoning-head addon for vLLM.

An opt-in vLLM *general plugin*: installing this package registers a
``vllm.general_plugins`` entry point (see pyproject) that vLLM loads in every
process, including TP workers. The plugin is a NO-OP unless a checkpoint is
provided, so it is safe to leave installed -- it only activates the reasoning
head when one exists.

Enable by pointing at a trained bundle:

    export VLLM_DS4_REASONING_CKPT=/path/to/reasoning_head_final.pt

The plugin-level injector loads on CPU; set ``VLLM_DS4_REASONING_DEVICE=cuda`` only
in single-process offline drivers that run the head themselves. Never set it for
``vllm serve`` -- see get_injector() for why that lands on cuda:0 in every process.

Then use the injector to feed the head's decoded latent into the DeepSeek-V4
backbone via vLLM's native ``--enable-prompt-embeds`` path:

    from vllm_ds4_reasoning import get_injector
    inj = get_injector()                       # loaded from the env checkpoint
    inj_vec = inj.hidden_to_injectable(h35)    # source-layer hidden -> hidden
    prompt = inj.build_embeds_prompt(token_ids, inj_vec)
    llm.generate(prompt)
"""

import logging
import os
from typing import Optional

from .checkpoint import ReasoningBundle, load_bundle
from .injector import ReasoningInjector
from .models import LatentDecoder, ReasoningCompressionHead

logger = logging.getLogger(__name__)

ENV_CKPT = "VLLM_DS4_REASONING_CKPT"
ENV_DEVICE = "VLLM_DS4_REASONING_DEVICE"

_injector: Optional[ReasoningInjector] = None
_registered = False

__all__ = [
    "register",
    "get_injector",
    "ReasoningInjector",
    "ReasoningBundle",
    "load_bundle",
    "ReasoningCompressionHead",
    "LatentDecoder",
]


def get_injector() -> Optional[ReasoningInjector]:
    """Return the process-global injector, loading it lazily from the env ckpt.

    Returns None when no checkpoint is configured (the addon stays dormant).
    """
    global _injector
    if _injector is not None:
        return _injector
    ckpt = os.environ.get(ENV_CKPT)
    if not ckpt:
        return None
    if not os.path.exists(ckpt):
        logger.warning("%s=%s does not exist; reasoning head disabled.",
                       ENV_CKPT, ckpt)
        return None
    # CPU by default -- do NOT touch the GPU here. register() runs in EVERY
    # process (API server, engine core, and every TP worker), and in the workers
    # it runs from init_worker, i.e. BEFORE GpuWorker.init_device() binds this
    # rank's device. A bare "cuda" therefore resolves to cuda:0 in all of them,
    # so each process pinned a ~660 MiB CUDA context + 152 MiB of head weights on
    # GPU 0 -- including TP rank 1, whose real device is cuda:1. That skewed the
    # memory profiler on rank 0 and, because kv_cache_utils levels every rank down
    # to the smallest block count, shrank the KV cache on BOTH ranks.
    #
    # These copies are dead weight on the serve path anyway: serving.py ships the
    # state dicts back to CPU (torch.save of .detach().cpu()) and the hot path runs
    # on the worker's own copy, which gpu_model_runner.ds4_setup_latent places on
    # next(model.parameters()).device. Offline drivers that do want a GPU injector
    # set VLLM_DS4_REASONING_DEVICE explicitly.
    device = os.environ.get(ENV_DEVICE) or "cpu"
    _injector = ReasoningInjector.from_checkpoint(ckpt, device=device)
    b = _injector.bundle
    logger.info(
        "DS4 reasoning head loaded from %s (hidden=%d latent=%d src_layer=%s "
        "tgt_layer=%s injectable=%s)",
        ckpt, b.hidden_size, b.latent_dim, b.source_layer, b.target_layer,
        b.injectable,
    )
    if not b.injectable:
        logger.warning(
            "Loaded head has no decoder; prefill injection is unavailable.")
    return _injector


def register() -> None:
    """vLLM general-plugin entry point.

    Called by ``vllm.plugins.load_general_plugins`` in every process. Must be
    safe to call repeatedly and cheap when dormant, so it only eagerly loads the
    checkpoint when one is configured.
    """
    global _registered
    if _registered:
        return
    _registered = True

    if os.environ.get(ENV_CKPT):
        # Warm the injector so a bad path/shape fails at startup, not mid-serve.
        inj = get_injector()
        # Phase 4: in the API-server process, monkeypatch the OpenAI chat handler
        # to drive closed-loop latent decode per request. No-op in engine/worker
        # processes (OpenAIServingChat not importable there) and when the head is
        # not injectable. Import lazily so a serving-API change can't break the
        # dormant/offline paths.
        if inj is not None and inj.bundle.injectable:
            try:
                from .serving import install_serving_hook
                install_serving_hook(inj)
            except Exception as e:  # noqa: BLE001
                logger.warning("DS4 serving hook not installed (%s); offline "
                               "capture/inject still works.", e)
        logger.info("DS4 reasoning-head plugin registered (active).")
    else:
        logger.debug(
            "DS4 reasoning-head plugin registered (dormant; set %s to enable).",
            ENV_CKPT,
        )
