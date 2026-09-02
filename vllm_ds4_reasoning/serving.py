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
requests a small ``max_tokens`` (e.g. a long-system-prompt client) can have its whole budget
spent thinking, yielding an empty text block (stop_reason=length). The hook
raises each request's output budget to at least this floor PLUS the request's
latent-step cap (each latent step bills one reserved accounting token), so the
floor bounds the answer alone; override per request via the
``x-ds4-min-output-tokens`` header, or set the floor to 0 to disable.
"""

import asyncio
import json
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
# (empty text block, stop_reason=length). Clients that send a modest
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
# Prepended system instruction that fixes the ANSWER REGISTER. Empty = disabled.
#
# WHY (measured 2026-08-19/20, n=8 and n=20 per arm on the SWE-bench-harness prompt):
# after the three latent-seam fixes the rider's output is token-clean and does not
# collapse, but the answer is written in the SCRATCHPAD voice -- "We need to design a
# harness... The user wants a detailed explanation" -- because the latent phase ends
# while the model is still mid-reasoning and the only place left to continue is the
# answer. Three training-side attempts failed to move this (two dspark Markov-head
# A/Bs, then H5 answer-register supervision, which converged to its own floor and
# measured 8.1 -> 9.5 markers/kw). It is not a head defect.
#
# scripts/probe_latent_depth.py showed the register tracks REASONING BUDGET: scratch
# markers fall monotonically with latent depth (12.6 -> 0.9 /kw over n=8..512) and
# with the learned-stop bar (11.8 -> 4.1 /kw over thr 0.5..0.99). Both of those
# knobs cost decode steps. This instruction buys most of the same effect for the
# price of a few prompt tokens (measured +0.0s serially, 11.0 -> 1.2 /kw), which is
# why it is the default half of the fix and a higher stop threshold is not.
#
# It is a SYSTEM message, prepended only when the request has no system message of
# its own -- overriding a client's system prompt would be a much bigger intervention
# than fixing a register. Set to the empty string to disable.
ENV_ANSWER_REGISTER = "VLLM_DS4_REASONING_ANSWER_REGISTER"
ANSWER_REGISTER_DEFAULT = (
    "Answer the user's question directly and in a clean expository register. "
    "Do not narrate your own reasoning process, and do not use phrases like "
    "\"we need to\", \"let's\", or \"the user asks\"."
)
# The literal reasoning-end token. DeepSeek's think tags are plain ASCII (unlike
# the FULL-WIDTH role markers) and tokenize to a single id (verified: 128822). The
# id is resolved from the ENGINE's tokenizer at arm time rather than hardcoded.
THINK_END_TOKEN = "</think>"

# --- streamed latent-tick signaling ------------------------------------------
# During the latent  thinking phase the runner emits no stream-visible tokens, so
# a client sees a dead connection until the first answer byte. To make the phase
# "active but empty" (approved UX: no visible text; the client's indicator is
# duration-driven on an OPEN thinking block), we synthesize per-latent-step ticks
# into the OpenAI chunk stream. The tick is carried in DeltaMessage.reasoning --
# NOT content -- so the Anthropic converter routes it into a thinking block and
# the client never surfaces it, reuses it, or counts it as an answer token. The
# value defaults to a zero-width space: non-empty (the Anthropic adapter drops
# empty reasoning deltas and never opens the block on ""), but renders as nothing.
# Set to "" for truly-empty reasoning deltas (OpenAI clients see `reasoning:""`;
# no thinking block opens on /v1/messages).
ENV_TICK_TEXT = "VLLM_DS4_REASONING_TICK_TEXT"
TICK_TEXT = os.environ.get(ENV_TICK_TEXT, "​")
# Poll cadence for the runner's per-request latent stats (latent steps run ~93 ms
# each, so 50 ms catches every step; fine under batching too).
_TICK_POLL_INTERVAL = 0.05
# How many consecutive polls with no runner entry for this prefix before we assume
# the phase never started (rider not armed / no latent) and stop the tick poller.
_TICK_MAX_SILENT_POLLS = 60


def _load_prometheus():
    """Import prometheus_client lazily; None when unavailable (offline processes
    that import this module but never serve, where /metrics does not exist)."""
    global _prom_client
    if _prom_client is None:
        try:
            import prometheus_client
            _prom_client = prometheus_client
        except Exception:  # noqa: BLE001 -- not a serve process
            _prom_client = False
    return _prom_client or None


_prom_client = None
LPS_COUNTER = None
LPS_HISTOGRAM = None
_LPS_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)


def _init_lps_metrics():
    """Register the LPS metrics on the default REGISTRY (where this serve exposes
    /metrics). Idempotent; swallows AlreadyRegisteredError on addon reloads."""
    global LPS_COUNTER, LPS_HISTOGRAM
    if LPS_COUNTER is not None:
        return
    prom = _load_prometheus()
    if prom is None:
        return
    if LPS_COUNTER is None:
        try:
            LPS_COUNTER = prom.Counter(
                "vllm:ds4_latent_steps_total",
                "Latent thinking steps executed by the closed-loop rider.",
            )
        except Exception:  # noqa: BLE001 -- already registered
            LPS_COUNTER = None
    if LPS_HISTOGRAM is None:
        try:
            LPS_HISTOGRAM = prom.Histogram(
                "vllm:ds4_latent_steps_per_request",
                "Latent thinking steps per request (closed-loop rider).",
                buckets=list(_LPS_BUCKETS),
            )
        except Exception:  # noqa: BLE001
            LPS_HISTOGRAM = None
    if LPS_COUNTER is None and LPS_HISTOGRAM is None:
        _prom_client = False  # registration failed; stop retrying


def _record_lps(stats_entry):
    """Increment the LPS metrics for THIS request's latent steps, if any.

    ``stats_entry`` is one runner {req_id -> {steps, ...}} hit (or None); the
    caller passes only the entry matched for this request. Never raises: a metrics
    hiccup must not break serving.
    """
    if not stats_entry:
        return
    steps = int(stats_entry.get("steps") or 0)
    if steps <= 0:
        return
    _init_lps_metrics()
    try:
        if LPS_COUNTER is not None:
            LPS_COUNTER.inc(steps)
        if LPS_HISTOGRAM is not None:
            LPS_HISTOGRAM.observe(steps)
    except Exception:  # noqa: BLE001
        pass


async def _record_lps_for_request(engine, internal_prefix):
    """Fetch the runner stats once and record LPS for the one matching request."""
    try:
        stats_ranks = await engine.collective_rpc(_rpc_latent_stats)
    except Exception:  # noqa: BLE001 -- never break serving
        return
    stats = next((s for s in stats_ranks if s), {})
    mine = {k: v for k, v in stats.items()
            if k.startswith(internal_prefix)}
    if not mine:
        return
    _record_lps(next(iter(mine.values())))


def _serialize_tick_chunk(rid, created, model, tick_text=TICK_TEXT):
    """Serialize one latent tick as an OpenAI streaming chunk string.

    ``delta.reasoning`` (not ``content``) so it lands in the thinking block on the
    Anthropic side. No ``usage``, no ``token_ids``, no ``logprobs``: the tick adds
    nothing to the client's billed/context token counts -- the real count ships in
    the engine's final usage chunk and already includes latent pad tokens once.
    """
    data = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"reasoning": tick_text},
                "logprobs": None,
                "finish_reason": None,
                "stop_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(data)}\n\n"


async def _wrap_with_latent_ticks(upstream, engine, internal_prefix):
    """Merge synthesized latent ticks into a streaming chunk generator.

    ``upstream`` yields serialized SSE chunks (from chat_completion_stream_generator
    or an equivalent). A background task polls the runner's per-request latent stats
    (keyed with the ``internal_prefix`` = "chatcmpl-<request_id>-") and enqueues a
    tick each time ``steps`` advances; the main loop drains queued ticks ahead of
    each upstream chunk and yields them as empty-but-active thinking deltas. The
    poller is bounded by the upstream's lifetime and stops early once stats mark the
    phase done. Never raises: any poll/queue hiccup just skips a tick.
    """
    queue: asyncio.Queue[None] = asyncio.Queue()
    stop_poll = asyncio.Event()
    last_steps = 0

    async def _poll():
        nonlocal last_steps  # mutate the enclosing step counter across polls
        silent = 0
        while not stop_poll.is_set():
            try:
                stats_ranks = await engine.collective_rpc(_rpc_latent_stats)
            except Exception:  # noqa: BLE001 -- transient RPC failure; retry
                await asyncio.sleep(_TICK_POLL_INTERVAL)
                continue
            stats = next((s for s in stats_ranks if s), {})
            mine = {k: v for k, v in stats.items()
                    if k.startswith(internal_prefix)}
            if not mine:
                silent += 1
                if silent >= _TICK_MAX_SILENT_POLLS:
                    return
                await asyncio.sleep(_TICK_POLL_INTERVAL)
                continue
            silent = 0
            s = next(iter(mine.values()))
            steps = int(s.get("steps") or 0)
            while steps > last_steps:
                last_steps += 1
                queue.put_nowait(None)
            if s.get("end") is not None:
                return
            await asyncio.sleep(_TICK_POLL_INTERVAL)

    first_id = None
    first_created = None
    first_model = None
    poller = asyncio.create_task(_poll())
    try:
        async for chunk in upstream:
            if first_id is None and chunk.startswith("data:"):
                data_str = chunk[5:].strip().rstrip("\n")
                if data_str == "[DONE]":
                    yield chunk
                    continue
                try:
                    obj = json.loads(data_str)
                    first_id = obj.get("id")
                    first_created = obj.get("created")
                    first_model = obj.get("model")
                except Exception:  # noqa: BLE001 -- malformed chunk; pass through
                    pass
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if first_id is None:
                    continue
                yield _serialize_tick_chunk(first_id, first_created, first_model)
            yield chunk
    finally:
        stop_poll.set()
        poller.cancel()
        # Record LPS once the stream is fully consumed: only now does the runner's
        # stat entry hold the completed step count. No-op when stats are absent.
        await _record_lps_for_request(engine, internal_prefix)

# --- per-request HTTP HEADERS -----------------------------------------------
# Clients that can set headers but not custom body fields (notably header-only clients
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
HDR_ANSWER_REGISTER = "x-ds4-answer-register"  # 0/1 -> answer-register system msg

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

    Enables clients that can only set headers (e.g. header-only clients via
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


def _apply_answer_register(request, raw_request, instr) -> None:
    """Add the answer-register instruction as a system message (ENV_ANSWER_REGISTER).

    APPENDS to an existing system message rather than skipping it. Skipping would be
    the more conservative choice, but it would make this knob inert for the client
    that motivated it: a header-only client always sends a large system prompt, so a
    "no system message" guard would fire on the measurement harness and never in
    production. Appending a two-sentence register note to the end of a system prompt
    leaves the client's own instructions in force and ahead of ours.

    Skipped when ``instr`` is empty (knob off), when ``x-ds4-answer-register`` is
    falsey (per-request opt-out), or when ``messages`` is missing / not a list (e.g. a
    completions-shaped request). Best-effort like the header path: an unrecognized
    request shape is left alone rather than failed. Runs BEFORE the template renders.

    A system message whose ``content`` is a content-part LIST (the Anthropic and
    newer OpenAI shape) gets an extra text part appended rather than a string concat,
    since ``str + list`` would raise and a replaced list would drop the client's
    prompt entirely.
    """
    if not instr:
        return
    headers = getattr(raw_request, "headers", None)
    if headers is not None:
        hv = headers.get(HDR_ANSWER_REGISTER)
        if hv is not None and not _truthy(hv):
            return
    msgs = getattr(request, "messages", None)
    if not isinstance(msgs, list):
        return

    def _role(m):
        return m.get("role") if isinstance(m, dict) else getattr(m, "role", None)

    def _text(m):
        c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        return ""

    # IDEMPOTENT: the patched handler can run more than once for one logical request
    # (client retry, streaming re-entry), and without this the note stacks up in the
    # system prompt on each pass.
    if any(instr in _text(m) for m in msgs if _role(m) == "system"):
        return

    try:
        out = list(msgs)
        idx = next((i for i, m in enumerate(out) if _role(m) == "system"), None)
        if idx is None:
            out.insert(0, {"role": "system", "content": instr})
        else:
            m = out[idx]
            cur = m.get("content") if isinstance(m, dict) else getattr(
                m, "content", None)
            if isinstance(cur, str):
                new = cur.rstrip() + "\n\n" + instr
            elif isinstance(cur, list):
                new = list(cur) + [{"type": "text", "text": instr}]
            else:
                # Unknown content shape -- add a separate system message instead of
                # guessing how to concatenate, so the client's prompt is untouched.
                out.insert(idx + 1, {"role": "system", "content": instr})
                request.messages = out
                return
            if isinstance(m, dict):
                out[idx] = {**m, "content": new}
            else:
                out[idx] = m.model_copy(update={"content": new}) if hasattr(
                    m, "model_copy") else m
                if out[idx] is m:      # could not copy -> don't mutate the original
                    out.insert(idx + 1, {"role": "system", "content": instr})
        request.messages = out
    except (AttributeError, ValueError, TypeError):  # frozen / validated field
        logger.warning("DS4 answer-register instruction could not be applied.")


def _apply_output_floor(request, raw_request, floor_default,
                        latent_default) -> None:
    """Raise the request's output-token budget to at least ``floor`` so the
    latent reasoning phase can't consume the whole budget inside <think> and
    leave no room for the answer (the failure mode: a modest max_tokens
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
        # Answer-register instruction (see ENV_ANSWER_REGISTER). Default ON: the
        # rider's measured failure mode without it is an answer in scratchpad voice.
        "answer_register": os.environ.get(ENV_ANSWER_REGISTER,
                                          ANSWER_REGISTER_DEFAULT),
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
            # how header-only clients -- which set headers but not custom body fields --
            # controls the reasoning head, including over /v1/messages.
            _apply_request_headers(request, raw_request)
            _apply_output_floor(request, raw_request, cfg["min_output_tokens"],
                                cfg["max_latent"])
            _apply_answer_register(request, raw_request, cfg["answer_register"])
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
        # Latent tick stream + LPS metrics -- per-request, best-effort; never break
        # serving. Streaming returns the chunk generator UNDRAINED (generation has
        # not started here), so ticks pace the latent phase via the runner-stats
        # poller and LPS is recorded on completion inside the wrapper; the
        # non-streaming response is already final, so record LPS immediately.
        # The external request id is deterministic pre-stream: it is
        # ``chatcmpl-{_base_request_id(raw, req.request_id)}``, and the runner
        # keys latent stats by ``f"{external_id}-{8hex}"`` (input_processor).
        internal_prefix = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}-"
        if request.stream:
            if result is not None:
                result = _wrap_with_latent_ticks(result, engine, internal_prefix)
        else:
            await _record_lps_for_request(engine, internal_prefix)
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
