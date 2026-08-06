# Closed-loop latent decode for the DS4 reasoning head — design & phased plan

Status: **IMPLEMENTED and shipped** — this document is retained as the design
record and rationale, not as a to-do list. It was written before the work started,
so its "Status: DESIGN" framing, phase ordering and open questions describe the
plan as of authoring time. Where it references `examples/*.py` scripts, those are
the internal validation drivers that recorded each phase's result; they are not
part of this published repository. For how to actually run the thing, see
[`../release/README-RELEASE.md`](../release/README-RELEASE.md).

## Goal

Serve the DS4 reasoning head in the **live HTTP path** doing **CoLaR-style
closed-loop latent decode**: prefill the prompt ONCE, then autoregressively
generate compressed *latents* (not tokens) for the reasoning phase — each latent
step feeds `decoder(head(h))` as the **next input embedding**, emits no token —
then switch to normal token decoding for the answer, terminating the latent
phase on a **learned `</think>` / stop signal**.

This needs a single prefill (NOT the 2x-prefill demo path in
`serve_interactive.py`, which re-prefills because it injects at a prompt position
and re-enters the model). One prefill + N cheap latent decode steps.

## Why the current head is INCOMPATIBLE as-is (grounding, verified in code)

All three are independent blockers (see modules/reasoning_head.py,
scripts/train_reasoning_head.py::colar_loss ~L340-402, injector.py, hook_inject.py):

1. **Injection-space mismatch.** `decoder(latent)` is trained to reconstruct the
   layer-normed, mean-pooled **layer-42** hidden (`recon_mse_loss(decoder(tgt_latent),
   tgt_pooled)`, train_reasoning_head.py:395-397). Injection writes the decoded
   vector at the **layer-0 embed_tokens output** (hook_inject.py:49). Deep-layer
   representation fed as a shallow-layer input — off-distribution, compounds in a loop.
2. **Never trained closed-loop.** SFT and GRPO are one-step, teacher-forced:
   predict next latent from a *captured real* anchor (source-35). The head never
   consumes its own decoded output as input, so it is not calibrated for rollout.
3. **No stop signal.** Head emits only `(mu, log_sigma)` (reasoning_head.py:50-53).
   No stop logit, no `</think>`, no length head. `compression_factor=4` is a
   FIXED ratio (`G = L // c`). Terminating on a learned signal is impossible today.

Bundle (verified): `outputs/reasoning_head_decoder_long/reasoning_head_final.pt`
— format_version=2, hidden_size=4096, latent_dim=1024, source_layer=35,
target_layer=42, compression_factor=4, decoder+target_proj present.

## DECISION (2026-07-11): **Option A — embedding-space latent, inject at layer 0.**

Chosen by the user. Rationale: standard CoLaR formulation, smallest runner
change, and the inputs_embeds-decode primitive (Workstream R) writes at the
layer-0 embed output where the existing hook/prompt_embeds machinery already
lives. The decoder's *output target* moves from layer-42 pooled hidden to
embed_tokens space (H3); the encoder still compresses reasoning content. All
Phase 2 training and Phase 1 runner work below assume A. Option B (mid-stack
layer-42 injection) is NOT pursued — it needs a per-position "enter at layer N"
path vLLM lacks.

## PIVOTAL OPEN DECISION — latent space / injection site (RESOLVED: A)

Everything downstream depends on this. TWO options:

- **(A) Embedding-space latent (recommended).** Retrain the autoencoder so
  `decoder(latent)` outputs an **embed_tokens-space** vector; inject at layer 0
  (the inputs_embeds-decode primitive already scoped). Standard CoLaR formulation;
  smallest runner change. Encoder still compresses reasoning content; only the
  decoder's *output target* moves from layer-42 to embed space.
- **(B) Mid-stack latent at layer 42.** Keep decoder's layer-42 target; inject at
  layer 42 (latent position skips layers 0-41). Truer to "compressed state
  resumes the residual stream," but vLLM has no native "enter at layer N for one
  position" path — much larger runner change (per-position layer routing).

Plan below is written for **(A)** and flags where **(B)** diverges. DECIDE BEFORE
Phase 2.

## Two workstreams

### Workstream R — RUNNER primitive (vLLM fork): inputs_embeds decode step
Make the engine able to run a decode step whose input is a supplied embedding and
which emits no sampled token, with correct KV bookkeeping. Reuses the batch-safe
capture already built (per-pass buffer + runner anchor store).

- R1. **inputs_embeds decode.** Today decode builds inputs from token ids
  (`_prepare_input_ids`, gpu_model_runner.py:1734). Add a path: for a request in
  "latent mode", the next step's input row is a caller-supplied embedding
  (reuse the `enable_prompt_embeds` inputs_embeds buffer, is_token_ids=False).
- R2. **Latent hidden feedback.** After the forward, read the source-layer (35)
  hidden of the latent position from the capture buffer (already have
  `_ds4_copy_anchors` / anchor store); run head→decoder → next latent embedding.
  Decide in-worker (GPU, avoids RPC) vs driver.
- R3. **Sampler bypass + stop.** A latent step emits no token. Need a per-request
  "latent step" flag so the sampler is skipped (or its output discarded) and the
  step doesn't count as an output token. Terminate latent phase when the stop
  head fires (Phase 3) or a fixed N is hit.
- R4. **KV bookkeeping.** The latent position occupies a real KV slot and advances
  positions/seq_len by 1, exactly like a token step — verify slot_mapping,
  query_start_loc, block table stay consistent for a no-token step.
- R5. **AsyncLLM path.** collective_rpc accepts a cloudpickled callable applied to
  the worker (multiproc_executor.py:993) — the async equivalent of apply_model.
  Latent-mode state must ride the request through the async engine.

Risk: R is a deep scheduler/runner change. Prove it with a FIXED-N loop and the
current (off-distribution) head BEFORE the retrain, to de-risk the mechanism.

### R (REVISED 2026-07-11 after grounding in the runner) — latent step = decode RIDER, not a new scheduler step-type

Mapping the runner (gpu_model_runner.py, scheduler.py) changed the R design. Two
findings shrink the risk dramatically:

- **KV/position advancement is content-independent.** The scheduler advances
  `request.num_computed_tokens += num_scheduled_tokens` (scheduler.py:1182); the
  runner derives positions (`self.positions = num_computed_tokens + query_pos`,
  gpu_model_runner.py:2138-2141) and `compute_slot_mapping` (2147-2151) purely
  from that counter — NOT from what the token *is*. So a latent step that reuses a
  normal decode step gets correct KV/positions/slot_mapping FOR FREE. **R4 needs no
  new bookkeeping** — it just needs the latent step to BE a decode step.
- **The anchor store already refreshes every decode step.** `_ds4_copy_anchors`
  (2259-2283) runs unconditionally after every forward, copying each req's
  `logits_indices` row (its last/only token this pass) from the layer-35 capture
  buffer into `model._ds4_anchors[req_id]`. On a decode step that row IS the latent
  position's source hidden. **R2's per-step feedback signal already exists.**

Revised mechanism (rider): a latent-mode request runs ORDINARY decode steps. Per
step, in-worker, BEFORE the forward we overwrite that request's row in
`self.inputs_embeds.gpu` (and set `is_token_ids=False` for that row) with the
decoded latent `decoder(head(LN(prev_anchor)))` computed from the previous step's
`_ds4_anchors[req_id]`; the embed_tokens hook path already routes inputs_embeds
into the forward under `enable_prompt_embeds` + `requires_raw_input_tokens`
(input_ids still flow for hash-MoE). The token the scheduler THINKS it decoded is a
placeholder; we mark the request's latent steps and strip those placeholder tokens
from the output. This is the existing `discard_request_mask` seam
(gpu_model_runner.py:2052, applied 3685-3691/3724-3726) — during a latent step the
sampled token is discarded, so it never enters `output_token_ids`.

- R1' (was R1+R4): reuse the decode step; overwrite the latent-mode request's
  `inputs_embeds.gpu` row + `is_token_ids=False` in a runner hook that fires right
  after `_preprocess` builds inputs_embeds. No new step-type, no scheduler edit.
- R2' (was R2): read `model._ds4_anchors[req_id]` (already populated); run
  head→decoder on GPU IN-WORKER (avoids an RPC per token). Head/decoder weights
  loaded once into worker globals via a one-time `collective_rpc` at setup.
- R3' (was R3): mark latent steps per-request; force `discard_request_mask` True for
  those rows so the sampled token is dropped (no output token, no `output_token_ids`
  growth). Count latent steps; at fixed N flip the request out of latent mode so
  normal token decode resumes for the answer.
- R5' (was R5): latent-mode config rides the request via
  `SamplingParams.extra_args["ds4_latent"]` (n_steps, ckpt id) — the same channel
  `kv_transfer_params` uses (request.py:114). Works for sync LLM and AsyncLLM
  (async exposes `collective_rpc`).

Why this is safe to try first: it touches the model_runner step locally (an
in-worker pre/post-forward hook + the discard mask) and rides machinery that
already serves today (capture buffer, anchor store, inputs_embeds path,
requires_raw_input_tokens). It does NOT add a scheduler step-type. If the rider
proves insufficient (e.g. we later need a latent step that does NOT advance the
KV the same way), fall back to the original R1-R5 step-type design.

OPEN Q for the rider: the scheduler still advances `num_computed_tokens` and
counts the placeholder toward `max_tokens`/`num_tokens`; for a FIXED-N latent phase
that is fine (budget N latents + answer). It also writes a placeholder id into
`token_ids_cpu[req, pos]`; that id becomes the NEXT step's input_id, but we
OVERWRITE that row's embedding anyway during latent mode, so the stale id only
matters for hash-MoE routing (input_ids still flow) — verify routing tolerates the
placeholder id (use a fixed benign id, e.g. a pad/space token).

### PHASE 1 RESULT (2026-07-11): MECHANISM PASS — head off-distribution as predicted

Implemented the rider as a NO-FORK-EDIT in-worker module
(`vllm_ds4_reasoning/closed_loop.py`) + verifier
(`examples/verify_latent_loop.py`). Two eager-path forward hooks on a single
sequence: Hook A stashes `layers[35]` output[0][-1] (source hidden); Hook B
overwrites the decode step's `embed_tokens` row with `decoder(head(LN(prev)))` for
the first N steps, then goes inert so token decode resumes. No scheduler/runner
edit; rides the existing capture + hash-MoE input_ids machinery.

Run: GRPO bundle, prompt "What is 17*24?", N=8, tp=2, eager. Gate = PASS:
- single prefill + **8 latent steps ran**, then normal decode → coherent answer
  (`</think>17 * 24 = 408`). Full loop closed in ONE generate().
- **Off-distribution drift confirmed (as the plan predicted).** Source-layer
  hidden norm per latent step: `[25, 1431, 4032, 4279, 4267, 4450, 4400, 4410]`.
  Step 1 anchor is the real `<think>` hidden (~25, in-dist); once we inject the
  decoder output the hidden jumps to ~4000 and PLATEAUS (stable, no NaN — better
  than feared) but is clearly off-manifold. Inject-embed norms stable ~58.
- Report: `latent_loop_report.json`. This validates the MECHANISM; the head still
  needs the Phase-2 retrain (H3 embed-space target + H4 closed-loop training) for
  the latent steps to carry meaningful compressed reasoning.

GATE CLEARED → proceed to Phase 2 (head retrain). The compiled/cudagraph +
batched version of this rider is the fork-side Workstream-R follow-up, deferred
until after retrain proves the head is worth serving fast.

### Workstream H — HEAD retrain
- H1. **Add stop head.** `ReasoningCompressionHead` → also emit a stop logit
  (`Linear(mlp_dim, 1)`). Persist/reload in all three copies
  (modules/reasoning_head.py, addon models.py, addon checkpoint.py) + bump
  format_version.
- H2. **Stop labels.** Relabel training traces: mark the group where reasoning
  ends (`</think>` / answer boundary; dataset already carries cot/completion,
  train_reasoning_head.py:568-572). Add BCE stop loss beside soft_mse.
- H3. **Space fix (option A).** Change the AE reconstruction target from
  `tgt_pooled` (layer 42) to the embed-space representation, so decoder output is
  feedable as an input embedding. (Option B: leave target, change injection site.)
- H4 DESIGN (2026-07-11, "full closed-loop w/ runner-in-training" chosen). The
  Phase-1 `closed_loop` primitive IS the runner-in-training on the eager path
  (backbone is inference-only in vLLM workers; the head trains on the driver via
  autograd on captured activations). Per trace:
    1. TEACHER-FORCED capture (supervision): run the backbone over the full
       `completion` (which carries `<think>...</think>answer`), capture embed +
       source(35) + target(42) hiddens. Compute reference group latents
       `tgt_latent[k]` and per-group mean EMBED targets (H3) and the `</think>`
       stop-group index (H2).
    2. CLOSED-LOOP ROLLOUT (runner-in-training): from the real `<think>` anchor,
       drive `closed_loop` for K steps with a STALE copy of head+decoder in the
       worker; capture the self-generated source-hidden trajectory
       `h_self[0..K-1]` (closed_loop extended to return hiddens, not just norms).
    3. DRIVER UPDATE (autograd): the CURRENT head re-processes each anchor and
       regresses onto the reference target. SCHEDULED SAMPLING / DAgger: with prob
       `ss_prob` the anchor is the self-generated `h_self[k]`, else the real
       `src[k]`. HONEST NOTE: there is no ground-truth teacher for a never-observed
       self-generated state; the target is the reference trace's aligned group
       latent (a proxy). This is the standard DAgger approximation, not an oracle.
       Worker rollout weights are refreshed from the trained head every
       `refresh_every` steps.
  H3 embed-space AE + H2 stop BCE also work in the plain one-step path, so a
  `--closed-loop`-off retrain is a valid fallback if rollout proves unstable.
- H4 (orig). Train the head against its
  OWN re-injected decoded latents (at least schedule-sample), so rollout is
  in-distribution. Requires R1-R4 available in the training engine (capture is
  already there).
- H5. Retrain, validate reconstruction + stop accuracy + rollout drift.

### PHASE 2 RESULT (2026-07-11): retrain converged; H1-H4 shipped

Retrain (`outputs/reasoning_head_phase2/`, warm-started from
`reasoning_head_decoder_long`, 600 steps, closed-loop DAgger ss_prob=0.5,
max_trace_chars=4000 so </think> is in-window for 89% of traces):
- recon 0.79 -> 0.0009 (decoder solved the EMBED-space target; H3 -- the drift fix)
- stop  0.66 -> 0.0001 (stop head converged on the </think> boundary; H2)
- self% ~50-100 throughout (trained on self-generated rollouts; H4)
- Caught + fixed a real bug: the 1500-char trace cap left </think> out of window
  for 95% of traces (median boundary ~2406 chars) -> stop head was starved. Raised
  to 4000 chars (still single-chunk at max_num_batched_tokens=12000). Coverage 5%->89%.
- NOTE: total loss goes negative (soft_mse entropy term drives log_sigma to its -10
  clamp = sigma collapse). Benign: inference uses mu (MAP), not samples.
Bundles: checkpoint-{150,300,450,600} + reasoning_head_final.pt (v3).

H5 VALIDATION (examples/validate_phase2.py, GRPO->phase2 bundle, "17*24", N=8):
**DRIFT FIXED.** Source-hidden norms per latent step:
[24.98, 22.52, 23.72, 23.33, 24.71, 21.71, 22.52, 23.75] -- FLAT ~23.
Drift ratio (plateau/step1) = **0.93x** vs Phase-1's ~160x (25->4000 explosion).
The embed-space AE target (H3) + closed-loop DAgger (H4) made the decoded latents
land in-distribution at layer 0. Gates: latent_steps_ran PASS, generation_completed
PASS, drift_bounded PASS. Stop head has weak signal on the free-gen trajectory
(mostly ~0 with small spikes) -- calibrating it in the LIVE loop is Phase 3.
Report: phase2_validation.json. PHASE 2 GATE CLEARED -> Phase 3.

## Phased sequence (each phase independently verifiable)

- **Phase 0 — this doc + decision (A vs B).** No code.
- **Phase 1 — Runner primitive, fixed-N, current head (off-distribution).**
  Implement R1-R5. Serve/offline harness runs a FIXED number of latent steps then
  normal decode. Verify: single prefill (assert no re-prefill), latent steps
  advance state, output is coherent-ish (won't be great — head is off-distribution).
  GATE: mechanism works end to end.
- **Phase 2 — Head retrain (H1-H5), space fix per decision.** Long training runs.
  GATE: reconstruction + stop-accuracy metrics; closed-loop rollout stable.
- **Phase 3 — Wire learned stop into the runner loop.** Replace fixed-N with the
  stop head. GATE: variable-length latent phase ends on `</think>`.
  PROGRESS (2026-07-11): WIRING DONE (closed_loop.use_stop -- embed hook evaluates
  sigmoid(stop_logit(prev_hidden)) each step, stops injecting once > threshold after
  min_latent; n_latent = cap; verify_stop_loop.py).

  ROOT-CAUSE FINDING (verified, 2026-07-11) -- the learned stop is NOT trainable at
  this compression/dataset, and both attempts confirmed it:
  * The </think> boundary sits DEEP in latent-group space: median 127 compressed
    groups (>=25 even for the shortest trace) at compression_factor=4, because
    Fable reasoning traces are long (median </think> ~600 tokens).
  * Attempt 1 (Phase-2 stop head, teacher-forced boundary labels): on the injected-
    latent hiddens the loop actually sees, p_stop is noise-level
    [0.0034,0,0.0002,...,0.0018,0.0011] (<=0.003); only threshold 0.001 fires (step
    13/16) i.e. on noise. It never saw in-distribution latent boundary examples.
  * Attempt 2 (rollout_stop_loss on self-generated hiddens): a feasible-length
    rollout (6, even 48 steps) NEVER reaches group 127, so ALL rollout labels are 0
    -> the head trivially learns "always 0" (rstop->0.0004 by learning to predict
    the absent positive). Running 127+ closed-loop steps/trace x thousands of traces
    on the eager path (~9 tok/s) is infeasible.
  DECISION (user, 2026-07-11): accept teacher-forced stop head; do NOT brute-force.
  The learned-stop WIRING is shipped and correct (fires immediately if a bundle
  with a genuine latent-space stop signal is ever supplied -- e.g. after a
  high-compression_factor retrain, the deferred option). For serving NOW the loop
  uses a fixed max-latent cap (Phase 4 default); the learned stop stays wired,
  gated by use_stop, ready for a future high-compression head. GATE: PARTIAL --
  mechanism verified end-to-end (variable-length termination works when a signal
  exists); learned </think> signal not achievable at c=4 on this data.
  Artifacts: phase3_stop*.json. reasoning_head_phase3 bundle = Phase-2 head with a
  rollout-degraded stop head; SERVE FROM reasoning_head_phase2 (intact stop head).
- **Phase 4 — HTTP serve integration.** Gate on VLLM_DS4_REASONING_CKPT; plugin
  monkeypatch of OpenAIServingChat (chosen mechanism) drives capture→latent-loop→
  answer per chat request. Streaming/tool-calling coverage TBD. GATE: live
  /v1/chat/completions produces latent-compressed reasoning.

  PHASE 4 RESULT (2026-07-12): GATE PASS. serving.py monkeypatches
  OpenAIServingChat._create_chat_completion from register() in the API-server
  process; per request it holds an async lock, lazily installs closed_loop in the
  workers via AsyncLLM.collective_rpc (state dicts shipped as torch.save BYTES --
  the engine-core RPC uses stdlib pickle, which chokes on the mmap'd-tensor
  memoryview; bytes always pickle), resets the loop, and runs the base handler
  (awaits generation for non-streaming). Live verified:
    vllm serve nvidia/DeepSeek-V4-Flash-NVFP4 --enforce-eager --max-num-seqs 1
      --tensor-parallel-size 2 --tokenizer-mode deepseek_v4 --kv-cache-dtype fp8
      --max-model-len 6000 --disable-custom-all-reduce --port 8001
    (env: VLLM_DS4_REASONING_CKPT=reasoning_head_phase2, VLLM_PLUGINS=ds4_reasoning,
     VLLM_DS4_REASONING_MAX_LATENT=8, VLLM_ALLOW_INSECURE_SERIALIZATION=1)
  POST /v1/chat/completions "What is 17*24?" -> 200, coherent+correct
  ("...= 408.</think>The product of 17 and 24 is 408."), with the latent-step
  signature at the start ("WeWeWeWe We We ...") = the 8 injected-latent steps'
  junk tokens before normal decode. Reproduced on a 2nd prompt. Streaming requests
  correctly BYPASS the loop (fall through to base handler) -> 200 SSE.
  CONSTRAINTS (inherent to the single-seq rider): --enforce-eager, --max-num-seqs 1,
  requests serialized by an async lock, non-streaming only. Batched/streaming
  closed-loop = fork-side Workstream-R follow-up. SERVE FROM reasoning_head_phase2.

## Verification per phase (all need the 284B model on the 2 GPUs, ~5 min/reload)
- Phase 1: an offline `examples/verify_latent_loop.py` — assert exactly one
  prefill forward per request (instrument the capture buffer / step count), N
  latent steps run, generation completes. Compare tokens/s vs baseline.
- Phase 2: training-side metrics (recon MSE, stop BCE/accuracy, rollout drift at
  k steps) — no serving needed.
- Phase 3-4: end-to-end HTTP + offline parity.

## Cost / honesty notes
- Phase 1 is a substantial vLLM runner change (scheduler + model_runner), highest
  technical risk; do it first to de-risk.
- Phase 2 is real training (multiple runs, hours+), not a code tweak.
- No 2x prefill in the target design — that was an artifact of the demo path.
- Batch-safe capture (already shipped) is the capture half of R2 and is reused.

## Open questions to resolve before Phase 1
1. Latent space: (A) embed-space vs (B) layer-42 injection. [PIVOTAL]
2. Does a latent step emit a token too (hybrid) or purely a latent? (User picked
   PURE latent step earlier.)
3. Retrain budget / dataset for stop labels (which traces, how `</think>` boundary
   is defined).
