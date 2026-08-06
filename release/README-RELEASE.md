# Serving the DS4 CoLaR Reasoning Head

Everything needed to serve the [CoLaR reasoning compression head](https://huggingface.co/nmitchko/deepseek-v4-Flash-CoLaR)
on a DeepSeek-V4-Flash backbone: pinned requirements, a serve script, and hardware
guidance.

The head makes the model **reason in a compressed latent space** instead of
spelling out a token-by-token chain of thought. The backbone prefills your prompt
once, then takes autoregressive *latent* steps — each step's input embedding is
`decoder(head(previous layer-35 hidden))`, not a token embedding — and a learned
stop head decides when reasoning is done, emits `</think>`, and lets the answer
decode as normal tokens.

- **Weights** — <https://huggingface.co/nmitchko/deepseek-v4-Flash-CoLaR>
- **Engine** — the DS4 SM120 vLLM fork, <https://github.com/nickmitchko/vllm-ds4-sm120> (branch `ds4-sm120-preview-dev`)
- **Addon** — this repository, <https://github.com/nickmitchko/ds4-reasoning-addon>

---

## Contents

| File | What it is |
|---|---|
| `requirements-serve.txt` | Exact pins this stack is served with. Ranges are not safe here. |
| `serve_ds4_reasoning.sh` | The serve entrypoint. Defaults are the recommended config. |
| `README-RELEASE.md` | This file. |

---

## Install

### 1. The engine (required — upstream vLLM will not work)

Upstream vLLM cannot serve this model. It needs the DS4 SM120 fork, which carries
the sm120 sparse-MLA path, DeepSeek-V4's hash-MoE routing, the per-layer MoE quant
dispatch, and the batched latent rider itself.

```bash
git clone https://github.com/nickmitchko/vllm-ds4-sm120.git && cd vllm-ds4-sm120
git checkout ds4-sm120-preview-dev

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0"

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e . --no-build-isolation      # a full build takes a while
```

### 2. Python dependencies

```bash
pip install -r requirements-serve.txt \
    --extra-index-url https://flashinfer.ai/whl/cu130/torch2.11
```

Two pins in there are load-bearing and worth repeating, because both fail in ways
that don't point at the cause:

- **The three `flashinfer-*` packages must share one version.** The sm120
  sparse-MLA kernel only exists in the 0.6.14 `jit_cache` layout. Skew gives you
  either a version-mismatch `RuntimeError` or
  `ModuleNotFoundError: No module named 'flashinfer.mla._sparse_mla_sm120'`, and
  an editable vLLM reinstall is what usually causes it — re-check after any
  `pip install -e`.
- **`kernels` must be `<0.15`.** Newer versions require an explicit
  version/revision on every `LayerRepository` call, which this `transformers`
  does not pass, so it's a hard error at import.

### 3. The reasoning addon

From the root of this repository:

```bash
pip install -e . --no-deps
```

Installing does nothing on its own — the plugin stays **dormant** until
`VLLM_DS4_REASONING_CKPT` is set, so it is safe to leave installed. If you'd
rather not install it, pass `ADDON_DIR=/path/to/ds4-reasoning-addon` to the serve
script and it goes on `PYTHONPATH` instead.

### 4. Weights

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download MJPansa/DeepSeek-V4-Flash-0731-NVFP4   # ~164 GiB
huggingface-cli download nmitchko/deepseek-v4-Flash-CoLaR       # ~136 MiB
```

---

## Serve

```bash
HEAD_BUNDLE=/path/to/reasoning_head_final.pt ./serve_ds4_reasoning.sh
```

### Requests must ask for thinking

```bash
curl localhost:8001/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "MJPansa/DeepSeek-V4-Flash-0731-NVFP4",
  "messages": [{"role": "user", "content": "Write a Python LRU cache."}],
  "chat_template_kwargs": {"thinking": true},
  "max_tokens": 2048
}'
```

**Without `chat_template_kwargs: {"thinking": true}` the output is garbage.** For
clients that can set headers but not body fields, `x-ds4-thinking: 1` does the
same thing, and works on `/v1/messages` too.

### Two startup behaviours that look like bugs

- **The first one or two requests after startup can come back as degenerate
  repetition** ("be be be") at low draft acceptance, even with the stop firing
  normally. It settles by itself and then stays correct. Send one throwaway
  request before trusting output, and don't lower `STOP_THRESHOLD` or force
  `SPEC_TOKENS=1` in response to it — neither is the cause.
- **An armed rider and a silently-dormant one produce identical server logs.**
  `DEBUG=1` is the only way to confirm injection is happening; it prints
  `steps=N stop_step=... max_p=... end=stop|cap` per completion, and says so
  explicitly when nothing was injected.

---

## GPU recommendations

The backbone is NVFP4 and lands at roughly **79 GiB of weights per GPU at TP=2**
(~158–164 GiB total, depending on base). The head itself is ~136 MiB and about 1%
of a decode step — the MoE dominates. So sizing is entirely about fitting weights
plus the fp8 KV cache.

### Verified configuration

| | |
|---|---|
| GPUs | 2× RTX PRO 6000 Blackwell Max-Q, 96 GiB each |
| Compute capability | sm120 (12.0) |
| TP | 2 |
| Context | 262,144 (443,012 fp8 KV tokens) — 524,288 also builds |
| `max_num_seqs` | 2 |
| `gpu_memory_utilization` | 0.95 |
| Throughput | 11.1 ms/token with the head + DSpark spec decode (1.40×) |
| Draft acceptance | 89–91% |

### Sizing guidance

| VRAM (total) | Verdict |
|---|---|
| **≥ 192 GiB** (2× 96, or 4× 48+) | **Recommended.** Long context (256k) with room for the KV cache. |
| 160–192 GiB | Workable. Weights fit; drop `MAX_MODEL_LEN` to 32k–64k so KV fits. |
| < 160 GiB | Not enough for NVFP4 weights at any context. |

Notes on scaling:

- **Blackwell (sm120/sm100) is the tested target.** Hopper and earlier are
  untested here and the sm120 sparse-MLA kernel path is specific to this fork.
- **`max_num_seqs` trades against context.** At 256k the fp8 KV cache only holds
  a couple of full-length sequences. Raising concurrency starves KV and either
  OOMs at startup or silently truncates usable context — raise it only alongside
  a lower `MAX_MODEL_LEN`.
- **If the engine fails its KV-cache check at startup**, lower `MAX_MODEL_LEN`
  first, then `MAX_NUM_SEQS`. `gpu_memory_utilization` has a narrow viable band
  in *both* directions — 0.95 is the tested value; 0.97 OOMs and lowering it can
  fail the KV check instead.
- **MoE backend: leave it alone.** On sm120 with this model's `swiglu_limit=10.0`,
  `FLASHINFER_CUTLASS` (the `auto` choice) is the only working option. `marlin`
  crashes with an illegal address as soon as a latent is injected; `trtllm`,
  `b12x` and both `cutedsl` variants refuse to start. There is nothing to tune.
- **`--tensor-parallel-size 1` needs a single GPU with ≥ 160 GiB** and is
  untested.

### Host RAM

Loading is what kills machines here, not serving. Reading a ~164 GiB checkpoint
fills the page cache, and on a 124 GB host `systemd-oomd` kills the process
**with no traceback** once user-slice pressure holds above 50% — which reads like
a model bug and isn't one (check `/var/log/syslog` for `systemd-oomd`). DSpark
makes it worse by re-reading all 48 shards for the drafter after the main model,
so the second pass starts with the cache already full.

If you have less than ~256 GB of RAM, cap the page cache the server can
accumulate rather than trusting it to behave:

```bash
systemd-run --user --scope -p MemoryHigh=48G -p ManagedOOMPreference=avoid \
    ./serve_ds4_reasoning.sh
```

`MemoryHigh` throttles and reclaims inside your own cgroup — page cache is
reclaimable, so the loader just re-reads — and slice-wide pressure never trips.
One caveat: this must wrap the **actual** vLLM process. Wrappers that re-exec
into their own scope (`uv run` does) leave the real worker uncapped in another
cgroup, so it looks applied and does nothing.

---

## Knob reference

Environment variables read by `serve_ds4_reasoning.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `HEAD_BUNDLE` | — | Head checkpoint. Required unless `RIDER=0`. |
| `MODEL` | `MJPansa/DeepSeek-V4-Flash-0731-NVFP4` | Base checkpoint. |
| `SPEC_METHOD` | `dspark` | `dspark` \| `mtp` \| `none`. Must match `MODEL`. |
| `TP` | `2` | Tensor-parallel size. |
| `MAX_MODEL_LEN` | `262144` | Context window. |
| `MAX_NUM_SEQS` | `2` | Concurrent sequences. Trades against context. |
| `GPU_UTIL` | `0.95` | `gpu_memory_utilization`. Narrow viable band. |
| `PORT` | `8001` | HTTP port. |
| `MAX_LATENT` | `256` | Safety cap on latent steps (stop-misfire bound). |
| `MIN_LATENT` | `4` | Floor on latent steps, so answers never no-think. |
| `USE_STOP` | `1` | Learned stop; `0` uses a fixed-N cap. |
| `STOP_THRESHOLD` | `0.5` | Stop-head threshold. `0.5` is correct. |
| `MIN_OUTPUT_TOKENS` | `4096` | Floor on the answer's token budget. |
| `RIDER` | `1` | `0` serves the bare backbone (A/B baseline). |
| `DEBUG` | `0` | `1` prints per-request latent stats. |
| `ADDON_DIR` | — | Path to this repo's root if the addon is not pip-installed. |
| `EXTRA_ARGS` | — | Appended verbatim to `vllm serve`. |

Per-request HTTP headers (these win over env defaults, and work on both
`/v1/chat/completions` and `/v1/messages`):

| Header | Effect |
|---|---|
| `x-ds4-thinking` | Enable thinking without a body field. |
| `x-ds4-max-latent` | Per-request latent cap. |
| `x-ds4-min-latent` | Per-request latent floor. |
| `x-ds4-use-stop` | Toggle the learned stop. |
| `x-ds4-stop-threshold` | Per-request stop threshold. |
| `x-ds4-min-output-tokens` | Per-request answer-budget floor. |

### Two knobs worth understanding

**`MAX_LATENT` is a safety cap, not a reasoning-depth dial.** The learned stop
normally ends the phase; this only bounds a misfire. `256` matches the head's
`K=256` training, and pushing far past what the head saw in training drifts into
garbage rather than thinking harder. The bundle's `compression_factor` is *inert*
at serve time — nothing in the rider reads it — so it describes what the head
learned, not a budget the loop enforces.

**`MIN_OUTPUT_TOKENS` exists because the latent phase and the answer share one
budget.** Each latent step bills one reserved accounting token, so a client
sending a modest `max_tokens` can spend the whole budget thinking and get an
empty answer back with `stop_reason=length`. The floor only ever *raises* a
client's `max_tokens`, and the learned stop still ends generation early when the
answer is done, so a generous floor doesn't force verbosity.

---

## How injection works (design note)

DeepSeek-V4 routes MoE experts by a **hash keyed on `input_ids`**. vLLM's native
`prompt_embeds` path nulls `input_ids` when embeddings are supplied, so
`--enable-prompt-embeds` alone would crash the engine with
`DeepSeek V4 hash MoE routing requires input_ids`.

Injection therefore overwrites the **`embed_tokens` output** at the target
position with the decoded latent, while token ids keep flowing normally so
hash-MoE routing still works. Latent steps carry a reserved pad token id purely
for accounting — a latent step occupies a real KV position, so it must advance
`num_tokens` in lockstep with `num_computed_tokens`, and the engine's invariant
`num_computed_tokens - num_tokens == in-flight draft tokens` depends on it. That
id never reaches the client, and its embedding is never read (the row is marked
`is_token_ids=False`, so the injected latent survives).

This runs at the `execute_model` seam, which is what keeps it on the **cudagraph
fast path** — no `enforce_eager`, and batched across concurrent requests.

Speculative drafting is suppressed *during* the latent phase
(`VLLM_DS4_SUPPRESS_LATENT_DRAFTS`, on by default). Leave it on: the rider
injects one embedding at the **last** query row, so a draft slot steals it and
the real position gets the pad token's embedding instead of the latent.
