# vllm-ds4-reasoning

Serve the **DS4 CoLaR reasoning compression head** on a DeepSeek-V4-Flash backbone
under vLLM — so the model **reasons in a compressed latent space** instead of
emitting a token-by-token chain of thought.

- **Weights** — <https://huggingface.co/nmitchko/deepseek-v4-Flash-CoLaR>
- **Engine** — <https://github.com/nickmitchko/vllm-ds4-sm120> (required; upstream vLLM cannot serve this model)
- **Start here** — [`release/README-RELEASE.md`](release/README-RELEASE.md) for install, GPU sizing and every knob

```bash
pip install -e . --no-deps
pip install -r release/requirements-serve.txt \
    --extra-index-url https://flashinfer.ai/whl/cu130/torch2.11
HEAD_BUNDLE=/path/to/reasoning_head_final.pt release/serve_ds4_reasoning.sh
```

Requires **≥192 GiB VRAM** (verified on 2× RTX PRO 6000, 96 GiB, sm120) and a CUDA
13.0 toolkit.

---

## What it does

The reasoning head predicts a compressed latent of the next reasoning step from a
source-layer (35) hidden state. On its own that latent has no path back into the
model — this addon supplies the missing half:

```
h_src (layer 35)  --layer_norm-->  head     -> mu   (latent, 1024-d)
mu                --layer_norm-->  decoder  -> hidden vector (4096-d)
```

At serve time the backbone prefills your prompt **once**, then takes autoregressive
*latent* decode steps — each step's input embedding is
`decoder(head(previous layer-35 hidden))`, fed back from the model's own hidden
state via a per-request anchor store — until a learned stop head ends the phase,
emits `</think>`, and the answer decodes as ordinary tokens.

This runs **batched across concurrent requests and on the cudagraph fast path**
(no `enforce_eager`, no `--max-num-seqs 1`), and supports streaming.

**Injection needs a checkpoint trained with a decoder.** Legacy head-only
checkpoints load but are not injectable.

## How injection works

DeepSeek-V4 routes MoE experts by a **hash keyed on `input_ids`**
(`gate.tid2eid`). vLLM's native `prompt_embeds` path nulls `input_ids` when
embeddings are supplied, so it cannot be used here — it fails at engine startup:

```
ValueError: DeepSeek V4 hash MoE routing requires input_ids.
```

So injection overwrites the **`embed_tokens` output** at the target positions with
the decoded latent, while token ids keep flowing normally and hash-MoE routing
still works. (`build_embeds_prompt` is retained for reference and for models
*without* hash routing; it is not the serving path.)

Latent steps carry a reserved pad token id purely for accounting: a latent step
occupies a real KV position, so it must advance `num_tokens` in lockstep with
`num_computed_tokens` — the engine's invariant
`num_computed_tokens - num_tokens == in-flight draft tokens` depends on it. That
id never reaches the client, and its embedding is never read (the row is marked
`is_token_ids=False`, so the injected latent survives).

Speculative drafting is suppressed *during* the latent phase
(`VLLM_DS4_SUPPRESS_LATENT_DRAFTS`, on by default). Leave it on: the rider injects
one embedding at the **last** query row, so a draft slot steals it and the real
position gets the pad token's embedding instead of the latent.

## Install

```bash
pip install -e . --no-deps
```

Installing does nothing by itself — the plugin registers under
`vllm.general_plugins` but stays **dormant** until `VLLM_DS4_REASONING_CKPT` is
set, so it is safe to leave installed.

If you would rather not install it, put the repo root on `PYTHONPATH` and pass
`ADDON_DIR=/path/to/this/repo` to the serve script.

## Enable

The supported entrypoint is [`release/serve_ds4_reasoning.sh`](release/serve_ds4_reasoning.sh),
which sets all of this up for you. Its defaults are the measured-good
configuration. To wire it by hand, the minimum is:

```bash
export VLLM_PLUGINS=ds4_reasoning
export VLLM_DS4_REASONING_CKPT=/path/to/reasoning_head_final.pt
export VLLM_DS4_REASONING_CAPTURE_LAYER=35   # anchor-store feedback signal
export VLLM_ALLOW_INSECURE_SERIALIZATION=1   # collective_rpc callable dispatch
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
```

Then serve **with `--enable-prompt-embeds`** — the rider overwrites
`inputs_embeds.gpu` rows, and without that flag the pure-token decode path never
reads the buffer, so latent injection *silently does nothing*.

See the [knob reference](release/README-RELEASE.md#knob-reference) for the full set
of env vars and the per-request `x-ds4-*` headers.

## Two things that look like bugs and are not

**Requests must ask for thinking.** Pass
`chat_template_kwargs={"thinking": true}` (or the header `x-ds4-thinking: 1`);
without it the output is garbage.

**An armed rider logs identically to a dormant one.** `VLLM_DS4_REASONING_DEBUG=1`
is the only way to confirm injection is happening — it prints
`steps=N stop_step=... max_p=... end=stop|cap` per completion, and says so
explicitly when nothing was injected.

Also expect the **first one or two requests after startup** to possibly return
degenerate repetition; it settles by itself. Don't lower the stop threshold in
response — that isn't the cause.

## Layout

| Path | Contents |
|---|---|
| `vllm_ds4_reasoning/` | The plugin: injector, checkpoint loader, hook injection, serving hook, closed loop. |
| `release/` | Serve script, pinned requirements, install + GPU-sizing docs. |
| `docs/` | Design notes on the closed-loop latent decode path. |

## Licence

Apache-2.0. See [`LICENSE`](LICENSE).
