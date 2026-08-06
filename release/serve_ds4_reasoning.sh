#!/usr/bin/env bash
# Serve the DS4 CoLaR reasoning head on DeepSeek-V4-Flash via `vllm serve`.
#
# What this runs: the backbone prefills the prompt ONCE, then takes autoregressive
# LATENT decode steps -- each step's input embedding is decoder(head(previous
# layer-35 hidden)) rather than a token embedding -- and a learned stop head ends
# the latent phase and emits </think>, after which the answer decodes as normal
# tokens. Batched across concurrent requests and cudagraph-safe (no enforce_eager).
#
# Quick start -- the defaults are the recommended configuration, and the default
# model ships its own head, so this needs no arguments:
#
#   ./serve_ds4_reasoning.sh
#
# Then, note the `thinking` kwarg -- WITHOUT it the model emits garbage:
#
#   curl localhost:8001/v1/chat/completions -H 'content-type: application/json' -d '{
#     "model": "nmitchko/DeepSeek-V4-Flash-0731-Latent-Reasoning",
#     "messages": [{"role":"user","content":"Write a Python LRU cache."}],
#     "chat_template_kwargs": {"thinking": true},
#     "max_tokens": 4096 }'
#
# Other useful invocations:
#   HEAD_BUNDLE=/path/to/head.safetensors ./serve_ds4_reasoning.sh   # explicit head
#   RIDER=0 ./serve_ds4_reasoning.sh                 # bare backbone A/B baseline
#   DEBUG=1 ./serve_ds4_reasoning.sh                 # per-request latent stats
#   MAX_MODEL_LEN=32768 TP=1 ./serve_ds4_reasoning.sh
#   MODEL=nvidia/DeepSeek-V4-Flash-NVFP4 SPEC_METHOD=mtp ./serve_ds4_reasoning.sh
#
# See README-RELEASE.md for GPU sizing and the full knob reference.
set -euo pipefail

# --- what to serve ------------------------------------------------------------
# The head bundle (.safetensors or .pt). Leave EMPTY with the default model and
# the script resolves the head that model repo ships (see below). Required only
# when serving a base that carries no head of its own.
HEAD_BUNDLE="${HEAD_BUNDLE:-}"

# Base checkpoint. All three supported bases have byte-identical bodies and
# tokenizers -- all 133,660 non-draft weight names match -- so ONE head bundle
# serves any of them. They differ in the speculative draft block they ship, which
# is why MODEL and SPEC_METHOD must move together:
#   nmitchko/DeepSeek-V4-Flash-0731-Latent-Reasoning  DSpark + BUNDLED HEAD -> dspark
#   MJPansa/DeepSeek-V4-Flash-0731-NVFP4              3-layer DSpark draft  -> dspark
#   nvidia/DeepSeek-V4-Flash-NVFP4                    1-layer MTP draft     -> mtp
MODEL="${MODEL:-nmitchko/DeepSeek-V4-Flash-0731-Latent-Reasoning}"
SPEC_METHOD="${SPEC_METHOD:-dspark}"

# Filename of the head inside a model repo that bundles one.
BUNDLED_HEAD_FILE="${BUNDLED_HEAD_FILE:-latent_reasoning_head.safetensors}"

PORT="${PORT:-8001}"
TP="${TP:-2}"
# 262144 verified on 2x96 GiB (443,012 fp8 KV tokens); 524288 also builds.
# Lower this first if the engine fails its KV-cache check at startup.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# At 256k context on 2x96 GiB the fp8 KV cache only fits a couple of full-length
# sequences. Raising this starves KV: it either OOMs at startup or silently
# truncates the usable context. Raise only alongside a lower MAX_MODEL_LEN.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
GPU_UTIL="${GPU_UTIL:-0.95}"

# --- latent reasoning knobs ---------------------------------------------------
# Runaway safety cap on latent steps. The learned stop ends the phase normally;
# this only bounds a stop MISFIRE. 256 matches the head's K=256 training -- going
# well beyond what the head saw in training drifts into garbage rather than
# thinking longer.
MAX_LATENT="${MAX_LATENT:-256}"
# Floor on latent steps before the stop may fire, so answers never no-think.
MIN_LATENT="${MIN_LATENT:-4}"
USE_STOP="${USE_STOP:-1}"                        # 0 = fixed-N cap instead
# 0.5 is correct. Do NOT lower it on the strength of one bad early answer -- see
# the WARMUP note below for what actually causes that.
STOP_THRESHOLD="${STOP_THRESHOLD:-0.5}"
# The latent <think> phase and the answer share ONE output-token budget, so a
# client sending a modest max_tokens can spend all of it thinking and return an
# empty answer. This floors each request's budget (it only ever RAISES a client's
# max_tokens); the learned stop still ends generation early when the answer is
# done, so a high floor does not force verbosity. Per-request override:
# x-ds4-min-output-tokens. 0 disables.
MIN_OUTPUT_TOKENS="${MIN_OUTPUT_TOKENS:-4096}"

# RIDER=0 serves the BARE backbone -- no head, no capture buffer, no hook. This
# is the honest A/B baseline. MAX_LATENT=0 is NOT equivalent: it still arms the
# head in every worker and runs the rider on every forward pass.
RIDER="${RIDER:-1}"
# DEBUG=1 prints "[ds4-debug] req ... steps=N stop_step=... max_p=... end=stop|cap"
# after each non-streaming completion. This is the ONLY way to confirm the rider
# is actually injecting: an armed rider and a silently-dormant one produce
# IDENTICAL server logs.
DEBUG="${DEBUG:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# DSpark drafts a block of dspark_block_size-1 tokens; MTP drafts 1.
case "$SPEC_METHOD" in
    mtp)    SPEC_TOKENS="${SPEC_TOKENS:-1}" ;;
    dspark) SPEC_TOKENS="${SPEC_TOKENS:-4}" ;;
    none)   SPEC_TOKENS="" ;;
    *) echo "ERROR: SPEC_METHOD must be mtp, dspark or none (got '$SPEC_METHOD')" >&2
       exit 1 ;;
esac

# Catch a base/drafter mismatch HERE rather than 20 minutes into a 160 GiB load,
# where it surfaces as an opaque weight-name error deep in the drafter loader.
case "$MODEL:$SPEC_METHOD" in
    nvidia/*:dspark)
        echo "ERROR: $MODEL ships a 1-layer MTP draft, not DSpark." >&2
        echo "       Use SPEC_METHOD=mtp (or none) with the nvidia base." >&2
        exit 1 ;;
esac

if [ "$RIDER" = "1" ] && [ -z "$HEAD_BUNDLE" ]; then
    # A local model directory containing the head: use it directly.
    if [ -f "$MODEL/$BUNDLED_HEAD_FILE" ]; then
        HEAD_BUNDLE="$MODEL/$BUNDLED_HEAD_FILE"
        echo "Using the head bundled with $MODEL"
    else
        # Otherwise pull just that one file from the Hub (~152 MiB). This is a
        # no-op once cached, and deliberately does NOT fetch the 164 GiB body --
        # vllm serve does that itself.
        echo "Resolving the head bundled with $MODEL ($BUNDLED_HEAD_FILE)..."
        HEAD_BUNDLE="$(python3 - "$MODEL" "$BUNDLED_HEAD_FILE" <<'PY' 2>/dev/null || true
import sys
try:
    from huggingface_hub import hf_hub_download
    print(hf_hub_download(sys.argv[1], sys.argv[2]))
except Exception:
    sys.exit(1)
PY
)"
        if [ -z "$HEAD_BUNDLE" ]; then
            echo "ERROR: could not resolve a head for $MODEL." >&2
            echo "       That model repo may not bundle one. Either point at a head:" >&2
            echo "         HEAD_BUNDLE=/path/to/head.safetensors $0" >&2
            echo "       or serve the bare backbone with no latent reasoning:" >&2
            echo "         RIDER=0 $0" >&2
            exit 1
        fi
    fi
fi

if [ "$RIDER" = "1" ] && [ ! -f "$HEAD_BUNDLE" ]; then
    echo "ERROR: head bundle not found: $HEAD_BUNDLE" >&2
    exit 1
fi

# --- environment --------------------------------------------------------------
# CUDA 13 toolkit: the sm120 MoE kernels JIT-compile on first load and an older
# nvcc fails with "No supported CUDA architectures found for major versions [12]".
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.0f}"
# Needed for the rider's collective_rpc callable dispatch (it arms the latent
# phase in every TP worker via apply_model, which is not msgpack-serializable).
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# The addon must be importable. If it is pip-installed, this is a no-op; if you
# are running from a source tree, point ADDON_DIR at the repo root (the directory
# containing vllm_ds4_reasoning/).
ADDON_DIR="${ADDON_DIR:-}"
if [ -n "$ADDON_DIR" ]; then
    export PYTHONPATH="$ADDON_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi

export VLLM_PLUGINS=ds4_reasoning
if [ "$RIDER" = "1" ]; then
    export VLLM_DS4_REASONING_CKPT="$HEAD_BUNDLE"
    # The anchor store's feedback signal. Read by the MODEL, not the plugin, so
    # dormancy requires BOTH this and _CKPT to be unset.
    export VLLM_DS4_REASONING_CAPTURE_LAYER=35
else
    unset VLLM_DS4_REASONING_CKPT VLLM_DS4_REASONING_CAPTURE_LAYER
fi
export VLLM_DS4_REASONING_MAX_LATENT="$MAX_LATENT"
export VLLM_DS4_REASONING_MIN_LATENT="$MIN_LATENT"
export VLLM_DS4_REASONING_STOP="$USE_STOP"
export VLLM_DS4_REASONING_STOP_THRESHOLD="$STOP_THRESHOLD"
export VLLM_DS4_REASONING_MIN_OUTPUT_TOKENS="$MIN_OUTPUT_TOKENS"
export VLLM_DS4_REASONING_DEBUG="$DEBUG"

echo "=============================================================="
echo "DS4 closed-loop latent decode (batched, compile-safe)"
echo "  model:        $MODEL"
if [ "$RIDER" = "1" ]; then
    echo "  head bundle:  $HEAD_BUNDLE"
    echo "  latent:       max=$MAX_LATENT min=$MIN_LATENT"
    echo "  learned stop: use_stop=$USE_STOP threshold=$STOP_THRESHOLD"
    echo "  answer floor: $MIN_OUTPUT_TOKENS output tokens"
else
    echo "  head bundle:  (RIDER=0 -- bare backbone, no latent reasoning)"
fi
echo "  serving:      port=$PORT tp=$TP max_model_len=$MAX_MODEL_LEN"
echo "                max_num_seqs=$MAX_NUM_SEQS gpu_util=$GPU_UTIL"
if [ "$SPEC_METHOD" = "none" ]; then
    echo "  spec decode:  disabled"
else
    echo "  spec decode:  $SPEC_METHOD (num_speculative_tokens=$SPEC_TOKENS)"
fi
[ "$DEBUG" = "1" ] && echo "  ds4 debug:    ON"
echo "=============================================================="
echo
echo "NOTE: requests MUST pass chat_template_kwargs={\"thinking\": true}"
echo "      (or the x-ds4-thinking: 1 header) or output will be garbage."
if [ "$RIDER" = "1" ]; then
    echo "NOTE: the first 1-2 requests after startup can return degenerate"
    echo "      repetition -- this is a warmup artifact that clears itself."
    echo "      Send one throwaway request before trusting output."
fi
echo

SPEC_ARGS=()
if [ "$SPEC_METHOD" != "none" ]; then
    SPEC_ARGS+=(--speculative-config \
        "{\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$SPEC_TOKENS}")
fi

# --enable-prompt-embeds is REQUIRED whenever the rider is on: the rider
# overwrites inputs_embeds.gpu rows, and the pure-token decode path never reads
# that buffer, so without the flag latent injection silently does NOTHING.
# Deliberately absent: --enforce-eager and --max-num-seqs 1. The rider runs at
# the execute_model seam, so it is cudagraph-safe and batched.
EMBEDS_ARGS=()
[ "$RIDER" = "1" ] && EMBEDS_ARGS+=(--enable-prompt-embeds)

exec vllm serve "$MODEL" \
    ${EMBEDS_ARGS[@]+"${EMBEDS_ARGS[@]}"} \
    --tensor-parallel-size "$TP" \
    --tokenizer-mode deepseek_v4 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --block-size 256 \
    --disable-custom-all-reduce \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --port "$PORT" \
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
    $EXTRA_ARGS
