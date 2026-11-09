#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server on a Runpod GPU pod.
#
# Run this *on the pod* (e.g. from the Runpod web terminal or via SSH):
#
#   MODEL=Qwen/Qwen3-32B PARSER=hermes ./start_vllm.sh
#   MODEL=meta-llama/Llama-3.3-70B-Instruct PARSER=llama3_json QUANT=fp8 ./start_vllm.sh
#   MODEL=Qwen/Qwen3-8B PARSER=hermes REASONING_PARSER=qwen3 ./start_vllm.sh
#   EXTRA_ARGS="--speculative-config '{...}'" ...   # anything else, verbatim
#
# Then on the control box:
#
#   export VLLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1
#   export VLLM_API_KEY=$API_KEY   # same value as below
#   inspect eval tasks/gaia_cone/task.py@gaia_cone --model vllm/$MODEL ...
#
# The tool-call parser is per model family and is part of what you are
# measuring. See parsers.yaml and record the value you used with the results.

set -euo pipefail

MODEL="${MODEL:?set MODEL to a HuggingFace model id}"
PARSER="${PARSER:?set PARSER to a vLLM tool-call parser name (see parsers.yaml)}"
PORT="${PORT:-8000}"
TP="${TP:-1}"                        # tensor parallel = number of GPUs
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
QUANT="${QUANT:-}"                   # e.g. fp8; empty for none
REASONING_PARSER="${REASONING_PARSER:-}"   # e.g. qwen3; see parsers.yaml
EXTRA_ARGS="${EXTRA_ARGS:-}"
API_KEY="${API_KEY:-inspectai}"
VLLM_VERSION="${VLLM_VERSION:-0.33.0}"   # pin; bump deliberately

if ! python -c "import vllm" 2>/dev/null; then
  pip install "vllm==${VLLM_VERSION}"
fi

ARGS=(
  --model "$MODEL"
  --port "$PORT"
  --host 0.0.0.0
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_MODEL_LEN"
  --enable-auto-tool-choice
  --tool-call-parser "$PARSER"
  --api-key "$API_KEY"
)
if [[ -n "$QUANT" ]]; then
  ARGS+=(--quantization "$QUANT")
fi
# For thinking models, the reasoning parser moves <think>...</think> out of the
# content before the tool parser runs. Without it, a tool call the model
# sketches while thinking can be parsed (or mis-parsed) as a real one, and the
# whole think block is replayed to the model as visible history every turn.
if [[ -n "$REASONING_PARSER" ]]; then
  ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi
if [[ -n "$EXTRA_ARGS" ]]; then
  eval "ARGS+=($EXTRA_ARGS)"
fi

echo "vllm ${VLLM_VERSION}  model=${MODEL}  parser=${PARSER}  tp=${TP}  quant=${QUANT:-none}  reasoning=${REASONING_PARSER:-none}"
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
