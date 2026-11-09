#!/usr/bin/env bash
# Run one task for one model and one tool tier.
#
#   [TASK=<name>] scripts/run.sh <inspect-model> [tier=search] [epochs=5] [extra inspect args...]
#
# Task names: `python -m tasks.registry list` (gaia_cone is the default).
#
# Examples:
#   scripts/run.sh anthropic/claude-sonnet-5-5
#   scripts/run.sh anthropic/claude-sonnet-5-5 none 10 --limit 5
#   TASK=objobssap_window scripts/run.sh anthropic/claude-sonnet-5-5 search 3 --limit 5
#   TASK=adql_translate scripts/run.sh anthropic/claude-haiku-5-5 none 3
#   scripts/run.sh vllm/Qwen/Qwen3-32B docs 5 --model-role judge=anthropic/claude-haiku-5-5

set -euo pipefail
cd "$(dirname "$0")/.."

TASK="${TASK:-gaia_cone}"
MODEL="${1:?usage: run.sh <model> [tier] [epochs] [extra args]}"
TIER="${2:-search}"
EPOCHS="${3:-5}"
shift $(( $# >= 3 ? 3 : $# ))

# Pre-flight: if the verifier's service is down, every sample scores NOANSWER
# and the run is a waste of money. Fail fast instead.
.venv/bin/python -m tasks.registry healthcheck "$TASK"

exec .venv/bin/inspect eval "tasks/$TASK/task.py@$TASK" \
  --model "$MODEL" \
  -T tool_tier="$TIER" \
  --epochs "$EPOCHS" \
  --log-dir logs \
  --max-sandboxes 4 \
  --max-samples 4 \
  "$@"
