#!/usr/bin/env bash
# Run the Gaia cone task for one model and one tool tier.
#
#   scripts/run.sh <inspect-model> [tier=search] [epochs=5] [extra inspect args...]
#
# Examples:
#   scripts/run.sh anthropic/claude-sonnet-5-5
#   scripts/run.sh anthropic/claude-sonnet-5-5 none 10 --limit 5
#   scripts/run.sh vllm/Qwen/Qwen3-32B docs 5 --model-role judge=anthropic/claude-haiku-5-5
#
# Add `-S failure_taxonomy` style scoring later with scripts/judge.sh once the
# verifier numbers are trusted; the taxonomy judge costs tokens on every sample.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:?usage: run.sh <model> [tier] [epochs] [extra args]}"
TIER="${2:-search}"
EPOCHS="${3:-5}"
shift $(( $# >= 3 ? 3 : $# ))

# Pre-flight: if MAST is down, every sample scores NOANSWER and the run is a
# waste of money. Fail fast instead.
.venv/bin/python - <<'EOF'
from harness.tap import GaiaVerifier
import sys
ok = GaiaVerifier().healthcheck()
print("MAST Gaia DR3 TAP healthcheck:", "ok" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF

exec .venv/bin/inspect eval tasks/gaia_cone/task.py@gaia_cone \
  --model "$MODEL" \
  -T tool_tier="$TIER" \
  --epochs "$EPOCHS" \
  --log-dir logs \
  --max-sandboxes 4 \
  --max-samples 4 \
  "$@"
