#!/usr/bin/env bash
# Run one task for one model and one tool tier.
#
#   [TASK=gaia_cone|objobssap_window] scripts/run.sh <inspect-model> [tier=search] [epochs=5] [extra inspect args...]
#
# Examples:
#   scripts/run.sh anthropic/claude-sonnet-5-5
#   scripts/run.sh anthropic/claude-sonnet-5-5 none 10 --limit 5
#   TASK=objobssap_window scripts/run.sh anthropic/claude-sonnet-5-5 search 3 --limit 5
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
TASK="$TASK" .venv/bin/python - <<'EOF'
import os, sys
task = os.environ["TASK"]
if task == "gaia_cone":
    from harness.tap import GaiaVerifier as Client
    label = "MAST Gaia DR3 TAP"
elif task == "objobssap_window":
    from harness.objobssap import ObjObsSAPClient as Client
    label = "vo-labs.dev ObjObsSAP"
else:
    sys.exit(f"unknown TASK {task!r}")
ok = Client().healthcheck()
print(f"{label} healthcheck:", "ok" if ok else "FAILED")
sys.exit(0 if ok else 1)
EOF

exec .venv/bin/inspect eval "tasks/$TASK/task.py@$TASK" \
  --model "$MODEL" \
  -T tool_tier="$TIER" \
  --epochs "$EPOCHS" \
  --log-dir logs \
  --max-sandboxes 4 \
  --max-samples 4 \
  "$@"
