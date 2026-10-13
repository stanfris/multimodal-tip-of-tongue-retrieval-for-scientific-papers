#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Resolve the saved login before the PBS worker switches HF_HOME for weights.
TOKEN_FILE="${HF_TOKEN_PATH:-}"
if [[ -z "$TOKEN_FILE" ]]; then
  PYTHON="${VTT_PYTHON:-$ROOT_DIR/.venv/bin/python}"
  TOKEN_FILE="$("$PYTHON" -c 'from huggingface_hub.constants import HF_TOKEN_PATH; print(HF_TOKEN_PATH)')"
fi

# Submit all pending train judgements with Gemma 4 and spawned engine workers.
# PBS workers receive these via job.sh; exporting them only on the login node
# would not reliably propagate them into the scheduled job.
exec "$ROOT_DIR/scripts/query_generation/judge_train_queries.sh" \
  split=train launcher=pbs_h200 model=gemma4_judge \
  model.provider=vllm visual_query.judgement.provider=vllm \
  stage.limit=null visual_query.judgement.batch_size=32 \
  launcher.vllm.max_num_seqs=16 ++launcher.vllm.enforce_eager=true \
  ++launcher.environment.VLLM_ENABLE_V1_MULTIPROCESSING=1 \
  ++launcher.environment.VLLM_WORKER_MULTIPROC_METHOD=spawn \
  ++launcher.environment.VLLM_USE_V2_MODEL_RUNNER=0 \
  ++launcher.environment.PYTHONUNBUFFERED=1 \
  "++launcher.environment.HF_TOKEN_PATH=$TOKEN_FILE" \
  launcher.walltime=24:00:00 "$@"
