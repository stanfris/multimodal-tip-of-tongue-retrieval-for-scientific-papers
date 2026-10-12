#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-session}"
if (( $# )); then shift; fi

case "$MODE" in
  session)
    echo 'After the allocation opens, run:'
    printf '  cd %q\n' "$ROOT_DIR"
    echo '  scripts/query_generation/interactive_judgements.sh probe'
    echo '  scripts/query_generation/interactive_judgements.sh run'
    exec qsub -I -N judge_debug \
      -P "${VTT_PBS_PROJECT:-gcb50357}" -q "${VTT_PBS_QUEUE:-rt_HG}" \
      -l "select=1:ncpus=16:mem=160gb:ngpus=1" \
      -l "walltime=${VTT_DEBUG_WALLTIME:-01:00:00}" "$@"
    ;;
  probe|run) ;;
  *)
    echo "Usage: $0 {session [qsub options]|probe [probe options]|run [Hydra overrides]}" >&2
    exit 2
    ;;
esac

PYTHON="${VTT_PYTHON:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "Main environment Python is missing: $PYTHON" >&2
  exit 2
fi
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Diagnostic defaults; preserve explicit settings for comparisons.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
# Keep the saved login at its original location when switching model caches.
if [[ -z "${HF_TOKEN_PATH:-}" ]]; then
  export HF_TOKEN_PATH="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/token"
fi
export HF_HOME="${HF_HOME:-${VTT_CACHE_ROOT:-${VTT_DATA_ROOT:-$ROOT_DIR/data}/cache}/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VTT_CACHE_ROOT:-${VTT_DATA_ROOT:-$ROOT_DIR/data}/cache}}"

if [[ "$MODE" == probe ]]; then
  # Put the engine in this process so faulthandler sees the stalled worker.
  export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
  export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
  exec "$PYTHON" -u "$ROOT_DIR/scripts/query_generation/probe_judge_startup.py" "$@"
fi

# Retain H200 paths/runtime, but execute on the allocated node without qsub.
exec "$PYTHON" -u -X faulthandler -m hydra_entry \
  stage=judge_queries launcher=pbs_h200 launcher.kind=local model.provider=vllm \
  visual_query.judgement.provider=vllm \
  stage.limit=2 visual_query.judgement.batch_size=1 \
  launcher.vllm.max_num_seqs=1 +launcher.vllm.enforce_eager=true "$@"
