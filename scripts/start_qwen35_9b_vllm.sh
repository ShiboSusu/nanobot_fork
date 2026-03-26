#!/usr/bin/env bash
set -euo pipefail

# Start Qwen3.5-9B vLLM with auto GPU detection.
# Designed for variable remote environments (0.5 MIG -> multi-GPU nodes).
#
# Usage:
#   bash scripts/start_qwen35_9b_vllm.sh
# Optional env:
#   MODEL_PATH=/mnt/afs/250010108/models/qwen/llm/Qwen3.5-9B-Instruct
#   SERVED_MODEL_NAME=Qwen3.5-9B-Instruct
#   PORT=8001
#   HOST=0.0.0.0
#   GPU_UTIL=0.85
#   MAX_MODEL_LEN=8192
#   TP_SIZE=1                # override tensor parallel size
#   MAX_TP_SIZE=4            # cap auto TP size
#   DTYPE=auto
#   VLLM_EXTRA_ARGS="--enforce-eager"

MODEL_PATH="${MODEL_PATH:-/mnt/afs/250010108/models/qwen/llm/Qwen3.5-9B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.5-9B-Instruct}"
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TP_SIZE="${MAX_TP_SIZE:-4}"
DTYPE="${DTYPE:-auto}"

detect_visible_gpu_count() {
  # 1) Respect CUDA_VISIBLE_DEVICES if set.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local cleaned
    cleaned="$(echo "${CUDA_VISIBLE_DEVICES}" | tr -d ' ')"
    if [[ -n "${cleaned}" ]]; then
      # Count non-empty entries split by comma.
      awk -F',' '{c=0; for(i=1;i<=NF;i++){ if(length($i)>0) c++ } print c}' <<< "${cleaned}"
      return 0
    fi
  fi

  # 2) Try torch if available.
  if command -v python >/dev/null 2>&1; then
    local torch_count
    torch_count="$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    print(int(torch.cuda.device_count()))
except Exception:
    print(0)
PY
)"
    if [[ "${torch_count}" =~ ^[0-9]+$ ]] && (( torch_count > 0 )); then
      echo "${torch_count}"
      return 0
    fi
  fi

  # 3) Fallback to nvidia-smi.
  if command -v nvidia-smi >/dev/null 2>&1; then
    local smi_count
    smi_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -cv '^[[:space:]]*$' || true)"
    if [[ "${smi_count}" =~ ^[0-9]+$ ]] && (( smi_count > 0 )); then
      echo "${smi_count}"
      return 0
    fi
  fi

  # Safe default.
  echo 1
}

detect_min_gpu_memory_mb() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return 0
  fi
  local min_mem
  min_mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1{m=$1} $1<m{m=$1} END{print m+0}' || true)"
  if [[ "${min_mem}" =~ ^[0-9]+$ ]]; then
    echo "${min_mem}"
  else
    echo 0
  fi
}

require_cmd() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: ${name}" >&2
    exit 1
  fi
}

require_cmd python

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[ERROR] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -iTCP:"${PORT}" -sTCP:LISTEN -nP >/dev/null 2>&1; then
    echo "[ERROR] Port ${PORT} is already in use. Please change PORT or stop existing process." >&2
    exit 1
  fi
fi

VISIBLE_GPU_COUNT="$(detect_visible_gpu_count)"
if [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] && (( VISIBLE_GPU_COUNT > 0 )); then
  :
else
  VISIBLE_GPU_COUNT=1
fi

if [[ -n "${TP_SIZE:-}" ]]; then
  TP="${TP_SIZE}"
else
  TP="${VISIBLE_GPU_COUNT}"
fi

if (( TP < 1 )); then
  TP=1
fi

if (( TP > MAX_TP_SIZE )); then
  TP="${MAX_TP_SIZE}"
fi

MIN_GPU_MEM_MB="$(detect_min_gpu_memory_mb)"
if (( MIN_GPU_MEM_MB > 0 && MIN_GPU_MEM_MB < 50000 )); then
  # Keep a conservative default for partitioned/half-card settings.
  if [[ "${GPU_UTIL}" == "0.85" ]]; then
    GPU_UTIL="0.80"
  fi
  if [[ "${MAX_MODEL_LEN}" == "8192" ]]; then
    MAX_MODEL_LEN="4096"
  fi
fi

echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "[INFO] HOST=${HOST} PORT=${PORT}"
echo "[INFO] VISIBLE_GPU_COUNT=${VISIBLE_GPU_COUNT} TP_SIZE=${TP}"
echo "[INFO] GPU_UTIL=${GPU_UTIL} MAX_MODEL_LEN=${MAX_MODEL_LEN} DTYPE=${DTYPE}"

cmd=(
  python -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --dtype "${DTYPE}"
)

if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${VLLM_EXTRA_ARGS})
  cmd+=("${extra_args[@]}")
fi

echo "[INFO] Launch command: ${cmd[*]}"
exec "${cmd[@]}"

