#!/usr/bin/env bash
set -euo pipefail

# Helper launcher:
# - points subagent runtime to local OpenAI-compatible vLLM
# - enables rule-based execution policy
# - preserves caller-supplied command line (agent/gateway/etc.)
#
# Examples:
#   bash scripts/run_nanobot_with_qwen_subagent.sh nanobot agent -m "hello"
#   bash scripts/run_nanobot_with_qwen_subagent.sh nanobot gateway

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://127.0.0.1:8001/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-Qwen3.5-9B-Instruct}"

export NANOBOT_SUBAGENT_API_KEY="${NANOBOT_SUBAGENT_API_KEY:-$OPENAI_API_KEY}"
export NANOBOT_SUBAGENT_API_BASE="${NANOBOT_SUBAGENT_API_BASE:-$OPENAI_API_BASE}"
export NANOBOT_SUBAGENT_MODEL="${NANOBOT_SUBAGENT_MODEL:-$OPENAI_MODEL}"

export NANOBOT_EXEC_POLICY_ENABLED="${NANOBOT_EXEC_POLICY_ENABLED:-1}"

echo "[INFO] OPENAI_API_BASE=${OPENAI_API_BASE}"
echo "[INFO] OPENAI_MODEL=${OPENAI_MODEL}"
echo "[INFO] NANOBOT_SUBAGENT_API_BASE=${NANOBOT_SUBAGENT_API_BASE}"
echo "[INFO] NANOBOT_SUBAGENT_MODEL=${NANOBOT_SUBAGENT_MODEL}"
echo "[INFO] NANOBOT_EXEC_POLICY_ENABLED=${NANOBOT_EXEC_POLICY_ENABLED}"

if (( $# == 0 )); then
  echo "[ERROR] No command provided." >&2
  echo "Usage: bash scripts/run_nanobot_with_qwen_subagent.sh <command...>" >&2
  exit 1
fi

exec "$@"

