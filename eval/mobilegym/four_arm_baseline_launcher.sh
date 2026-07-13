#!/usr/bin/env bash
set -u

BASE="${BASE:-/home/ma-user/work/susu/fastslow}"
TASK_N="${TASK_N:-150}"
REPEAT_N="${REPEAT_N:-3}"
PASS_K="${PASS_K:-1}"
ARMS="${ARMS:-s1_fast s1_slow s2_fast s2_slow}"
FAST_PARALLEL="${FAST_PARALLEL:-8}"
SLOW_PARALLEL="${SLOW_PARALLEL:-2}"
SLOW_TOKENS="${SLOW_TOKENS:-2048}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

API_KEY="${MA_TOKEN:?MA_TOKEN is required}"
S1_BASE="${S1_BASE_URL:-http://192.168.200.27/v2/infer/96885984-4acb-4917-84f8-21707a38c7d6/v1}"
S2_BASE="${S2_BASE_URL:-http://192.168.200.27/v2/infer/18a80e40-ebe1-40b4-ae18-4b6d96f22844/v1}"

cd "$BASE/mobilegym" || exit 1

GROUP_ROOT="$BASE/ws14fix3_traces/four_arm_baseline_${TASK_N}x${REPEAT_N}_${TS}"
SPLIT="$GROUP_ROOT/tasks_${TASK_N}.txt"
mkdir -p "$GROUP_ROOT"
sed -n "1,${TASK_N}p" "$BASE/mobilegym/bench_env/splits/test.txt" > "$SPLIT"
env | sort > "$GROUP_ROOT/env.before"
printf '%s\n' "$GROUP_ROOT" > "$BASE/ws14fix3_traces/LATEST_four_arm_baseline"
echo "$GROUP_ROOT"

run_arm() {
  local arm="$1"
  local actor="${arm%%_*}"
  local think="${arm##*_}"
  local root="$GROUP_ROOT/$arm"
  local parallel="$FAST_PARALLEL"
  if [ "$think" = "slow" ]; then
    parallel="$SLOW_PARALLEL"
  fi
  mkdir -p "$root"
  cp "$SPLIT" "$root/tasks_${TASK_N}.txt"
  {
    date -Is
    echo "GROUP_ROOT=$GROUP_ROOT"
    echo "ARM=$arm"
    echo "TASK_N=$TASK_N"
    echo "REPEAT_N=$REPEAT_N"
    echo "PASS_K=$PASS_K"
    echo "parallel=$parallel"
    echo "slow_tokens=$SLOW_TOKENS"
    df -h /home/ma-user/work "$BASE" 2>&1 || true
  } > "$root/PERSISTENCE_CHECK.txt"

  local start_epoch end_epoch status
  local start_iso end_iso
  start_epoch="$(date +%s)"
  start_iso="$(date -Is)"

  PYTHONPATH="$BASE/nanobot:$BASE/mobilegym" \
  PLAYWRIGHT_BROWSERS_PATH="$BASE/ms-playwright" \
  MOBILE_GYM_TO_THREAD_WORKERS=128 \
  OPENGUI_EVAL_ARM="$arm" \
  OPENGUI_MOBILEGYM_STEP_MODE=per_step OPENGUI_RECORD_PER_STEP=1 \
  OPENGUI_AGENT_PROFILE=general_e2e \
  OPENGUI_SYSTEM_PROMPT_PROFILE=slim \
  OPENGUI_HISTORY_IMAGE_WINDOW=1 OPENGUI_HISTORY_TEXT_WINDOW=3 \
  OPENGUI_STEP_CONTEXT_HISTORY_LIMIT=3 OPENGUI_STEP_CONTEXT_SUMMARY_MAX_CHARS=600 \
  OPENGUI_FAST_MAX_TOKENS=256 OPENGUI_SLOW_MAX_TOKENS="$SLOW_TOKENS" OPENGUI_MAX_TOKENS=1024 \
  OPENGUI_PURE_GUI_STRICT=1 OPENGUI_QUERY_ACTION_GUARDS_ENABLED=0 \
  OPENGUI_QUERY_AUTO_ENTER_CANDIDATE=0 OPENGUI_QUERY_AUTO_SUBMIT_AFTER_ENTRY=0 \
  OPENGUI_QUERY_AUTO_CORRECT_ANSWER=0 OPENGUI_TASK_HELPERS_ENABLED=0 \
  OPENGUI_REPEAT_NAVIGATION_GUARD_LIMIT=0 OPENGUI_MOBILEGYM_ALLOW_QUERY_DONE=0 \
  OPENGUI_MOBILEGYM_DONE_GUARD=0 OPENGUI_LARGE_TRANSFER_FORCE_ANSWER_AFTER_SWIPES=0 \
  MOBILEGYM_EARLY_STOP_ON_JUDGE_SUCCESS=1 \
  OPENGUI_S1_BASE_URL="$S1_BASE" OPENGUI_S1_API_KEY="$API_KEY" OPENGUI_S1_MODEL="qwen3.5-9b" \
  OPENGUI_S2_BASE_URL="$S2_BASE" OPENGUI_S2_API_KEY="$API_KEY" OPENGUI_S2_MODEL="qwen3.5-397b-a17b" \
  OPENGUI_S2_ENABLED=1 OPENGUI_FORCE_ACTOR="$actor" OPENGUI_FORCE_THINK="$think" \
  OPENGUI_QUADRANT_ROUTER_ENABLED=0 OPENGUI_QUADRANT_ROUTER_KIND=rpr_v1 \
  OPENGUI_RPR_COST_CONTROLLER_ENABLED=0 OPENGUI_RPR_GUIDANCE_LEASE_ENABLED=0 \
  HTTPS_PROXY="" HTTP_PROXY="" https_proxy="" http_proxy="" ALL_PROXY="" all_proxy="" \
  NO_PROXY="127.0.0.1,localhost,192.168.200.27,172.16.*" no_proxy="127.0.0.1,localhost,192.168.200.27,172.16.*" \
  python3 -m bench_env.run --agent opengui_s1s2 \
    --model-base-url "$S1_BASE" --model-name qwen3.5-9b --model-api-key "$API_KEY" \
    --split "$SPLIT" --repeat-n "$REPEAT_N" --pass-k "$PASS_K" \
    --env-url https://127.0.0.1:4180 --headless \
    --parallel "$parallel" --processes 1 --browsers 0 --isolation pages \
    --runs-dir "$root/mobilegym_runs" --quiet 2>&1 | tee "$root/run.log"
  status=${PIPESTATUS[0]}

  end_epoch="$(date +%s)"
  end_iso="$(date -Is)"
  printf '{"start":"%s","end":"%s","elapsed_seconds":%s,"exit_code":%s}\n' \
    "$start_iso" "$end_iso" "$((end_epoch - start_epoch))" "$status" > "$root/timing.json"
  python3 "$BASE/scripts/summarize_mobilegym_run.py" "$root" > "$root/stats.stdout.json" || true
  return "$status"
}

overall=0
for arm in $ARMS; do
  echo "=== $arm ==="
  run_arm "$arm" || overall=1
done

echo "$GROUP_ROOT"
exit "$overall"
