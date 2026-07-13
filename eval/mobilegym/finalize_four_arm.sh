#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/home/ma-user/work/susu/fastslow}"
ROOT="${1:?usage: finalize_four_arm.sh RUN_ROOT}"
NAME="$(basename "$ROOT")"
OUT_DIR="${OUT_DIR:-$BASE/results_archive/$NAME}"

cd "$BASE"
mkdir -p "$OUT_DIR"

for ARM in s1_fast s1_slow s2_fast s2_slow; do
  if [ -d "$ROOT/$ARM" ]; then
    python3 scripts/summarize_mobilegym_run.py "$ROOT/$ARM" > "$ROOT/$ARM/stats.stdout.json"
  fi
done

python3 eval/mobilegym/aggregate.py \
  --arm "s1_fast=$ROOT/s1_fast" \
  --arm "s1_slow=$ROOT/s1_slow" \
  --arm "s2_fast=$ROOT/s2_fast" \
  --arm "s2_slow=$ROOT/s2_slow" \
  --k 3 \
  --json-out "$OUT_DIR/aggregate.json" \
  --md-out "$OUT_DIR/aggregate.md" \
  > "$OUT_DIR/aggregate.stdout.md"

python3 eval/mobilegym/cost.py "$OUT_DIR/aggregate.json" \
  --json-out "$OUT_DIR/cost.json" \
  --md-out "$OUT_DIR/cost.md" \
  --pareto-out "$OUT_DIR/pareto.jsonl" \
  > "$OUT_DIR/cost.stdout.md"

python3 eval/analysis/oracle.py "$OUT_DIR/cost.json" \
  --json-out "$OUT_DIR/oracle.json" \
  --md-out "$OUT_DIR/oracle.md" \
  --svg-out "$OUT_DIR/oracle.svg" \
  > "$OUT_DIR/oracle.stdout.md"

printf '\n- finalized four-arm run: `%s`\n' "$OUT_DIR" >> results_archive/RESULT_INDEX.md
echo "$OUT_DIR"
