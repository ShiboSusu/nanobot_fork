# Phase 0: Signal Validation Pilot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether thinking token length and confidence score are effective escalation signals for Qwen3VL-8B on GUI tasks — producing validated thresholds or a go/no-go decision for each signal.

**Architecture:** A standalone eval script (`eval/phase0_signal_pilot.py`) runs the small model on ~25 validation tasks with thinking mode enabled and structured confidence output. Per-step metrics (reasoning_tokens, confidence, action correctness) are saved to a JSONL file. A separate analysis script (`eval/phase0_analysis.py`) computes correlations, plots ROC curves, and determines thresholds. Requires a one-line change to the provider to extract `reasoning_tokens` from the usage response.

**Tech Stack:** Python 3.10+, scipy (Spearman), sklearn (ROC/AUC), matplotlib (plots), nanobot SDK (Nanobot.from_config + gui_task), dashscope API (qwen3-vl-8b or equivalent small VL model with thinking support)

---

## File Structure

```
nanobot/providers/openai_compat_provider.py  — MODIFY: extract reasoning_tokens from usage
eval/phase0_signal_pilot.py                  — CREATE: run validation tasks, collect per-step signals
eval/phase0_analysis.py                      — CREATE: correlation + ROC + threshold determination
eval/phase0_config.json                      — CREATE: config with small model + thinking enabled
eval/datasets/phase0_validation_25.csv       — CREATE: 25-task validation subset from 63-task set
```

---

### Task 1: Extract reasoning_tokens from provider usage response

**Files:**
- Modify: `nanobot/providers/openai_compat_provider.py:787-833`

- [ ] **Step 1: Add reasoning_tokens extraction to `_extract_usage()`**

After the `cached_tokens` extraction block (line 831), add reasoning_tokens extraction:

```python
        # --- reasoning_tokens (thinking/reasoning models) ---
        # Try nested path first (OpenAI-style), then top-level (Qwen/dashscope)
        for path in (
            ("completion_tokens_details", "reasoning_tokens"),  # OpenAI o1/o3
            ("reasoning_tokens",),                              # Qwen thinking mode
        ):
            reasoning = cls._get_nested_int(usage_map, path)
            if not reasoning and usage_obj:
                reasoning = cls._get_nested_int(usage_obj, path)
            if reasoning:
                result["reasoning_tokens"] = reasoning
                break

        return result
```

Replace the final `return result` at line 833 with the above block (which includes its own `return result`).

- [ ] **Step 2: Verify the change doesn't break existing tests**

Run: `cd /Users/su4o_/Documents/Codes/nanobot_fork && /opt/anaconda3/envs/nanobot/bin/python -m pytest tests/ -x -q --timeout=30 2>/dev/null || echo "No tests or tests failed"`

Expected: Existing tests pass (or no test suite exists for this method).

- [ ] **Step 3: Commit**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
git add nanobot/providers/openai_compat_provider.py
git commit -m "feat(providers): extract reasoning_tokens from usage response"
```

---

### Task 2: Create validation dataset (25 tasks from 63-task set)

**Files:**
- Create: `eval/datasets/phase0_validation_25.csv`

- [ ] **Step 1: Select 25 tasks from the 63-task tested set**

Select tasks that exercise GUI operations (not pure MCP/web tasks) with a mix of complexity levels:

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
/opt/anaconda3/envs/nanobot/bin/python -c "
import csv, random
random.seed(42)
with open('eval/datasets/clawbench_63_tested.csv') as f:
    reader = csv.DictReader(f)
    tasks = list(reader)
# Select 25 tasks deterministically
selected = random.sample(tasks, min(25, len(tasks)))
with open('eval/datasets/phase0_validation_25.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=tasks[0].keys())
    writer.writeheader()
    writer.writerows(selected)
print(f'Selected {len(selected)} tasks for Phase 0 validation')
"
```

- [ ] **Step 2: Verify the CSV is valid**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
/opt/anaconda3/envs/nanobot/bin/python -c "
import csv
with open('eval/datasets/phase0_validation_25.csv') as f:
    rows = list(csv.DictReader(f))
print(f'Tasks: {len(rows)}')
print(f'Columns: {list(rows[0].keys())}')
print(f'First task: {rows[0][\"task_id\"]}')"
```

Expected: 25 tasks with columns `task_id`, `instruction`, `instruction_ch`.

- [ ] **Step 3: Commit**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
git add eval/datasets/phase0_validation_25.csv
git commit -m "data: add 25-task validation split for Phase 0 signal pilot"
```

---

### Task 3: Create Phase 0 config with thinking mode enabled

**Files:**
- Create: `eval/phase0_config.json`

- [ ] **Step 1: Create config file**

This config uses the dashscope small VL model with `enable_thinking: true`. The specific model name depends on what dashscope offers — `qwen-vl-plus` is the smallest VL model available via API. For local Qwen3VL-8B deployment (NPU server), the provider would be `vllm` with a local endpoint.

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.nanobot/workspace",
      "model": "qwen-vl-plus",
      "provider": "dashscope_thinking",
      "maxTokens": 4096,
      "temperature": 0,
      "maxToolIterations": 15
    }
  },
  "providers": {
    "dashscope_thinking": {
      "apiKey": "${DASHSCOPE_API_KEY}",
      "apiBase": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "extra_body": {
        "enable_thinking": true
      }
    }
  },
  "gui": {
    "model": "qwen-vl-plus",
    "provider": "dashscope_thinking",
    "backend": "adb",
    "maxSteps": 15,
    "artifactsDir": "~/.nanobot/workspace/phase0_runs",
    "stagnationLimit": 3,
    "adb": {
      "serial": "APH0219524034229"
    },
    "agentProfile": "default",
    "capture_ttft": true,
    "imageScaleRatio": 1.0
  }
}
```

**Note:** Before running, verify which dashscope model supports thinking mode for VL tasks. If `qwen-vl-plus` doesn't support it, try `qwen3-vl-8b` or check dashscope docs. The model name may need adjustment.

- [ ] **Step 2: Commit**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
git add eval/phase0_config.json
git commit -m "config: add Phase 0 config with thinking mode enabled"
```

---

### Task 4: Create the Phase 0 signal collection script

**Files:**
- Create: `eval/phase0_signal_pilot.py`

- [ ] **Step 1: Write the signal collection script**

This script runs tasks from the validation set, collects per-step metrics from the trajectory, and saves structured results. It does NOT require confidence in the prompt yet (Task 5 adds that). This version collects reasoning_tokens and timing metrics.

```python
#!/opt/anaconda3/envs/nanobot/bin/python
"""
Phase 0 Signal Validation Pilot — Collect per-step metrics from small model.

Runs validation tasks with thinking mode enabled, extracts per-step:
- reasoning_tokens (from usage)
- model_output (raw text, to later extract confidence)
- action details
- task-level success/failure

Usage:
    python eval/phase0_signal_pilot.py eval/datasets/phase0_validation_25.csv \
        --config eval/phase0_config.json \
        --output eval/phase0_results.jsonl
"""

from __future__ import annotations

import asyncio
import csv
import json
import sys
import time
from pathlib import Path

TASK_TIMEOUT_S = 300
INTER_TASK_PAUSE_S = 5
TRACES_DIR = Path("~/.nanobot/workspace/phase0_runs").expanduser()

DEVICE_PREFIX = (
    "你是一个手机助手agent，已连接鸿蒙手机（ADB serial: APH0219524034229）。\n"
    "可以使用所有可用工具（gui_task）完成任务。\n"
    "需要在手机屏幕上操作时使用 gui_task。\n"
    "【鸿蒙系统】打开App：在桌面向下滑动唤出搜索框，输入App名称打开；没有应用抽屉，禁止向上滑。\n"
    "直接执行，不要询问确认。\n\n"
    "任务：\n"
)


def find_latest_trace(before_ts: float) -> Path | None:
    """Find the most recently modified trace file created after before_ts."""
    if not TRACES_DIR.exists():
        return None
    traces = sorted(
        TRACES_DIR.rglob("trace.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for t in traces:
        if t.stat().st_mtime > before_ts:
            return t
    return None


def extract_step_signals(trace_path: Path) -> list[dict]:
    """Extract per-step signals from a trajectory JSONL file."""
    steps = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        event_type = event.get("type") or event.get("event")
        if event_type != "step":
            continue

        usage = event.get("token_usage") or {}
        steps.append({
            "step_index": event.get("step_index"),
            "action": event.get("action", {}),
            "model_output": event.get("model_output", ""),
            "reasoning_tokens": usage.get("reasoning_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "duration_s": event.get("duration_s"),
            "chat_latency_s": event.get("chat_latency_s"),
            "ttft_s": event.get("ttft_s"),
            "foreground_app": (event.get("observation") or {}).get("foreground_app"),
        })
    return steps


async def run_single_task(agent, task_id: str, instruction: str) -> dict:
    """Run a single task and return structured result with per-step signals."""
    before_ts = time.time()
    response = ""
    error = None

    try:
        result = await asyncio.wait_for(
            agent.process_direct(
                DEVICE_PREFIX + instruction,
                session_key=f"phase0:{task_id}",
                channel="cli",
                chat_id="phase0_pilot",
            ),
            timeout=TASK_TIMEOUT_S,
        )
        response = result.content if hasattr(result, "content") else str(result)
    except asyncio.TimeoutError:
        error = "TIMEOUT"
        response = "TIMEOUT"
    except Exception as exc:
        error = str(exc)
        response = f"ERROR: {exc}"

    # Find and parse trace
    trace_path = find_latest_trace(before_ts)
    step_signals = extract_step_signals(trace_path) if trace_path else []

    return {
        "task_id": task_id,
        "instruction": instruction,
        "response": response[:500],
        "error": error,
        "trace_path": str(trace_path) if trace_path else None,
        "num_steps": len(step_signals),
        "steps": step_signals,
        "duration_s": time.time() - before_ts,
    }


async def main(csv_path: str, config_path: str, output_path: str) -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from nanobot.nanobot import Nanobot

    agent = Nanobot.from_config(config_path)

    with open(csv_path) as f:
        tasks = list(csv.DictReader(f))

    print(f"Phase 0 Signal Pilot: {len(tasks)} tasks")
    print(f"Config: {config_path}")
    print(f"Output: {output_path}")
    print("=" * 60)

    results = []
    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        instruction = task.get("instruction_ch") or task["instruction"]
        print(f"\n[{i+1}/{len(tasks)}] {task_id}")

        result = await run_single_task(agent, task_id, instruction)
        results.append(result)

        # Write incrementally
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"  Steps: {result['num_steps']}, Error: {result['error']}")
        if result["steps"]:
            reasoning_tokens = [s["reasoning_tokens"] for s in result["steps"]]
            print(f"  Reasoning tokens per step: {reasoning_tokens}")

        await asyncio.sleep(INTER_TASK_PAUSE_S)

    # Summary
    total_steps = sum(r["num_steps"] for r in results)
    has_reasoning = sum(1 for r in results for s in r["steps"] if s["reasoning_tokens"] > 0)
    print(f"\n{'=' * 60}")
    print(f"Complete: {len(results)} tasks, {total_steps} total steps")
    print(f"Steps with reasoning_tokens > 0: {has_reasoning}/{total_steps}")
    if has_reasoning == 0:
        print("WARNING: No reasoning tokens detected. Thinking mode may not be supported.")
        print("Proceed with confidence-only signal path.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 0 Signal Validation Pilot")
    parser.add_argument("csv_path", help="Path to validation task CSV")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--output", default="eval/phase0_results.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    # Clear output file
    Path(args.output).unlink(missing_ok=True)
    asyncio.run(main(args.csv_path, args.config, args.output))
```

- [ ] **Step 2: Verify syntax**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
/opt/anaconda3/envs/nanobot/bin/python -c "import ast; ast.parse(open('eval/phase0_signal_pilot.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
git add eval/phase0_signal_pilot.py
git commit -m "feat(eval): add Phase 0 signal collection script"
```

---

### Task 5: Create the Phase 0 analysis script

**Files:**
- Create: `eval/phase0_analysis.py`

- [ ] **Step 1: Write the analysis script**

This script reads the results JSONL from Task 4, computes correlations and ROC curves, and outputs threshold recommendations.

```python
#!/opt/anaconda3/envs/nanobot/bin/python
"""
Phase 0 Analysis — Compute signal-failure correlations and determine thresholds.

Reads phase0_results.jsonl (output of phase0_signal_pilot.py) and produces:
- Spearman correlation between reasoning_tokens and step failure
- ROC curve for reasoning_tokens as escalation signal
- Threshold recommendations (T1, theta2)
- Go/no-go decision for each signal

Usage:
    python eval/phase0_analysis.py eval/phase0_results.jsonl \
        --output-dir eval/phase0_report/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def load_step_data(results_path: str) -> list[dict]:
    """Load all step-level data from results JSONL.

    Labels steps as correct/incorrect based on task outcome:
    - If task succeeded: all steps labeled correct
    - If task failed: last 30% of steps labeled incorrect (heuristic)
    """
    all_steps = []
    for line in Path(results_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        task_success = result.get("error") is None  # No error = likely success
        steps = result.get("steps", [])
        num_steps = len(steps)

        for i, step in enumerate(steps):
            # Heuristic: if task failed, last 30% of steps are "incorrect"
            if task_success:
                step_correct = True
            else:
                failure_start = int(num_steps * 0.7)
                step_correct = i < failure_start

            all_steps.append({
                "task_id": result["task_id"],
                "step_index": step["step_index"],
                "reasoning_tokens": step["reasoning_tokens"],
                "completion_tokens": step["completion_tokens"],
                "duration_s": step.get("duration_s") or 0,
                "step_correct": step_correct,
                "task_success": task_success,
            })
    return all_steps


def compute_correlations(steps: list[dict]) -> dict:
    """Compute Spearman correlation between signals and step failure."""
    from scipy import stats

    reasoning_tokens = np.array([s["reasoning_tokens"] for s in steps])
    step_failed = np.array([0 if s["step_correct"] else 1 for s in steps])

    results = {}

    # Signal 1: reasoning_tokens vs failure
    if reasoning_tokens.max() > 0:
        corr, p_value = stats.spearmanr(reasoning_tokens, step_failed)
        results["reasoning_tokens"] = {
            "spearman_corr": float(corr),
            "p_value": float(p_value),
            "significant": p_value < 0.05 and abs(corr) >= 0.3,
            "mean_correct": float(reasoning_tokens[step_failed == 0].mean()),
            "mean_incorrect": float(reasoning_tokens[step_failed == 1].mean()),
        }
    else:
        results["reasoning_tokens"] = {
            "spearman_corr": None,
            "p_value": None,
            "significant": False,
            "note": "No reasoning tokens detected — thinking mode not supported",
        }

    # Signal proxy: completion_tokens vs failure (alternative if no reasoning_tokens)
    completion_tokens = np.array([s["completion_tokens"] for s in steps])
    if completion_tokens.max() > 0:
        corr, p_value = stats.spearmanr(completion_tokens, step_failed)
        results["completion_tokens"] = {
            "spearman_corr": float(corr),
            "p_value": float(p_value),
            "significant": p_value < 0.05 and abs(corr) >= 0.3,
        }

    return results


def compute_roc(steps: list[dict], signal_key: str) -> dict:
    """Compute ROC curve and optimal threshold for a signal."""
    from sklearn.metrics import roc_curve, roc_auc_score

    signal_values = np.array([s[signal_key] for s in steps])
    step_failed = np.array([0 if s["step_correct"] else 1 for s in steps])

    if signal_values.max() == 0 or step_failed.sum() == 0 or step_failed.sum() == len(step_failed):
        return {"auc": None, "optimal_threshold": None, "note": "Insufficient data for ROC"}

    auc = float(roc_auc_score(step_failed, signal_values))
    fpr, tpr, thresholds = roc_curve(step_failed, signal_values)

    # Optimal threshold: maximize Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[optimal_idx])

    return {
        "auc": auc,
        "optimal_threshold": optimal_threshold,
        "optimal_tpr": float(tpr[optimal_idx]),
        "optimal_fpr": float(fpr[optimal_idx]),
        "youden_j": float(j_scores[optimal_idx]),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def plot_roc(roc_data: dict, signal_name: str, output_path: Path) -> None:
    """Plot ROC curve and save to file."""
    import matplotlib.pyplot as plt

    if roc_data.get("auc") is None:
        return

    plt.figure(figsize=(8, 6))
    plt.plot(roc_data["fpr"], roc_data["tpr"], "b-", linewidth=2,
             label=f'{signal_name} (AUC = {roc_data["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.scatter([roc_data["optimal_fpr"]], [roc_data["optimal_tpr"]],
                color="red", s=100, zorder=5,
                label=f'Optimal T={roc_data["optimal_threshold"]:.0f}')
    plt.xlabel("False Positive Rate (unnecessary escalations)")
    plt.ylabel("True Positive Rate (caught failures)")
    plt.title(f"ROC Curve: {signal_name} as Escalation Signal")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved ROC plot: {output_path}")


def generate_report(steps: list[dict], correlations: dict, roc_results: dict, output_dir: Path) -> dict:
    """Generate the final Phase 0 report with go/no-go decisions."""
    report = {
        "summary": {
            "total_steps": len(steps),
            "total_tasks": len(set(s["task_id"] for s in steps)),
            "task_success_rate": sum(1 for s in steps if s["task_success"]) / len(steps) if steps else 0,
            "step_failure_rate": sum(1 for s in steps if not s["step_correct"]) / len(steps) if steps else 0,
        },
        "correlations": correlations,
        "roc": {k: {kk: vv for kk, vv in v.items() if kk not in ("fpr", "tpr", "thresholds")}
                for k, v in roc_results.items()},
        "decisions": {},
    }

    # Decision: Signal 1 (reasoning_tokens)
    rt_corr = correlations.get("reasoning_tokens", {})
    if rt_corr.get("significant"):
        rt_roc = roc_results.get("reasoning_tokens", {})
        report["decisions"]["reasoning_tokens"] = {
            "go": True,
            "threshold_T1": rt_roc.get("optimal_threshold"),
            "rationale": f"Significant correlation (r={rt_corr['spearman_corr']:.3f}, p={rt_corr['p_value']:.4f}), AUC={rt_roc.get('auc', 'N/A'):.3f}",
        }
    else:
        report["decisions"]["reasoning_tokens"] = {
            "go": False,
            "threshold_T1": None,
            "rationale": rt_corr.get("note", "Correlation not significant (r<0.3 or p>0.05)"),
        }

    # Decision: Signal 2 placeholder (confidence — not yet collected, requires prompt modification)
    report["decisions"]["confidence"] = {
        "go": "pending",
        "threshold_theta2": None,
        "rationale": "Confidence score not yet collected. Requires prompt modification to request structured output with confidence field. Will be validated in Phase 0.5 if reasoning_tokens signal is insufficient.",
    }

    # Overall recommendation
    if report["decisions"]["reasoning_tokens"]["go"]:
        report["recommendation"] = "PROCEED with reasoning_tokens as primary escalation signal. Threshold T1 determined from ROC analysis."
    else:
        report["recommendation"] = "PROCEED with confidence-only path. Implement structured confidence output in Phase 0.5 before Phase 1."

    return report


def main(results_path: str, output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Phase 0 Analysis")
    print("=" * 60)

    # Load data
    steps = load_step_data(results_path)
    print(f"Loaded {len(steps)} steps from {len(set(s['task_id'] for s in steps))} tasks")
    print(f"Step failure rate: {sum(1 for s in steps if not s['step_correct'])/len(steps):.1%}")
    print()

    # Correlations
    print("Correlations:")
    correlations = compute_correlations(steps)
    for signal, data in correlations.items():
        if data.get("spearman_corr") is not None:
            print(f"  {signal}: r={data['spearman_corr']:.3f}, p={data['p_value']:.4f}, significant={data['significant']}")
        else:
            print(f"  {signal}: {data.get('note', 'N/A')}")
    print()

    # ROC analysis
    print("ROC Analysis:")
    roc_results = {}
    for signal_key in ["reasoning_tokens", "completion_tokens"]:
        signal_values = [s[signal_key] for s in steps]
        if max(signal_values) > 0:
            roc = compute_roc(steps, signal_key)
            roc_results[signal_key] = roc
            if roc.get("auc") is not None:
                print(f"  {signal_key}: AUC={roc['auc']:.3f}, optimal_threshold={roc['optimal_threshold']:.0f}")
                plot_roc(roc, signal_key, output_dir / f"roc_{signal_key}.png")
            else:
                print(f"  {signal_key}: {roc.get('note', 'N/A')}")
    print()

    # Report
    report = generate_report(steps, correlations, roc_results, output_dir)
    report_path = output_dir / "phase0_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved: {report_path}")
    print()
    print("DECISION:")
    for signal, decision in report["decisions"].items():
        print(f"  {signal}: {'GO' if decision['go'] is True else 'NO-GO' if decision['go'] is False else 'PENDING'}")
        print(f"    {decision['rationale']}")
    print()
    print(f"RECOMMENDATION: {report['recommendation']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 0 Signal Analysis")
    parser.add_argument("results_path", help="Path to phase0_results.jsonl")
    parser.add_argument("--output-dir", default="eval/phase0_report/", help="Output directory")
    args = parser.parse_args()
    main(args.results_path, args.output_dir)
```

- [ ] **Step 2: Verify syntax and imports**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
/opt/anaconda3/envs/nanobot/bin/python -c "import ast; ast.parse(open('eval/phase0_analysis.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: Verify dependencies available**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
/opt/anaconda3/envs/nanobot/bin/python -c "
import numpy; print(f'numpy {numpy.__version__}')
from scipy import stats; print('scipy OK')
from sklearn.metrics import roc_curve, roc_auc_score; print('sklearn OK')
import matplotlib; print(f'matplotlib {matplotlib.__version__}')
"
```

Expected: All imports succeed. If any fail, install with: `/opt/anaconda3/envs/nanobot/bin/pip install scipy scikit-learn matplotlib`

- [ ] **Step 4: Commit**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
git add eval/phase0_analysis.py
git commit -m "feat(eval): add Phase 0 analysis script (correlation + ROC + thresholds)"
```

---

### Task 6: Verify thinking mode support and run pilot

**Files:** None created. This is an execution task.

- [ ] **Step 1: Check if dashscope model supports thinking mode**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
/opt/anaconda3/envs/nanobot/bin/python -c "
import json, urllib.request as req

# Test a simple call with enable_thinking=true
payload = json.dumps({
    'model': 'qwen-vl-plus',
    'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'What is 2+2? Reply briefly.'}]}],
    'extra_body': {'enable_thinking': True},
    'max_tokens': 200,
}).encode()
r = req.Request(
    'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
    data=payload,
    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer sk-1232a27eebe04114a13f62024966e0de'},
    method='POST',
)
try:
    with req.urlopen(r, timeout=30) as resp:
        data = json.loads(resp.read())
    print('Response:', json.dumps(data.get('usage', {}), indent=2))
    print('Has reasoning_tokens:', 'reasoning_tokens' in json.dumps(data))
except Exception as e:
    print(f'Error: {e}')
    print('Try different model name or check if thinking mode is supported')
"
```

If this fails or returns no `reasoning_tokens`:
- Try model names: `qwen3-vl-8b`, `qwen-vl-plus-latest`, `qwen3-vl-plus`
- Check dashscope docs for which VL models support thinking mode
- If NO VL model supports thinking: **Signal 1 is NO-GO**, proceed with confidence-only path

- [ ] **Step 2: Run the pilot on 2-3 tasks first (smoke test)**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
# Create a mini CSV with just 3 tasks
head -4 eval/datasets/phase0_validation_25.csv > /tmp/phase0_mini.csv
/opt/anaconda3/envs/nanobot/bin/python eval/phase0_signal_pilot.py /tmp/phase0_mini.csv \
    --config eval/phase0_config.json \
    --output /tmp/phase0_mini_results.jsonl
```

Verify output has per-step data with reasoning_tokens.

- [ ] **Step 3: Run full pilot (25 tasks)**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
/opt/anaconda3/envs/nanobot/bin/python eval/phase0_signal_pilot.py \
    eval/datasets/phase0_validation_25.csv \
    --config eval/phase0_config.json \
    --output eval/phase0_results.jsonl
```

This will take ~30-60 minutes (25 tasks × 1-3 min each). Monitor progress.

- [ ] **Step 4: Run analysis**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
/opt/anaconda3/envs/nanobot/bin/python eval/phase0_analysis.py \
    eval/phase0_results.jsonl \
    --output-dir eval/phase0_report/
```

- [ ] **Step 5: Review results and commit report**

Check `eval/phase0_report/phase0_report.json` for the go/no-go decision.

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
cat eval/phase0_report/phase0_report.json | python -m json.tool
git add eval/phase0_results.jsonl eval/phase0_report/
git commit -m "data: Phase 0 signal validation results and analysis report"
```

---

### Task 7: Decision checkpoint — determine next steps

**Files:** None. This is a decision point.

- [ ] **Step 1: Evaluate Phase 0 outcomes**

Read `eval/phase0_report/phase0_report.json` and decide:

**If reasoning_tokens signal is GO** (correlation ≥ 0.3, p < 0.05, AUC > 0.6):
- ✅ Use the determined T1 threshold
- Proceed to Phase 1 with reasoning_tokens as primary signal
- Update `eval/phase0_config.json` with the threshold value

**If reasoning_tokens signal is NO-GO**:
- Signal 1 abandoned
- Need Phase 0.5: modify the GUI agent prompt to request structured confidence output
- Then re-run pilot with confidence extraction
- Proceed to Phase 1 with confidence as primary signal

**If BOTH signals are NO-GO** (unlikely but possible):
- Fall back to risk-rules-only escalation (Signal 3 alone)
- Still valuable as a paper contribution if recovery and distillation work

- [ ] **Step 2: Update spec with findings**

```bash
cd /Users/su4o_/Documents/Codes/nanobot_fork
# Update the spec's Phase 0 section with actual results
# (Manually edit based on the analysis output)
git add docs/superpowers/specs/2026-05-14-fastslow-gui-controller-design.md
git commit -m "docs: update spec with Phase 0 signal validation results"
```

---

## Execution Order Summary

```
Task 1: Provider change (reasoning_tokens extraction)     — 5 min
Task 2: Validation dataset creation                       — 5 min
Task 3: Phase 0 config                                    — 5 min
Task 4: Signal collection script                          — 15 min
Task 5: Analysis script                                   — 15 min
Task 6: Run pilot + analysis                              — 60-90 min (mostly waiting)
Task 7: Decision checkpoint                               — 10 min
```

**Total estimated wall-clock time**: ~2 hours (mostly inference waiting time in Task 6).

**Dependencies**: Tasks 1-5 are independent of each other and can be done in parallel. Task 6 depends on all of 1-5. Task 7 depends on 6.
