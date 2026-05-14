# FastSlow-GUI Controller Design Spec

## 1. Problem Statement

Current GUI agents use a single model for all operations — either a large model (high accuracy, high cost, high latency) or a small model (fast, cheap, but unreliable on complex tasks). There is no mechanism to:

- Dynamically allocate model capacity based on operation complexity and risk
- Detect and recover from execution errors mid-task (only post-hoc failure)
- Distill successful complex reasoning into reusable fast-execution patterns

### 1.1 Novelty Positioning: Why This Is Not "GUI-RouteLLM"

**Existing LLM cascade/routing work** (FrugalGPT, RouteLLM, Hybrid LLM, AutoMix) solves an **open-loop** problem: route a query to the right model, get a response, done. There is no feedback from execution outcomes, no error recovery, no learning from past routing decisions.

**Our core novelty is the closed-loop**: routing → execution → error detection → recovery → distillation → improved routing. The three components (adaptive routing, runtime recovery, skill distillation) are not independent contributions — they form an **indivisible feedback loop** where:

1. **Routing without recovery** = open-loop cascade (FrugalGPT territory, incremental)
2. **Recovery without routing** = always use big model for recovery (defeats cost purpose)
3. **Distillation without routing+recovery** = no data source (distillation requires S2 interventions to learn from)
4. **All three together** = closed-loop adaptive system that improves over time

**The paper's single claim**: *In interactive GUI environments, closed-loop model routing — where routing decisions are informed by execution feedback and refined through experience distillation — outperforms both fixed-model and open-loop routing approaches in the success-cost Pareto frontier.*

This positions us clearly against:
- **LLM cascade literature** (FrugalGPT, RouteLLM, AutoMix): We add the closed loop — their systems never look back at execution results
- **GUI agent literature** (CogAgent, AppAgent, DigiRL): We add adaptive model allocation — they use a single model throughout
- **Error recovery literature** (ReAct, Reflexion, SelfRefine): We add the cost dimension — they always use the same expensive model for recovery

### 1.2 On the System 1/System 2 Framing

We **do NOT claim** a cognitive science contribution. The Kahneman analogy is used only as intuition-building vocabulary, not as a theoretical framework.

In the paper, we use precise terminology:
- **"Fast executor"** (not "System 1") — small model + skill cache, optimized for speed
- **"Slow reasoner"** (not "System 2") — large model with extended reasoning, invoked on demand
- **"Adaptive routing with closed-loop correction"** — the actual technical contribution

If a reviewer asks "remove the S1/S2 labels, does anything change?" — the answer is yes: the system implements **metacognitive escalation** (the fast executor monitors its own uncertainty and decides when to escalate), which is a specific dual-process mechanism, not just "use small model first, big model second". But we do not overstate this — the paper's strength is the engineering contribution and empirical results, not a cognitive theory.

In this spec, we continue using S1/S2 for brevity. The paper will use the precise terms above.

### 1.3 Formal Problem Definition

**GUI Task as POMDP (simplified to MDP)**: A GUI automation task is formally a Partially Observable MDP, since the agent observes screenshots but not the full app state (internal data, network responses, background processes). Following standard practice in vision-based sequential decision making (Mnih et al., 2015), we approximate the POMDP as an MDP where observations are treated as states — the agent conditions on the most recent screenshot as if it were the full state.

Under this approximation, the task is a finite-horizon MDP (S, A, T, R, H):

- **S**: Observed state = (screenshot, foreground_app, UI_tree_if_available). Note: this is technically an observation o ∈ Ω, but we treat o as s for tractability.
- **A**: GUI action space = {tap(x,y), type(text), scroll(direction), swipe(start,end), press_home, press_back, done}
- **T**: State transition T(s'|s,a) — deterministic but unknown (governed by OS/app)
- **R**: Sparse reward: R=1 if task completed successfully at horizon H, else R=0
- **H**: Step budget (max_steps, typically 15)

**Dual-Model Routing Problem**: Given:
- Two policies: π_f (fast executor, per-step cost c_f) and π_s (slow reasoner, per-step cost c_s), where c_f << c_s
- A routing function σ: S × History → {π_f, π_s}
- A safety constraint function U: S × A → {U0, U1, U2, U3, U4}

Objective: find σ* that maximizes:

```
J(σ) = E[R(τ)] - λ · E[Cost(τ)]
subject to: ∀(s,a) in τ: U(s,a) ≥ U2 ⟹ σ(s) = π_s
```

where τ is the trajectory under routing policy σ, and λ is the cost-accuracy tradeoff parameter.

**Why heuristic cascade, not learned routing**: Direct optimization of J(σ) is intractable in our setting — the GUI environment is non-differentiable, rewards are sparse (only at task completion), safety constraints are hard, and the action space is continuous (tap coordinates). Learning σ via RL or bandit methods would require thousands of task episodes for convergence, which is infeasible on real devices. We propose a structured heuristic cascade as an approximation, and evaluate its empirical proximity to oracle routing (Config O) as an upper bound. Future work could use the collected trajectories to train a learned router as a warm-started policy.

**Interpretation**: Our three escalation signals (risk classification, reasoning effort, confidence) are features of σ. The cascade logic is a structured approximation of σ*. The ablation experiments (C1-C6) measure which feature combinations best approximate the optimal routing. The Pareto frontier analysis (Section 9.5) shows how different operating points of σ trade off cost vs accuracy across the λ spectrum.

**Two closed loops at different time scales**:

The system operates two distinct feedback loops, each contributing differently:

| Loop | Time Scale | Mechanism | What It Improves |
|------|-----------|-----------|-----------------|
| **Intra-task loop** | Within a single task (step-level) | Route → execute → detect anomaly → recover | Current task's success rate |
| **Inter-task loop** | Across tasks (task-level) | Distill successful S2 paths → skill library → S1 reuses | Future tasks' S2 call rate and latency |

These are **not the same loop**. The intra-task loop (Config D vs C) saves the current task from failure. The inter-task loop (Config E vs D) makes the system cheaper over time. Both are necessary for the full claim: the intra-task loop generates the successful recovery trajectories that the inter-task loop distills into skills. Without recovery, there are fewer successful S2 trajectories to learn from; without distillation, recovered tasks don't benefit future execution.

**Recovery as trajectory correction**: When the monitor detects an anomaly at step t, the system switches to π_s and attempts to find an action sequence that returns to a state s' ∈ S_viable (the set of states from which π_f can complete the task). This is a sub-problem:

```
Recovery(s_error) = argmin_{a_{t:t+k}} k  s.t. s_{t+k} ∈ S_viable, k ≤ budget_recovery
```

**Distillation as policy transfer**: After a successful trajectory τ where π_s intervened, extract a skill ψ = (precondition, action_sequence) such that π_f can execute ψ without escalation in future similar states. Over time, this reduces the expected escalation rate.

## 2. Core Concept: System 1 & System 2 for GUI Agents

Inspired by Kahneman's dual-process theory:

- **System 1 (S1)**: Fast, experience-based execution. Small model (Qwen3VL-8B, local) + cached skills. Low latency, low cost. Handles routine GUI operations.
- **System 2 (S2)**: Slow, deliberative reasoning. Large model (qwen3.6-35b-a3b, API) with thinking mode. Handles complex planning, risk assessment, and error recovery.

S1 and S2 are NOT two separate systems — they are two roles within the same execution flow, with dynamic switching per step.

**Key differentiator**: S2's core mission is **backtracking and correction** (not just planning). The system can detect errors mid-task and recover, rather than failing and restarting.

**Closed loop**: S2's successful complex decisions are distilled into skills that S1 can execute directly next time — the system self-evolves.

## 3. Architecture

### 3.1 Implementation Strategy: Wrapper Controller (Option B)

No modifications to opengui's `GuiAgent` internals. All fast/slow logic lives in the `gui_task` tool layer through:

1. A custom LLM adapter that routes calls dynamically
2. An event callback that monitors execution state
3. External controllers for backtracking and skill distillation

```
gui_task.execute(task)
  |
  +-- FastSlowLLMAdapter (replaces NanobotLLMAdapter)
  |   +-- S1 Provider: Qwen3VL-8B (local, no thinking or light thinking)
  |   +-- S2 Provider: qwen3.6-35b-a3b (API, full thinking mode)
  |   +-- EscalationPolicy: decides routing per chat() call
  |
  +-- StepMonitor (via event_callback)
  |   +-- Listens to every step: action, token usage, duration, confidence
  |   +-- Computes escalation signals
  |   +-- Triggers backtrack assessment
  |
  +-- BacktrackController
  |   +-- CheckpointManager: saves state snapshots at key transitions
  |   +-- Forward recovery: attempt correction from current state
  |   +-- Checkpoint rollback: navigate back to saved state
  |   +-- Abort & replan: give up current approach
  |
  +-- SkillDistiller
      +-- Monitors successful trajectories
      +-- Extracts structured operation sequences + semantic descriptions
      +-- Writes to SkillLibrary for S1 reuse
```

### 3.2 Why Wrapper (not invasive or rewrite)

- **Isolation from colleague's framework work**: No changes to opengui core code
- **Reuses existing capabilities**: opengui's grounding, trajectory recorder, skill executor
- **Pluggable**: Can be disabled to fall back to single-model behavior
- **Gradual migration path**: Can go deeper into opengui internals later if needed

### 3.3 Key Hook Points in Existing Code

| Hook | Location | Usage |
|------|----------|-------|
| `NanobotLLMAdapter` | `gui_adapter.py` | Subclass to create FastSlowLLMAdapter |
| `event_callback` | `TrajectoryRecorder._write_event()` | Real-time step monitoring |
| `progress_callback` | `GuiAgent._run_step()` | Step-level progress tracking |
| `intervention_handler` | `GuiAgent.__init__` | Could be used for S2 intervention |
| `SkillLibrary` | `opengui/skills/library.py` | Write distilled skills |
| `SkillExecutor` | `opengui/skills/executor.py` | Execute distilled skills via S1 |

## 4. S1/S2 Switching Mechanism

### 4.1 Three Escalation Signals

#### Signal 1: Reasoning Effort as Uncertainty Proxy (hypothesis — requires validation)

When S1 (small model) has thinking mode enabled, the length of `<think>...</think>` output reflects reasoning effort. The **hypothesis** is that for a fixed-capacity model, longer reasoning chains on GUI tasks correlate with higher failure probability — not because reasoning is bad, but because tasks that force a small model to reason extensively are likely beyond its reliable operating range.

- **Source**: `response.usage.reasoning_tokens` or parsed `<think>` segment length
- **Critical caveat**: Longer reasoning does NOT automatically mean the model is struggling. It correlates with task complexity. The claim is specifically: *for a given small model, there exists a threshold beyond which additional reasoning yields diminishing returns and higher error rates*. This must be empirically validated before use (see Phase 0 below).
- **If Qwen3VL-8B does not support thinking mode**: This signal is unavailable. In that case, Signal 2 (confidence) becomes the primary dynamic signal alongside Signal 3 (risk rules). This is NOT a fallback — it is an equal-status alternative path.

**Phase 0 validation requirement** (before Phase 1 implementation):
1. Run S1 (small model) on 40+ ClawBench tasks with thinking mode enabled
2. Record per-step: reasoning_tokens, task success/failure, action correctness
3. Compute Spearman correlation between reasoning_tokens and step failure rate
4. If correlation < 0.3 or p > 0.05: abandon this signal, use confidence-based path
5. If correlation significant: determine T1 via ROC curve (optimal operating point)

#### Signal 2: Explicit Confidence Score (equal-status signal)

Prompt requires structured output including a confidence field:

```json
{
  "confidence": 0.72,
  "action": {"action_type": "tap", "target": "confirm button"}
}
```

If `confidence < theta2`, escalate.

- **Caveat**: Small models are often poorly calibrated (overconfident on wrong answers). Need ECE (Expected Calibration Error) measurement and potential post-hoc calibration (temperature scaling on a held-out validation set).
- **Advantage**: Interpretable, easy to ablate, works regardless of thinking mode support.
- **Theta2 determination**: Not a magic number — derived from pilot experiment ROC analysis on validation split. Initial sweep range: [0.5, 0.6, 0.7, 0.8].

#### Signal 3: Page-Context-Aware Risk Classification

Risk assessment using the U0-U4 taxonomy. **Critical**: the same action type can have different risk levels depending on page context (e.g., "tap confirm" on a notification vs on a payment page).

- **U0 (safe)**: Open app, scroll, navigate — S1 handles
- **U1 (reversible)**: Type in search field, toggle setting — S1 handles
- **U2 (user-visible)**: Send message, post content — force S2
- **U3 (irreversible)**: Delete account, confirm payment — force S2
- **U4 (privacy/security)**: Access credentials, share location — force S2

**Implementation** (NOT keyword matching alone):

The risk classifier takes as input the **full step context**: (screenshot, proposed_action, foreground_app, page_description). It outputs a risk level. Two implementation options:

**Option A (lightweight, preferred for Phase 1)**: S1 itself outputs a `risk_assessment` field alongside confidence and action. The prompt explicitly asks: "Rate the irreversibility of this action: U0/U1/U2/U3/U4". This piggybacks on the existing S1 inference call — zero additional latency.

**Option B (more accurate, Phase 2)**: A dedicated lightweight classifier (e.g., fine-tuned on human-annotated risk labels from Phase 0 trajectories). Runs in parallel with S1 inference.

**Validation**: Report risk classification accuracy as a sub-experiment. Ground truth: human annotation of 200+ step-level risk labels from Phase 0 trajectories. Report confusion matrix between predicted and actual risk levels.

### 4.2 Cascade Decision Logic

```python
def should_escalate(step_result, risk_level, mode="production") -> EscalationDecision:
    # U3/U4: hard safety constraint, always S2
    if risk_level >= U3:
        return EscalationDecision(escalate=True, reason="safety_hard_rule")

    # U2: production forces S2; experiment mode runs S1 in shadow
    if risk_level == U2:
        if mode == "experiment":
            # Shadow mode: S1 generates action but does NOT execute.
            # S2 generates and executes. Both recorded for counterfactual analysis.
            return EscalationDecision(
                escalate=True,
                reason="risk_u2",
                shadow_s1=True,  # Record what S1 would have done
            )
        return EscalationDecision(escalate=True, reason="risk_u2")

    # Dynamic signals (only for U0-U1)
    if step_result.reasoning_tokens and step_result.reasoning_tokens > T1:
        return EscalationDecision(escalate=True, reason="thinking_overflow")

    if step_result.confidence and step_result.confidence < theta2:
        return EscalationDecision(escalate=True, reason="low_confidence")

    return EscalationDecision(escalate=False)
```

**Shadow mode for U2** (experiment only): S1 still runs inference on U2 steps but its action is NOT executed. S2's action is executed. Both outputs are logged, enabling post-hoc analysis of whether S1 could have handled U2 tasks. This provides the counterfactual data needed to evaluate escalation precision at U2 level. In production mode, U2+ always uses S2.

Cascade (not weighted) because U3/U4 safety constraints must not be diluted by other signals.

### 4.3 S2 -> S1 Fallback: Sub-goal Granularity

~~After S2 intervention, default back to S1 on the next step.~~

**Revised**: S2 does not hand back control step-by-step. Instead, S2 operates at **sub-goal granularity**:

1. When S2 is invoked, it first assesses the remaining task and decomposes it into sub-goals
2. S2 returns a `handoff` decision:
   - `{"retain": 3, "reason": "next 3 steps are a coordinated sequence (fill form + submit)"}` — S2 keeps control for N steps
   - `{"retain": 0, "reason": "isolated correction done, S1 can resume"}` — immediate handoff
   - `{"retain": -1, "reason": "task requires sustained complex reasoning"}` — S2 keeps control until sub-goal completion signal, **subject to max_s2_retain_steps hard cap**
3. S1 resumes only after S2 explicitly releases control

**Hard cap**: `max_s2_retain_steps` (default: 8). If S2 has retained control for this many consecutive steps without releasing, control is forcibly returned to S1. This prevents S2 from consuming the entire step budget on a single sub-goal. If S2 hits the cap, a warning is logged and the system treats it as an implicit handoff — S1 resumes, and if S1 immediately re-escalates, that counts as a new S2 session with a fresh budget.

**S2 multi-step context strategy**: During an S2 retain period, S2 operates within opengui's native conversation history — each step sees the full message history from GuiAgent's `_build_messages()`, including its own prior actions and screenshots. The `EscalationContext` (Section 4.5) is only used for the initial S1→S2 handoff. Subsequent S2 steps are standard GuiAgent iterations routed through the large model, preserving reasoning continuity.

This prevents oscillation: a 5-step complex sequence stays entirely in S2, not bouncing S1→S2→S1→S2→S1.

### 4.4 Small Model's Own Fast/Slow

Within S1 itself, the small model operates in two modes:

- **S1-Fast**: Direct action output, minimal thinking tokens. For routine operations where skills match or the step is obvious.
- **S1-Slow**: Thinking mode enabled, more reasoning tokens. For steps that need some deliberation but don't warrant full S2 escalation.

The thinking token threshold T1 is the boundary between "S1-Slow is sufficient" and "need S2".

### 4.5 S1 → S2 Context Passing Protocol

When escalation occurs, S2 must receive enough context to make an effective decision without re-doing all of S1's work. Too little context and S2 is blind; too much and we pay unnecessary latency/token costs.

**Escalation context payload**:

```python
@dataclass
class EscalationContext:
    # Always included (lightweight)
    task_description: str                  # Original task
    current_screenshot: bytes              # Current screen state
    current_step_index: int                # Where we are in the trajectory
    escalation_reason: str                 # Why S1 escalated
    foreground_app: str                    # Current app
    s1_proposed_action: dict | None        # What S1 was going to do (may be None if S1 couldn't decide)
    s1_confidence: float | None            # S1's confidence if available

    # Included for backtracking (adds ~2-3 screenshots)
    recent_trajectory: list[StepSummary]   # Last 3 steps: {action, screenshot_path, success}
    checkpoints: list[CheckpointSummary]   # Available rollback points: {step_index, app, screenshot_path}

    # NOT included (too expensive)
    # full_trajectory: ...                 # All steps from the beginning — too many tokens
    # raw_model_outputs: ...              # S1's full reasoning — irrelevant to S2
```

**S2 prompt template** (injected as system context):

```
You are the slow reasoner of a dual-model GUI agent. The fast executor has
escalated control to you because it encountered a situation beyond its
reliable operating range.

Task: {task_description}
Current step: {current_step_index}/{max_steps}
Escalation reason: {escalation_reason}
Fast executor's proposed action: {s1_proposed_action} (confidence: {s1_confidence})
Current app: {foreground_app}

Recent trajectory (last 3 steps):
{formatted_recent_trajectory}

Available checkpoints for recovery:
{formatted_checkpoints}

Your responsibilities:
1. Assess the current situation from the screenshot
2. Decide: execute the proposed action / override with better action / initiate recovery
3. Indicate how many subsequent steps you need to retain control (0 = hand back immediately)
```

**Token budget**: Escalation context is capped at ~2000 tokens text + 3 images (current + last 2 screenshots). This is a fixed overhead per escalation, not per step.

## 5. Backtracking & Correction

### 5.1 Anomaly Detection (StepMonitor)

StepMonitor watches for these signals via `event_callback`:

- **Stagnation**: Same action repeated 2+ consecutive steps
- **No-effect action**: Action executed but state unchanged. Detection uses **structured state comparison** (primary) + pixel similarity (supplementary):
  - Primary: `foreground_app` unchanged AND no new UI elements detected (if UI tree available via accessibility service)
  - Supplementary: screenshot structural similarity (SSIM) > threshold, BUT filtered for known false positives: loading spinners, clock updates, cursor blink, toast notifications. These are detected by checking if changes are confined to small, known-volatile screen regions.
  - Note: Some effective actions produce visually subtle changes (e.g., toggling a small switch). Structured state comparison catches these; pixel comparison alone does not.
- **Unexpected app switch**: `foreground_app` changed without a navigation action
- **Confidence crash**: S1 confidence drops by > 0.3 compared to previous step
- **Max step approaching**: Current step > 70% of max_steps without significant progress

### 5.2 Three-Level Recovery

#### Level 1: Forward Recovery

S2 analyzes the current screenshot and generates a recovery sub-goal.

- Example: Accidentally opened wrong menu -> S2 says "go back to main page" -> executes 1-2 corrective actions
- Implementation: Reuse opengui's existing `subgoal_runner` (already supports mini vision-action loops for recovery)
- Trigger: Mild anomaly (stagnation, no-effect action)

#### Level 2: Navigational Recovery to Checkpoint

S2 determines forward recovery is insufficient. **Navigate back** to a previously observed state through GUI actions (NOT a memory-level state restore — GUI state is not serializable).

**Critical distinction**: This is "navigational recovery", not "state rollback". We cannot restore() a GUI state. We can only attempt to navigate back to a state that visually and functionally matches a saved checkpoint. This navigation itself may fail or produce side effects.

- Example: 3 steps deep in wrong app -> press home -> reopen correct app -> navigate to target page
- Implementation: `CheckpointManager` saves state metadata at key transitions:
  - Entering a new app
  - Completing a sub-step
  - Before high-risk operations
- Storage: Screenshot + foreground_app + page features + step index
- Max 5 checkpoints retained (circular buffer)
- Trigger: Forward recovery failed, or severe anomaly (wrong app, confidence crash)

**Recovery verification**: After navigational recovery, S2 MUST verify arrival by comparing current screenshot against the checkpoint screenshot (using VLM-based comparison, not pixel matching — the page may have different dynamic content but same functional state). If verification fails, the recovery attempt is marked as failed and falls through to Level 3.

**Recovery step budget**: Navigational recovery is capped at 5 steps. If the agent cannot reach a checkpoint-like state in 5 steps, the attempt is abandoned.

**Recovery failure path**: If navigational recovery fails (verification fails or step budget exceeded), proceed to Level 3 (abort). Do NOT attempt a second checkpoint — cascading recovery attempts waste step budget.

#### Level 3: Abort & Replan

Current approach is fundamentally wrong. Return control to the upper layer.

- Example: App doesn't support the required operation -> need alternative app/approach
- Implementation: Return `task_failed` with structured reason to AgentLoop
- AgentLoop can decide to retry with different strategy or report failure
- Trigger: Checkpoint rollback failed, or S2 explicitly determines no viable path

### 5.3 Checkpoint Strategy

```python
class CheckpointManager:
    max_checkpoints: int = 5

    def should_checkpoint(self, event: dict) -> bool:
        """Save checkpoint at state transitions, not every step."""
        return (
            event.get("foreground_app") != self.last_app  # App changed
            or event.get("phase") != self.last_phase       # Phase changed
            or self.steps_since_checkpoint >= 3             # Periodic safety net
        )

    def save(self, screenshot, foreground_app, step_index, page_features):
        """Circular buffer, oldest checkpoint dropped when full."""
        ...

    def get_nearest(self) -> Checkpoint:
        """Get most recent valid checkpoint for rollback."""
        ...
```

## 6. Skill Distillation (S2 -> S1 Feedback Loop)

### 6.1 Extraction Conditions

A trajectory is eligible for skill distillation when ALL of:

- Task completed successfully (`success=True`)
- S2 was invoked at least once during execution (trajectory has learning value)
- Total steps <= 10 (longer sequences too fragile to replay)
- No abort/replan occurred (clean execution path)

### 6.2 Distillation Process

1. **Structured recording**: Extract per-step `{action_type, semantic_target_description, parameters}` from trajectory. **Coordinates are NOT stored** — they are resolution-dependent and layout-dependent.
2. **Semantic description**: S2 (large model) generates natural language skill summary:
   - "Search for contact in WeChat and send message: tap search icon -> type name in search field -> select contact from results -> type content in message input -> tap send button"
3. **Preconditions**: Starting app, starting page state description
4. **Parameterization**: Extract variable parts as parameters (contact_name, message_content)
5. **Grounding strategy**: Skills store **semantic targets** (e.g., "search icon", "send button"), NOT coordinates. At execution time, S1 re-grounds each semantic target to screen coordinates using the current screenshot. This is consistent with opengui's existing `SkillExecutor` which already supports LLM-based action grounding per step (`grounding_mode="llm"`). No new grounding capability needed — we reuse what exists.

### 6.3 Storage Format

Reuse opengui's existing `Skill` and `SkillStep` data structures:

```python
Skill(
    skill_id="distilled_wechat_search_contact_001",
    name="Search and message contact in WeChat",
    description="Open WeChat, search for a contact by name, and send a text message",
    app="weixin",
    platform="harmony",
    steps=(
        SkillStep(action_type="tap", target="search_icon", expected_state="search page visible"),
        SkillStep(action_type="type", target="search_field", parameters={"text": "{contact_name}"}),
        SkillStep(action_type="tap", target="{contact_name}_result", expected_state="chat page"),
        SkillStep(action_type="type", target="message_input", parameters={"text": "{message}"}),
        SkillStep(action_type="tap", target="send_button", expected_state="message sent"),
    ),
    parameters=("contact_name", "message"),
    tags=("distilled", "wechat", "messaging"),
    success_count=1,
    success_streak=1,
)
```

Additional metadata field: `source: "distilled"` to distinguish from manually defined skills.

### 6.4 Skill Lifecycle

- **Creation**: Distilled after first successful S2-assisted execution
- **Validation**: success_count / failure_count tracked across uses
- **Invalidation**: 3 consecutive failures -> skill marked inactive, S2 re-engaged
- **Proactive invalidation**: If app version change detected (via `adb shell dumpsys package` versionCode or equivalent), all skills for that app are marked "unverified". Next use triggers a single-step validation before full execution. This avoids wasting 3 runs on skills broken by UI updates.
- **Update**: If S2 finds a better path for an existing skill, update the skill steps

### 6.6 What Skills Actually Save (not inference count)

A 5-step skill still requires 5 S1 inference calls for per-step grounding. The value of skills is NOT reducing inference count. It is:

1. **Eliminating action planning uncertainty**: S1 only needs to ground "where" (visual grounding), not decide "what to do next" (action planning). This removes the escalation trigger — S1 stays below the confidence/thinking-token thresholds.
2. **Reducing per-step reasoning tokens**: Grounding-only inference uses fewer tokens than planning+grounding inference. Estimated saving: 30-50% tokens per step (to be measured in Phase 1).
3. **Providing verification checkpoints**: Each skill step has `expected_state`, enabling early error detection without waiting for task-level failure.
4. **Primary value: reducing S2 call rate**: The main measurable benefit is fewer escalations to the expensive large model. Tasks that previously required S2 intervention now execute entirely on S1.

If latency savings from (2) are not significant in practice, the paper should frame skill distillation's contribution as **cost reduction** (fewer S2 calls), not speed improvement.

### 6.5 S1 Reuse Flow

**Skill matching mechanism**: Embedding cosine similarity between the incoming task description and stored skill descriptions. Uses the embedding model already configured in opengui (`NanobotEmbeddingAdapter`, initialized in `GuiSubagentTool.__init__`). Match threshold: cosine similarity > 0.85 (tunable). This adds ~50ms per task start (single embedding call), not per step.

If no embedding model is configured, falls back to opengui's existing `SkillReuser` which uses an LLM call for matching (~1-2s, but more accurate for ambiguous queries).

```
New task arrives
  -> SkillReuser.find() semantic match against SkillLibrary
  -> Match found (similarity > threshold)
     -> S1 executes skill via SkillExecutor (existing opengui pipeline)
     -> Each step validated against expected_state
     -> If validation fails -> invoke S2 for recovery (Level 1)
  -> No match
     -> Normal S1 execution with escalation monitoring
```

## 7. New Files & Components

All new code goes into a new directory: `nanobot/agent/fastslow/`

```
nanobot/agent/fastslow/
  __init__.py
  adapter.py          # FastSlowLLMAdapter (subclass of NanobotLLMAdapter)
  policy.py           # EscalationPolicy (cascade decision logic)
  monitor.py          # StepMonitor (event_callback handler)
  checkpoint.py       # CheckpointManager (state snapshots)
  backtrack.py        # BacktrackController (3-level recovery)
  distiller.py        # SkillDistiller (trajectory -> skill extraction)
  config.py           # FastSlowConfig (thresholds, feature flags)
```

Modifications to existing files:

- `nanobot/agent/tools/gui.py`: In `GuiSubagentTool.__init__()` and `_run_task()`, optionally use `FastSlowLLMAdapter` instead of `NanobotLLMAdapter` when `fastslow.enabled=True` in config
- `nanobot/config/schema.py`: Add `FastSlowConfig` section to the config schema

No changes to opengui internals.

## 8. Configuration

```json
{
  "fastslow": {
    "enabled": true,
    "s1_model": "qwen3vl-8b",
    "s1_provider": "openai_compat",
    "s1_api_base": "http://localhost:8000/v1",
    "s2_model": "qwen3.6-35b-a3b",
    "s2_provider": "openai_compat",
    "s2_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "thinking_token_threshold": null,
    "confidence_threshold": null,
    "risk_escalation_level": "U2",
    "max_checkpoints": 5,
    "navigational_recovery_step_budget": 5,
    "s2_handoff_mode": "subgoal",
    "max_s2_retain_steps": 8,
    "shadow_mode": false,
    "skill_distillation_enabled": true,
    "skill_max_steps_for_distillation": 10,
    "skill_failure_invalidation_streak": 3
  }
}
```

**Threshold values** (`thinking_token_threshold`, `confidence_threshold`): Set to `null` initially. Determined by Phase 0 pilot experiment via ROC analysis. Will be populated with empirically validated values before Phase 1 deployment. Sensitivity analysis across +/-20% of optimal threshold will be reported in the paper.

## 9. Experiment Design

### 9.1 Baselines

**Internal baselines** (ablation of our system):

| Config | Description |
|--------|-------------|
| A: Large-only | Pure qwen3.6-35b-a3b, no S1/S2 switching (upper bound accuracy) |
| B: Small-only | Pure Qwen3VL-8B, no escalation (lower bound accuracy, best cost) |
| C: S1+S2 switch only | Escalation enabled, no backtracking, no skill distillation |
| D: S1+S2+backtrack | + backtracking/recovery |
| E: Full system | + skill distillation (complete FastSlow-GUI) |
| R: Random escalation | Each step escalates to S2 with probability p (matched to C's escalation rate). Proves signals outperform random |
| O: Oracle escalation | Human-annotated per-step labels: "S1 sufficient" vs "needs S2". Measures the ceiling — how good could escalation be with perfect information |

**External baselines** (SOTA GUI agents, mandatory):

| Agent | Why Compare | How to Run |
|-------|-------------|------------|
| AppAgent (Yang et al., 2024) | Competing "learning from exploration" mechanism — their offline learning vs our runtime distillation | Open-source. Run on same benchmark with same device/backend |
| CogAgent (Hong et al., 2024) | Single-model SOTA grounding — must prove dual-model routing beats a stronger single model | Open-source. Run CogAgent-18B as single-model baseline |
| DigiRL (Bai et al., 2024) | RL-based self-correction — competing recovery mechanism | If code available; otherwise cite published numbers on overlapping benchmarks |

**Minimum for submission**: AppAgent + CogAgent on the same test set with identical evaluation protocol. DigiRL numbers from their paper if direct comparison on same benchmark is infeasible.

**Oracle annotation protocol** (Config O):
- Scope: 40 test tasks × ~10 steps/task ≈ 400 step-level annotations
- Annotators: Minimum 2 independent annotators
- Annotation guide: For each step, annotator sees (screenshot, S1-proposed action, S2-proposed action, task description) and labels: "S1 sufficient" (S1's action would succeed) or "needs S2" (S1's action would fail or cause harm). Annotators do NOT see which action was actually executed.
- Agreement: Report Cohen's κ; require κ > 0.6 for inclusion. Disagreements resolved by third annotator.
- If only 1 annotator available (resource constraint): acknowledge in Limitations, report intra-annotator consistency via re-annotation of 20% sample after 1 week.

### 9.2 Metrics

| Metric | Description | Why |
|--------|-------------|-----|
| Task Success Rate | % of tasks completed correctly | Core effectiveness |
| Big Model Call Rate | % of steps that used S2 | Cost efficiency |
| Avg Latency per Task | Wall-clock time per task | Speed |
| Cost per Task | API token cost per task | Practical deployment |
| Escalation Precision | When escalated, was S2 actually needed? | Signal quality |
| Escalation Recall | When S2 was needed, did we escalate? | Signal quality |
| Recovery Success Rate | % of backtrack attempts that recovered | Correction value |
| Skill Reuse Rate | % of tasks that used distilled skills | Learning value |
| Escalation F1 | Harmonic mean of precision and recall | Overall signal quality |
| Effective Step Utilization | productive_steps / total_steps (recovery steps excluded) | Whether recovery wastes budget |

**Escalation ground truth definition**: A step "needed S2" if: (a) S1 produced an incorrect action (verified by oracle annotation or by comparing S1-shadow output against task outcome), OR (b) the step was in a recovery sequence triggered by a prior S1 error. Oracle baseline (Config O) provides the reference labels.

**Statistical rigor**:
- Each configuration: minimum 5 independent runs (different random seeds for any stochastic components)
- Report mean +/- 95% confidence interval for all metrics
- Use paired bootstrap test (n=10000) for significance between configurations
- With 40 tasks x 5 runs = 200 task-runs per config, power analysis at effect size d=0.4, alpha=0.05 yields power > 0.8 for success rate comparisons

### 9.3 Benchmarks and Dataset Split

**Primary benchmark: AndroidWorld** (Android emulator, ADB backend)

AndroidWorld (Rawles et al., 2024) is the current standard benchmark in the GUI agent community (116 tasks, Google-backed, widely adopted). Running on Android emulator is fully compatible with opengui's `adb` backend. This is the main reported benchmark for CCF-A submission.

**Secondary benchmark: ClawBench** (HarmonyOS + cross-platform validation)

ClawBench (our eval pipeline, 40-task fullmodel split) serves as:
- Cross-platform generalization evidence (HarmonyOS, different from AndroidWorld's Android)
- Additional task diversity (different app ecosystem)
- If ClawBench is not yet published: describe task distribution in detail (app types, step count distribution, risk level distribution) and commit to open-sourcing

**Dataset split for threshold selection** (preventing data leakage):

| Split | Source | Size | Purpose |
|-------|--------|------|---------|
| Validation | 63-task ClawBench tested set | ~25 tasks | Phase 0: ROC analysis, threshold selection, signal correlation |
| Test (primary) | AndroidWorld | 116 tasks | Phase 1+: All main reported metrics, SOTA comparison |
| Test (secondary) | ClawBench 40-task fullmodel split | 40 tasks | Cross-platform generalization |

Thresholds (T1, theta2) are determined ONLY on the validation split and frozen before any test-set evaluation.

**Platform priority**:
- **Android (emulator)**: Co-primary platform. All SOTA baselines (AppAgent, CogAgent) run here. Main reported numbers.
- **HarmonyOS (real device)**: Cross-platform evidence. Shows the system is device-agnostic.
- **iOS (real device)**: Additional cross-platform evidence if time permits.

### 9.4 Evaluation Runs

Use existing ClawBench eval pipeline (`eval/direct_eval.py`) with multi-run evaluation on held-out test set. Each configuration runs 5 independent times. Results reported as mean ± 95% CI.

### 9.5 Generalization Experiments

To demonstrate the system is not overfit to one model pair or one platform:

| Dimension | Primary | Secondary | Purpose |
|-----------|---------|-----------|---------|
| S1 model | Qwen3VL-8B (local) | InternVL2-8B (local) | Prove routing works across model architectures |
| S2 model | qwen3.6-35b-a3b | GPT-4o (if budget allows) | Prove S2 is substitutable |
| Platform | Android emulator (ADB) | HarmonyOS (HDC, real) + iOS (real) | Cross-platform: Android is primary for SOTA comparison |

**Minimum requirement for submission**: All main experiments on Android (AndroidWorld). At least one secondary S1 model on the same test set. HarmonyOS results as cross-platform evidence.

**Android emulator**: Use Android Studio Emulator or Genymotion with ADB backend. opengui's `adb` backend is directly compatible. This gives us 3 platforms without additional hardware.

### 9.6 Pareto Frontier Analysis

Plot a **cost-accuracy Pareto curve** (x-axis: avg cost per task, y-axis: task success rate). Each configuration is a point:

- Config A (large-only): high accuracy, high cost (upper-right)
- Config B (small-only): low cost, low accuracy (lower-left)
- Configs C/D/E: should be on or near the Pareto frontier (better tradeoff than A and B)
- Config R (random): should be dominated by C (below the frontier)

To trace the full frontier, vary the escalation threshold (sweep theta2 from 0.0 to 1.0, or vary T1). Each threshold setting produces a different (cost, accuracy) operating point. Connect these points to form the curve.

**Why this matters**: This is the standard visualization in LLM cascade literature (FrugalGPT Fig.3, RouteLLM Fig.2). Reviewers expect it. It directly addresses the λ tradeoff parameter from the formal objective — different points on the curve correspond to different effective λ values.

### 9.7 Recovery vs Restart Comparison

**Mandatory experiment**: Compare recovery strategies to determine if backtracking is actually worthwhile vs simply restarting from scratch.

| Strategy | Description |
|----------|-------------|
| No-recovery | Error detected → task fails immediately |
| Restart-from-scratch | Error detected → reset to home screen, restart entire task from step 0 |
| Recovery (ours) | Error detected → Level 1/2/3 recovery as designed |

Key metrics to compare:
- Task success rate (does recovery save tasks that would otherwise fail?)
- Average total steps consumed (is recovery cheaper than restart?)
- Step efficiency: successful_steps / total_steps (does recovery waste budget on doomed attempts?)

**If recovery only delays inevitable failure** (similar success rate but more steps consumed), it is not a valid contribution. This experiment MUST show that recovery recovers tasks that restart cannot (e.g., tasks where context built in earlier steps is lost on restart).

### 9.8 Ablation on Escalation Signals

| Config | Signals Used |
|--------|-------------|
| C1 | Risk rules only (Signal 3) |
| C2 | Thinking tokens only (Signal 1) |
| C3 | Confidence only (Signal 2) |
| C4 | Signals 1+3 |
| C5 | Signals 2+3 |
| C6 | All three signals (cascade) |

### 9.9 Skill Distillation Learning Curve

Report **per-run** metrics (not just averages) to show the learning curve:

| Run | Skill Library Size | Skill Reuse Rate | Task Success Rate | S2 Call Rate |
|-----|-------------------|-------------------|-------------------|-------------|
| 1 | 0 (cold start) | 0% | (baseline) | (baseline) |
| 2 | N₁ | ... | ... | ... |
| ... | ... | ... | ... | ... |
| 5 | N₅ | ... | ... | ... |

This answers: How many runs until skill distillation starts paying off? If runs 1-3 show no benefit and only runs 4-5 improve, the claim of "self-evolution" is weak for practical deployment. If learning curve is too slow, consider seeding the skill library with manually defined skills (warm start).

### 9.10 Escalation Ground Truth Definition (precise)

A step is labeled "needed S2" if **both** conditions hold:
1. S1 (in shadow mode) produced an action **different** from the ground-truth successful action at that step
2. The S1 action, when evaluated by a VLM judge against the task goal, is judged as **incorrect or suboptimal** (not just "different but equally valid")

This avoids circular reasoning: we don't use "S2 succeeded" as proof that "S2 was needed" — S1 might also have succeeded with a different action. The VLM judge provides an independent assessment.

For steps where neither S1 nor S2 produces the ground-truth action, the step is labeled "ambiguous" and excluded from precision/recall calculation.

## 10. Paper Structure (Target: CCF-A)

The spec is an engineering document. The paper requires a fundamentally different narrative structure.

### Target Structure (~9 pages)

1. **Introduction** (1.5 pages)
   - Open with: GUI agents face a cost-accuracy dilemma. Existing cascade approaches are open-loop.
   - Core claim: closed-loop routing (route → execute → detect → recover → distill → improve routing) outperforms open-loop in GUI tasks.
   - Contribution list: ONE contribution with three inseparable components, NOT three independent contributions.
   - "We propose X, a closed-loop adaptive GUI agent framework that integrates (1) uncertainty-aware model routing, (2) runtime error recovery, and (3) experience distillation into a self-improving execution loop."

2. **Related Work** (1 page)
   - Four positioning axes:
     - **LLM Cascade/Routing**: FrugalGPT, RouteLLM, AutoMix, Hybrid LLM → all open-loop, no execution feedback
     - **GUI Agents**: CogAgent, AppAgent, DigiRL, OS-Copilot → single model, no adaptive routing
     - **Error Recovery in GUI Agents**: Must explicitly differentiate (see table below)
     - **LLM Self-Correction**: Reflexion, SelfRefine, ReAct → always use same model, no cost optimization
   - Our gap: intersection of cascade + GUI + runtime recovery

   **Recovery differentiation table** (must appear in Related Work):
   | System | Recovery Type | Our Difference |
   |--------|-------------|----------------|
   | AppAgent | Offline: learns what NOT to do from failed explorations | Ours is runtime: correct errors as they happen, not after the fact |
   | DigiRL | Learned: RL reward shaping drives policy toward self-correction | Ours is zero-shot: S2 reasons about recovery without RL training |
   | Reflexion | Verbal: generates text feedback, retries entire episode | Ours is structured: checkpoint-based navigational recovery within the same episode |
   | OS-Copilot | Re-observation: retakes screenshot and retries | Ours is multi-level: forward correction → navigational recovery → abort, not just retry |

   Core distinction: our recovery is **runtime, model-heterogeneous** (a different, stronger model recovers from the weaker model's errors) and **multi-level** (not just retry).

3. **Problem Formulation** (0.5 pages)
   - MDP definition (Section 1.3 of this spec)
   - Routing objective: maximize E[R] - λ·E[Cost] subject to safety constraints
   - Define "closed-loop" formally

4. **Method** (3 pages)
   - 4.1 Adaptive Routing (escalation signals, cascade logic)
   - 4.2 Runtime Recovery (anomaly detection, three-level recovery)
   - 4.3 Experience Distillation (skill extraction, storage, reuse)
   - 4.4 Closed-Loop Integration (how 4.1-4.3 form the feedback loop)
   - Architecture diagram showing the loop

5. **Experiments** (2.5 pages)
   - 5.1 Setup (models, devices, datasets, metrics)
   - 5.2 Main Results (Table: A/B/C/D/E/R/O comparison)
   - 5.3 Ablation: Synergy analysis (removing any one component degrades significantly)
   - 5.4 Ablation: Signal effectiveness (C1-C6)
   - 5.5 Analysis: Recovery vs Restart, Learning Curve, Risk Classification Accuracy
   - 5.6 Generalization (different S1 model, different device)

6. **Conclusion** (0.5 pages)

### Key Narrative Decisions

- **Synergy over independence**: The ablation table (C < D < E with significant gaps) is the central evidence. Frame as "three capabilities that only work together", not "three techniques we combined".
- **Open-loop vs closed-loop**: This is the sharpest differentiation from existing work. Every comparison should highlight: "RouteLLM stops after routing. We route, watch, recover, learn."
- **Don't oversell S1/S2 analogy**: Use it in the introduction for intuition, then switch to precise terms throughout the paper.

## 11. Scope & Phasing

### Phase 0 (Prerequisite): Signal Validation Pilot

Before building the controller, validate that escalation signals actually work:

1. Deploy Qwen3VL-8B locally, verify thinking mode support (enable_thinking=True)
2. Run small model on **validation split** (see Dataset Split below), record per-step:
   - reasoning_tokens, confidence score, action correctness, task outcome
3. Compute signal-failure correlations (Spearman), plot ROC curves
4. Determine thresholds (T1, theta2) from ROC optimal operating points
5. If thinking mode unsupported: proceed with confidence-only signal path
6. **Output**: Validated threshold values, signal effectiveness report, go/no-go for each signal

Estimated effort: 2-3 days (mostly waiting for inference runs)

### Phase 1: GUI Operation Layer S1/S2

- FastSlowLLMAdapter + EscalationPolicy
- StepMonitor
- Basic backtracking (Level 1 forward recovery)
- Evaluate on ClawBench

### Phase 2: Full Backtracking + Skill Distillation

- CheckpointManager + Level 2/3 recovery
- SkillDistiller
- Extended evaluation with skill reuse metrics

### Phase 3 (Future): Conversation Layer S1/S2

- Extend to AgentLoop level
- S1/S2 switching for dialogue-level planning (not just GUI operations)
- Cross-task skill transfer

## 12. Grounding, Latency, and Deployment Considerations

### 12.1 GUI Grounding: Set-of-Mark (SoM) Usage

Set-of-Mark prompting (overlaying numbered labels on UI elements in screenshots) is the de facto standard for GUI grounding in 2024-2025 (used by CogAgent, OS-Atlas, UGround).

**Current status in opengui**: Check whether opengui's grounding pipeline uses SoM or direct coordinate prediction. This determines the grounding approach:

- **If opengui uses SoM**: Document in Method. S1 and S2 both receive SoM-annotated screenshots. Skill re-grounding maps semantic targets to SoM element IDs, then to coordinates.
- **If opengui uses direct coordinate prediction**: Document in Method and explain why (e.g., no dependency on external UI element detection, works on any device without accessibility tree). Consider adding a SoM vs non-SoM comparison experiment if time allows.

### 12.2 End-to-End Latency Breakdown (mandatory for paper)

The "fast executor" claim requires latency evidence. Report a per-step latency decomposition table:

| Component | S1 Mode | S2 Mode | Notes |
|-----------|---------|---------|-------|
| Screenshot capture | ~W ms | ~W ms | Same for both |
| S1 inference | ~X ms | N/A | Local GPU, measured |
| S2 inference | N/A | ~Y ms | API call, measured |
| Escalation overhead | N/A | ~Z ms | Context assembly + provider switch |
| Skill matching | ~50 ms | N/A | Per-task, not per-step |
| Action execution | ~V ms | ~V ms | ADB/HDC command |
| Post-action settle | ~300-1000 ms | ~300-1000 ms | Wait for UI update |

**What to measure in Phase 1**: Actual values for X, Y, Z. If S1=200ms and S2=2000ms, escalation avoidance saves ~1800ms/step. If escalation overhead Z=500ms, net saving per avoided escalation is ~1300ms. Report total task latency for each config (A through E).

### 12.3 Action Space Limitations

Current action space: {tap, type, scroll, swipe, press_home, press_back, done}

**Not currently supported** (should be stated in Limitations):
- `long_press`: Triggers context menus in many apps. Common but not in current opengui action schema.
- `pinch/zoom`: Required for maps, image viewing. Rare in task benchmarks.
- `drag`: Required for sliders, reordering. Moderate frequency.
- System permission dialogs: These interrupt task flow. Currently handled by opengui's existing dialog detection (verify in Phase 1).

These limitations apply equally to all baselines (AppAgent, CogAgent use similar action spaces). The adaptive routing mechanism is orthogonal to action space — extending actions does not affect the S1/S2 switching design.

### 12.4 Multi-App Coordination

For cross-app tasks (e.g., "copy address from email → paste in maps → share route to WeChat"):

- **Planning**: S1/S2 handles the full cross-app plan at the AgentLoop level (Phase 3). In Phase 1-2 (GUI operation layer only), cross-app coordination relies on the existing AgentLoop's planning.
- **Checkpointing**: CheckpointManager already saves state at app transitions. Each app segment has its own checkpoint chain.
- **Clipboard**: Not explicitly tracked. If clipboard state is critical, S2 can verify by performing a paste action and checking the result. This is a known limitation — stated in paper.
- **Skill scope**: Skills are per-app (the `app` field in Skill dataclass). Cross-app sequences are NOT distilled as a single skill — they decompose into per-app skill invocations.

## 13. Risk & Mitigation

| Risk | Mitigation |
|------|-----------|
| Qwen3VL-8B doesn't support thinking mode | Phase 0 tests this first. If unsupported, confidence becomes primary signal (equal status, not fallback) |
| Small model confidence poorly calibrated | Phase 0 measures ECE; apply temperature scaling if needed; thinking tokens as complementary signal |
| Thinking tokens correlate with complexity, not failure | Phase 0 validates correlation; abandon signal if Spearman < 0.3 or p > 0.05 |
| Navigational recovery reaches wrong state | VLM-based verification after recovery; 5-step budget cap; fall through to abort on failure |
| Navigational recovery side effects | Recovery step budget limits damage; only navigate via safe actions (back, home, app switch) |
| Skill distillation produces fragile skills | Semantic targets + runtime re-grounding (no coordinates); failure streak invalidation; state validation per step |
| S1/S2 oscillation wastes context and steps | Sub-goal granularity handoff: S2 retains control until sub-goal complete |
| U2 escalation precision unknown | Shadow mode in experiments: S1 runs but doesn't execute, providing counterfactual data |
| Framework changes by colleague break wrapper | All code in separate `fastslow/` directory; no opengui changes |
| Threshold sensitivity | Phase 0 ROC analysis + sensitivity sweep (+/-20%); report in paper |
| 40 tasks insufficient statistical power | 5 runs per config (200 task-runs); paired bootstrap tests; report 95% CI |
| Single device limits generalizability | 3 platforms: Android emulator (co-primary for SOTA comparison), HarmonyOS (real), iOS (real). Adapter architecture is device-agnostic |
| Recovery wastes step budget on doomed tasks | Recovery vs Restart experiment (Section 9.7) provides empirical evidence; if recovery doesn't help, demote to minor contribution |
| No SOTA GUI agent baselines | AppAgent + CogAgent on AndroidWorld as external baselines; DigiRL numbers from published paper |
| ClawBench only 40 tasks, not community-validated | AndroidWorld (116 tasks) as primary benchmark; ClawBench as cross-platform secondary |
| App UI updates break distilled skills | Proactive invalidation via versionCode detection; mark skills "unverified" on app update |
| Skill distillation doesn't actually save latency | Frame as S2 call rate reduction (cost), not latency. Measure per-step token reduction in Phase 1 |
