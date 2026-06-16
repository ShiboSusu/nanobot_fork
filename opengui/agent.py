"""
opengui.agent
=============
Core GUI automation agent with a vision-action loop.

``GuiAgent`` orchestrates a multi-step loop: observe the screen, call an LLM
with the screenshot, parse the tool-call response into an ``Action``, execute
it on the backend, and repeat until the task is done or max steps is reached.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from opengui.action import Action, ActionError, describe_action, parse_action
from opengui.agent_profiles import (
    build_mobileworld_messages,
    canonicalize_agent_profile,
    coordinate_mode_for_profile,
    normalize_profile_response_for_observation,
    normalize_profile_response_for_screen,
    profile_uses_native_tools,
    prompt_contract_for_profile,
)
from opengui.interfaces import (
    DeviceBackend,
    InterventionHandler,
    InterventionRequest,
    LLMProvider,
    LLMResponse,
    ProgressCallback,
    ToolCall,
)
from opengui.observation import Observation
from opengui.s2_policy import (
    S2Mode,
    S2Trigger,
    S2TriggerEvent,
    S2Usage,
    decide_s2_mode,
)
from opengui.evidence import extract_gui_evidence
from opengui.skills.compact_prompt import (
    ALWAYS_ON_SKILL_TAG,
    COMPOSITE_ACTION_DEFINITIONS,
    CompositeActionInfo,
    CompactPromptParts,
    build_compact_prompt_parts,
    composite_action_infos_from_skills,
    is_always_on_skill,
    is_shortcut_skill,
    skill_info_from_flat_skill,
)
from opengui.skills.deeplink import AppShortcutProfile
from opengui.skills.normalization import (
    find_android_app_in_text,
    is_browser_bundle,
    normalize_adb_app_identifier,
    normalize_app_identifier,
    task_explicitly_allows_browser,
)
from opengui.skills.state_contract import evaluate_state_contract, infer_interaction_target
from opengui.tool_schemas import (
    COMPUTER_USE_TOOL,
    build_shortcut_tool_defs,
    image_dimensions,
    minimal_tool_schema,
)
from opengui.trajectory.recorder import ExecutionPhase, TrajectoryRecorder
from opengui.trajectory.summarizer import strip_completed_status_prefix
from opengui.trajectory.summarizer import build_state_note, is_state_note

logger = logging.getLogger(__name__)

S2_NO_THINKING_POLICY = """/no_think

Do not output thinking.
Do not output analysis.
Do not output markdown.
Do not output code fences.
Do not include text before or after the required output.
Keep the output concise.
"""

S2_TAKEOVER_ACTION_POLICY = (
    "Return only the required GUI action output.\n"
    "Do not explain the action.\n"
    "Follow the active GUI action contract exactly."
)
_SKILL_PARAM_EXTRACTION_TIMEOUT_SECONDS = 60.0

_DONE_FAILURE_HINTS: tuple[str, ...] = (
    "fail",
    "failed",
    "failure",
    "unable",
    "cannot",
    "can't",
    "error",
    "not completed",
    "incomplete",
    "失败",
    "无法",
    "不能",
    "错误",
    "未完成",
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StepResult:
    """Result of a single vision-action step."""

    action: Action
    tool_call_id: str
    tool_result: str
    assistant_message: dict[str, Any]
    action_summary: str
    action_intent: str | None = None
    state_summary: str | None = None
    next_observation: Observation | None = None
    interaction_target: dict[str, Any] | None = None
    action_debug: dict[str, Any] | None = None
    prompt_snapshot: dict[str, Any] | None = None
    model_snapshot: dict[str, Any] | None = None
    execution_snapshot: dict[str, Any] | None = None
    intervention_requested: bool = False
    done: bool = False
    step_usage: dict[str, int] = dataclasses.field(default_factory=dict)
    duration_s: float = 0.0
    chat_latency_s: float | None = None
    ttft_s: float | None = None


@dataclass(frozen=True)
class HistoryTurn:
    """One completed step kept in the prompt history window."""

    step_index: int
    observation: Observation
    assistant_message: dict[str, Any]
    tool_result_message: dict[str, Any]
    action_summary: str
    action_intent: str | None = None
    state_summary: str | None = None
    raw_response_content: str | None = None


@dataclass(frozen=True)
class AgentResult:
    """Final result of a complete GUI task run (possibly with retries)."""

    success: bool
    summary: str
    model_summary: str | None = None
    trace_path: str | None = None
    steps_taken: int = 0
    error: str | None = None
    attempt_summary: str | None = None
    token_usage: dict[str, int] = dataclasses.field(default_factory=dict)
    s2_usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    answer_candidates: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    evidence: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class _ScreenFingerprint:
    """Compact screen signature for loop-stagnation detection."""

    app: str | None
    method: str
    digest: str


class _StepExecutionError(RuntimeError):
    """Runtime error raised from _run_step with optional model snapshot context."""

    def __init__(
        self,
        message: str,
        *,
        model_snapshot: dict[str, Any] | None = None,
        attempt_summary: str | None = None,
    ) -> None:
        super().__init__(message)
        self.model_snapshot = model_snapshot
        self.attempt_summary = attempt_summary


# ---------------------------------------------------------------------------
# Agent-side protocol implementations for SkillExecutor
# ---------------------------------------------------------------------------

from opengui.skills.action_grounder import ActionGrounder as _AgentActionGrounder  # noqa: E402
from opengui.skills.subgoal_runner import SubgoalRunner as _AgentSubgoalRunner  # noqa: E402
from opengui.skills.observation_provider import AgentScreenshotProvider as _AgentScreenshotProvider  # noqa: E402


# ---------------------------------------------------------------------------
# GuiAgent
# ---------------------------------------------------------------------------


def _clean_inferred_param(value: str) -> str:
    return value.strip().strip('`"\'“”‘’').strip()


def _looks_like_date_param(value: str) -> bool:
    return bool(re.search(r"\d+\s*(?:月|号|日|/|-)", value))


_SKILL_PARAM_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_ANDROID_LAUNCHER_PACKAGES = frozenset({
    "com.android.launcher",
    "com.android.launcher3",
    "com.google.android.apps.nexuslauncher",
    "com.miui.home",
})


def _skill_param_names(skill: Any) -> list[str]:
    return [
        str(name).strip()
        for name in (getattr(skill, "parameters", None) or ())
        if str(name).strip()
    ]


def _skill_required_param_names(skill: Any) -> list[str]:
    names = _skill_param_names(skill)
    if not names:
        return []
    placeholders: set[str] = set()
    for step in tuple(getattr(skill, "steps", ()) or ()):
        placeholders.update(_extract_template_placeholders(getattr(step, "target", None)))
        placeholders.update(_extract_template_placeholders(getattr(step, "valid_state", None)))
        placeholders.update(_extract_template_placeholders(getattr(step, "parameters", None)))
        placeholders.update(_extract_template_placeholders(getattr(step, "fixed_values", None)))
        placeholders.update(_extract_template_placeholders(getattr(step, "state_contract", None)))
    required = [name for name in names if name in placeholders]
    return required or names


def _missing_skill_params(skill: Any, params: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for name in _skill_required_param_names(skill):
        value = params.get(name)
        if value is None or not str(value).strip():
            missing.append(name)
    return missing


def _extract_template_placeholders(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {match.group(1) for match in _SKILL_PARAM_PLACEHOLDER_RE.finditer(value)}
    if isinstance(value, dict):
        found: set[str] = set()
        for key, item in value.items():
            found.update(_extract_template_placeholders(key))
            found.update(_extract_template_placeholders(item))
        return found
    if isinstance(value, (list, tuple, set)):
        found: set[str] = set()
        for item in value:
            found.update(_extract_template_placeholders(item))
        return found
    return set()


def _skill_entry_targets_android_launcher(skill: Any, first_step: Any) -> bool:
    platform = str(getattr(skill, "platform", "") or "").strip().lower()
    if platform != "android":
        return False
    candidates: list[str] = [
        str(getattr(skill, "app", "") or ""),
        str(getattr(first_step, "target", "") or ""),
    ]
    fixed_values = getattr(first_step, "fixed_values", None)
    if isinstance(fixed_values, dict):
        for key in ("text", "package", "app", "target"):
            value = fixed_values.get(key)
            if value:
                candidates.append(str(value))
    for candidate in candidates:
        normalized = normalize_app_identifier("android", candidate)
        if normalized in _ANDROID_LAUNCHER_PACKAGES:
            return True
    return False


def _render_skill_template(text: str, params: dict[str, str]) -> str:
    """Substitute ``{{param}}`` placeholders in *text* with provided values."""
    return _SKILL_PARAM_PLACEHOLDER_RE.sub(
        lambda m: str(params.get(m.group(1), m.group(0))), str(text or "")
    )


def _observation_ui_text_blob(observation: Any) -> str:
    """Lowercased concatenation of on-screen text from an observation.

    Pulls visible text / content descriptions from the ``ui_tree`` and any
    ``*_text`` lists in ``observation.extra``. Used for a cheap, deterministic
    "is this control on screen" check without an LLM call.
    """
    extra = getattr(observation, "extra", None)
    if not isinstance(extra, dict):
        return ""
    parts: list[str] = []
    ui_tree = extra.get("ui_tree")
    if isinstance(ui_tree, list):
        for node in ui_tree:
            if isinstance(node, dict):
                for key in ("text", "content_desc"):
                    value = node.get(key)
                    if value:
                        parts.append(str(value))
    for key in ("clickable_text", "visible_text", "content_desc"):
        value = extra.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        elif value:
            parts.append(str(value))
    return " ".join(parts).lower()


def _prompt_skill_entry_allows(skill: Any, observation: Any, params: dict[str, str]) -> bool:
    """Cheap deterministic entry guard for a prompt-selected ``use_skill``.

    Fail-closed: a skill whose first step cannot be confirmed against the current
    screen is rejected so the agent keeps control instead of having the grounder
    blindly tap a wrong screen. Entry actions (``open_app`` / deeplink / intent)
    are trusted to navigate themselves; otherwise a first-step ``state_contract``
    must match, or — when no contract exists — the rendered first-step target text
    must appear on screen.
    """
    steps = tuple(getattr(skill, "steps", ()) or ())
    if not steps:
        return False
    first = steps[0]
    first_action = str(getattr(first, "action_type", "") or "").strip().lower()
    if first_action == "open_app":
        return not _skill_entry_targets_android_launcher(skill, first)
    if first_action in {"open_deeplink", "open_intent"}:
        return True

    contract = getattr(first, "state_contract", None)
    if contract is not None:
        return evaluate_state_contract(contract, observation=observation) is True

    target = _render_skill_template(str(getattr(first, "target", "") or ""), params).strip().lower()
    if not target:
        return False  # nothing deterministic to check → fail closed
    blob = _observation_ui_text_blob(observation)
    return bool(blob) and target in blob


class GuiAgent:
    """Standalone GUI automation agent with vision-action loop.

    Args:
        llm: LLM provider conforming to :class:`~opengui.interfaces.LLMProvider`.
        backend: Device backend conforming to :class:`~opengui.interfaces.DeviceBackend`.
        model: Model name string (used for prompt customisation).
        artifacts_root: Root directory for run artifacts (traces, screenshots).
        max_steps: Maximum steps per single attempt.
        step_timeout: Timeout in seconds for each step (LLM + execute + observe).
        history_image_window: Number of recent screenshot turns kept as full
            image context, including the current screen.
        include_date_context: Whether to include today's date in the task framing text.
        progress_callback: Optional async callback for progress reporting.
        stagnation_limit: Consecutive unchanged-screen transitions before abort.
    """

    _MAX_TOOL_RETRIES = 3
    _COORDINATE_ACTIONS = frozenset({"tap", "double_tap", "long_press", "swipe", "drag", "scroll"})
    _POST_ACTION_SETTLE_SECONDS = 0.50
    _OPEN_APP_SETTLE_SECONDS = 5.00
    _POST_ACTION_STABILITY_WINDOW_SECONDS = 2.0
    _POST_ACTION_STABILITY_POLL_SECONDS = 0.15
    _POST_ACTION_STABILITY_MAX_ATTEMPTS = 4
    _POST_ACTION_STABILITY_FRAMES_REQUIRED = 2
    _POST_ACTION_OBSERVE_TIMEOUT_SECONDS = 8.0
    _NO_SETTLE_ACTIONS = frozenset({"wait", "done", "request_intervention"})
    _STAGNATION_SSIM_SIZE = 64
    _STAGNATION_SSIM_THRESHOLD = 0.985
    _IOS_TRANSIENT_SYSTEM_APPS = frozenset({
        "com.apple.springboard",
        "com.apple.ScreenshotServicesService",
        "com.apple.notificationcenterui",
        "com.apple.usernotificationsui",
    })
    _TRANSIENT_OVERLAY_TERMS = frozenset({
        "notification",
        "banner",
        "popup",
        "pop-up",
        "permission",
        "alert",
        "allow",
        "don't allow",
        "not now",
        "later",
        "close",
        "skip",
        "通知",
        "横幅",
        "弹窗",
        "弹出",
        "权限",
        "提醒",
        "允许",
        "不允许",
        "暂不",
        "稍后",
        "关闭",
        "跳过",
    })

    def __init__(
        self,
        llm: LLMProvider,
        backend: DeviceBackend,
        trajectory_recorder: TrajectoryRecorder,
        model: str = "",
        artifacts_root: Path | str = ".opengui/runs",
        max_steps: int = 15,
        step_timeout: float = 30.0,
        history_image_window: int = 3,
        include_date_context: bool = True,
        history_text_window: int = 8,
        progress_callback: ProgressCallback | None = None,
        memory_retriever: Any = None,
        skill_library: Any = None,
        skill_executor: Any = None,
        skill_reuser: Any = None,
        memory_top_k: int = 5,
        skill_threshold: float = 0.35,
        installed_apps: list[str] | None = None,
        shortcut_backend: DeviceBackend | None = None,
        shortcut_cache_dir: Path | str | None = None,
        intervention_handler: InterventionHandler | None = None,
        policy_context: str | None = None,
        memory_store: Any = None,
        agent_profile: str | None = None,
        image_scale_ratio: float = 0.5,
        stagnation_limit: int = 2,
        enable_prompt_skill_selection: bool = False,
        prompt_skill_top_k: int = 5,
        prompt_shortcut_only: bool = False,
        always_on_skill_tags: list[str] | tuple[str, ...] | None = None,
        s2_llm: LLMProvider | None = None,
        s2_model: str | None = None,
        s2_enabled: bool = False,
        s1_reasoning_effort: str | None = None,
        s2_reasoning_effort: str | None = None,
        s2_hint_enabled: bool = True,
        s2_takeover_enabled: bool = True,
        s2_max_hints: int = 1,
        s2_takeover_after_hints: int = 1,
        s2_max_takeover_steps: int = 8,
        s2_trigger_on_stagnation: bool = True,
        s2_trigger_on_max_steps_near: bool = False,
        s2_trigger_on_done_missing_evidence: bool = True,
        s2_trigger_on_step_error: bool = True,
        s2_trigger_on_wrong_app: bool = True,
        s2_max_steps_near_margin: int = 3,
        s2_prompt_max_chars: int = 6000,
        evidence_request: Any = None,
    ) -> None:
        self.llm = llm
        self.backend = backend
        self.model = model
        self.agent_profile = canonicalize_agent_profile(agent_profile)
        self.artifacts_root = Path(artifacts_root)
        self.max_steps = max_steps
        self.step_timeout = step_timeout
        self.history_image_window = max(1, history_image_window)
        self.history_text_window = max(1, history_text_window)
        self.include_date_context = include_date_context
        self.progress_callback = progress_callback
        self._trajectory_recorder = trajectory_recorder
        self._memory_retriever = memory_retriever
        self._policy_context = policy_context
        self._evidence_request = evidence_request
        self._skill_library = skill_library
        self._skill_executor = skill_executor
        self._memory_top_k = memory_top_k
        if skill_reuser is None and skill_library is not None:
            from opengui.skills.reuser import SkillReuser
            skill_reuser = SkillReuser(llm, threshold=skill_threshold)
        self._skill_reuser = skill_reuser
        self._skill_threshold = skill_threshold
        self._installed_apps = installed_apps
        self._shortcuts: dict[str, AppShortcutProfile] = {}
        self._shortcut_backend = shortcut_backend
        self._shortcut_cache_dir = Path(shortcut_cache_dir) if shortcut_cache_dir else None
        self._shortcut_tools: list[dict[str, Any]] = []
        self._shortcut_action_map: dict[str, tuple[str, str, str, str | None]] = {}
        self._intervention_handler = intervention_handler
        self._memory_store = memory_store
        self._active_retry_summaries: tuple[str, ...] = ()
        self._image_scale_ratio = image_scale_ratio
        self._enable_prompt_skill_selection = bool(enable_prompt_skill_selection)
        try:
            parsed_prompt_skill_top_k = int(prompt_skill_top_k)
        except (TypeError, ValueError):
            parsed_prompt_skill_top_k = 5
        self._prompt_skill_top_k = max(0, parsed_prompt_skill_top_k)
        self._prompt_shortcut_only = bool(prompt_shortcut_only)
        self._always_on_skill_tags = tuple(
            str(tag)
            for tag in (always_on_skill_tags if always_on_skill_tags is not None else (ALWAYS_ON_SKILL_TAG,))
            if str(tag)
        )
        self._prompt_skills_by_id: dict[str, Any] = {}
        self._prompt_composite_aliases: set[str] = set()
        try:
            parsed_stagnation_limit = int(stagnation_limit)
        except (TypeError, ValueError):
            parsed_stagnation_limit = 0
        self.stagnation_limit = max(0, parsed_stagnation_limit)
        self._s2_llm = s2_llm
        self._s2_model = s2_model or ""
        self._s2_enabled = bool(s2_enabled and s2_llm is not None)
        self._s1_reasoning_effort = s1_reasoning_effort
        self._s2_reasoning_effort = s2_reasoning_effort
        self._s2_hint_enabled = bool(s2_hint_enabled)
        self._s2_takeover_enabled = bool(s2_takeover_enabled)
        self._s2_max_hints = max(0, int(s2_max_hints))
        self._s2_takeover_after_hints = max(0, int(s2_takeover_after_hints))
        self._s2_max_takeover_steps = max(1, int(s2_max_takeover_steps))
        self._s2_trigger_on_stagnation = bool(s2_trigger_on_stagnation)
        self._s2_max_steps_near_margin = max(1, int(s2_max_steps_near_margin))
        self._s2_allowed_triggers = frozenset(
            trigger
            for trigger, enabled in (
                (S2Trigger.STAGNATION, s2_trigger_on_stagnation),
                (S2Trigger.MAX_STEPS_NEAR, s2_trigger_on_max_steps_near),
                (S2Trigger.MISSING_EVIDENCE, s2_trigger_on_done_missing_evidence),
                (S2Trigger.STEP_ERROR, s2_trigger_on_step_error),
                (S2Trigger.WRONG_APP, s2_trigger_on_wrong_app),
            )
            if enabled
        )
        # Follow-up: wire S2Trigger.WRONG_APP into the wrong-app relaunch failure
        # path. WS-2 batch 1 only hooks max_steps_near, missing_evidence, and step_error.
        self._s2_prompt_max_chars = max(1000, int(s2_prompt_max_chars))
        self._s2_usage = S2Usage(enabled=self._s2_enabled)
        self._s2_guidance_notes: list[str] = []
        self._s2_pending_takeover_steps = 0
        self._s2_last_takeover_reason = ""
        self._s2_last_stagnation_streak = 0
        self._s2_current_task = ""
        self._s2_run_dir: Path | None = None
        self._s2_actor_before = "s1"

    def _build_tools_list(self) -> list[dict[str, Any]]:
        tools = [COMPUTER_USE_TOOL]
        tools.extend(self._shortcut_tools)
        return tools

    async def _ensure_shortcuts_for_app(self, foreground_app: str) -> None:
        """Lazy-load shortcuts when a new foreground app is detected."""
        app = str(foreground_app or "").strip()
        if not app or not self._shortcut_cache_dir or not self._shortcut_backend:
            return
        if str(getattr(self._shortcut_backend, "platform", self.backend.platform)).lower() != "android":
            return
        app = normalize_adb_app_identifier(app)
        if not app or app == "unknown":
            return
        if app in self._shortcuts:
            return

        cache_file = self._shortcut_cache_dir / f"{app}.json"
        try:
            if cache_file.exists():
                profile = AppShortcutProfile.from_dict(json.loads(cache_file.read_text(encoding="utf-8")))
            else:
                from opengui.skills.deeplink import extract_app_shortcuts

                profile = await extract_app_shortcuts(self._shortcut_backend, app)
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(
                    json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            logger.warning("Shortcut discovery failed for %s: %s", app, exc)
            try:
                self._trajectory_recorder.record_event(
                    "shortcut_discovery_failed",
                    app=app,
                    cache_file=str(cache_file),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception as record_exc:
                logger.debug("Could not record shortcut discovery failure: %s", record_exc)
            self._shortcuts[app] = AppShortcutProfile(
                package=app,
                manifest_meta={
                    "status": "shortcut_discovery_failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )
            return

        self._shortcuts[app] = profile
        if profile.deep_links or profile.deep_intents:
            self._shortcut_tools, self._shortcut_action_map = build_shortcut_tool_defs(self._shortcuts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        *,
        max_retries: int = 3,
        app_hint: str | None = None,
        expected_bundle_id: str | None = None,
    ) -> AgentResult:
        """Run the task with retry logic.

        Returns an :class:`AgentResult` summarising the outcome. On failure
        after all retries, ``success`` is ``False`` and ``error`` contains the
        last error message.
        """
        # 1. Start trajectory recording
        self._trajectory_recorder.start(phase=ExecutionPhase.AGENT)

        # 2. Retrieve memory context (once)
        memory_context = await self._retrieve_memory(task)

        skill_context: str | None = None
        skill_app_filter = self._skill_app_filter(task, app_hint)
        prompt_skill_parts: CompactPromptParts | None = None
        if self._enable_prompt_skill_selection:
            prompt_skill_parts = await self._build_prompt_skill_parts(
                task,
                app=skill_app_filter,
            )

        # 3. Search the flat skill library once; LLM-gated when SkillReuser is available.
        reuser_usage: dict[str, int] = {}
        if self._enable_prompt_skill_selection:
            skill_match = None
            self._trajectory_recorder.record_event(
                "skill_search",
                task=task,
                source="prompt_skill_selection",
                matched=False,
                reason="deferred_to_model_prompt",
                app_filter=skill_app_filter,
            )
        elif self._skill_reuser is not None and self._skill_library is not None:
            skill_match = await self._skill_reuser.find(
                task,
                self._skill_library,
                self.backend.platform,
                app=skill_app_filter,
                trajectory_recorder=self._trajectory_recorder,
            )
            reuser_usage = self._skill_reuser.drain_usage()
        else:
            skill_match = await self._search_skill(task, app=skill_app_filter)

        matched_skill: Any | None = None
        final_score: float | None = None
        skill_match_for_maintenance: Any | None = skill_match
        if skill_match is not None:
            if hasattr(skill_match, "layer"):
                matched_skill = skill_match.skill
                final_score = skill_match.score
            else:
                matched_skill, final_score = skill_match

        # 4. If skill matched, attempt skill execution first.
        skill_result: Any | None = None
        if matched_skill is not None and self._skill_executor is not None and final_score is not None:
            entry_allowed = await self._skill_entry_allows_current_state(matched_skill)
            if not entry_allowed:
                skill_match_for_maintenance = None
                matched_skill = None
            else:
                memory_context = await self._inject_skill_memory_context(matched_skill, memory_context)
        elif matched_skill is not None:
            memory_context = await self._inject_skill_memory_context(matched_skill, memory_context)

        if matched_skill is not None and self._skill_executor is not None and final_score is not None:
            self._trajectory_recorder.set_phase(
                ExecutionPhase.SKILL,
                reason=f"Matched skill: {matched_skill.name} (score={final_score:.2f})",
            )
            try:
                skill_result = await self._execute_skill_with_params(task, matched_skill)
                execution_summary = getattr(skill_result, "execution_summary", None)
                skill_context = execution_summary if isinstance(execution_summary, str) else None
                if skill_result.state.value == "succeeded":
                    # Skill succeeded — fall through to agent for confirmation
                    self._trajectory_recorder.set_phase(
                        ExecutionPhase.AGENT,
                        reason="Skill complete, agent confirms",
                    )
                else:
                    # Skill partially succeeded — agent completes the rest
                    self._trajectory_recorder.set_phase(
                        ExecutionPhase.AGENT, reason="Skill partially succeeded, agent completes"
                    )
            except Exception:
                # Skill failed — fall back to free exploration
                self._trajectory_recorder.set_phase(
                    ExecutionPhase.AGENT, reason="Skill execution failed, falling back"
                )

        # 5. Retry loop with free exploration
        last_error: str | None = None
        last_model_summary: str | None = None
        last_trace_path: str | None = None
        last_steps_taken = 0
        result: AgentResult | None = None
        retry_summaries: list[str] = []
        # Seed total_usage with tokens consumed during skill retrieval + execution (if any)
        skill_token_usage: dict[str, int] = {}
        if skill_result is not None:
            raw_skill_token_usage = getattr(skill_result, "token_usage", None)
            if isinstance(raw_skill_token_usage, dict):
                skill_token_usage = dict(raw_skill_token_usage)
        total_usage: dict[str, int] = dict(skill_token_usage)
        for k, v in reuser_usage.items():
            total_usage[k] = total_usage.get(k, 0) + v

        try:
            for attempt in range(max_retries):
                self._active_retry_summaries = tuple(retry_summaries)
                run_dir = self._make_run_dir(task, attempt)
                last_trace_path = str(run_dir)

                await self._log_attempt_event(
                    run_dir,
                    "attempt_start",
                    attempt=attempt,
                    max_retries=max_retries,
                    task=task,
                )
                try:
                    result = await self._run_once(
                        task, app_hint=app_hint, run_dir=run_dir,
                        expected_bundle_id=expected_bundle_id,
                        memory_context=memory_context,
                        skill_context=skill_context,
                        prompt_skill_parts=prompt_skill_parts,
                    )
                    for k, v in result.token_usage.items():
                        total_usage[k] = total_usage.get(k, 0) + v
                    await self._log_attempt_event(
                        run_dir,
                        "attempt_result",
                        attempt=attempt,
                        success=result.success,
                        summary=result.summary,
                        model_summary=result.model_summary,
                        error=result.error,
                        steps_taken=result.steps_taken,
                        trace_path=result.trace_path,
                    )
                    if result.success:
                        break
                    last_error = result.error
                    last_model_summary = result.model_summary
                    last_trace_path = result.trace_path or last_trace_path
                    last_steps_taken = result.steps_taken
                    attempt_summary = result.attempt_summary or self._build_attempt_summary(
                        failure_reason=result.error or result.summary,
                        result_summary=result.summary,
                        model_summary=result.model_summary,
                        action_summaries=(),
                    )
                    retry_summaries.append(attempt_summary)
                    if result.error and (
                        result.error.startswith("intervention_cancelled")
                        or result.error == "stagnation_detected"
                    ):
                        break
                    if attempt < max_retries - 1:
                        await self._log_attempt_event(
                            run_dir,
                            "retry",
                            attempt=attempt,
                            next_attempt=attempt + 1,
                            reason=result.error or result.summary,
                        )
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    model_snapshot = getattr(exc, "model_snapshot", None)
                    attempt_summary = getattr(exc, "attempt_summary", None) or self._build_attempt_summary(
                        failure_reason=last_error,
                        result_summary="Attempt ended with an exception before completion.",
                        action_summaries=(),
                    )
                    retry_summaries.append(attempt_summary)
                    await self._log_attempt_event(
                        run_dir,
                        "attempt_exception",
                        attempt=attempt,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        model_response=model_snapshot,
                    )
                    if attempt < max_retries - 1:
                        await self._log_attempt_event(
                            run_dir,
                            "retry",
                            attempt=attempt,
                            next_attempt=attempt + 1,
                            reason=last_error,
                        )
        finally:
            self._active_retry_summaries = ()

        if result is None:
            status = "blocked" if last_error and any(
                keyword in last_error.lower()
                for keyword in ("stagnation", "intervention", "preflight")
            ) else "partial"
            result = AgentResult(
                success=False,
                summary=self._build_state_note(
                    status=status,
                    history=[],
                    current_observation=None,
                    error=last_error or f"Failed after {max_retries} attempt(s).",
                ),
                model_summary=last_model_summary,
                trace_path=last_trace_path,
                steps_taken=last_steps_taken,
                error=last_error,
                token_usage=total_usage,
            )
        else:
            result = dataclasses.replace(
                result,
                token_usage=total_usage,
                trace_path=last_trace_path or result.trace_path,
                error=last_error or result.error,
            )

        result = dataclasses.replace(
            result,
            model_summary=strip_completed_status_prefix(result.model_summary),
        )
        result = await self._attach_produced_evidence(result)

        # 6. Finish trajectory
        self._trajectory_recorder.finish(
            success=result.success,
            error=result.error,
            token_usage=result.token_usage or None,
        )

        # 7. Post-run skill maintenance — use skill execution outcome,
        #    not overall task result, so prefix skills aren't penalised
        #    for agent failures after their steps complete.
        skill_exec_success = (
            skill_result is not None and skill_result.state.value == "succeeded"
        ) if skill_result is not None else result.success

        # 7b. Agent compensation detection — if the skill "succeeded" but the
        #     agent needed significantly more steps than the skill itself had,
        #     the skill didn't actually advance the task.  Demote to failure.
        if skill_exec_success and skill_result is not None and result is not None:
            skill_step_count = len(getattr(skill_result, "step_results", ()) or ())
            agent_steps = getattr(result, "steps_taken", 0)
            if agent_steps > skill_step_count + 1:
                skill_exec_success = False

        await self._skill_maintenance(skill_match_for_maintenance, skill_exec_success)

        return result

    async def _attach_produced_evidence(self, result: AgentResult) -> AgentResult:
        """Populate AgentResult evidence from the latest trace step using declared keys."""
        if result.answer_candidates or self._evidence_request is None:
            return result
        allowed_keys = self._declared_evidence_keys(self._evidence_request)
        if not allowed_keys:
            return result
        latest_step = self._load_latest_trace_step(result.trace_path)
        if latest_step is None:
            return result
        produced = await self._produce_answer_candidates(
            request=self._evidence_request,
            latest_step=latest_step,
            allowed_keys=allowed_keys,
        )
        candidates = produced.get("answer_candidates") if isinstance(produced, dict) else None
        if not candidates:
            return result
        extracted = extract_gui_evidence(
            request=self._evidence_request,
            payload={"answer_candidates": candidates},
            latest_step=latest_step,
        )
        evidence = extracted.get("evidence") if isinstance(extracted, dict) else {}
        if isinstance(evidence, dict):
            evidence = {**evidence, "producer": "llm", "allowed_keys": allowed_keys}
        return dataclasses.replace(
            result,
            answer_candidates=list(candidates),
            evidence=evidence if isinstance(evidence, dict) else {},
        )

    async def _produce_answer_candidates(
        self,
        *,
        request: Any,
        latest_step: dict[str, Any],
        allowed_keys: list[str],
    ) -> dict[str, Any]:
        system_prompt = (
            "Extract structured GUI task outputs from the latest step only. "
            "If a value is not explicitly present, omit the key. Do not guess. "
            "Return JSON object using only these keys: "
            f"{', '.join(allowed_keys)}."
        )
        user_payload = {
            "task": self._evidence_request_task(request),
            "allowed_keys": allowed_keys,
            "latest_step": latest_step,
        }
        try:
            response = await self.llm.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
                ],
                max_tokens=512,
            )
        except Exception:
            logger.debug("GUI evidence producer failed", exc_info=True)
            return {"answer_candidates": []}
        raw = self._parse_json_object(response.content)
        if not isinstance(raw, dict):
            return {"answer_candidates": []}
        candidates: list[dict[str, Any]] = []
        for key in allowed_keys:
            value = raw.get(key)
            if value is None:
                continue
            text = self._candidate_text(value)
            if not text:
                continue
            candidates.append(
                {
                    "key": key,
                    "text": text,
                    "type": "answer",
                    "confidence": 1.0,
                    "source": "llm_evidence_producer",
                    "evidence_refs": ["latest_step"],
                }
            )
        return {"answer_candidates": candidates}

    @staticmethod
    def _declared_evidence_keys(request: Any) -> list[str]:
        success_condition = getattr(request, "success_condition", None)
        required_key = getattr(success_condition, "required_key", None)
        key = str(required_key or "").strip()
        if key:
            return [key]
        output_mode = getattr(getattr(request, "output_mode", None), "value", getattr(request, "output_mode", None))
        task_type = getattr(getattr(request, "task_type", None), "value", getattr(request, "task_type", None))
        if output_mode == "answer_required" or task_type in {"information_query", "mixed_query_and_action"}:
            return ["answer"]
        return []

    @staticmethod
    def _evidence_request_task(request: Any) -> str:
        return str(getattr(request, "task", "") or "")

    @staticmethod
    def _load_latest_trace_step(trace_path: str | None) -> dict[str, Any] | None:
        if not trace_path:
            return None
        path = Path(trace_path)
        if path.is_dir():
            path = path / "trace.jsonl"
        if not path.exists():
            return None
        latest: dict[str, Any] | None = None
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if isinstance(event, dict) and event.get("event") == "step":
                    latest = event
        except Exception:
            logger.debug("Failed to read latest GUI trace step from %s", path, exc_info=True)
            return None
        return latest

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any] | None:
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _candidate_text(value: Any) -> str:
        if isinstance(value, dict):
            for key in ("text", "answer", "value", "title"):
                if value.get(key) is not None:
                    return str(value[key]).strip()
            return ""
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    # ------------------------------------------------------------------
    # Single attempt
    # ------------------------------------------------------------------

    async def _run_once(
        self,
        task: str,
        *,
        app_hint: str | None,
        expected_bundle_id: str | None = None,
        run_dir: Path,
        memory_context: str | None = None,
        skill_context: str | None = None,
        prompt_skill_parts: CompactPromptParts | None = None,
    ) -> AgentResult:
        """Execute one full attempt of the task."""
        s2_usage = S2Usage(enabled=self._s2_enabled)
        self._s2_usage = s2_usage
        self._s2_guidance_notes = []
        self._s2_pending_takeover_steps = 0
        self._s2_last_takeover_reason = ""
        self._s2_last_stagnation_streak = 0
        self._s2_current_task = task
        self._s2_run_dir = run_dir
        # 1. Preflight
        try:
            await self.backend.preflight()
        except Exception as exc:
            return AgentResult(
                success=False,
                summary=self._build_state_note(
                    status="blocked",
                    history=[],
                    current_observation=None,
                    error=f"Preflight failed: {exc}",
                ),
                trace_path=str(run_dir),
                error=str(exc),
                s2_usage=s2_usage.to_dict(),
            )

        # 2. Initial observation
        obs = await self.backend.observe(
            run_dir / "screenshots" / "step_000.png",
            timeout=self.step_timeout,
        )
        browser_relaunch_attempted = False
        if self._should_relaunch_expected_app(obs, expected_bundle_id=expected_bundle_id, task=task):
            obs = await self._relaunch_expected_app(
                obs,
                expected_bundle_id=expected_bundle_id,
                run_dir=run_dir,
                screenshot_name="step_000_native_relaunch.png",
            )
            browser_relaunch_attempted = True

        history: list[HistoryTurn] = []
        previous_fingerprint: _ScreenFingerprint | None = None
        previous_action_type: str | None = None
        stagnation_streak = 0
        s2_guidance_notes = self._s2_guidance_notes
        takeover_remaining = 0
        last_s2_takeover_reason = ""
        last_s2_stagnation_streak = 0
        if self.stagnation_limit > 0:
            previous_fingerprint = self._build_screen_fingerprint(obs)

        # 4. Step loop
        steps_taken = 0
        total_usage: dict[str, int] = {}
        for step in range(self.max_steps):
            step_index = step + 1
            if (
                S2Trigger.MAX_STEPS_NEAR in self._s2_allowed_triggers
                and self._s2_enabled
                and not s2_usage.takeover_used
                and (self.max_steps - step_index) <= self._s2_max_steps_near_margin
            ):
                self._s2_actor_before = "s1"
                s2_mode = await self._maybe_invoke_s2(
                    trigger=S2Trigger.MAX_STEPS_NEAR,
                    step_index=step_index,
                    history=history,
                    reason=(
                        f"Approaching max_steps: step_index={step_index}, "
                        f"max_steps={self.max_steps}, margin={self._s2_max_steps_near_margin}."
                    ),
                    observation=obs,
                    metadata={
                        "max_steps": self.max_steps,
                        "remaining_steps": max(self.max_steps - step_index, 0),
                        "margin": self._s2_max_steps_near_margin,
                    },
                )
                if s2_mode == S2Mode.TAKEOVER:
                    takeover_remaining = self._s2_pending_takeover_steps
                    last_s2_takeover_reason = self._s2_last_takeover_reason
                    last_s2_stagnation_streak = self._s2_last_stagnation_streak
            effective_memory_context = self._memory_context_with_s2_guidance(
                memory_context,
                s2_guidance_notes,
            )
            messages = self._build_messages(
                task=task,
                current_observation=obs,
                history=history,
                expected_bundle_id=expected_bundle_id,
                memory_context=effective_memory_context,
                skill_context=skill_context,
                prompt_skill_parts=prompt_skill_parts,
            )
            prompt_snapshot = self._snapshot_step_prompt(
                task=task,
                step_index=step_index,
                messages=messages,
                current_observation=obs,
                history=history,
            )

            if takeover_remaining > 0 and self._s2_llm is not None:
                llm_override: LLMProvider | None = self._s2_llm
                actor = "s2_takeover"
                takeover_remaining -= 1
                takeover_context = self._build_s2_takeover_context(
                    task=task,
                    trigger_reason=last_s2_takeover_reason or "stagnation",
                    stagnation_streak=last_s2_stagnation_streak,
                    stagnation_limit=self.stagnation_limit,
                    recent_history=self._recent_history_for_takeover(history),
                    s2_guidance_notes=s2_guidance_notes,
                )
                messages = self._with_s2_takeover_context(messages, takeover_context)
            else:
                llm_override = None
                actor = "s1"

            try:
                result = await asyncio.wait_for(
                    self._run_step(
                        messages=messages,
                        prompt_snapshot=prompt_snapshot,
                        step_index=step_index,
                        total_steps=self.max_steps,
                        current_observation=obs,
                        llm_override=llm_override,
                        actor=actor,
                    ),
                    timeout=self.step_timeout * 3,
                )
            except asyncio.TimeoutError:
                await self._write_trace(run_dir / "trace.jsonl", {
                    "event": "timeout", "step_index": step_index,
                    "timestamp": time.time(),
                })
                self._s2_actor_before = actor
                s2_mode = await self._maybe_invoke_s2(
                    trigger=S2Trigger.STEP_ERROR,
                    step_index=step_index,
                    history=history,
                    reason=f"Step {step_index} timed out before producing a usable GUI action.",
                    observation=obs,
                    metadata={"error": "step_timeout"},
                )
                if s2_mode in {S2Mode.HINT, S2Mode.TAKEOVER}:
                    if s2_mode == S2Mode.TAKEOVER:
                        takeover_remaining = self._s2_pending_takeover_steps
                        last_s2_takeover_reason = self._s2_last_takeover_reason
                        last_s2_stagnation_streak = self._s2_last_stagnation_streak
                    continue
                return AgentResult(
                    success=False,
                    summary=self._build_state_note(
                        status="partial",
                        history=history,
                        current_observation=obs,
                        error="step_timeout",
                    ),
                    model_summary=None,
                    trace_path=str(run_dir),
                    steps_taken=step_index,
                    error="step_timeout",
                    attempt_summary=self._build_attempt_summary(
                        failure_reason="step_timeout",
                        result_summary=self._build_state_note(
                            status="partial",
                            history=history,
                            current_observation=obs,
                            error="step_timeout",
                        ),
                        action_summaries=tuple(turn.action_summary for turn in history),
                    ),
                    token_usage=total_usage,
                    s2_usage=s2_usage.to_dict(),
                )
            except _StepExecutionError as exc:
                self._s2_actor_before = actor
                s2_mode = await self._maybe_invoke_s2(
                    trigger=S2Trigger.STEP_ERROR,
                    step_index=step_index,
                    history=history,
                    reason=f"Step {step_index} failed with {type(exc).__name__}: {exc}",
                    observation=obs,
                    metadata={
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
                if s2_mode in {S2Mode.HINT, S2Mode.TAKEOVER}:
                    if s2_mode == S2Mode.TAKEOVER:
                        takeover_remaining = self._s2_pending_takeover_steps
                        last_s2_takeover_reason = self._s2_last_takeover_reason
                        last_s2_stagnation_streak = self._s2_last_stagnation_streak
                    continue
                raise _StepExecutionError(
                    str(exc),
                    model_snapshot=exc.model_snapshot,
                    attempt_summary=self._build_attempt_summary(
                        failure_reason=f"{type(exc).__name__}: {exc}",
                        result_summary="Attempt ended with an exception before completion.",
                        action_summaries=tuple(turn.action_summary for turn in history),
                    ),
                ) from exc

            steps_taken = step_index
            if actor == "s2_takeover":
                s2_usage.s2_steps += 1
                s2_usage.takeover_steps += 1
                for k, v in result.step_usage.items():
                    s2_usage.token_usage[k] = s2_usage.token_usage.get(k, 0) + v
            else:
                s2_usage.s1_steps += 1
            for k, v in result.step_usage.items():
                total_usage[k] = total_usage.get(k, 0) + v

            intervention_cancelled = False
            if result.intervention_requested:
                request = InterventionRequest(
                    task=task,
                    reason=result.action.text or "",
                    step_index=step_index,
                    platform=self.backend.platform,
                    foreground_app=obs.foreground_app,
                    target=dict(obs.extra),
                )
                await self._log_attempt_event(
                    run_dir,
                    "intervention_requested",
                    step_index=step_index,
                    platform=request.platform,
                    foreground_app=request.foreground_app,
                    reason=request.reason,
                    target=request.target,
                )
                if self._intervention_handler is None:
                    intervention_cancelled = True
                    cancellation_note = "missing_intervention_handler"
                else:
                    resolution = await self._intervention_handler.request_intervention(request)
                    if resolution.resume_confirmed:
                        await self._log_attempt_event(
                            run_dir,
                            "intervention_resumed",
                            step_index=step_index,
                            note=resolution.note,
                        )
                        next_screenshot = run_dir / "screenshots" / f"step_{step_index:03d}.png"
                        next_observation = await self.backend.observe(
                            next_screenshot,
                            timeout=self.step_timeout,
                        )
                        result = replace(
                            result,
                            tool_result="intervention_resumed",
                            next_observation=next_observation,
                            execution_snapshot={
                                "tool_result": "intervention_resumed",
                                "intervention": {
                                    "requested": True,
                                    "note": resolution.note,
                                },
                                "next_observation": self._serialize_observation(next_observation),
                                "done": False,
                            },
                        )
                    else:
                        intervention_cancelled = True
                        cancellation_note = resolution.note or "resume_not_confirmed"

                if intervention_cancelled:
                    scrubbed_cancellation_note = self._scrub_sensitive_text(cancellation_note)
                    scrubbed_intervention_summary = (
                        self._scrub_text_for_artifact_action(
                            result.state_summary or result.action_summary,
                            result.action,
                        )
                        or result.state_summary
                        or result.action_summary
                    )
                    scrubbed_action_summary = (
                        self._scrub_text_for_artifact_action(result.action_summary, result.action)
                        or result.action_summary
                    )
                    await self._log_attempt_event(
                        run_dir,
                        "intervention_cancelled",
                        step_index=step_index,
                        note=scrubbed_cancellation_note,
                    )
                    result = replace(
                        result,
                        tool_result="intervention_cancelled",
                        execution_snapshot={
                            "tool_result": "intervention_cancelled",
                            "intervention": {
                                "requested": True,
                                "note": scrubbed_cancellation_note,
                            },
                            "next_observation": None,
                            "done": False,
                        },
                    )
                    summary_history = history + [
                        HistoryTurn(
                            step_index=step_index,
                            observation=obs,
                            assistant_message=self._scrub_assistant_message_for_log(
                                result.assistant_message,
                                result.action,
                            ),
                            tool_result_message={
                                "role": "tool",
                                "tool_call_id": result.tool_call_id,
                                "content": self._scrub_text_for_action(
                                    result.tool_result,
                                    result.action,
                                ),
                            },
                            action_summary=(
                                self._scrub_text_for_action(
                                    result.action_summary,
                                    result.action,
                                )
                                or result.action_summary
                            ),
                            action_intent=(
                                self._scrub_text_for_action(
                                    result.action_intent,
                                    result.action,
                                )
                                or result.action_intent
                            ),
                            state_summary=(
                                self._scrub_text_for_action(
                                    result.state_summary,
                                    result.action,
                                )
                                or result.state_summary
                            ),
                            raw_response_content=(
                                result.model_snapshot.get("raw_content")
                                if isinstance(result.model_snapshot, dict)
                                else None
                            ),
                        )
                    ]
                    summary_observation = result.next_observation or obs

            if (
                result.next_observation is not None
                and self._is_transient_expected_app_interruption(
                    current_observation=obs,
                    next_observation=result.next_observation,
                    expected_bundle_id=expected_bundle_id,
                )
            ):
                result = await self._refresh_transient_expected_app_interruption(
                    result,
                    current_observation=obs,
                    expected_bundle_id=expected_bundle_id,
                    run_dir=run_dir,
                    step_index=step_index,
                )

            if (
                result.next_observation is not None
                and not browser_relaunch_attempted
                and self._classify_expected_app_observation(
                    result.next_observation,
                    expected_bundle_id=expected_bundle_id,
                    task=task,
                ).get("reason") == "wrong_app_browser"
            ):
                previous_validation = self._classify_expected_app_observation(
                    result.next_observation,
                    expected_bundle_id=expected_bundle_id,
                    task=task,
                )
                next_observation = await self._relaunch_expected_app(
                    result.next_observation,
                    expected_bundle_id=expected_bundle_id,
                    run_dir=run_dir,
                    screenshot_name=f"step_{step_index:03d}_native_relaunch.png",
                )
                result = replace(
                    result,
                    next_observation=next_observation,
                    execution_snapshot={
                        **(result.execution_snapshot or {}),
                        "native_app_validation": {
                            **previous_validation,
                            "previous_foreground_app": result.next_observation.foreground_app,
                            "foreground_app": next_observation.foreground_app,
                        },
                        "next_observation": self._serialize_observation(next_observation),
                    },
                )
                browser_relaunch_attempted = True

            if result.next_observation is not None:
                execution_snapshot = result.execution_snapshot or {}
                if "native_app_validation" not in execution_snapshot:
                    validation = self._classify_expected_app_observation(
                        result.next_observation,
                        expected_bundle_id=expected_bundle_id,
                        task=task,
                    )
                    if validation.get("event") in {"app_mismatch", "transient_overlay"}:
                        result = replace(
                            result,
                            execution_snapshot={
                                **execution_snapshot,
                                "native_app_validation": validation,
                                "next_observation": self._serialize_observation(result.next_observation),
                            },
                        )

            # Write trace entry
            await self._write_trace(
                run_dir / "trace.jsonl",
                self._scrub_for_artifact({
                    "event": "step",
                    "step_index": step_index,
                    "prompt": result.prompt_snapshot,
                    "model_output": result.model_snapshot,
                    "execution": result.execution_snapshot,
                    "action": self._serialize_action(result.action),
                    "action_summary": self._scrub_text_for_artifact_action(result.action_summary, result.action),
                    "action_intent": self._scrub_text_for_artifact_action(result.action_intent, result.action),
                    "state_summary": self._scrub_text_for_artifact_action(result.state_summary, result.action),
                    "screenshot_path": (
                        result.next_observation.screenshot_path
                        if result.next_observation and result.next_observation.screenshot_path
                        else obs.screenshot_path
                    ),
                    "done": result.done,
                    "timestamp": time.time(),
                }),
            )
            self._write_mobileworld_traj(
                run_dir=run_dir,
                task=task,
                step_index=step_index,
                result=result,
                current_observation=obs,
                total_usage=total_usage,
            )

            # Record trajectory step
            self._trajectory_recorder.record_step(
                action=self._scrub_for_artifact(self._serialize_action(result.action)),
                model_output=(
                    self._scrub_text_for_artifact_action(
                        result.action_intent or result.action_summary,
                        result.action,
                    )
                    or ""
                ),
                screenshot_path=(
                    str(result.next_observation.screenshot_path)
                    if result.next_observation and result.next_observation.screenshot_path
                    else str(obs.screenshot_path) if obs.screenshot_path is not None else None
                ),
                foreground_app=(
                    result.next_observation.foreground_app
                    if result.next_observation else obs.foreground_app
                ),
                screen_width=(
                    result.next_observation.screen_width
                    if result.next_observation else obs.screen_width
                ),
                screen_height=(
                    result.next_observation.screen_height
                    if result.next_observation else obs.screen_height
                ),
                platform=(
                    result.next_observation.platform
                    if result.next_observation else obs.platform
                ),
                observation_extra=(
                    self._scrub_for_artifact(result.next_observation.extra)
                    if result.next_observation else self._scrub_for_artifact(obs.extra)
                ),
                interaction_target=self._scrub_for_artifact(result.interaction_target),
                token_usage=result.step_usage or None,
                duration_s=result.duration_s or None,
                chat_latency_s=result.chat_latency_s,
                ttft_s=result.ttft_s,
            )

            if intervention_cancelled:
                termination_summary = await self._generate_termination_summary(
                    task=task,
                    termination_reason=f"Task was interrupted by policy: {scrubbed_cancellation_note}",
                    history=summary_history,
                    run_dir=run_dir,
                )
                result_summary = self._build_state_note(
                    status="blocked",
                    history=summary_history,
                    current_observation=summary_observation,
                    error="intervention_cancelled",
                )
                return AgentResult(
                    success=False,
                    summary=termination_summary or result_summary,
                    model_summary=scrubbed_intervention_summary,
                    trace_path=str(run_dir),
                    steps_taken=steps_taken,
                    error=f"intervention_cancelled: {scrubbed_cancellation_note}",
                    attempt_summary=self._build_attempt_summary(
                        failure_reason=f"intervention_cancelled: {scrubbed_cancellation_note}",
                        result_summary=result_summary,
                        model_summary=scrubbed_intervention_summary,
                        action_summaries=tuple(
                            list(turn.action_summary for turn in history) + [scrubbed_action_summary]
                        ),
                    ),
                    token_usage=total_usage,
                    s2_usage=s2_usage.to_dict(),
                )

            if result.done:
                success = self._resolve_done_status(result.action) == "success"
                if (
                    success
                    and S2Trigger.MISSING_EVIDENCE in self._s2_allowed_triggers
                    and self._done_missing_evidence(
                        task=task,
                        result=result,
                        current_observation=obs,
                    )
                ):
                    evidence_observation = result.next_observation or obs
                    self._s2_actor_before = actor
                    s2_mode = await self._maybe_invoke_s2(
                        trigger=S2Trigger.MISSING_EVIDENCE,
                        step_index=step_index,
                        history=history,
                        reason=(
                            "Done action did not provide required answer_candidates "
                            "or visible evidence for an information query."
                        ),
                        observation=evidence_observation,
                        metadata={"done_status": "success"},
                    )
                    if s2_mode in {S2Mode.HINT, S2Mode.TAKEOVER}:
                        if result.next_observation is not None:
                            obs = result.next_observation
                        if s2_mode == S2Mode.TAKEOVER:
                            takeover_remaining = self._s2_pending_takeover_steps
                            last_s2_takeover_reason = self._s2_last_takeover_reason
                            last_s2_stagnation_streak = self._s2_last_stagnation_streak
                        continue
                return AgentResult(
                    success=success,
                    summary=self._build_state_note(
                        status="completed" if success else "blocked",
                        history=history,
                        current_observation=obs,
                        current_action_summary=result.state_summary or result.action_summary,
                        error=None if success else result.tool_result,
                    ),
                    model_summary=result.state_summary or result.action_summary,
                    trace_path=str(run_dir),
                    steps_taken=steps_taken,
                    error=None if success else result.tool_result,
                    attempt_summary=None if success else self._build_attempt_summary(
                        failure_reason=result.tool_result,
                        result_summary=self._build_state_note(
                            status="blocked",
                            history=history,
                            current_observation=obs,
                            current_action_summary=result.state_summary or result.action_summary,
                            error=result.tool_result,
                        ),
                        model_summary=result.state_summary or result.action_summary,
                        action_summaries=tuple(
                            list(turn.action_summary for turn in history) + [result.action_summary]
                        ),
                    ),
                    token_usage=total_usage,
                    s2_usage=s2_usage.to_dict(),
                )

            if self.stagnation_limit > 0 and result.next_observation is not None:
                current_fingerprint = self._build_screen_fingerprint(result.next_observation)
                if (
                    previous_fingerprint is not None
                    and current_fingerprint is not None
                    and (previous_action_type is None or previous_action_type == result.action.action_type)
                    and self._is_same_screen(previous_fingerprint, current_fingerprint)
                ):
                    stagnation_streak += 1
                else:
                    stagnation_streak = 0
                previous_fingerprint = current_fingerprint
                previous_action_type = result.action.action_type

                if stagnation_streak >= self.stagnation_limit:
                    app_label = (
                        result.next_observation.foreground_app
                        or obs.foreground_app
                        or "unknown"
                    )
                    history_with_current_step = history + [
                        HistoryTurn(
                            step_index=step_index,
                            observation=obs,
                            assistant_message=self._scrub_assistant_message_for_log(
                                result.assistant_message,
                                result.action,
                            ),
                            tool_result_message={
                                "role": "tool",
                                "tool_call_id": result.tool_call_id,
                                "content": self._scrub_text_for_action(
                                    result.tool_result,
                                    result.action,
                                ),
                            },
                            action_summary=(
                                self._scrub_text_for_action(
                                    result.action_summary,
                                    result.action,
                                )
                                or result.action_summary
                            ),
                            action_intent=(
                                self._scrub_text_for_action(
                                    result.action_intent,
                                    result.action,
                                )
                                or result.action_intent
                            ),
                            state_summary=(
                                self._scrub_text_for_action(
                                    result.state_summary,
                                    result.action,
                                )
                                or result.state_summary
                            ),
                            raw_response_content=(
                                result.model_snapshot.get("raw_content")
                                if isinstance(result.model_snapshot, dict)
                                else None
                            ),
                        )
                    ]
                    stagnation_reason = (
                        "Detected unchanged screen state for "
                        f"{stagnation_streak} consecutive step(s) in app {app_label}; "
                        "task would stop to avoid repeating the same action loop."
                    )
                    self._s2_actor_before = actor
                    s2_mode = await self._maybe_invoke_s2(
                        trigger=S2Trigger.STAGNATION,
                        step_index=step_index,
                        history=history_with_current_step,
                        reason=stagnation_reason,
                        observation=result.next_observation or obs,
                        metadata={
                            "foreground_app": app_label,
                            "stagnation_streak": stagnation_streak,
                            "stagnation_limit": self.stagnation_limit,
                        },
                    )
                    if s2_mode == S2Mode.HINT:
                        history.append(history_with_current_step[-1])
                        if result.next_observation is not None:
                            obs = result.next_observation
                        stagnation_streak = 0
                        previous_fingerprint = self._build_screen_fingerprint(obs)
                        previous_action_type = None
                        continue

                    if s2_mode == S2Mode.TAKEOVER:
                        takeover_remaining = self._s2_pending_takeover_steps
                        last_s2_takeover_reason = self._s2_last_takeover_reason
                        last_s2_stagnation_streak = self._s2_last_stagnation_streak
                        history.append(history_with_current_step[-1])
                        if result.next_observation is not None:
                            obs = result.next_observation
                        stagnation_streak = 0
                        previous_fingerprint = self._build_screen_fingerprint(obs)
                        previous_action_type = None
                        continue

                    termination_summary = await self._generate_termination_summary(
                        task=task,
                        termination_reason=(
                            "Detected unchanged screen state for "
                            f"{stagnation_streak} consecutive step(s) in app {app_label}; "
                            "task stopped to avoid repeating the same action loop."
                        ),
                        history=history_with_current_step,
                        run_dir=run_dir,
                    )
                    await self._log_attempt_event(
                        run_dir,
                        "stagnation_detected",
                        step_index=step_index,
                        stagnation_streak=stagnation_streak,
                        stagnation_limit=self.stagnation_limit,
                        foreground_app=app_label,
                    )
                    return AgentResult(
                        success=False,
                        summary=termination_summary or self._build_state_note(
                            status="blocked",
                            history=history_with_current_step,
                            current_observation=result.next_observation or obs,
                            error="stagnation_detected",
                        ),
                        model_summary=result.state_summary or result.action_summary,
                        trace_path=str(run_dir),
                        steps_taken=steps_taken,
                        error="stagnation_detected",
                        attempt_summary=self._build_attempt_summary(
                            failure_reason="stagnation_detected",
                            result_summary=self._build_state_note(
                                status="blocked",
                                history=history_with_current_step,
                                current_observation=result.next_observation or obs,
                                error="stagnation_detected",
                            ),
                            model_summary=result.state_summary or result.action_summary,
                            action_summaries=tuple(
                                list(turn.action_summary for turn in history) + [result.action_summary]
                            ),
                        ),
                        token_usage=total_usage,
                        s2_usage=s2_usage.to_dict(),
                    )

            history.append(
                HistoryTurn(
                    step_index=step_index,
                    observation=obs,
                    assistant_message=self._scrub_assistant_message_for_log(
                        result.assistant_message,
                        result.action,
                    ),
                    tool_result_message={
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": self._scrub_text_for_action(result.tool_result, result.action),
                    },
                    action_summary=(
                        self._scrub_text_for_action(result.action_summary, result.action)
                        or result.action_summary
                    ),
                    action_intent=(
                        self._scrub_text_for_action(result.action_intent, result.action)
                        or result.action_intent
                    ),
                    state_summary=(
                        self._scrub_text_for_action(result.state_summary, result.action)
                        or result.state_summary
                    ),
                    raw_response_content=(
                        result.model_snapshot.get("raw_content")
                        if isinstance(result.model_snapshot, dict)
                        else None
                    ),
                )
            )

            if result.next_observation is not None:
                obs = result.next_observation

        termination_summary = await self._generate_termination_summary(
            task=task,
            termination_reason=f"Reached maximum step limit ({self.max_steps})",
            history=history,
            run_dir=run_dir,
        )
        return AgentResult(
            success=False,
            summary=termination_summary or self._build_state_note(
                status="partial",
                history=history,
                current_observation=obs,
                error="max_steps_exceeded",
            ),
            model_summary=None,
            trace_path=str(run_dir),
            steps_taken=steps_taken,
            error="max_steps_exceeded",
            attempt_summary=self._build_attempt_summary(
                failure_reason="max_steps_exceeded",
                result_summary=self._build_state_note(
                    status="partial",
                    history=history,
                    current_observation=obs,
                    error="max_steps_exceeded",
                ),
                action_summaries=tuple(turn.action_summary for turn in history),
            ),
            token_usage=total_usage,
            s2_usage=s2_usage.to_dict(),
        )

    @staticmethod
    def _observation_matches_expected_bundle(
        observation: Observation | None,
        *,
        expected_bundle_id: str | None,
    ) -> bool:
        if not observation or not expected_bundle_id:
            return False
        actual = str(observation.foreground_app or "").casefold()
        expected = str(expected_bundle_id or "").casefold()
        return bool(actual and expected and actual == expected)

    @staticmethod
    def _observation_visible_text_blob(observation: Observation | None) -> str:
        if observation is None:
            return ""
        extra = getattr(observation, "extra", None)
        if not isinstance(extra, dict):
            return ""
        parts: list[str] = []
        for key in ("visible_text", "texts", "labels", "ui_text"):
            value = extra.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, (list, tuple, set)):
                parts.extend(str(item) for item in value if item is not None)
        return " ".join(parts).casefold()

    def _transient_overlay_kind(
        self,
        observation: Observation | None,
        *,
        expected_bundle_id: str | None,
    ) -> str | None:
        if observation is None:
            return None
        platform = str(getattr(observation, "platform", "") or self.backend.platform or "").lower()
        foreground_app = str(observation.foreground_app or "")
        if platform == "ios" and foreground_app in self._IOS_TRANSIENT_SYSTEM_APPS:
            return "ios_system_overlay"
        if self._observation_matches_expected_bundle(observation, expected_bundle_id=expected_bundle_id):
            blob = self._observation_visible_text_blob(observation)
            if blob and any(term.casefold() in blob for term in self._TRANSIENT_OVERLAY_TERMS):
                return "transient_overlay"
        return None

    def _classify_expected_app_observation(
        self,
        observation: Observation | None,
        *,
        expected_bundle_id: str | None,
        task: str,
    ) -> dict[str, Any]:
        if observation is None:
            return {
                "event": "no_observation",
                "expected_bundle_id": expected_bundle_id,
                "foreground_app": None,
                "matches_expected_app": False,
                "app_mismatch": False,
            }
        foreground_app = str(observation.foreground_app or "")
        matches = self._observation_matches_expected_bundle(
            observation,
            expected_bundle_id=expected_bundle_id,
        )
        transient_kind = self._transient_overlay_kind(
            observation,
            expected_bundle_id=expected_bundle_id,
        )
        base = {
            "expected_bundle_id": expected_bundle_id,
            "foreground_app": foreground_app or None,
            "matches_expected_app": matches,
            "app_mismatch": False,
        }
        if not expected_bundle_id or task_explicitly_allows_browser(task):
            return {**base, "event": "not_app_constrained"}
        if transient_kind is not None:
            return {**base, "event": "transient_overlay", "reason": transient_kind}
        if matches:
            return {**base, "event": "expected_app"}
        if not foreground_app:
            return {**base, "event": "unknown_foreground_app"}
        if is_browser_bundle(foreground_app):
            return {
                **base,
                "event": "app_mismatch",
                "reason": "wrong_app_browser",
                "app_mismatch": True,
            }
        return {
            **base,
            "event": "app_mismatch",
            "reason": "wrong_app",
            "app_mismatch": True,
        }

    def _is_transient_expected_app_interruption(
        self,
        *,
        current_observation: Observation,
        next_observation: Observation | None,
        expected_bundle_id: str | None,
    ) -> bool:
        if not expected_bundle_id or next_observation is None:
            return False
        if not self._observation_matches_expected_bundle(
            current_observation,
            expected_bundle_id=expected_bundle_id,
        ):
            return False
        kind = self._transient_overlay_kind(
            next_observation,
            expected_bundle_id=expected_bundle_id,
        )
        return kind == "ios_system_overlay"

    async def _refresh_transient_expected_app_interruption(
        self,
        result: StepResult,
        *,
        current_observation: Observation,
        expected_bundle_id: str | None,
        run_dir: Path,
        step_index: int,
    ) -> StepResult:
        next_observation = result.next_observation
        if not self._is_transient_expected_app_interruption(
            current_observation=current_observation,
            next_observation=next_observation,
            expected_bundle_id=expected_bundle_id,
        ):
            return result
        await asyncio.sleep(0.25)
        screenshot_dir = run_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        try:
            refreshed = await self.backend.observe(
                screenshot_dir / f"step_{step_index:03d}_transient_reobserve.png",
                timeout=self.step_timeout,
            )
        except Exception as exc:
            await self._log_attempt_event(
                run_dir,
                "transient_overlay_reobserve_failed",
                monitor_event="transient_overlay",
                reason="ios_system_overlay",
                expected_bundle_id=expected_bundle_id,
                current_observation=self._serialize_observation(current_observation),
                next_observation=self._serialize_observation(next_observation),
                error=f"{type(exc).__name__}: {exc}",
            )
            return result

        validation = self._classify_expected_app_observation(
            refreshed,
            expected_bundle_id=expected_bundle_id,
            task="",
        )
        await self._log_attempt_event(
            run_dir,
            "transient_overlay_reobserved",
            monitor_event="transient_overlay",
            reason="ios_system_overlay",
            expected_bundle_id=expected_bundle_id,
            current_observation=self._serialize_observation(current_observation),
            next_observation=self._serialize_observation(next_observation),
            refreshed_observation=self._serialize_observation(refreshed),
            accepted=validation.get("matches_expected_app") is True,
        )
        if validation.get("matches_expected_app") is not True:
            return result
        return replace(
            result,
            next_observation=refreshed,
            execution_snapshot={
                **(result.execution_snapshot or {}),
                "native_app_validation": {
                    "event": "transient_overlay",
                    "reason": "ios_system_overlay",
                    "expected_bundle_id": expected_bundle_id,
                    "previous_foreground_app": (
                        next_observation.foreground_app if next_observation else None
                    ),
                    "foreground_app": refreshed.foreground_app,
                    "app_mismatch": False,
                },
                "next_observation": self._serialize_observation(refreshed),
            },
        )

    @staticmethod
    def _should_relaunch_expected_app(
        observation: Observation,
        *,
        expected_bundle_id: str | None,
        task: str,
    ) -> bool:
        if not expected_bundle_id or task_explicitly_allows_browser(task):
            return False
        return is_browser_bundle(observation.foreground_app)

    async def _relaunch_expected_app(
        self,
        observation: Observation,
        *,
        expected_bundle_id: str | None,
        run_dir: Path,
        screenshot_name: str,
    ) -> Observation:
        if not expected_bundle_id:
            return observation
        previous_foreground_app = observation.foreground_app
        try:
            await self.backend.execute(
                Action(action_type="open_app", text=expected_bundle_id),
                timeout=self.step_timeout,
            )
            screenshot_dir = run_dir / "screenshots"
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            next_observation = await self.backend.observe(
                screenshot_dir / screenshot_name,
                timeout=self.step_timeout,
            )
            await self._log_attempt_event(
                run_dir,
                "native_app_relaunch",
                monitor_event="app_mismatch",
                reason="wrong_app_browser",
                expected_bundle_id=expected_bundle_id,
                previous_foreground_app=previous_foreground_app,
                foreground_app=next_observation.foreground_app,
            )
            return next_observation
        except Exception as exc:
            await self._log_attempt_event(
                run_dir,
                "native_app_relaunch_failed",
                monitor_event="app_mismatch",
                reason="wrong_app_browser",
                expected_bundle_id=expected_bundle_id,
                previous_foreground_app=previous_foreground_app,
                error=f"{type(exc).__name__}: {exc}",
            )
            return observation

    async def _maybe_invoke_s2(
        self,
        *,
        trigger: S2Trigger,
        step_index: int,
        history: list[HistoryTurn],
        reason: str,
        observation: Observation,
        metadata: dict[str, Any] | None = None,
    ) -> S2Mode:
        usage = self._s2_usage
        mode = decide_s2_mode(
            enabled=self._s2_enabled,
            trigger=trigger,
            hints_used=usage.hints_used,
            max_hints=self._s2_max_hints,
            hint_enabled=self._s2_hint_enabled,
            takeover_enabled=self._s2_takeover_enabled,
            takeover_after_hints=self._s2_takeover_after_hints,
            takeover_used=usage.takeover_used,
            allowed_triggers=self._s2_allowed_triggers,
        )
        if mode == S2Mode.OFF:
            return mode

        event_metadata = dict(metadata or {})
        app_label = observation.foreground_app or event_metadata.get("foreground_app") or "unknown"
        event_metadata.setdefault("foreground_app", str(app_label))
        actor_before = self._s2_actor_before or "s1"

        if mode == S2Mode.HINT:
            last_action_summary = None
            if history:
                last_turn = history[-1]
                last_action_summary = last_turn.state_summary or last_turn.action_summary
            hint = await self._request_s2_hint(
                task=self._s2_current_task,
                step_index=step_index,
                current_observation=observation,
                history=history,
                last_action_summary=last_action_summary,
                reason=reason,
            )
            usage.hints_used += 1
            event_metadata["hint_available"] = bool(hint)
            usage.triggers.append(
                S2TriggerEvent(
                    step_index=step_index,
                    trigger=trigger,
                    mode=S2Mode.HINT,
                    reason=reason,
                    actor_before=actor_before,
                    metadata=event_metadata,
                )
            )
            if hint:
                self._s2_guidance_notes.append(hint)
            if self._s2_run_dir is not None:
                await self._log_attempt_event(
                    self._s2_run_dir,
                    "s2_hint",
                    step_index=step_index,
                    trigger=trigger.value,
                    reason=reason,
                    foreground_app=app_label,
                    hint_available=bool(hint),
                )
            return mode

        if mode == S2Mode.TAKEOVER and self._s2_llm is not None:
            remaining_steps = max(self.max_steps - step_index, 0)
            takeover_steps = min(self._s2_max_takeover_steps, remaining_steps)
            if takeover_steps <= 0:
                return S2Mode.OFF
            self._s2_pending_takeover_steps = takeover_steps
            self._s2_last_takeover_reason = reason
            self._s2_last_stagnation_streak = int(event_metadata.get("stagnation_streak") or 0)
            usage.takeover_used = True
            usage.triggers.append(
                S2TriggerEvent(
                    step_index=step_index,
                    trigger=trigger,
                    mode=S2Mode.TAKEOVER,
                    reason=reason,
                    actor_before=actor_before,
                    metadata=event_metadata,
                )
            )
            if self._s2_run_dir is not None:
                await self._log_attempt_event(
                    self._s2_run_dir,
                    "s2_takeover",
                    step_index=step_index,
                    trigger=trigger.value,
                    reason=reason,
                    foreground_app=app_label,
                    takeover_remaining=takeover_steps,
                )
            return mode

        return S2Mode.OFF

    @staticmethod
    def _task_likely_requires_answer(task: str) -> bool:
        lowered = str(task or "").casefold()
        return any(
            marker in lowered
            for marker in (
                "告诉我",
                "看看",
                "查",
                "查询",
                "是什么",
                "是谁",
                "哪个",
                "哪家",
                "多少",
                "第几",
                "热搜",
                "榜",
                "排名",
                "price",
                "status",
                "what",
                "which",
                "who",
            )
        )

    def _done_missing_evidence(
        self,
        *,
        task: str,
        result: StepResult,
        current_observation: Observation,
    ) -> bool:
        if not self._task_likely_requires_answer(task):
            return False
        request = {
            "task": task,
            "task_type": "information_query",
            "output_mode": "answer_required",
            "evidence_requirements": {"answer_candidates_required": True},
            "success_condition": {"required_key": "answer"},
        }
        payload = {
            "success": True,
            "summary": result.state_summary or result.action_summary,
            "model_summary": result.state_summary or result.action_summary,
            "answer_candidates": [],
        }
        observation = result.next_observation or current_observation
        latest_step = {
            "text": result.state_summary or result.action_summary,
            "observation": {
                "foreground_app": observation.foreground_app,
                "text": result.tool_result,
                "extra": observation.extra,
            },
        }
        extracted = extract_gui_evidence(
            request=request,
            payload=payload,
            latest_step=latest_step,
        )
        return not bool(extracted.get("answer_candidates"))

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    async def _run_step(
        self,
        messages: list[dict[str, Any]],
        prompt_snapshot: dict[str, Any] | None,
        step_index: int,
        total_steps: int,
        current_observation: Observation,
        *,
        llm_override: LLMProvider | None = None,
        actor: str = "s1",
    ) -> StepResult:
        """Execute a single vision-action step with retries on malformed calls."""
        _step_start = time.monotonic()
        fg = str(current_observation.foreground_app or "").strip() if current_observation else ""
        await self._ensure_shortcuts_for_app(fg)
        retries_left = self._MAX_TOOL_RETRIES + 1
        step_usage: dict[str, int] = {}
        step_chat_latency_s: float = 0.0
        step_ttft_s: float | None = None

        while retries_left > 0:
            retries_left -= 1

            # Call LLM
            native_tools_enabled = profile_uses_native_tools(self.agent_profile)
            active_llm = llm_override or self.llm
            request_messages = (
                self._messages_with_s2_takeover_policy(messages)
                if actor == "s2_takeover"
                else messages
            )
            reasoning_effort = (
                self._s2_reasoning_effort
                if actor == "s2_takeover"
                else self._s1_reasoning_effort
            )
            chat_kwargs: dict[str, Any] = {
                "messages": request_messages,
                "tools": self._build_tools_list() if native_tools_enabled else None,
                "tool_choice": "required" if native_tools_enabled else None,
            }
            if reasoning_effort is not None:
                chat_kwargs["reasoning_effort"] = reasoning_effort
            response: LLMResponse = await active_llm.chat(**chat_kwargs)
            for k, v in (response.usage or {}).items():
                step_usage[k] = step_usage.get(k, 0) + v
            if response.latency_s is not None:
                step_chat_latency_s += response.latency_s
            if step_ttft_s is None and response.ttft_s is not None:
                step_ttft_s = response.ttft_s
            raw_response_snapshot = self._snapshot_failed_model_response(response)

            try:
                response = normalize_profile_response_for_observation(
                    self.agent_profile,
                    response,
                    current_observation,
                    model_name=self.model,
                )
            except ValueError as exc:
                detail = f"{exc}. Follow the required response format exactly."
                feedback = self._build_tool_format_error(
                    native_tools=native_tools_enabled,
                    detail=detail,
                )
                if retries_left > 0:
                    messages.append({
                        "role": "user",
                        "content": feedback,
                    })
                    continue
                raise _StepExecutionError(
                    f"Failed to parse profile response after retries: {exc}",
                    model_snapshot=raw_response_snapshot,
                ) from exc

            # Append assistant message
            assistant_msg = self._build_assistant_message(
                response,
                include_tool_calls=native_tools_enabled,
            )
            messages.append(assistant_msg)
            assistant_snapshot = self._snapshot_failed_model_response(
                response,
                assistant_message=assistant_msg,
            )

            # Validate tool call
            if not response.tool_calls or len(response.tool_calls) == 0:
                if retries_left > 0:
                    feedback = self._build_tool_format_error(
                        native_tools=native_tools_enabled,
                        detail=(
                            "No action payload found. "
                            "Return one `Thought:` + `Action:` response "
                            "in the configured profile format."
                        ),
                    )
                    messages.append({
                        "role": "user",
                        "content": feedback,
                    })
                    continue
                raise _StepExecutionError(
                    "LLM did not return a computer_use tool call after retries.",
                    model_snapshot=assistant_snapshot,
                )

            tool_call = response.tool_calls[0]
            if native_tools_enabled and tool_call.name != "computer_use" and tool_call.name not in self._shortcut_action_map:
                if retries_left > 0:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: unexpected tool '{tool_call.name}'.",
                    })
                    continue
                raise _StepExecutionError(
                    f"LLM called unexpected tool '{tool_call.name}'.",
                    model_snapshot=assistant_snapshot,
                )
            if tool_call.name in self._shortcut_action_map:
                action_type, text, component, mime_type = self._shortcut_action_map[tool_call.name]
                arguments = {
                    "action_type": action_type,
                    "text": text,
                    "component": component,
                    "intent": f"Shortcut: {tool_call.name}",
                    "summary": f"Executing shortcut {tool_call.name}",
                }
                if action_type == "open_intent":
                    arguments["intent_action"] = text
                    if mime_type:
                        arguments["mime_type"] = mime_type
                tool_call = replace(tool_call, arguments=arguments)
            action_intent, state_summary = self._tool_call_semantics(tool_call)

            special_action_type = str(
                (tool_call.arguments or {}).get("action_type")
                or (tool_call.arguments or {}).get("action")
                or ""
            ).strip().lower()
            if special_action_type == "use_skill" or special_action_type in self._prompt_composite_aliases:
                try:
                    special_result = await self._dispatch_prompt_special_action(
                        tool_call=tool_call,
                        response=response,
                        assistant_message=assistant_msg,
                        prompt_snapshot=prompt_snapshot,
                        assistant_snapshot=assistant_snapshot,
                        current_observation=current_observation,
                        step_index=step_index,
                        step_usage=step_usage,
                        step_start=_step_start,
                        step_chat_latency_s=step_chat_latency_s,
                        step_ttft_s=step_ttft_s,
                        action_intent=action_intent,
                        state_summary=state_summary,
                    )
                    return self._tag_step_actor(special_result, actor)
                except ActionError as exc:
                    if retries_left > 0:
                        messages.append({
                            "role": "user",
                            "content": f"Format error: {exc}. Please return a listed skill/composite action or a normal GUI action.",
                        })
                        continue
                    raise _StepExecutionError(
                        f"Failed to dispatch prompt special action after retries: {exc}",
                        model_snapshot=assistant_snapshot,
                    ) from exc

            # Parse action
            try:
                action = parse_action(tool_call.arguments)
                action = self._normalize_relative_coordinates(action)
            except ActionError as exc:
                if retries_left > 0:
                    if native_tools_enabled:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error parsing action: {exc}. Please fix and retry.",
                        })
                    else:
                        messages.append({
                            "role": "user",
                            "content": f"Format error: {exc}. Please return {tool_call.name} output in the configured profile format.",
                        })
                    continue
                raise _StepExecutionError(
                    f"Failed to parse action after retries: {exc}",
                    model_snapshot=assistant_snapshot,
                ) from exc

            # Report progress
            if self.progress_callback is not None:
                await self.progress_callback(
                    f"GUI step {step_index}/{total_steps}: {describe_action(action)}"
                )

            action_text = self._normalize_action_text(
                response.content,
                action,
                tool_summary=action_intent or state_summary,
            )
            action_summary = action_intent or self._action_summary(action_text)
            assistant_message = self._build_assistant_message(
                response,
                content_override=action_text,
                include_tool_calls=native_tools_enabled,
            )
            model_snapshot = self._snapshot_model_response(
                response=response,
                action=action,
                assistant_message=assistant_message,
                action_text=action_text,
                action_intent=action_summary,
                state_summary=state_summary,
            )

            # Handle terminal action (done)
            if action.action_type == "done":
                done_status = self._resolve_done_status(action)
                if action.status != done_status:
                    action = replace(action, status=done_status)
                tool_result = f"Task terminated with status: {done_status}"
                return self._tag_step_actor(StepResult(
                    action=action,
                    tool_call_id=tool_call.id,
                    tool_result=tool_result,
                    assistant_message=assistant_message,
                    action_summary=action_summary,
                    action_intent=action_summary,
                    state_summary=state_summary,
                    prompt_snapshot=prompt_snapshot,
                    model_snapshot=model_snapshot,
                    execution_snapshot={
                        "tool_result": tool_result,
                        "next_observation": None,
                        "done": True,
                    },
                    done=True,
                    step_usage=step_usage,
                    duration_s=time.monotonic() - _step_start,
                    chat_latency_s=step_chat_latency_s or None,
                    ttft_s=step_ttft_s,
                ), actor)

            if action.action_type == "request_intervention":
                return self._tag_step_actor(StepResult(
                    action=action,
                    tool_call_id=tool_call.id,
                    tool_result="intervention_requested",
                    assistant_message=assistant_message,
                    action_summary=action_summary,
                    action_intent=action_summary,
                    state_summary=state_summary,
                    prompt_snapshot=prompt_snapshot,
                    model_snapshot=model_snapshot,
                    execution_snapshot={
                        "tool_result": "intervention_requested",
                        "next_observation": None,
                        "done": False,
                    },
                    intervention_requested=True,
                    step_usage=step_usage,
                    duration_s=time.monotonic() - _step_start,
                    chat_latency_s=step_chat_latency_s or None,
                    ttft_s=step_ttft_s,
                ), actor)

            # Normalize app identifiers for mobile open/close actions.
            if (
                action.action_type in ("open_app", "close_app")
                and action.text
                and self.backend.platform in ("android", "ios")
            ):
                resolved = self._normalize_backend_app_identifier(action.text)
                if resolved != action.text:
                    logger.debug(
                        "Resolved %s app name %r -> %r",
                        self.backend.platform,
                        action.text,
                        resolved,
                    )
                    action = replace(action, text=resolved)

            interaction_target = infer_interaction_target(action, current_observation)

            # Execute action on backend
            try:
                result_text = await self.backend.execute(action, timeout=self.step_timeout)
            except Exception as exc:
                result_text = f"Action failed: {exc}"

            settle_seconds = self._post_action_settle_seconds(action)
            if settle_seconds > 0:
                await asyncio.sleep(settle_seconds)

            # Observe next state
            run_dir = Path(current_observation.screenshot_path or ".").parent.parent
            next_screenshot = run_dir / "screenshots" / f"step_{step_index:03d}.png"
            next_observation, _observe_error = await self._observe_after_action(
                next_screenshot,
                previous_observation=current_observation,
                action=action,
                timeout=self.step_timeout,
            )

            return self._tag_step_actor(StepResult(
                action=action,
                tool_call_id=tool_call.id,
                tool_result=result_text,
                assistant_message=assistant_message,
                action_summary=action_summary,
                action_intent=action_summary,
                state_summary=state_summary,
                next_observation=next_observation,
                interaction_target=interaction_target,
                prompt_snapshot=prompt_snapshot,
                model_snapshot=model_snapshot,
                execution_snapshot={
                    "tool_result": self._scrub_text_for_action(result_text, action),
                    "next_observation": self._serialize_observation(next_observation),
                    "done": False,
                },
                step_usage=step_usage,
                duration_s=time.monotonic() - _step_start,
                chat_latency_s=step_chat_latency_s or None,
                ttft_s=step_ttft_s,
            ), actor)

        raise RuntimeError("GUI model did not return a valid computer_use call after retries.")

    @staticmethod
    def _tag_step_actor(result: StepResult, actor: str) -> StepResult:
        model_snapshot = dict(result.model_snapshot or {})
        model_snapshot["actor"] = actor
        execution_snapshot = dict(result.execution_snapshot or {})
        execution_snapshot["actor"] = actor
        return replace(
            result,
            model_snapshot=model_snapshot,
            execution_snapshot=execution_snapshot,
        )

    async def _dispatch_prompt_special_action(
        self,
        *,
        tool_call: ToolCall,
        response: LLMResponse,
        assistant_message: dict[str, Any],
        prompt_snapshot: dict[str, Any] | None,
        assistant_snapshot: dict[str, Any],
        current_observation: Observation,
        step_index: int,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
        action_intent: str | None,
        state_summary: str | None,
    ) -> StepResult:
        del assistant_message, assistant_snapshot
        arguments = tool_call.arguments or {}
        action_type = str(arguments.get("action_type") or arguments.get("action") or "").strip().lower()
        if action_type == "use_skill":
            return await self._execute_prompt_skill_action(
                tool_call=tool_call,
                response=response,
                prompt_snapshot=prompt_snapshot,
                current_observation=current_observation,
                step_index=step_index,
                step_usage=step_usage,
                step_start=step_start,
                step_chat_latency_s=step_chat_latency_s,
                step_ttft_s=step_ttft_s,
                action_intent=action_intent,
                state_summary=state_summary,
            )
        if action_type in self._prompt_composite_aliases:
            return await self._execute_prompt_composite_action(
                tool_call=tool_call,
                response=response,
                prompt_snapshot=prompt_snapshot,
                current_observation=current_observation,
                step_index=step_index,
                step_usage=step_usage,
                step_start=step_start,
                step_chat_latency_s=step_chat_latency_s,
                step_ttft_s=step_ttft_s,
                action_intent=action_intent,
                state_summary=state_summary,
            )
        raise ActionError(f"Unknown prompt special action {action_type!r}.")

    async def _execute_prompt_skill_action(
        self,
        *,
        tool_call: ToolCall,
        response: LLMResponse,
        prompt_snapshot: dict[str, Any] | None,
        current_observation: Observation,
        step_index: int,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
        action_intent: str | None,
        state_summary: str | None,
    ) -> StepResult:
        arguments = tool_call.arguments or {}
        skill_id = str(arguments.get("skill_id") or "").strip()
        if not skill_id:
            raise ActionError("use_skill requires a listed 'skill_id'.")
        skill = self._prompt_skills_by_id.get(skill_id)
        if skill is None:
            raise ActionError(f"use_skill skill_id {skill_id!r} is not in the prompt skill list.")
        if self._skill_executor is None:
            raise ActionError("use_skill requires a configured SkillExecutor.")

        raw_params = arguments.get("arguments") or arguments.get("params") or {}
        if not isinstance(raw_params, dict):
            raise ActionError("use_skill 'arguments' must be an object.")
        params = {str(key): self._stringify_skill_argument(value) for key, value in raw_params.items()}
        action = Action(action_type="use_skill", text=skill_id)
        skill_name = str(getattr(skill, "name", "") or skill_id)
        summary = f"use_skill {skill_name}"
        assistant_message = self._build_assistant_message(
            response,
            content_override=self._normalize_action_text(
                response.content,
                action,
                tool_summary=action_intent or state_summary or summary,
            ),
            include_tool_calls=profile_uses_native_tools(self.agent_profile),
        )

        self._trajectory_recorder.record_event(
            "prompt_skill_selected",
            skill_id=skill_id,
            skill_name=skill_name,
            arguments=params,
        )
        self._trajectory_recorder.set_phase(
            ExecutionPhase.SKILL,
            reason=f"Prompt-selected skill: {skill_name}",
        )
        try:
            try:
                missing_params = _missing_skill_params(skill, params)
                if missing_params:
                    self._trajectory_recorder.record_event(
                        "prompt_skill_rejected",
                        skill_id=skill_id,
                        skill_name=skill_name,
                        reason="missing_required_params",
                        missing_params=missing_params,
                    )
                    raise ActionError(
                        f"use_skill {skill_name} is missing required params: "
                        f"{', '.join(missing_params)}"
                    )
                if not _prompt_skill_entry_allows(skill, current_observation, params):
                    self._trajectory_recorder.record_event(
                        "prompt_skill_rejected",
                        skill_id=skill_id,
                        skill_name=skill_name,
                        reason="entry_precondition_not_met",
                        first_action=str(
                            getattr(skill.steps[0], "action_type", "") if skill.steps else ""
                        ),
                    )
                    raise ActionError(
                        f"use_skill {skill_name} is not applicable on the current screen; "
                        "use manual GUI actions instead"
                    )
                skill_result = await self._skill_executor.execute(
                    skill,
                    params=params,
                    timeout=self.step_timeout,
                )
            except Exception as exc:
                return await self._prompt_skill_exception_result(
                    exc,
                    skill_id=skill_id,
                    skill_name=skill_name,
                    action=action,
                    tool_call=tool_call,
                    response=response,
                    assistant_message=assistant_message,
                    prompt_snapshot=prompt_snapshot,
                    current_observation=current_observation,
                    step_index=step_index,
                    step_usage=step_usage,
                    step_start=step_start,
                    step_chat_latency_s=step_chat_latency_s,
                    step_ttft_s=step_ttft_s,
                    action_intent=action_intent,
                )
        finally:
            self._trajectory_recorder.set_phase(
                ExecutionPhase.AGENT,
                reason="Prompt-selected skill finished",
            )

        merged_usage = dict(step_usage)
        raw_skill_usage = getattr(skill_result, "token_usage", None)
        if isinstance(raw_skill_usage, dict):
            for key, value in raw_skill_usage.items():
                if isinstance(value, int):
                    merged_usage[key] = merged_usage.get(key, 0) + value

        next_observation = self._observation_from_skill_result(skill_result)
        if next_observation is None:
            run_dir = Path(current_observation.screenshot_path or ".").parent.parent
            next_screenshot = run_dir / "screenshots" / f"step_{step_index:03d}.png"
            next_observation = await self.backend.observe(next_screenshot, timeout=self.step_timeout)

        succeeded = getattr(getattr(skill_result, "state", None), "value", "") == "succeeded"
        result_text = getattr(skill_result, "execution_summary", "") or ""
        if not succeeded and getattr(skill_result, "error", None):
            result_text = f"{result_text}\nError: {skill_result.error}".strip()
        action_summary = f"{summary}: {'succeeded' if succeeded else 'failed'}"
        rich_action_intent = (response.content or "").strip() or action_intent or action_summary
        model_snapshot = self._snapshot_model_response(
            response=response,
            action=action,
            assistant_message=assistant_message,
            action_text=action_summary,
            action_intent=rich_action_intent,
            state_summary=result_text,
        )
        return StepResult(
            action=action,
            tool_call_id=tool_call.id,
            tool_result=result_text,
            assistant_message=assistant_message,
            action_summary=action_summary,
            action_intent=rich_action_intent,
            state_summary=result_text,
            next_observation=next_observation,
            prompt_snapshot=prompt_snapshot,
            model_snapshot=model_snapshot,
            execution_snapshot={
                "tool_result": self._scrub_text_for_action(result_text, action),
                "skill": {
                    "skill_id": skill_id,
                    "skill_name": skill_name,
                    "state": getattr(getattr(skill_result, "state", None), "value", None),
                    "error": getattr(skill_result, "error", None),
                },
                "next_observation": self._serialize_observation(next_observation),
                "done": False,
            },
            step_usage=merged_usage,
            duration_s=time.monotonic() - step_start,
            chat_latency_s=step_chat_latency_s or None,
            ttft_s=step_ttft_s,
        )

    async def _prompt_skill_exception_result(
        self,
        exc: Exception,
        *,
        skill_id: str,
        skill_name: str,
        action: Action,
        tool_call: ToolCall,
        response: LLMResponse,
        assistant_message: dict[str, Any],
        prompt_snapshot: dict[str, Any] | None,
        current_observation: Observation,
        step_index: int,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
        action_intent: str | None,
    ) -> StepResult:
        run_dir = Path(current_observation.screenshot_path or ".").parent.parent
        next_screenshot = run_dir / "screenshots" / f"step_{step_index:03d}.png"
        next_observation = await self.backend.observe(next_screenshot, timeout=self.step_timeout)
        result_text = f"Prompt-selected skill failed with {type(exc).__name__}: {exc}"
        action_summary = f"use_skill {skill_name}: failed"
        rich_action_intent = (response.content or "").strip() or action_intent or action_summary
        model_snapshot = self._snapshot_model_response(
            response=response,
            action=action,
            assistant_message=assistant_message,
            action_text=action_summary,
            action_intent=rich_action_intent,
            state_summary=result_text,
        )
        return StepResult(
            action=action,
            tool_call_id=tool_call.id,
            tool_result=result_text,
            assistant_message=assistant_message,
            action_summary=action_summary,
            action_intent=rich_action_intent,
            state_summary=result_text,
            next_observation=next_observation,
            prompt_snapshot=prompt_snapshot,
            model_snapshot=model_snapshot,
            execution_snapshot={
                "tool_result": self._scrub_text_for_action(result_text, action),
                "skill": {
                    "skill_id": skill_id,
                    "skill_name": skill_name,
                    "state": "failed",
                    "error": str(exc),
                },
                "next_observation": self._serialize_observation(next_observation),
                "done": False,
            },
            step_usage=step_usage,
            duration_s=time.monotonic() - step_start,
            chat_latency_s=step_chat_latency_s or None,
            ttft_s=step_ttft_s,
        )

    async def _execute_prompt_composite_action(
        self,
        *,
        tool_call: ToolCall,
        response: LLMResponse,
        prompt_snapshot: dict[str, Any] | None,
        current_observation: Observation,
        step_index: int,
        step_usage: dict[str, int],
        step_start: float,
        step_chat_latency_s: float,
        step_ttft_s: float | None,
        action_intent: str | None,
        state_summary: str | None,
    ) -> StepResult:
        arguments = tool_call.arguments or {}
        alias = str(arguments.get("action_type") or arguments.get("action") or "").strip().lower()
        if alias not in self._prompt_composite_aliases:
            raise ActionError(f"Composite action {alias!r} is not listed in the prompt.")
        if alias not in COMPOSITE_ACTION_DEFINITIONS and alias != "click_and_type":
            raise ActionError(f"Composite action {alias!r} has no executor.")

        action = Action(action_type=alias, text=self._composite_action_text(arguments))
        assistant_message = self._build_assistant_message(
            response,
            content_override=self._normalize_action_text(
                response.content,
                action,
                tool_summary=action_intent or state_summary or describe_action(action),
            ),
            include_tool_calls=profile_uses_native_tools(self.agent_profile),
        )
        executed_actions = self._build_composite_actions(alias, arguments, current_observation)
        result_lines: list[str] = []
        for index, inner_action in enumerate(executed_actions):
            if index > 0:
                await asyncio.sleep(self._POST_ACTION_SETTLE_SECONDS)
            try:
                result_lines.append(await self.backend.execute(inner_action, timeout=self.step_timeout))
            except Exception as exc:
                result_lines.append(f"Action failed: {exc}")
                break

        await asyncio.sleep(self._POST_ACTION_SETTLE_SECONDS)
        run_dir = Path(current_observation.screenshot_path or ".").parent.parent
        next_screenshot = run_dir / "screenshots" / f"step_{step_index:03d}.png"
        next_observation, _observe_error = await self._observe_after_action(
            next_screenshot,
            previous_observation=current_observation,
            action=action,
            timeout=self.step_timeout,
        )
        result_text = "\n".join(result_lines)
        action_summary = f"{alias}: {self._composite_action_summary(alias, arguments)}"
        rich_action_intent = (response.content or "").strip() or action_intent or action_summary
        model_snapshot = self._snapshot_model_response(
            response=response,
            action=action,
            assistant_message=assistant_message,
            action_text=action_summary,
            action_intent=rich_action_intent,
            state_summary=state_summary,
        )
        return StepResult(
            action=action,
            tool_call_id=tool_call.id,
            tool_result=result_text,
            assistant_message=assistant_message,
            action_summary=action_summary,
            action_intent=rich_action_intent,
            state_summary=state_summary,
            next_observation=next_observation,
            interaction_target={"composite_action": alias},
            prompt_snapshot=prompt_snapshot,
            model_snapshot=model_snapshot,
            execution_snapshot={
                "tool_result": self._scrub_text_for_action(result_text, action),
                "inner_actions": [
                    self._serialize_action(inner_action) for inner_action in executed_actions
                ],
                "next_observation": self._serialize_observation(next_observation),
                "done": False,
            },
            step_usage=step_usage,
            duration_s=time.monotonic() - step_start,
            chat_latency_s=step_chat_latency_s or None,
            ttft_s=step_ttft_s,
        )

    def _build_composite_actions(
        self,
        alias: str,
        arguments: dict[str, Any],
        observation: Observation,
    ) -> list[Action]:
        if alias in {"click_then_type", "click_and_type"}:
            x, y = self._point_from_payload(arguments, observation)
            text = str(arguments.get("text") or "")
            if not text:
                raise ActionError(f"{alias} requires 'text'.")
            return [
                Action(action_type="tap", x=x, y=y),
                Action(
                    action_type="input_text",
                    text=text,
                    auto_enter=self._bool_argument(arguments.get("auto_enter"), False),
                ),
            ]
        if alias == "click_multi":
            points = arguments.get("coordinates") or arguments.get("points") or []
            if not isinstance(points, list) or not points:
                single = arguments.get("coordinate")
                points = [single] if single is not None else []
            actions = [
                Action(action_type="tap", x=x, y=y)
                for x, y in (self._point_to_screen(point, observation) for point in points)
            ]
            if not actions:
                raise ActionError("click_multi requires non-empty 'coordinates'.")
            return actions
        raise ActionError(f"Unsupported composite action {alias!r}.")

    def _point_from_payload(
        self,
        payload: dict[str, Any],
        observation: Observation,
    ) -> tuple[int, int]:
        if "coordinate" in payload:
            return self._point_to_screen(payload["coordinate"], observation)
        if "x" in payload and "y" in payload:
            return self._point_to_screen([payload["x"], payload["y"]], observation)
        raise ActionError("Composite action requires 'coordinate' or 'x'/'y'.")

    @staticmethod
    def _point_to_screen(point: Any, observation: Observation) -> tuple[int, int]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ActionError(f"Invalid coordinate {point!r}.")
        try:
            raw_x = float(point[0])
            raw_y = float(point[1])
        except (TypeError, ValueError) as exc:
            raise ActionError(f"Invalid coordinate {point!r}.") from exc
        width = int(observation.screen_width or 1000)
        height = int(observation.screen_height or 1000)
        if 0 <= raw_x <= 999 and 0 <= raw_y <= 999:
            return (
                int(round(raw_x / 999 * (width - 1))),
                int(round(raw_y / 999 * (height - 1))),
            )
        return int(raw_x), int(raw_y)

    @staticmethod
    def _composite_action_text(arguments: dict[str, Any]) -> str:
        text = arguments.get("text")
        if isinstance(text, str) and text:
            return text
        return str(arguments.get("coordinates") or arguments.get("coordinate") or "")

    @staticmethod
    def _composite_action_summary(alias: str, arguments: dict[str, Any]) -> str:
        if alias in {"click_then_type", "click_and_type"}:
            return "tap target and type text"
        if alias == "click_multi":
            points = arguments.get("coordinates") or arguments.get("points") or []
            return f"tap {len(points) if isinstance(points, list) else 1} target(s)"
        return alias

    @staticmethod
    def _bool_argument(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return bool(value)

    @staticmethod
    def _stringify_skill_argument(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return "" if value is None else str(value)
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _observation_from_skill_result(skill_result: Any) -> Observation | None:
        for step_result in reversed(getattr(skill_result, "step_results", ()) or ()):
            raw = getattr(step_result, "observation", None)
            if not isinstance(raw, dict):
                continue
            screenshot_path = raw.get("screenshot_path")
            if not screenshot_path:
                continue
            return Observation(
                screenshot_path=str(screenshot_path),
                screen_width=int(raw.get("screen_width") or 1000),
                screen_height=int(raw.get("screen_height") or 1000),
                foreground_app=raw.get("foreground_app"),
                platform=raw.get("platform") or "unknown",
                extra=raw.get("extra") if isinstance(raw.get("extra"), dict) else {},
            )
        return None

    def _coordinate_mode(self) -> str:
        return coordinate_mode_for_profile(self.agent_profile, self.model)

    def _model_uses_relative_grid(self) -> bool:
        return self._coordinate_mode() == "relative_999"

    def _normalize_relative_coordinates(self, action: Action) -> Action:
        if action.relative or action.action_type not in self._COORDINATE_ACTIONS:
            return action
        if not self._model_uses_relative_grid():
            return action
        coords = [value for value in (action.x, action.y, action.x2, action.y2) if value is not None]
        if coords and all(0 <= value <= 999 for value in coords):
            return replace(action, relative=True)
        return action

    def _post_action_settle_seconds(self, action: Action) -> float:
        if action.action_type in self._NO_SETTLE_ACTIONS:
            return 0.0
        if action.action_type == "open_app":
            return self._OPEN_APP_SETTLE_SECONDS
        return self._POST_ACTION_SETTLE_SECONDS

    async def _observe_after_action(
        self,
        screenshot_path: Path,
        *,
        previous_observation: Observation | None = None,
        action: Action | None = None,
        timeout: float,
    ) -> tuple[Observation | None, str | None]:
        # Wait for the UI to reach a short stable state before returning the
        # observation used by the next planner turn.
        max_attempts = self._POST_ACTION_STABILITY_MAX_ATTEMPTS
        if action is not None and action.action_type in self._NO_SETTLE_ACTIONS:
            max_attempts = 1
        if action is None or self._post_action_settle_seconds(action) <= 0:
            max_attempts = min(max_attempts, 1)

        window_seconds = min(timeout, self._POST_ACTION_STABILITY_WINDOW_SECONDS)
        poll_interval = self._POST_ACTION_STABILITY_POLL_SECONDS
        observe_timeout = min(timeout, self._POST_ACTION_OBSERVE_TIMEOUT_SECONDS)
        attempts_within_window = max(
            1,
            int(window_seconds / max(poll_interval, 1e-6)),
        )
        max_attempts = max(1, min(max_attempts, attempts_within_window))

        previous_fingerprint = (
            self._build_screen_fingerprint(previous_observation)
            if previous_observation is not None
            else None
        )
        stable_count = 0
        stable_required = self._POST_ACTION_STABILITY_FRAMES_REQUIRED
        last_error: str | None = None
        last_observation: Observation | None = None
        deadline = time.monotonic() + window_seconds

        for attempt in range(max_attempts):
            try:
                observation = await self.backend.observe(
                    screenshot_path,
                    timeout=observe_timeout,
                )
                last_observation = observation
                current_fingerprint = self._build_screen_fingerprint(observation)

                if previous_fingerprint is not None and current_fingerprint is not None:
                    if self._is_same_screen(previous_fingerprint, current_fingerprint):
                        stable_count += 1
                    else:
                        stable_count = 1
                        previous_fingerprint = current_fingerprint
                else:
                    # Fingerprints are best-effort. If unavailable, just use the
                    # latest sampled observation as the fallback signal.
                    stable_count = min(stable_count + 1, stable_required)

                if stable_count >= stable_required:
                    return observation, None
            except Exception as exc:
                last_error = str(exc)

            if max_attempts > 1 and time.monotonic() < deadline:
                await asyncio.sleep(poll_interval)

        if last_observation is not None:
            return last_observation, last_error
        return None, last_error

    def _build_screen_fingerprint(self, observation: Observation) -> _ScreenFingerprint | None:
        screenshot = observation.screenshot_path
        if not screenshot:
            return None
        screenshot_path = Path(screenshot)
        if not screenshot_path.exists():
            return None

        app_name = self._normalize_stagnation_app(observation.foreground_app)
        try:
            data = screenshot_path.read_bytes()
        except OSError:
            return None

        try:
            from PIL import Image

            with Image.open(screenshot_path) as img:
                resampling = getattr(Image, "Resampling", Image)
                ssim_size = self._STAGNATION_SSIM_SIZE
                grayscale = img.convert("L").resize(
                    (ssim_size, ssim_size),
                    resampling.BILINEAR,
                )
                pixels = list(grayscale.tobytes())
                if pixels:
                    return _ScreenFingerprint(
                        app=app_name,
                        method="ssim",
                        digest=base64.b64encode(bytes(pixels)).decode("ascii"),
                    )
        except Exception:
            pass

        return _ScreenFingerprint(
            app=app_name,
            method="sha256",
            digest=hashlib.sha256(data).hexdigest(),
        )

    @classmethod
    def _is_same_screen(
        cls,
        previous: _ScreenFingerprint,
        current: _ScreenFingerprint,
    ) -> bool:
        if previous.app and current.app and previous.app != current.app:
            return False

        if previous.method == "ssim" and current.method == "ssim":
            try:
                return cls._ssim_is_similar(previous.digest, current.digest)
            except Exception:
                return previous.digest == current.digest

        if previous.method != current.method:
            return False
        return previous.digest == current.digest

    @classmethod
    def _ssim_is_similar(cls, previous_digest: str, current_digest: str) -> bool:
        previous_pixels = base64.b64decode(previous_digest)
        current_pixels = base64.b64decode(current_digest)
        if len(previous_pixels) != len(current_pixels) or len(previous_pixels) == 0:
            return False
        return cls._ssim_score(previous_pixels, current_pixels) >= cls._STAGNATION_SSIM_THRESHOLD

    @classmethod
    def _ssim_score(cls, previous_pixels: bytes, current_pixels: bytes) -> float:
        del cls

        if len(previous_pixels) != len(current_pixels) or len(previous_pixels) == 0:
            return 0.0

        n = len(previous_pixels)
        previous_values = [value for value in previous_pixels]
        current_values = [value for value in current_pixels]

        previous_mean = sum(previous_values) / n
        current_mean = sum(current_values) / n

        previous_variance = sum((value - previous_mean) ** 2 for value in previous_values) / n
        current_variance = sum((value - current_mean) ** 2 for value in current_values) / n
        covariance = sum(
            (previous_value - previous_mean) * (current_value - current_mean)
            for previous_value, current_value in zip(previous_values, current_values)
        ) / n

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        denominator = (previous_mean * previous_mean + current_mean * current_mean + c1) * (
            previous_variance + current_variance + c2
        )

        if denominator == 0:
            return 1.0 if previous_mean == current_mean else 0.0

        numerator = (2 * previous_mean * current_mean + c1) * (2 * covariance + c2)
        return numerator / denominator

    def _normalize_stagnation_app(self, app: str | None) -> str | None:
        if not app:
            return None
        normalized = self._normalize_backend_app_identifier(app)
        return None if normalized == "unknown" else normalized

    def _normalize_backend_app_identifier(self, app: str) -> str:
        if self.backend.platform == "android" and hasattr(self.backend, "_run"):
            return normalize_adb_app_identifier(app)
        return normalize_app_identifier(self.backend.platform, app)

    # ------------------------------------------------------------------
    # S2 recovery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _memory_context_with_s2_guidance(
        memory_context: str | None,
        s2_guidance_notes: list[str],
    ) -> str | None:
        if not s2_guidance_notes:
            return memory_context
        lines: list[str] = []
        if memory_context and memory_context.strip():
            lines.append(memory_context.strip())
            lines.append("")
        lines.append("Slow-model recovery hint:")
        for note in s2_guidance_notes[-3:]:
            for line in str(note).splitlines():
                clean = line.strip()
                if clean:
                    lines.append(f"- {clean}")
        lines.append("")
        lines.append("This is advisory. Continue using the current screen evidence.")
        return "\n".join(lines)

    def _recent_history_for_takeover(
        self,
        history: list[Any],
        *,
        max_items: int = 5,
    ) -> list[dict[str, str]]:
        del self
        recent: list[dict[str, str]] = []
        for item in history[-max_items:]:
            if isinstance(item, dict):
                action_summary = str(item.get("action_summary") or "").strip()
                state_summary = str(item.get("state_summary") or "").strip()
                actor = str(item.get("actor") or "s1").strip()
            else:
                action_summary = str(getattr(item, "action_summary", "") or "").strip()
                state_summary = str(getattr(item, "state_summary", "") or "").strip()
                actor = str(getattr(item, "actor", "") or "s1").strip()
            if not action_summary and not state_summary:
                continue
            recent.append(
                {
                    "actor": actor or "s1",
                    "action_summary": action_summary[:500],
                    "state_summary": state_summary[:500],
                }
            )
        return recent

    def _build_s2_takeover_context(
        self,
        *,
        task: str,
        trigger_reason: str,
        stagnation_streak: int,
        stagnation_limit: int,
        recent_history: list[dict[str, str]],
        s2_guidance_notes: list[str],
    ) -> str:
        lines: list[str] = []
        if self._s2_uses_no_think():
            lines.extend([S2_NO_THINKING_POLICY.strip(), ""])
        lines.extend([
            "S2 takeover context:",
            "You are now taking over from the small GUI executor because the previous GUI steps reached a stagnation condition.",
            "Use the current screen and the recent step history to recover.",
            "Continue from the current screen. Do not restart the whole task unless it is clearly necessary.",
            "Do not repeat the last failed action pattern.",
            "Do not invent screen content.",
            "Do not directly answer the user.",
            "Return only the required GUI action output.",
            "",
            f"Original task: {task}",
            "Trigger: stagnation",
            f"Trigger reason: {trigger_reason}",
            f"Stagnation streak: {stagnation_streak}",
            f"Stagnation limit: {stagnation_limit}",
        ])

        if recent_history:
            lines.append("")
            lines.append("Recent S1 step history:")
            for idx, item in enumerate(recent_history, start=1):
                action_summary = item.get("action_summary") or ""
                state_summary = item.get("state_summary") or ""
                actor = item.get("actor") or "s1"
                lines.append(
                    f"{idx}. actor={actor}; action={action_summary}; state={state_summary}"
                )

        if s2_guidance_notes:
            lines.append("")
            lines.append("Previous S2 recovery hints:")
            for idx, note in enumerate(s2_guidance_notes[-3:], start=1):
                lines.append(f"{idx}. {str(note).strip()[:800]}")

        lines.append("")
        lines.append("Safety constraints:")
        lines.append("- Do not send, pay, submit, delete, authorize, or enter secrets.")
        lines.append("- If a sensitive action is required, stop before execution and request user confirmation.")
        lines.append("- For information queries, do not mark done unless the requested answer is visible or explicitly extracted.")

        return "\n".join(lines).strip()

    def _s2_uses_no_think(self) -> bool:
        effort = self._s2_reasoning_effort
        return effort is None or str(effort).strip().casefold() == "none"

    def _s2_prefixed_policy(self, policy: str) -> str:
        policy = policy.strip()
        if self._s2_uses_no_think():
            return f"{S2_NO_THINKING_POLICY.strip()}\n\n{policy}".strip()
        return policy

    def _with_s2_takeover_context(
        self,
        messages: list[dict[str, Any]],
        takeover_context: str,
    ) -> list[dict[str, Any]]:
        del self
        if not takeover_context:
            return messages

        copied = [dict(item) for item in messages]
        for item in copied:
            if item.get("role") == "system":
                item["content"] = f"{item.get('content', '')}\n\n{takeover_context}".strip()
                return copied

        return [{"role": "system", "content": takeover_context}, *copied]

    async def _request_s2_hint(
        self,
        *,
        task: str,
        step_index: int,
        current_observation: Observation,
        history: list[HistoryTurn],
        last_action_summary: str | None,
        reason: str,
    ) -> str | None:
        if not self._s2_enabled or self._s2_llm is None:
            return None
        recent_actions = [
            f"Step {turn.step_index}: {turn.action_intent or turn.action_summary}"
            for turn in history[-6:]
        ]
        observation_payload = {
            "foreground_app": current_observation.foreground_app,
            "platform": current_observation.platform,
            "screen_width": current_observation.screen_width,
            "screen_height": current_observation.screen_height,
            "extra": self._scrub_for_log(current_observation.extra),
        }
        user_content = (
            f"Original task:\n{task}\n\n"
            f"Step index: {step_index}\n"
            f"Current foreground app: {current_observation.foreground_app or 'unknown'}\n"
            f"Current visible/screen summary:\n"
            f"{self._truncate_text(self._json_for_prompt(observation_payload), self._s2_prompt_max_chars)}\n\n"
            f"Recent action summaries:\n{chr(10).join(recent_actions) if recent_actions else 'None'}\n\n"
            f"Last action summary: {last_action_summary or 'None'}\n"
            f"Stagnation reason: {reason}"
        )
        user_content = self._truncate_text(user_content, self._s2_prompt_max_chars)
        system_content = self._s2_prefixed_policy(
            "You are the slow-reasoning recovery model for a GUI agent.\n\n"
            "You must not directly answer the user.\n"
            "You must not invent screen content.\n"
            "You must provide a concise recovery hint for the small GUI executor.\n\n"
            "Return ONLY one minified JSON object.\n"
            "Keep the JSON under 120 tokens.\n"
            'Schema: {"diagnosis":"...","next_subgoal":"...",'
            '"avoid":["..."],"stop_condition":"..."}'
        )
        chat_kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "tools": None,
            "max_tokens": 180,
        }
        if self._s2_reasoning_effort is not None:
            chat_kwargs["reasoning_effort"] = self._s2_reasoning_effort
        try:
            response = await self._s2_llm.chat(**chat_kwargs)
        except Exception:
            logger.warning("GUI S2 hint request failed.", exc_info=True)
            return None

        text = (response.content or "").strip()
        if not text:
            return None
        payload = self._extract_json_object(text)
        if isinstance(payload, dict):
            lines: list[str] = []
            for key in ("diagnosis", "next_subgoal", "stop_condition"):
                value = payload.get(key)
                if value:
                    lines.append(f"{key}: {str(value).strip()}")
            avoid = payload.get("avoid")
            if isinstance(avoid, list):
                avoid_items = [str(item).strip() for item in avoid if str(item).strip()]
                if avoid_items:
                    lines.append(f"avoid: {', '.join(avoid_items)}")
            elif avoid:
                lines.append(f"avoid: {str(avoid).strip()}")
            if lines:
                return self._truncate_text("\n".join(lines), self._s2_prompt_max_chars)
        return self._truncate_text(text, self._s2_prompt_max_chars)

    @staticmethod
    def _json_for_prompt(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                payload = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        *,
        task: str,
        current_observation: Observation,
        history: list[HistoryTurn],
        expected_bundle_id: str | None = None,
        memory_context: str | None = None,
        skill_context: str | None = None,
        prompt_skill_parts: CompactPromptParts | None = None,
    ) -> list[dict[str, Any]]:
        task_context: list[str] = [task]
        if expected_bundle_id:
            task_context.extend([
                "",
                f"Expected native app bundle id: {expected_bundle_id}",
                (
                    "For named app tasks, use the native installed app when available. "
                    "Do not use browser or web search results as a substitute unless "
                    "the user explicitly asks for a web/browser task or the native app is unavailable."
                ),
            ])
        if memory_context:
            task_context.extend(["", "Relevant Knowledge:", memory_context])
        if skill_context:
            task_context.extend([
                "",
                "Previous skill execution (already completed):",
                skill_context,
                "",
                "Check the current screen. If the task is now complete, report completion; otherwise continue with any remaining steps.",
            ])
        return build_mobileworld_messages(
            self.agent_profile,
            task="\n".join(task_context),
            current_observation=current_observation,
            history=history,
            model_name=self.model,
            history_image_window=self.history_image_window,
            compact_prompt_parts=prompt_skill_parts,
        )

    def _messages_with_s2_takeover_policy(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        policy = self._s2_prefixed_policy(S2_TAKEOVER_ACTION_POLICY)
        if not messages:
            return [{"role": "system", "content": policy}]
        updated = [dict(message) for message in messages]
        first = dict(updated[0])
        content = first.get("content")
        if isinstance(content, str):
            first["content"] = f"{content}\n\n{policy}"
            updated[0] = first
            return updated
        updated.insert(0, {"role": "system", "content": policy})
        return updated

    def _build_instruction_prompt(
        self,
        *,
        task: str,
        current_observation: Observation,
        history: list[HistoryTurn],
        app_hint: str | None,
        skill_context: str | None = None,
    ) -> str:
        """Build the text prompt that frames the current step."""
        recent_intents = self._format_recent_intents(
            history,
            window=self.history_text_window,
        )
        latest_summary = self._latest_state_summary(history)
        lines = [
            "Please generate the next move according to the UI screenshot, instruction and recent progress context.",
            "",
        ]

        if self.include_date_context:
            lines.append(f"Today's date is: {datetime.now().strftime('%Y-%m-%d %A')}.")

        lines.append(f"Instruction: {task}")
        lines.append(f"Platform: {self.backend.platform}")

        app_name = app_hint or current_observation.foreground_app
        if app_name:
            lines.append(f"Foreground app hint: {app_name}")

        retry_summaries = self._format_retry_attempt_summaries(self._active_retry_summaries)
        if retry_summaries:
            lines.extend([
                "",
                "Previous attempt summaries:",
                retry_summaries,
                "",
                "Continue from the current screen state. Reuse the progress above and avoid blindly repeating the same failed action sequence.",
            ])

        lines.extend([
            "",
            "Recent intents:",
            recent_intents,
        ])
        if latest_summary:
            lines.extend([
                "",
                f"Latest state summary: {latest_summary}",
            ])

        if skill_context:
            lines.extend([
                "",
                "Previous skill execution (already completed):",
                skill_context,
                "",
                "Check the current screen. If the task is now complete, call "
                "done(status=\"success\"). Otherwise, continue with any remaining steps.",
            ])
        return "\n".join(lines)

    @staticmethod
    def _format_recent_intents(history: list[HistoryTurn], *, window: int = 8) -> str:
        if not history:
            return "None"
        recent_history = history[-max(1, window):]
        return "\n".join(
            f"Step {turn.step_index}: {turn.action_intent or turn.action_summary}"
            for turn in recent_history
        )

    @staticmethod
    def _latest_state_summary(history: list[HistoryTurn]) -> str | None:
        for turn in reversed(history):
            if turn.state_summary and turn.state_summary.strip():
                return turn.state_summary.strip()
        return None

    @staticmethod
    def _format_retry_attempt_summaries(attempt_summaries: tuple[str, ...]) -> str:
        if not attempt_summaries:
            return ""
        return "\n\n".join(
            f"Attempt {index}:\n{summary}"
            for index, summary in enumerate(attempt_summaries, start=1)
        )

    @staticmethod
    def _build_state_note(
        *,
        status: str,
        history: list[HistoryTurn],
        current_observation: Observation | None,
        current_action_summary: str | None = None,
        error: str | None = None,
    ) -> str:
        return build_state_note(
            status=status,
            done=GuiAgent._summarize_progress(history, current_action_summary),
            remaining=GuiAgent._remaining_hint(status=status, error=error),
            current=GuiAgent._describe_observation_state(current_observation),
            resume=GuiAgent._resume_hint(status=status, error=error),
        )

    @staticmethod
    def _summarize_progress(history: list[HistoryTurn], current_action_summary: str | None = None) -> str:
        summaries = [
            (turn.state_summary or turn.action_summary).strip()
            for turn in history
            if (turn.state_summary or turn.action_summary).strip()
        ]
        if current_action_summary and current_action_summary.strip():
            summaries.append(current_action_summary.strip())
        if not summaries:
            return "No GUI actions were completed."
        return "; ".join(summary.rstrip(".") for summary in summaries[-3:])

    @staticmethod
    def _describe_observation_state(observation: Observation | None) -> str:
        if observation is None:
            return "Current screen state unavailable."
        parts: list[str] = []
        if observation.foreground_app and observation.foreground_app.strip():
            parts.append(observation.foreground_app.strip())
        if isinstance(observation.screen_width, int) and isinstance(observation.screen_height, int):
            parts.append(f"{observation.screen_width}x{observation.screen_height}")
        if parts:
            return " ".join(parts)
        if observation.platform:
            return observation.platform
        return "Current screen state unavailable."

    @staticmethod
    def _remaining_hint(*, status: str, error: str | None) -> str:
        error_text = (error or "").lower()
        if status == "completed":
            return "none"
        if "stagnation_detected" in error_text:
            return "Change the action sequence from the current screen."
        if "intervention_cancelled" in error_text or "interruption" in error_text:
            return "Wait for the intervention blocker to be resolved."
        if "step_timeout" in error_text:
            return "Retry the timed-out step from the current screen."
        if "max_steps_exceeded" in error_text:
            return "Continue the remaining task from the current screen."
        if status == "blocked":
            return "Resolve the blocker before retrying."
        return "Continue from the current screen."

    @staticmethod
    def _resume_hint(*, status: str, error: str | None) -> str:
        error_text = (error or "").lower()
        if status == "completed":
            return "No further action needed."
        if "stagnation_detected" in error_text:
            return "Resume by trying a different action on the same screen."
        if "intervention_cancelled" in error_text or "interruption" in error_text:
            return "Resolve the intervention blocker, then continue from the current screen."
        if "step_timeout" in error_text:
            return "Resume from the current screen after the timeout clears."
        if "max_steps_exceeded" in error_text:
            return "Resume from the current screen and finish the remaining steps."
        if status == "blocked":
            return "Resolve the blocker, then continue from the current screen."
        return "Resume from the current screen."

    async def _generate_termination_summary(
        self,
        *,
        task: str,
        termination_reason: str,
        history: list[HistoryTurn],
        run_dir: Path,
    ) -> str | None:
        """Ask the LLM for a brief state note when the task terminates abnormally."""
        steps_text = "\n".join(
            f"  {i}. {turn.action_summary}" for i, turn in enumerate(history, 1)
        ) or "  (no steps completed)"
        fallback_status = "blocked" if any(
            keyword in termination_reason.lower()
            for keyword in ("interrupted", "cancel", "loop")
        ) else "partial"
        observation: Observation | None = None
        try:
            # Take a fresh screenshot for the summary
            screenshot_path = run_dir / "screenshots" / "termination_summary.png"
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            observation = await self.backend.observe(
                screenshot_path, timeout=self.step_timeout,
            )
            prompt_text = (
                "Return a compact GUI state note for the terminated task.\n\n"
                f"Task: {task}\n"
                f"Termination reason: {termination_reason}\n"
                f"Steps executed:\n{steps_text}\n\n"
                "Use exactly these 5 labels and keep each value short:\n"
                "Status: completed|partial|blocked\n"
                "Done: ...\n"
                "Remaining: ...\n"
                "Current: ...\n"
                "Resume: ...\n\n"
                "Rules:\n"
                "- Do not add bullets, markdown, or extra lines.\n"
                "- Use a clear resume hint if continuation is still possible.\n"
                "- Use 'none' for Remaining when the task is completed."
            )
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
            if observation.screenshot_path and Path(observation.screenshot_path).exists():
                content.append(self._image_block(Path(observation.screenshot_path)))

            response = await self.llm.chat(
                messages=[{"role": "user", "content": content}],
                tools=None,
            )
            text = response.content.strip()
            if text and is_state_note(text):
                return text
        except Exception as exc:
            logger.warning("Failed to generate termination summary: %s", exc)

        return self._build_state_note(
            status=fallback_status,
            history=history,
            current_observation=observation,
            error=termination_reason,
        )

    @staticmethod
    def _build_attempt_summary(
        *,
        failure_reason: str,
        result_summary: str | None,
        action_summaries: tuple[str, ...],
        model_summary: str | None = None,
    ) -> str:
        lines = [f"Failure reason: {failure_reason}"]
        if result_summary and result_summary != failure_reason:
            lines.append(f"Attempt result: {result_summary}")
        if model_summary:
            lines.append(f"Latest model summary: {model_summary}")

        if action_summaries:
            lines.append("Completed GUI actions before the failure:")
            trimmed_actions = action_summaries[-6:]
            omitted_count = len(action_summaries) - len(trimmed_actions)
            if omitted_count > 0:
                lines.append(f"- ... {omitted_count} earlier step(s) omitted")
            start_index = len(action_summaries) - len(trimmed_actions) + 1
            for step_offset, action_summary in enumerate(trimmed_actions, start=start_index):
                lines.append(f"- Step {step_offset}: {action_summary}")
        else:
            lines.append("No completed GUI actions were recorded before the failure.")

        lines.append(
            "Retry guidance: Continue from the current screen state. Reuse the progress above and avoid blindly repeating the same failed action sequence."
        )
        return "\n".join(lines)

    def _history_user_message(
        self,
        observation: Observation,
        prompt_text: str | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        if prompt_text:
            content.append({"type": "text", "text": prompt_text})
        if observation.screenshot_path and Path(observation.screenshot_path).exists():
            content.append(self._image_block(Path(observation.screenshot_path)))
        return {"role": "user", "content": content}

    @staticmethod
    def _history_image_prompt(turn: HistoryTurn) -> str:
        return (
            f"Historical screen before Step {turn.step_index}. "
            "The next assistant message summarizes the action taken from this screen."
        )

    @staticmethod
    def _history_assistant_message(turn: HistoryTurn) -> dict[str, Any]:
        lines = [f"Step {turn.step_index}: {turn.action_intent or turn.action_summary}"]
        if turn.state_summary and turn.state_summary.strip():
            lines.append(f"State summary: {turn.state_summary.strip()}")
        tool_result = turn.tool_result_message.get("content")
        if isinstance(tool_result, str) and tool_result.strip():
            lines.append(f"Tool result: {tool_result.strip()}")
        return {"role": "assistant", "content": "\n".join(lines)}

    def _current_user_message(
        self,
        observation: Observation,
        *,
        task: str,
        step_index: int,
        app_hint: str | None,
        prompt_text: str | None = None,
    ) -> dict[str, Any]:
        if self._coordinate_mode() == "relative_999":
            coord_inst = (
                "Use relative coordinates in [0, 999] for both x and y, "
                "and set relative=true."
            )
        else:
            coord_inst = "Prefer absolute pixel coordinates."

        content: list[dict[str, Any]] = []
        if prompt_text:
            content.append({"type": "text", "text": prompt_text})
        content.append({
            "type": "text",
            "text": observation.to_user_text(
                task,
                step_index=step_index,
                app_hint=app_hint,
                coordinate_instruction=coord_inst,
                include_extra=False,
            ),
        })
        if observation.screenshot_path and Path(observation.screenshot_path).exists():
            content.append(self._image_block(Path(observation.screenshot_path)))
        return {"role": "user", "content": content}

    @staticmethod
    def _build_assistant_message(
        response: LLMResponse,
        *,
        content_override: str | None = None,
        include_tool_calls: bool = True,
    ) -> dict[str, Any]:
        """Build an assistant message dict from an LLM response."""
        msg: dict[str, Any] = {"role": "assistant"}

        content = content_override if content_override is not None else response.content
        if content:
            msg["content"] = content

        if include_tool_calls and response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                        if isinstance(tc.arguments, dict) else str(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]

        return msg

    @staticmethod
    def _extract_action_line_from_response(content: str) -> str | None:
        parts = content.split("Action:", 1)
        if len(parts) != 2:
            return None

        action_line = parts[1].strip().splitlines()
        if not action_line:
            return None

        first_line = action_line[0].strip()
        if not first_line:
            return None

        first_line = first_line.strip('"')
        return first_line.strip()

    @staticmethod
    def _build_tool_format_error(
        *,
        native_tools: bool,
        detail: str,
    ) -> str:
        if native_tools:
            return f"Error: {detail}"
        return "Format error: " + detail

    def _snapshot_step_prompt(
        self,
        *,
        task: str,
        step_index: int,
        messages: list[dict[str, Any]],
        current_observation: Observation,
        history: list[HistoryTurn],
    ) -> dict[str, Any]:
        return {
            "task": task,
            "step_index": step_index,
            "messages": self._scrub_for_artifact(messages),
            "history": [
                {
                    "step_index": turn.step_index,
                    "action_summary": turn.action_summary,
                    "action_intent": turn.action_intent,
                    "state_summary": turn.state_summary,
                    "raw_response_content": turn.raw_response_content,
                    "observation": self._serialize_observation(turn.observation),
                    "tool_result": turn.tool_result_message.get("content"),
                }
                for turn in history
            ],
            "current_observation": self._serialize_observation(current_observation),
        }

    def _snapshot_model_response(
        self,
        *,
        response: LLMResponse,
        action: Action,
        assistant_message: dict[str, Any],
        action_text: str,
        action_intent: str | None = None,
        state_summary: str | None = None,
    ) -> dict[str, Any]:
        return {
            "raw_content": self._scrub_text_for_artifact_action(response.content, action),
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": self._scrub_for_artifact(tool_call.arguments),
                }
                for tool_call in (response.tool_calls or [])
            ],
            "assistant_message": self._scrub_assistant_message_for_artifact(assistant_message, action),
            "parsed_action": self._scrub_for_artifact(self._serialize_action(action)),
            "action_text": self._scrub_text_for_artifact_action(action_text, action),
            "action_summary": self._scrub_text_for_artifact_action(self._action_summary(action_text), action),
            "action_intent": self._scrub_text_for_artifact_action(action_intent, action),
            "state_summary": self._scrub_text_for_artifact_action(state_summary, action),
        }

    def _snapshot_failed_model_response(
        self,
        response: LLMResponse,
        *,
        assistant_message: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "raw_content": self._scrub_text_for_artifact_action(response.content, None),
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": self._scrub_for_artifact(tool_call.arguments),
                }
                for tool_call in (response.tool_calls or [])
            ],
        }
        if assistant_message is not None:
            snapshot["assistant_message"] = self._scrub_for_artifact(assistant_message)
        return snapshot

    @staticmethod
    def _normalize_action_text(
        content: str,
        action: Action,
        *,
        tool_summary: str | None = None,
    ) -> str:
        summary = GuiAgent._clean_action_summary(tool_summary)
        if summary:
            return f"Action: {summary}"
        text = content.strip() if content else ""
        action_line = GuiAgent._extract_action_line_from_response(text)
        if action_line:
            return f"Action: {action_line}"
        if text:
            first_line = text.splitlines()[0].strip()
            if first_line.lower().startswith("action:"):
                return first_line
            return f"Action: {first_line}"
        return f"Action: {describe_action(action)}"

    @staticmethod
    def _tool_call_summary(tool_call: ToolCall) -> str | None:
        intent, summary = GuiAgent._tool_call_semantics(tool_call)
        return intent or summary

    @staticmethod
    def _tool_call_semantics(tool_call: ToolCall) -> tuple[str | None, str | None]:
        arguments = tool_call.arguments or {}
        intent = GuiAgent._clean_action_summary(arguments.get("intent"))
        summary = GuiAgent._clean_action_summary(arguments.get("summary"))
        return intent, summary

    @staticmethod
    def _clean_action_summary(value: Any) -> str | None:
        if value is None:
            return None
        text = " ".join(str(value).split()).strip()
        if not text:
            return None
        lowered = text.casefold()
        if lowered.startswith("action:"):
            text = text.split(":", 1)[1].strip()
        text = text.strip("`\"'")
        return text or None

    @staticmethod
    def _action_summary(action_text: str) -> str:
        if action_text.lower().startswith("action:"):
            return action_text.split(":", 1)[1].strip()
        return action_text.strip()

    @staticmethod
    def _resolve_done_status(action: Action) -> str:
        """Resolve terminal status for done actions with safe fallback rules."""
        if action.status in {"success", "failure"}:
            return action.status
        text = (action.text or "").strip().lower()
        if any(hint in text for hint in _DONE_FAILURE_HINTS):
            return "failure"
        # Missing status is common for some providers; default to success so
        # we do not retry already-completed tasks.
        return "success"

    def _image_block(self, path: Path) -> dict[str, Any]:
        """Create a base64 image content block for an LLM message."""
        from opengui.skills.executor import _scale_image
        b64 = base64.b64encode(
            _scale_image(path.read_bytes(), scale_ratio=self._image_scale_ratio)
        ).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        }

    @staticmethod
    def _serialize_action(action: Action) -> dict[str, Any]:
        payload = dataclasses.asdict(action)
        return {
            key: value
            for key, value in payload.items()
            if value is not None and not (key == "relative" and value is False)
        }

    @staticmethod
    def _serialize_observation(observation: Observation | None) -> dict[str, Any] | None:
        if observation is None:
            return None
        return {
            "screenshot_path": (
                str(observation.screenshot_path)
                if observation.screenshot_path is not None
                else None
            ),
            "screen_width": observation.screen_width,
            "screen_height": observation.screen_height,
            "foreground_app": observation.foreground_app,
            "platform": observation.platform,
            "extra": GuiAgent._scrub_for_log(observation.extra),
        }

    @staticmethod
    def _scrub_for_log(value: Any) -> Any:
        return GuiAgent._scrub_value(value, redact_input_text=True)

    @staticmethod
    def _scrub_for_artifact(value: Any) -> Any:
        return GuiAgent._scrub_value(value, redact_input_text=False)

    @staticmethod
    def _scrub_value(value: Any, *, redact_input_text: bool) -> Any:
        if isinstance(value, dict):
            scrubbed: dict[str, Any] = {}
            action_type = value.get("action_type") if isinstance(value.get("action_type"), str) else None
            intervention_text = (
                value.get("text")
                if action_type == "request_intervention" and isinstance(value.get("text"), str)
                else None
            )
            for key, item in value.items():
                if key == "url" and isinstance(item, str) and item.startswith("data:image/"):
                    scrubbed[key] = "<omitted:image-data-url>"
                elif redact_input_text and action_type == "input_text" and key == "text":
                    scrubbed[key] = "<redacted:input_text>"
                elif (action_type == "request_intervention" and key == "text") or key == "reason":
                    scrubbed[key] = "<redacted:intervention_reason>"
                elif any(token in key.lower() for token in ("password", "secret", "token", "otp", "credential")):
                    scrubbed[key] = "<redacted:sensitive_field>"
                elif intervention_text and isinstance(item, str):
                    scrubbed[key] = GuiAgent._scrub_sensitive_text(item).replace(
                        intervention_text,
                        "<redacted:intervention_reason>",
                    )
                else:
                    scrubbed[key] = GuiAgent._scrub_value(item, redact_input_text=redact_input_text)
            return scrubbed
        if isinstance(value, list):
            return [GuiAgent._scrub_value(item, redact_input_text=redact_input_text) for item in value]
        if isinstance(value, str):
            return GuiAgent._scrub_sensitive_text(value)
        return value

    @staticmethod
    def _scrub_text_for_action(text: str | None, action: Action | None) -> str | None:
        return GuiAgent._scrub_text(text, action, redact_input_text=True)

    @staticmethod
    def _scrub_text_for_artifact_action(text: str | None, action: Action | None) -> str | None:
        return GuiAgent._scrub_text(text, action, redact_input_text=False)

    @staticmethod
    def _scrub_text(text: str | None, action: Action | None, *, redact_input_text: bool) -> str | None:
        if text is None:
            return None
        scrubbed = GuiAgent._scrub_sensitive_text(text)
        if action is None:
            return scrubbed
        if redact_input_text and action.action_type == "input_text" and action.text:
            scrubbed = scrubbed.replace(action.text, "<redacted:input_text>")
        if action.action_type == "request_intervention" and action.text:
            scrubbed = scrubbed.replace(action.text, "<redacted:intervention_reason>")
        return scrubbed

    @staticmethod
    def _scrub_sensitive_text(text: str) -> str:
        return re.sub(
            r"(?i)(\b[\w-]*(?:password|secret|token|otp|credential)[\w-]*\b\s*[:=]\s*)([^\s,}\]]+)",
            r"\1<redacted:sensitive_field>",
            text,
        )

    @classmethod
    def _scrub_assistant_message_for_log(
        cls,
        assistant_message: dict[str, Any],
        action: Action,
    ) -> dict[str, Any]:
        scrubbed = cls._scrub_for_log(assistant_message)
        content = scrubbed.get("content")
        if isinstance(content, str):
            scrubbed["content"] = cls._scrub_text_for_action(content, action)
        for tool_call in scrubbed.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue
            function_payload = tool_call.get("function")
            if not isinstance(function_payload, dict):
                continue
            arguments = function_payload.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                function_payload["arguments"] = json.dumps(
                    cls._scrub_for_log(json.loads(arguments)),
                    ensure_ascii=False,
                )
            except json.JSONDecodeError:
                function_payload["arguments"] = cls._scrub_text_for_action(arguments, action)
        return scrubbed

    @classmethod
    def _scrub_assistant_message_for_artifact(
        cls,
        assistant_message: dict[str, Any],
        action: Action,
    ) -> dict[str, Any]:
        scrubbed = cls._scrub_for_artifact(assistant_message)
        content = scrubbed.get("content")
        if isinstance(content, str):
            scrubbed["content"] = cls._scrub_text_for_artifact_action(content, action)
        for tool_call in scrubbed.get("tool_calls", []):
            if not isinstance(tool_call, dict):
                continue
            function_payload = tool_call.get("function")
            if not isinstance(function_payload, dict):
                continue
            arguments = function_payload.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                function_payload["arguments"] = json.dumps(
                    cls._scrub_for_artifact(json.loads(arguments)),
                    ensure_ascii=False,
                )
            except json.JSONDecodeError:
                function_payload["arguments"] = cls._scrub_text_for_artifact_action(arguments, action)
        return scrubbed

    # ------------------------------------------------------------------
    # Run directory and trace
    # ------------------------------------------------------------------

    def _make_run_dir(self, task: str, attempt: int) -> Path:
        """Create a unique run directory for this task attempt."""
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", task)[:48].strip("_") or "gui_task"
        name = f"{slug}_{int(time.time() * 1000)}_{attempt}"
        run_dir = self.artifacts_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "screenshots").mkdir(exist_ok=True)
        return run_dir

    @staticmethod
    async def _write_trace(path: Path, payload: dict[str, Any]) -> None:
        """Append a JSON line to the trace file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    @classmethod
    def _write_mobileworld_traj(
        cls,
        *,
        run_dir: Path,
        task: str,
        step_index: int,
        result: StepResult,
        current_observation: Observation,
        total_usage: dict[str, int],
    ) -> None:
        """Write an inspectable MobileWorld-style trajectory snapshot."""
        traj_path = run_dir / "traj.json"
        task_id = "0"
        if traj_path.exists():
            try:
                log_data = json.loads(traj_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log_data = {}
        else:
            log_data = {}

        task_log = log_data.setdefault(task_id, {"tools": None, "traj": []})
        marked_screenshot = cls._write_marked_screenshot(
            run_dir=run_dir,
            step_index=step_index,
            action=result.action,
            screenshot_path=current_observation.screenshot_path,
        )
        step_payload = {
            "task_goal": task,
            "step": step_index,
            "prediction": result.model_snapshot.get("action_text") or result.action_summary,
            "action": cls._scrub_for_artifact(cls._serialize_action(result.action)),
            "intent": cls._scrub_text_for_artifact_action(result.action_intent, result.action),
            "summary": cls._scrub_text_for_artifact_action(result.state_summary, result.action),
            "action_summary": cls._scrub_text_for_artifact_action(result.action_summary, result.action),
            "tool_call": _first_tool_call(result.model_snapshot),
            "tool_result": cls._scrub_text_for_artifact_action(result.tool_result, result.action),
            "done": result.done,
            "screenshot": _relative_path(current_observation.screenshot_path, run_dir),
            "next_screenshot": _relative_path(
                result.next_observation.screenshot_path
                if result.next_observation else None,
                run_dir,
            ),
            "marked_screenshot": marked_screenshot,
            "observation": cls._serialize_observation(current_observation),
            "next_observation": (
                cls._serialize_observation(result.next_observation)
                if result.next_observation else None
            ),
            "duration_s": round(result.duration_s, 3),
            "token_usage": result.step_usage or None,
        }
        task_log["traj"].append(cls._scrub_for_artifact(step_payload))
        task_log["token_usage"] = dict(total_usage)

        traj_path.write_text(
            json.dumps(log_data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _write_marked_screenshot(
        *,
        run_dir: Path,
        step_index: int,
        action: Action,
        screenshot_path: str | None,
    ) -> str | None:
        if action.action_type not in {"tap", "double_tap", "long_press", "drag", "swipe"}:
            return None
        if not screenshot_path or action.x is None or action.y is None:
            return None
        source = Path(screenshot_path)
        if not source.exists():
            return None
        marked_dir = run_dir / "marked_screenshots"
        marked_dir.mkdir(parents=True, exist_ok=True)
        target = marked_dir / f"marked-step_{step_index:03d}.png"
        try:
            from PIL import Image, ImageDraw

            with Image.open(source) as image:
                image = image.convert("RGB")
                width, height = image.size
                draw = ImageDraw.Draw(image)
                x1, y1 = _image_point(action.x, action.y, width, height, relative=action.relative)
                radius = max(4, min(width, height) // 50)
                if action.action_type in {"drag", "swipe"} and action.x2 is not None and action.y2 is not None:
                    x2, y2 = _image_point(action.x2, action.y2, width, height, relative=action.relative)
                    draw.line((x1, y1, x2, y2), fill="blue", width=max(2, radius // 2))
                    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill="green")
                    draw.ellipse((x2 - radius, y2 - radius, x2 + radius, y2 + radius), fill="red")
                else:
                    draw.ellipse((x1 - radius, y1 - radius, x1 + radius, y1 + radius), fill="red")
                image.save(target)
        except Exception as exc:
            logger.debug("Could not write marked screenshot %s: %s", target, exc)
            return None
        return target.relative_to(run_dir).as_posix()

    async def _log_attempt_event(
        self,
        run_dir: Path,
        event: str,
        **payload: Any,
    ) -> None:
        scrubbed_payload = self._scrub_for_log(payload)
        await self._write_trace(run_dir / "trace.jsonl", {
            "event": event,
            "timestamp": time.time(),
            **scrubbed_payload,
        })
        self._trajectory_recorder.record_event(event, **scrubbed_payload)

    # ------------------------------------------------------------------
    # Memory / skill / trajectory helpers
    # ------------------------------------------------------------------

    async def _retrieve_memory(self, task: str) -> str | None:
        """Return memory context for the current task.

        When ``_policy_context`` is set (nanobot path), policy entries are injected
        directly without embedding search — guaranteeing full policy coverage.  The
        legacy ``_memory_retriever`` path (opengui CLI) is preserved for backward
        compatibility when ``_policy_context`` is not provided.
        """
        if self._policy_context is not None:
            self._log_policy_injection(self._policy_context)
            return self._policy_context

        # Existing retriever-based path — used by the opengui CLI and any callers that
        # construct GuiAgent directly with a memory_retriever.
        if self._memory_retriever is None:
            return None
        from opengui.memory.types import MemoryType

        # Fetch relevant entries by query
        results = await self._memory_retriever.search(task, top_k=self._memory_top_k + 10)

        # Separate POLICY entries from search results
        policies = [(e, s) for e, s in results if e.memory_type == MemoryType.POLICY]
        others = [(e, s) for e, s in results if e.memory_type != MemoryType.POLICY][
            : self._memory_top_k
        ]

        # Also fetch all POLICY entries separately (they must always be included)
        policy_results = await self._memory_retriever.search(
            task, memory_type=MemoryType.POLICY, top_k=50,
        )
        # Merge: add any POLICY entries not already in the list
        seen_ids = {e.entry_id for e, _ in policies}
        for entry, score in policy_results:
            if entry.entry_id not in seen_ids:
                policies.append((entry, score))
                seen_ids.add(entry.entry_id)

        memory_entries = policies + others
        if not memory_entries:
            logger.info("Memory retrieval: no hits for task=%r", task)
            self._trajectory_recorder.record_event(
                "memory_retrieval",
                task=task,
                hit_count=0,
                hits=[],
                context="",
            )
            return None
        context = self._memory_retriever.format_context(memory_entries)
        self._log_memory_retrieval(task, memory_entries, context)
        return context

    def _log_policy_injection(self, context: str) -> None:
        """Record a trajectory event for direct policy context injection."""
        line_count = context.count("\n") + 1
        logger.info("Policy context injected directly: %d line(s)", line_count)
        self._trajectory_recorder.record_event(
            "memory_retrieval",
            task="(policy_direct_injection)",
            hit_count=line_count,
            hits=[],
            context=context[:200],
        )

    def _log_memory_retrieval(
        self,
        task: str,
        memory_entries: list[tuple[Any, float]],
        context: str,
    ) -> None:
        hits: list[dict[str, Any]] = []
        logger.info("Memory retrieval: %d hit(s) for task=%r", len(memory_entries), task)
        for entry, score in memory_entries:
            preview = re.sub(r"\s+", " ", entry.content).strip()[:160]
            hit = {
                "entry_id": entry.entry_id,
                "memory_type": entry.memory_type.value,
                "platform": entry.platform,
                "app": entry.app,
                "score": round(float(score), 4),
                "content_preview": preview,
            }
            hits.append(hit)
            logger.info(
                "Memory hit id=%s type=%s score=%.4f platform=%s app=%s content=%s",
                entry.entry_id,
                entry.memory_type.value,
                float(score),
                entry.platform,
                entry.app or "-",
                preview,
            )

        self._trajectory_recorder.record_event(
            "memory_retrieval",
            task=task,
            hit_count=len(hits),
            hits=hits,
            context=context,
        )

    @staticmethod
    def _task_without_advisory_hints(task: str) -> str:
        """Return the user task text without injected advisory memory hints."""
        return str(task or "").split("\n\nAdvisory hints from past GUI memory:", 1)[0]

    def _skill_app_filter(self, task: str, app_hint: str | None) -> str | None:
        platform = self.backend.platform
        if app_hint:
            normalized = normalize_app_identifier(platform, app_hint)
            if normalized and normalized != "unknown":
                return normalized
        if platform.strip().lower() == "android":
            return find_android_app_in_text(self._task_without_advisory_hints(task))
        return None

    async def _build_prompt_skill_parts(
        self,
        task: str,
        *,
        app: str | None = None,
    ) -> CompactPromptParts:
        """Retrieve prompt-visible skills and always-on composite actions."""
        self._prompt_skills_by_id = {}
        self._prompt_composite_aliases = set()
        if self._skill_library is None:
            self._trajectory_recorder.record_event(
                "prompt_skill_context",
                task=task,
                hit_count=0,
                composite_aliases=[],
                reason="no_library",
                app_filter=app,
            )
            return CompactPromptParts()

        all_skills = [
            skill
            for skill in self._skill_library.list_all(platform=self.backend.platform)
            if self._skill_matches_app_filter(skill, app)
        ]
        composite_actions = composite_action_infos_from_skills(
            all_skills,
            tags=self._always_on_skill_tags,
        )
        composite_actions.extend(self._legacy_composite_action_infos(all_skills, composite_actions))
        self._prompt_composite_aliases = {action.alias for action in composite_actions}

        skill_query = self._task_without_advisory_hints(task)
        retrieved_infos = []
        if self._prompt_skill_top_k > 0:
            search_k = max(
                self._prompt_skill_top_k,
                self._prompt_skill_top_k * 5
                if self._prompt_shortcut_only or self._always_on_skill_tags
                else self._prompt_skill_top_k,
            )
            results = await self._skill_library.search(
                skill_query,
                platform=self.backend.platform,
                app=app,
                top_k=search_k,
            )
            for skill, score in results:
                if is_always_on_skill(skill, self._always_on_skill_tags):
                    continue
                if self._prompt_shortcut_only and not is_shortcut_skill(skill):
                    continue
                skill_id = str(getattr(skill, "skill_id", "") or "")
                if not skill_id:
                    continue
                self._prompt_skills_by_id[skill_id] = skill
                retrieved_infos.append(skill_info_from_flat_skill(skill, score=score))
                if len(retrieved_infos) >= self._prompt_skill_top_k:
                    break

        parts = build_compact_prompt_parts(
            retrieved_skills=retrieved_infos,
            composite_actions=composite_actions,
        )
        self._trajectory_recorder.record_event(
            "prompt_skill_context",
            task=task,
            hit_count=len(retrieved_infos),
            skill_ids=list(parts.skill_ids),
            composite_aliases=list(parts.composite_aliases),
            app_filter=app,
            shortcut_only=self._prompt_shortcut_only,
        )
        return parts

    @staticmethod
    def _legacy_composite_action_infos(
        skills: list[Any],
        existing_actions: list[CompositeActionInfo],
    ) -> list[CompositeActionInfo]:
        existing_aliases = {action.alias for action in existing_actions}
        if "click_and_type" in existing_aliases:
            return []
        for skill in skills:
            tags = {str(tag).strip().lower() for tag in (getattr(skill, "tags", ()) or ())}
            name = str(getattr(skill, "name", "") or "").strip().lower()
            if "action_alias:click_and_type" in tags or name == "click_and_type":
                return [
                    CompositeActionInfo(
                        alias="click_and_type",
                        description=(
                            "Legacy alias for click_then_type: tap a visible text field and type text."
                        ),
                        example='`{"action_type":"click_and_type","coordinate":[x,y],"text":"Hello","auto_enter":false}`',
                        source_skill_id=str(getattr(skill, "skill_id", "") or "") or None,
                    )
                ]
        return []

    @staticmethod
    def _skill_matches_app_filter(skill: Any, app: str | None) -> bool:
        if not app:
            return True
        skill_app = str(getattr(skill, "app", "") or "").strip()
        return skill_app in {"", "*", "any", "unknown"} or skill_app == app

    async def _search_skill(self, task: str, *, app: str | None = None) -> Any | None:
        """Search the skill library and return the top match when above threshold."""
        if self._skill_library is None:
            self._trajectory_recorder.record_event(
                "skill_search",
                task=task,
                source="none",
                matched=False,
                reason="no_library",
                threshold=self._skill_threshold,
                app_filter=app,
            )
            return None
        from opengui.skills.data import compute_confidence

        search_results = await self._skill_library.search(
            task, platform=self.backend.platform, app=app, top_k=1,
        )
        if not search_results:
            self._trajectory_recorder.record_event(
                "skill_search",
                task=task,
                source="legacy",
                matched=False,
                reason="no_results",
                threshold=self._skill_threshold,
                app_filter=app,
            )
            return None
        skill, relevance = search_results[0]
        confidence = compute_confidence(skill)
        final_score = relevance
        if final_score >= self._skill_threshold:
            self._trajectory_recorder.record_event(
                "skill_search",
                task=task,
                source="legacy",
                matched=True,
                skill_id=skill.skill_id,
                skill_name=skill.name,
                score=round(final_score, 4),
                confidence=round(confidence, 4),
                relevance=round(relevance, 4),
                threshold=self._skill_threshold,
                app_filter=app,
            )
            return (skill, final_score)
        self._trajectory_recorder.record_event(
            "skill_search",
            task=task,
            source="legacy",
            matched=False,
            reason="below_threshold",
            skill_name=skill.name,
            score=round(final_score, 4),
            confidence=round(confidence, 4),
            relevance=round(relevance, 4),
            threshold=self._skill_threshold,
            app_filter=app,
        )
        return None

    async def _execute_skill_with_params(self, task: str, skill: Any) -> Any:
        """Execute a skill, extracting task parameters only when needed."""
        if self._skill_executor is None:
            raise RuntimeError("skill executor is not configured")
        skill_params: dict[str, str] = {}
        if getattr(skill, "parameters", None):
            skill_params = await self._extract_skill_params(task, skill)
            missing_params = _missing_skill_params(skill, skill_params)
            if missing_params:
                recorder = getattr(self, "_trajectory_recorder", None)
                if recorder is not None:
                    recorder.record_event(
                        "skill_param_extraction_failed",
                        skill_id=str(getattr(skill, "skill_id", "") or ""),
                        skill_name=str(getattr(skill, "name", "") or ""),
                        reason="missing_required_params",
                        missing_params=missing_params,
                        extracted_params=dict(skill_params),
                    )
                raise RuntimeError(
                    "missing required skill params: " + ", ".join(missing_params)
                )
        return await self._skill_executor.execute(skill, params=skill_params)

    async def _skill_entry_allows_current_state(self, skill: Any) -> bool:
        """Fail closed on mid-flow skills unless their first contract matches now."""
        steps = tuple(getattr(skill, "steps", ()) or ())
        skill_id = str(getattr(skill, "skill_id", "") or "")
        skill_name = str(getattr(skill, "name", "") or "")
        if not steps:
            self._trajectory_recorder.record_event(
                "skill_entry_rejected",
                skill_id=skill_id,
                skill_name=skill_name,
                reason="empty_skill",
            )
            return False

        first_step = steps[0]
        first_action = str(getattr(first_step, "action_type", "") or "")
        if first_action == "open_app" and _skill_entry_targets_android_launcher(skill, first_step):
            self._trajectory_recorder.record_event(
                "skill_entry_rejected",
                skill_id=skill_id,
                skill_name=skill_name,
                first_action=first_action,
                reason="launcher_entry_package",
                app=str(getattr(skill, "app", "") or "") or None,
                target=str(getattr(first_step, "target", "") or "") or None,
            )
            return False
        if first_action in {"open_app", "open_deeplink", "open_intent"}:
            self._trajectory_recorder.record_event(
                "skill_entry_accepted",
                skill_id=skill_id,
                skill_name=skill_name,
                first_action=first_action,
                reason="entry_action",
            )
            return True

        contract = getattr(first_step, "state_contract", None)
        if contract is None:
            self._trajectory_recorder.record_event(
                "skill_entry_rejected",
                skill_id=skill_id,
                skill_name=skill_name,
                first_action=first_action,
                reason="missing_first_step_state_contract",
            )
            return False

        screenshot_path = self.artifacts_root / "skill_entry_gate" / "current.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            observation = await self.backend.observe(
                screenshot_path,
                timeout=self.step_timeout,
            )
        except Exception as exc:
            logger.warning("Skill entry observation failed for %s: %s", skill_name, exc)
            self._trajectory_recorder.record_event(
                "skill_entry_rejected",
                skill_id=skill_id,
                skill_name=skill_name,
                first_action=first_action,
                reason="observe_failed",
                error=str(exc),
            )
            return False

        contract_result = evaluate_state_contract(contract, observation=observation)
        if contract_result is True:
            self._trajectory_recorder.record_event(
                "skill_entry_accepted",
                skill_id=skill_id,
                skill_name=skill_name,
                first_action=first_action,
                reason="state_contract_matched",
            )
            return True

        self._trajectory_recorder.record_event(
            "skill_entry_rejected",
            skill_id=skill_id,
            skill_name=skill_name,
            first_action=first_action,
            reason="state_contract_unevaluable" if contract_result is None else "state_contract_failed",
        )
        return False

    async def _inject_skill_memory_context(
        self,
        skill: Any,
        existing_context: str | None,
    ) -> str | None:
        """No-op: legacy Skill objects do not carry a memory_context_id."""
        return existing_context

    async def _extract_skill_params(
        self,
        task: str,
        skill: Any,
    ) -> dict[str, str]:
        """Extract runtime parameter values from the task description via LLM.

        Uses the skill's declared ``parameters`` list as a schema and asks the
        LLM to pull matching values from the task string.  Returns partial
        values when some parameters can be inferred deterministically.
        """
        param_names: list[str] = _skill_param_names(skill)
        if not param_names:
            return {}
        params = {
            name: value
            for name in param_names
            if (value := self._guess_skill_param(task, name)) is not None
        }
        if len(params) == len(param_names):
            return params
        json_template = "{" + ", ".join(f'"{p}": "value"' for p in param_names) + "}"
        prompt = (
            f"Task: {task}\n\n"
            f"Skill: {skill.name} — {skill.description}\n"
            f"Parameters to extract: {param_names}\n\n"
            f"Extract the value for each parameter from the task description.\n"
            f"Return JSON only, with exactly these keys: {json_template}"
        )
        try:
            response = await asyncio.wait_for(
                self.llm.chat([{"role": "user", "content": prompt}]),
                timeout=_SKILL_PARAM_EXTRACTION_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            logger.warning("Skill param extraction LLM call failed: %s", exc)
            recorder = getattr(self, "_trajectory_recorder", None)
            if recorder is not None:
                recorder.record_event(
                    "skill_param_extraction_failed",
                    skill_id=str(getattr(skill, "skill_id", "") or ""),
                    skill_name=str(getattr(skill, "name", "") or ""),
                    reason="llm_call_failed",
                    error=str(exc),
                    exception_type=type(exc).__name__,
                    missing_params=[name for name in param_names if name not in params],
                    extracted_params=dict(params),
                )
            return params
        text = (response.content or "").strip()
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                for key, value in obj.items():
                    if key in param_names and key not in params:
                        text_value = str(value).strip()
                        if text_value:
                            params[key] = text_value
                return params
            except json.JSONDecodeError:
                pass
        logger.warning("Could not parse skill param extraction response: %r", text[:120])
        recorder = getattr(self, "_trajectory_recorder", None)
        if recorder is not None:
            recorder.record_event(
                "skill_param_extraction_failed",
                skill_id=str(getattr(skill, "skill_id", "") or ""),
                skill_name=str(getattr(skill, "name", "") or ""),
                reason="parse_failed",
                response=text[:240],
                missing_params=[name for name in param_names if name not in params],
                extracted_params=dict(params),
            )
        return params

    @staticmethod
    def _guess_skill_param(task: str, param_name: str) -> str | None:
        name = param_name.strip().casefold()
        normalized = " ".join((task or "").split())
        if not normalized:
            return None
        quoted = re.search(r"[“\"']([^”\"']{1,80})[”\"']", normalized)
        text_like_names = {
            "search_query",
            "search_term",
            "query",
            "keyword",
            "title",
            "subject",
            "note",
            "memo",
            "message",
            "description",
            "text",
            "name",
            "contact_name",
            "person",
            "item",
        }
        if quoted is not None and name in text_like_names:
            return quoted.group(1).strip()
        if name in {"phone", "phone_number", "mobile", "mobile_phone", "tel", "telephone"}:
            for pattern in (
                r"(?:手机号|电话号码|联系电话|电话)\s*(?:是|为|:|：)?\s*([+()0-9][0-9()+\-\s]{5,24})",
                r"(?:phone|mobile|tel(?:ephone)?)\s*(?:number)?\s*(?:is|:)?\s*([+()0-9][0-9()+\-\s]{5,24})",
            ):
                match = re.search(pattern, normalized, flags=re.IGNORECASE)
                if match is not None:
                    value = re.sub(r"\s+", "", match.group(1)).strip(".,;，。；")
                    if value:
                        return value
            return None
        if name in {"amount", "price", "cost", "total", "value", "money"}:
            for pattern in (
                r"(?:金额|价格|花费|费用|总额|支出)\s*(?:是|为|:|：)?\s*(?:¥|￥|\$)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:amount|price|cost|total|value)\s*(?:is|:)?\s*(?:\$)?\s*([0-9]+(?:\.[0-9]+)?)",
                r"(?:¥|￥|\$)\s*([0-9]+(?:\.[0-9]+)?)",
            ):
                match = re.search(pattern, normalized, flags=re.IGNORECASE)
                if match is not None:
                    return match.group(1).strip()
            return None
        if name in {"date", "day"}:
            match = re.search(
                r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?|\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?|\d{1,2}[-/]\d{1,2})",
                normalized,
            )
            return _clean_inferred_param(match.group(1)) if match else None
        if name in {"time", "clock_time"}:
            match = re.search(
                r"(\d{1,2}\s*[:：]\s*\d{2}(?:\s*(?:AM|PM|am|pm))?|\d{1,2}\s*(?:点|时)(?:\s*\d{1,2}\s*分?)?)",
                normalized,
            )
            return _clean_inferred_param(match.group(1)) if match else None
        if name in {"name", "contact_name", "person"}:
            for pattern in (
                r"(?:联系人|姓名|名字)\s*(?:是|为|叫|:|：)?\s*([^，,。.!?；;、]{1,40})",
                r"(?:named|called|name(?:d)?\s+is)\s+([^,.;]{1,60})",
            ):
                match = re.search(pattern, normalized, flags=re.IGNORECASE)
                if match is not None:
                    value = _clean_inferred_param(match.group(1))
                    if value:
                        return value
            return None
        if name in {"city", "location", "destination", "place", "area"}:
            for pattern in (
                r"(?:搜索一下|搜一下|搜索|查找|找一下)\s*([^，,。.!?；;、]{2,30}?)(?:周边|附近|的酒店|酒店|民宿)",
                r"(?:去|到)\s*([^，,。.!?；;、]{2,16}?)(?:附近|周边|住|的|，|,|要)",
                r"住在\s*([^，,。.!?；;、]{2,16}?)(?:附近|周边|，|,|要)",
            ):
                for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                    value = _clean_inferred_param(match.group(1))
                    if value and not _looks_like_date_param(value):
                        return value
            return None
        if name not in {"search_query", "search_term", "query", "keyword"}:
            return None
        account = re.search(r"(?:找一下|找|搜索|搜)\s*([^，,。.!?；;、]{1,30}?)(?:的账号|账号)", normalized)
        if account is not None:
            value = _clean_inferred_param(account.group(1))
            if value:
                return value
        for pattern in (
            r"(?:搜索一下|搜一下|搜索|查找)\s*([^\s，,。.!?；;、]+)",
            r"(?:search\s+for|search|find)\s*([^\s，,。.!?；;、]+)",
        ):
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match is None:
                continue
            value = _clean_inferred_param(match.group(1))
            if value:
                return value
        return None

    async def _skill_maintenance(
        self, skill_match: Any | None, success: bool
    ) -> None:
        """Post-run: update confidence while keeping failed skills evolvable."""
        if skill_match is None or self._skill_library is None:
            return
        if hasattr(skill_match, "layer"):
            return
        skill, _ = skill_match
        if success:
            updated = replace(
                skill,
                success_count=skill.success_count + 1,
                success_streak=skill.success_streak + 1,
                failure_streak=0,
            )
        else:
            updated = replace(
                skill,
                failure_count=skill.failure_count + 1,
                failure_streak=skill.failure_streak + 1,
                success_streak=0,
            )

        self._skill_library.update(skill.skill_id, updated)


def _first_tool_call(model_snapshot: dict[str, Any]) -> dict[str, Any] | None:
    tool_calls = model_snapshot.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first = tool_calls[0]
        if isinstance(first, dict):
            return first
    return None


def _relative_path(path: str | None, root: Path) -> str | None:
    if not path:
        return None
    target = Path(path)
    try:
        return target.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _image_point(
    x: float,
    y: float,
    width: int,
    height: int,
    *,
    relative: bool,
) -> tuple[int, int]:
    return (
        _image_coordinate(x, width, relative=relative),
        _image_coordinate(y, height, relative=relative),
    )


def _image_coordinate(value: float, extent: int, *, relative: bool) -> int:
    if extent <= 1:
        return 0
    if relative:
        pixel = round(float(value) / 999 * (extent - 1))
    else:
        pixel = round(float(value))
    return max(0, min(pixel, extent - 1))
