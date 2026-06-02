"""GuiSubagentTool: exposes opengui's GuiAgent as a nanobot tool."""

from __future__ import annotations

import inspect
import json
import logging
import sys
from collections.abc import Awaitable, Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import litellm
import numpy as np

from nanobot.agent.gui_adapter import NanobotEmbeddingAdapter, NanobotLLMAdapter
from nanobot.agent.tools.base import Tool
from opengui.agent import GuiAgent
from opengui.interfaces import InterventionHandler, InterventionRequest, InterventionResolution
from opengui.postprocessing import EvaluationConfig, PostRunProcessor
from opengui.skills.normalization import get_gui_skill_store_root, resolve_ios_bundle
from opengui.trajectory.recorder import TrajectoryRecorder

if TYPE_CHECKING:
    from nanobot.config.schema import GuiConfig
    from nanobot.providers.base import LLMProvider


logger = logging.getLogger(__name__)
DEFAULT_OPENGUI_MEMORY_DIR = Path.home() / ".opengui" / "memory"
_EMBEDDING_BATCH_SIZE = 10
_SAFE_INTERVENTION_TARGET_KEYS = frozenset(
    {"display_id", "monitor_index", "desktop_name", "width", "height", "platform"}
)
WindowsIsolatedBackend = None
probe_isolated_background_support = None
resolve_run_mode = None
log_mode_resolution = None
WINDOWS_TARGET_APP_CLASSES = ("classic-win32", "uwp", "directx", "gpu-heavy", "electron-gpu")


class GuiSubagentTool(Tool):
    """Run a GUI automation task through opengui."""

    def __init__(
        self,
        *,
        gui_config: "GuiConfig | None",
        provider: "LLMProvider",
        model: str,
        workspace: Path,
        s2_provider: "LLMProvider | None" = None,
        s2_model: str | None = None,
        gui_event_callback: Any | None = None,
        gui_frame_callback: Any | None = None,
    ) -> None:
        if gui_config is None:
            raise ValueError("GuiSubagentTool requires gui_config")

        self._gui_config = gui_config
        self._provider = provider
        self._model = model
        self._s2_provider = s2_provider
        self._s2_model = s2_model
        self._workspace = Path(workspace)
        self._gui_event_callback = gui_event_callback
        self._gui_frame_callback = gui_frame_callback
        self._llm_adapter = NanobotLLMAdapter(
            provider, model, capture_ttft=gui_config.capture_ttft,
        )
        self._s2_llm_adapter = (
            NanobotLLMAdapter(s2_provider, s2_model, capture_ttft=gui_config.capture_ttft)
            if s2_provider is not None and s2_model
            else None
        )
        self._embedding_signature: str | None = self._resolve_embedding_signature()
        self._embedding_adapter = self._build_embedding_adapter() if gui_config.embedding_model else None
        self._skill_libraries: dict[str, Any] = {}

        self._backend = self._build_backend(gui_config.backend)
        self._skill_library = (
            self._get_skill_library(self._backend.platform, embedding_signature=self._embedding_signature)
            if gui_config.enable_skill_execution
            else None
        )
        self._postprocessor = PostRunProcessor(
            llm=self._llm_adapter,
            merge_llm=self._llm_adapter,
            embedding_provider=self._embedding_adapter,
            embedding_signature=self._embedding_signature,
            skill_store_root=get_gui_skill_store_root(self._workspace),
            enable_skill_extraction=gui_config.enable_skill_extraction,
            evaluation=EvaluationConfig(
                enabled=gui_config.evaluation.enabled,
                judge_model=gui_config.evaluation.judge_model,
                api_key=gui_config.evaluation.api_key,
                api_base=gui_config.evaluation.api_base,
            ),
        )

    @property
    def name(self) -> str:
        return "gui_task"

    @property
    def description(self) -> str:
        return (
            "Execute a GUI automation task on a device. The task is performed by "
            "a vision-action agent that observes screenshots and executes actions. "
            "Returns a structured result with success status, summary, and trace path."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The GUI task to perform.",
                },
                "backend": {
                    "type": "string",
                    "enum": ["adb", "ios", "hdc", "local", "dry-run"],
                    "description": "Optional backend override. Defaults to the configured GUI backend.",
                },
                "require_background_isolation": {
                    "type": "boolean",
                    "description": "Block instead of falling back when isolated background execution is unavailable.",
                },
                "acknowledge_background_fallback": {
                    "type": "boolean",
                    "description": "Explicitly acknowledge foreground fallback when isolated background execution is unavailable.",
                },
                "target_app_class": {
                    "type": "string",
                    "enum": list(WINDOWS_TARGET_APP_CLASSES),
                    "description": "Optional Windows app class hint for isolated background probing.",
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        backend: str | None = None,
        require_background_isolation: bool = False,
        acknowledge_background_fallback: bool = False,
        target_app_class: str | None = None,
        **kwargs: Any,
    ) -> str:
        active_backend = self._select_backend(backend)
        try:
            if self._gui_config.background:
                probe_fn = probe_isolated_background_support
                resolve_fn = resolve_run_mode
                log_fn = log_mode_resolution
                if probe_fn is None:
                    from opengui.backends.background_runtime import (
                        probe_isolated_background_support as runtime_probe_isolated_background_support,
                    )

                    probe_fn = runtime_probe_isolated_background_support
                if resolve_fn is None:
                    from opengui.backends.background_runtime import (
                        resolve_run_mode as runtime_resolve_run_mode,
                    )

                    resolve_fn = runtime_resolve_run_mode
                if log_fn is None:
                    from opengui.backends.background_runtime import (
                        log_mode_resolution as runtime_log_mode_resolution,
                    )

                    log_fn = runtime_log_mode_resolution

                resolved_target_app_class = self._resolve_probe_target_app_class(
                    backend,
                    target_app_class,
                    sys_platform=sys.platform,
                )
                probe = probe_fn(
                    sys_platform=sys.platform,
                    target_app_class=resolved_target_app_class,
                )
                decision = resolve_fn(
                    probe,
                    require_isolation=require_background_isolation,
                    require_acknowledgement_for_fallback=True,
                )
                log_fn(logger, decision, owner="nanobot", task=task)

                if decision.mode == "blocked":
                    return self._background_json_failure(decision.message)

                if decision.mode == "fallback" and not acknowledge_background_fallback:
                    return self._background_json_failure(
                        f"{decision.message} Re-run with acknowledge_background_fallback=true to continue in foreground."
                    )

                if decision.mode == "isolated":
                    try:
                        mgr = self._build_isolated_display_manager(probe)
                    except RuntimeError as exc:
                        return self._background_json_failure(str(exc))
                    if probe.backend_name == "windows_isolated_desktop":
                        wrapped_backend = None
                        try:
                            backend_cls = WindowsIsolatedBackend
                            if backend_cls is None:
                                from opengui.backends.windows_isolated import (
                                    WindowsIsolatedBackend as ImportedWindowsIsolatedBackend,
                                )

                                backend_cls = ImportedWindowsIsolatedBackend
                            wrapped_backend = backend_cls(
                                active_backend,
                                mgr,
                                run_metadata={"owner": "nanobot", "task": task, "model": self._model},
                            )
                            return await self._run_task(wrapped_backend, task, **kwargs)
                        except RuntimeError as exc:
                            return self._background_json_failure(str(exc))
                        finally:
                            if wrapped_backend is not None:
                                await wrapped_backend.shutdown()
                    else:
                        from opengui.backends.background import BackgroundDesktopBackend

                        wrapped_backend = BackgroundDesktopBackend(
                            active_backend,
                            mgr,
                            run_metadata={"owner": "nanobot", "task": task, "model": self._model},
                        )
                        try:
                            return await self._run_task(wrapped_backend, task, **kwargs)
                        finally:
                            await wrapped_backend.shutdown()

            return await self._run_task(active_backend, task, **kwargs)
        finally:
            await self._shutdown_android_backend(active_backend)

    async def _run_task(self, active_backend: Any, task: str, **kwargs: Any) -> str:
        policy_context, memory_store = self._load_policy_context_and_memory_store(
            platform=active_backend.platform,
        )
        memory_retriever = await self._build_memory_retriever()
        installed_apps = await self._load_installed_apps(active_backend)
        skill_library = None
        if self._gui_config.enable_skill_execution:
            self._refresh_cached_skill_stores()
            skill_library = self._get_skill_library(
                active_backend.platform,
                embedding_signature=self._embedding_signature,
            )
        run_dir = self._make_run_dir()
        recorder = TrajectoryRecorder(
            output_dir=run_dir,
            task=task,
            platform=active_backend.platform,
            event_callback=self._gui_event_callback,
        )

        skill_executor = None
        if self._gui_config.enable_skill_execution:
            from opengui.agent import (
                _AgentActionGrounder,
                _AgentScreenshotProvider,
                _AgentSubgoalRunner,
            )
            from opengui.skills.executor import LLMStateValidator, SkillExecutor

            validator_llm = (
                NanobotLLMAdapter(self._provider, self._gui_config.validator_model)
                if self._gui_config.validator_model
                else self._llm_adapter
            )
            grounder_llm = (
                NanobotLLMAdapter(self._provider, self._gui_config.grounder_model)
                if self._gui_config.grounder_model
                else self._llm_adapter
            )
            grounder_model = self._gui_config.grounder_model or self._model

            state_validator = LLMStateValidator(
                validator_llm,
                image_scale_ratio=self._gui_config.image_scale_ratio,
            )
            skill_executor = SkillExecutor(
                backend=active_backend,
                state_validator=state_validator,
                action_grounder=_AgentActionGrounder(
                    llm=grounder_llm,
                    model=grounder_model,
                    agent_profile=self._gui_config.agent_profile,
                    image_scale_ratio=self._gui_config.image_scale_ratio,
                ),
                subgoal_runner=_AgentSubgoalRunner(
                    llm=self._llm_adapter,
                    backend=active_backend,
                    state_validator=state_validator,
                    model=self._model,
                    artifacts_root=run_dir,
                    trajectory_recorder=recorder,
                    agent_profile=self._gui_config.agent_profile,
                    step_timeout=30.0,
                    image_scale_ratio=self._gui_config.image_scale_ratio,
                ),
                screenshot_provider=_AgentScreenshotProvider(
                    backend=active_backend,
                    artifacts_root=run_dir,
                ),
                trajectory_recorder=recorder,
                stop_on_failure=True,
                max_recovery_steps=3,
            )

        skill_reuser = None
        if skill_library is not None and self._gui_config.reuser_model:
            from opengui.skills.reuser import SkillReuser

            reuser_llm = NanobotLLMAdapter(self._provider, self._gui_config.reuser_model)
            skill_reuser = SkillReuser(reuser_llm, threshold=self._gui_config.skill_threshold)

        agent = GuiAgent(
            llm=self._llm_adapter,
            backend=active_backend,
            trajectory_recorder=recorder,
            model=self._model,
            artifacts_root=run_dir,
            max_steps=self._gui_config.max_steps,
            policy_context=policy_context,
            memory_retriever=memory_retriever,
            skill_library=skill_library,
            skill_threshold=self._gui_config.skill_threshold,
            skill_executor=skill_executor,
            skill_reuser=skill_reuser,
            intervention_handler=self._build_intervention_handler(active_backend, task),
            memory_store=memory_store,
            agent_profile=self._gui_config.agent_profile,
            image_scale_ratio=self._gui_config.image_scale_ratio,
            stagnation_limit=self._gui_config.stagnation_limit,
            installed_apps=installed_apps,
            s2_llm=self._s2_llm_adapter,
            s2_model=self._s2_model,
        )

        app_hint = self._resolve_app_hint(
            task,
            platform=active_backend.platform,
            installed_apps=installed_apps,
        )
        result = await agent.run(task=task, app_hint=app_hint)
        summary = result.summary
        error = result.error
        if error and error.startswith("intervention_cancelled:"):
            error = "intervention_cancelled"
        trace_path = self._resolve_trace_path(recorder_path=recorder.path, agent_trace_path=result.trace_path)
        post_run_state = self._build_post_run_state(
            trace_path=trace_path,
            success=result.success,
            summary=summary,
            error=error,
        )
        self._postprocessor.schedule(trace_path, is_success=result.success, platform=active_backend.platform, task=task)

        return json.dumps(
            {
                "success": result.success,
                "summary": summary,
                "model_summary": result.model_summary,
                "trace_path": str(trace_path) if trace_path is not None else result.trace_path,
                "steps_taken": result.steps_taken,
                "error": error,
                "post_run_state": post_run_state,
            },
            ensure_ascii=False,
        )

    async def _load_installed_apps(self, active_backend: Any) -> list[str] | None:
        list_apps = getattr(active_backend, "list_apps", None)
        if not callable(list_apps):
            return None
        try:
            apps = await list_apps()
        except Exception:
            logger.debug("Failed to list GUI backend apps for app hint resolution", exc_info=True)
            return None
        if not isinstance(apps, list):
            return None
        return [str(app) for app in apps if str(app).strip()]

    @staticmethod
    def _resolve_app_hint(
        task: str,
        *,
        platform: str,
        installed_apps: list[str] | None,
    ) -> str | None:
        if platform != "ios":
            return None
        normalized_task = GuiAgent._normalize_system_action_task(task)
        target = GuiAgent._extract_ios_app_target(normalized_task)
        if target is None:
            return None
        bundle_id = resolve_ios_bundle(target, installed_apps)
        if bundle_id == target and "." not in bundle_id:
            return None
        return bundle_id

    def _build_post_run_state(
        self,
        *,
        trace_path: Path | None,
        success: bool,
        summary: str,
        error: str | None,
    ) -> dict[str, Any]:
        step_event = self._load_latest_step_event(trace_path)
        observation = self._extract_latest_observation(step_event)
        last_action = step_event.get("action") if isinstance(step_event, dict) else None
        last_action_summary = None
        latest_screenshot_path = None
        if isinstance(step_event, dict):
            last_action_summary = self._string_or_none(step_event.get("model_output"))
            latest_screenshot_path = self._string_or_none(step_event.get("screenshot_path"))
        if latest_screenshot_path is None and observation:
            latest_screenshot_path = self._string_or_none(observation.get("screenshot_path"))

        foreground_app = self._string_or_none(observation.get("foreground_app")) if observation else None
        platform = self._string_or_none(observation.get("platform")) if observation else None
        resolution = self._format_resolution(observation)
        current_state = self._describe_current_state(
            success=success,
            summary=summary,
            error=error,
            foreground_app=foreground_app,
            resolution=resolution,
        )

        return {
            "trace_read": trace_path is not None and trace_path.exists(),
            "latest_screenshot_path": latest_screenshot_path,
            "last_action": last_action,
            "last_action_summary": last_action_summary,
            "last_foreground_app": foreground_app,
            "platform": platform,
            "screen_resolution": resolution,
            "current_state": current_state,
            "completion_assessment": "completed" if success else "not_completed",
        }

    async def _wait_for_pending_postprocessing(self) -> None:
        """Drain all pending post-processing background tasks."""
        await self._postprocessor.drain()

    @staticmethod
    def _load_latest_step_event(trace_path: Path | None) -> dict[str, Any]:
        if trace_path is None or not trace_path.exists():
            return {}

        latest_step: dict[str, Any] = {}
        latest_observed_step: dict[str, Any] = {}
        try:
            with open(trace_path, encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and (event.get("type") == "step" or event.get("event") == "step"):
                        latest_step = event
                        if GuiSubagentTool._step_has_observation_evidence(event):
                            latest_observed_step = event
        except OSError:
            logger.warning("Could not read GUI trace for post-run state: %s", trace_path, exc_info=True)
        if latest_step and latest_observed_step and latest_step is not latest_observed_step:
            merged = dict(latest_step)
            if not merged.get("screenshot_path"):
                screenshot_path = latest_observed_step.get("screenshot_path")
                if not screenshot_path:
                    observation = GuiSubagentTool._extract_latest_observation(latest_observed_step)
                    screenshot_path = observation.get("screenshot_path")
                if screenshot_path:
                    merged["screenshot_path"] = screenshot_path
            if not merged.get("observation"):
                observation = GuiSubagentTool._extract_latest_observation(latest_observed_step)
                if observation:
                    merged["observation"] = observation
            latest_step = merged
        return latest_step

    @staticmethod
    def _step_has_observation_evidence(step_event: dict[str, Any]) -> bool:
        if step_event.get("screenshot_path") or isinstance(step_event.get("observation"), dict):
            return True
        execution = step_event.get("execution")
        if isinstance(execution, dict) and isinstance(execution.get("next_observation"), dict):
            return True
        prompt = step_event.get("prompt")
        return isinstance(prompt, dict) and isinstance(prompt.get("current_observation"), dict)

    @staticmethod
    def _extract_latest_observation(step_event: dict[str, Any]) -> dict[str, Any]:
        execution = step_event.get("execution") if isinstance(step_event, dict) else None
        if isinstance(execution, dict):
            next_observation = execution.get("next_observation")
            if isinstance(next_observation, dict):
                return next_observation

        observation = step_event.get("observation") if isinstance(step_event, dict) else None
        if isinstance(observation, dict):
            return observation

        prompt = step_event.get("prompt") if isinstance(step_event, dict) else None
        if isinstance(prompt, dict):
            current_observation = prompt.get("current_observation")
            if isinstance(current_observation, dict):
                return current_observation

        return {}

    @staticmethod
    def _format_resolution(observation: dict[str, Any]) -> str | None:
        width = observation.get("screen_width")
        height = observation.get("screen_height")
        if isinstance(width, int) and isinstance(height, int):
            return f"{width}x{height}"
        return None

    @staticmethod
    def _string_or_none(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return str(value)

    @staticmethod
    def _describe_current_state(
        *,
        success: bool,
        summary: str,
        error: str | None,
        foreground_app: str | None,
        resolution: str | None,
    ) -> str:
        note = summary.strip()
        if note:
            return note

        parts = ["GUI task completed successfully." if success else "GUI task did not complete successfully."]
        if error:
            parts.append(f"Error: {error}.")
        if foreground_app:
            parts.append(f"Latest visible app: {foreground_app}.")
        if resolution:
            parts.append(f"Screen resolution: {resolution}.")
        return " ".join(parts)

    def _select_backend(self, backend: str | None) -> Any:
        if backend is None or backend == self._gui_config.backend:
            return self._backend
        return self._build_backend(backend)

    async def _build_memory_retriever(self) -> Any | None:
        """Build a memory retriever indexed with non-policy GUI memory entries.

        Policy entries are injected in full through ``policy_context``.  OS, app,
        and icon guides stay selective so they do not crowd out the GUI prompt.
        """
        if self._embedding_adapter is None:
            return None

        from opengui.memory.retrieval import MemoryRetriever
        from opengui.memory.store import MemoryStore
        from opengui.memory.types import MemoryType

        try:
            memory_store = MemoryStore(DEFAULT_OPENGUI_MEMORY_DIR)
            entries = [
                entry
                for entry in memory_store.list_all()
                if entry.memory_type != MemoryType.POLICY
            ]
            if not entries:
                return None
            memory_retriever = MemoryRetriever(embedding_provider=self._embedding_adapter, top_k=5)
            await memory_retriever.index(entries)
            return memory_retriever
        except Exception:
            logger.warning(
                "GUI memory retriever initialization failed for %s",
                DEFAULT_OPENGUI_MEMORY_DIR,
                exc_info=True,
            )
            return None

    def _load_policy_context(self) -> str | None:
        policy_context, _ = GuiSubagentTool._load_policy_context_and_memory_store(self)
        return policy_context

    def _load_policy_context_and_memory_store(
        self,
        *,
        platform: str | None = None,
    ) -> tuple[str | None, Any | None]:
        return GuiSubagentTool._load_memory_context_and_memory_store(
            self,
            platform=platform,
        )

    def _load_memory_context_and_memory_store(
        self,
        *,
        platform: str | None = None,
    ) -> tuple[str | None, Any | None]:
        """Load always-on GUI policy memory for direct injection into the prompt.

        Policies must always be present regardless of task relevance, so they are loaded
        in full without embedding-based search filtering.  Other memory types are
        retrieved selectively by ``_build_memory_retriever``.
        """
        from opengui.memory.store import MemoryStore
        from opengui.memory.types import MemoryType

        try:
            memory_store = MemoryStore(DEFAULT_OPENGUI_MEMORY_DIR)
            lines: list[str] = []

            policy_entries = memory_store.list_all(memory_type=MemoryType.POLICY)
            lines.extend(f"- [policy] {entry.content}" for entry in policy_entries)

            del platform

            return ("\n".join(lines) if lines else None), memory_store
        except Exception:
            logger.warning("Failed to load GUI memory", exc_info=True)
            return None, None

    def _build_embedding_adapter(self) -> NanobotEmbeddingAdapter:
        """Build a NanobotEmbeddingAdapter backed by litellm.aembedding.

        The embed function:
        - resolves the configured model name via the provider when supported
        - passes provider credentials to litellm so the correct API is used
        - normalises the response into an (n, dim) float32 numpy array
        """
        resolved_model = self._resolve_embedding_model()

        embedding_model = resolved_model

        direct_client = getattr(self._provider, "_client", None)
        if direct_client is not None and hasattr(direct_client, "embeddings"):
            direct_model = self._normalize_direct_embedding_model(embedding_model)

            async def _embed_direct(texts: list[str]) -> np.ndarray:
                async def _request_batch(batch: list[str]) -> list[list[float]]:
                    response = await direct_client.embeddings.create(
                        model=direct_model,
                        input=batch,
                    )
                    return [item.embedding for item in response.data]

                return await self._embed_texts_in_batches(texts, _request_batch)

            return NanobotEmbeddingAdapter(_embed_direct)

        provider = self._provider

        async def _embed(texts: list[str]) -> np.ndarray:
            async def _request_batch(batch: list[str]) -> list[list[float]]:
                # Collect provider credentials — only pass keys that are truthy to avoid
                # sending empty strings that some backends reject.
                kwargs: dict[str, Any] = {"model": resolved_model, "input": batch}
                # OpenAI-compatible embedding endpoints (including DashScope compatible-mode)
                # expect an explicit encoding_format. LiteLLM may default to an unsupported
                # format for some providers if omitted.
                kwargs["encoding_format"] = "float"
                api_key = getattr(provider, "api_key", None)
                if api_key:
                    kwargs["api_key"] = api_key
                api_base = getattr(provider, "api_base", None)
                if not api_base:
                    api_base = self._default_embedding_api_base(resolved_model)
                if api_base:
                    kwargs["api_base"] = api_base
                extra_headers = getattr(provider, "extra_headers", None)
                if extra_headers:
                    kwargs["extra_headers"] = extra_headers

                response = await litellm.aembedding(**kwargs)
                vectors: list[list[float]] = []
                for item in response.data:
                    if isinstance(item, dict):
                        embedding = item.get("embedding")
                    else:
                        embedding = getattr(item, "embedding", None)
                    if embedding is None:
                        raise ValueError("Embedding response item missing 'embedding' field")
                    vectors.append(embedding)
                return vectors

            return await self._embed_texts_in_batches(texts, _request_batch)

        return NanobotEmbeddingAdapter(_embed)

    def _resolve_embedding_model(self) -> str:
        embedding_model = self._gui_config.embedding_model
        if not embedding_model:
            raise ValueError("embedding model is required to build embedding adapter")

        resolve = getattr(self._provider, "_resolve_model", None)
        resolved = resolve(embedding_model) if callable(resolve) else embedding_model

        # DashScope's compatible-mode endpoint is OpenAI-style under LiteLLM.
        # When users configure bare model names like "text-embedding-v4", normalize
        # to "openai/<model>" so provider routing is deterministic.
        if (
            isinstance(resolved, str)
            and "/" not in resolved
            and (self._gui_config.provider or "").strip().lower() == "dashscope"
        ):
            return f"openai/{resolved}"
        return resolved

    def _resolve_embedding_signature(self) -> str | None:
        embedding_model = self._gui_config.embedding_model
        if not embedding_model:
            return None

        resolved_model = self._resolve_embedding_model()
        direct_client = getattr(self._provider, "_client", None)
        if direct_client is not None and hasattr(direct_client, "embeddings"):
            resolved_model = self._normalize_direct_embedding_model(resolved_model)

        gateway = getattr(self._provider, "_gateway", None)
        provider_name = (
            getattr(gateway, "name", None)
            or getattr(gateway, "litellm_prefix", None)
            or self._provider.__class__.__name__
        )
        api_base = getattr(self._provider, "api_base", None)
        if not api_base:
            api_base = getattr(self._provider, "_api_base", None)
        if not api_base:
            api_base = self._default_embedding_api_base(resolved_model)

        parts = [str(provider_name)]
        if api_base:
            parts.append(str(api_base))
        parts.append(resolved_model)
        return "|".join(parts)

    def _default_embedding_api_base(self, resolved_model: str) -> str | None:
        """Return provider-specific fallback API base for embedding requests."""
        provider_name = (self._gui_config.provider or "").strip().lower()
        if provider_name == "dashscope" and resolved_model.startswith("openai/"):
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        return None

    async def _embed_texts_in_batches(
        self,
        texts: list[str],
        request_batch: Callable[[list[str]], Awaitable[list[list[float]]]],
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBEDDING_BATCH_SIZE):
            batch = texts[start : start + _EMBEDDING_BATCH_SIZE]
            vectors.extend(await request_batch(batch))
        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _normalize_direct_embedding_model(model: str) -> str:
        """Strip LiteLLM-style provider prefixes for direct OpenAI-compatible clients."""
        if "/" not in model:
            return model
        return model.split("/", 1)[1]

    def _build_backend(self, backend_name: str) -> Any:
        if backend_name == "adb":
            from opengui.backends.adb import AdbBackend

            return AdbBackend(
                serial=self._gui_config.adb.serial,
                scrcpy_max_fps=self._gui_config.scrcpy.max_fps,
                scrcpy_jpeg_quality=self._gui_config.scrcpy.jpeg_quality,
                scrcpy_frame_timeout_ms=self._gui_config.scrcpy.frame_timeout_ms,
                scrcpy_max_frame_age_ms=self._gui_config.scrcpy.max_frame_age_ms,
                on_jpeg_frame=self._gui_frame_callback,
            )

        if backend_name == "ios":
            from opengui.backends.ios_wda import WdaBackend

            return WdaBackend(wda_url=self._gui_config.ios.wda_url)

        if backend_name == "hdc":
            from opengui.backends.hdc import HdcBackend

            return HdcBackend(serial=self._gui_config.hdc.serial)

        if backend_name == "dry-run":
            from opengui.backends.dry_run import DryRunBackend

            return DryRunBackend()

        if backend_name == "local":
            from opengui.backends.desktop import LocalDesktopBackend

            return LocalDesktopBackend()

        raise ValueError(f"Unsupported GUI backend: {backend_name}")

    def _build_isolated_display_manager(self, probe: Any) -> Any:
        if probe.backend_name == "xvfb":
            from opengui.backends.displays.xvfb import XvfbDisplayManager

            display_num = self._gui_config.display_num if self._gui_config.display_num is not None else 99
            return XvfbDisplayManager(
                display_num=display_num,
                width=self._gui_config.display_width,
                height=self._gui_config.display_height,
            )

        if probe.backend_name == "cgvirtualdisplay":
            from opengui.backends.displays.cgvirtualdisplay import CGVirtualDisplayManager

            return CGVirtualDisplayManager(
                width=self._gui_config.display_width,
                height=self._gui_config.display_height,
            )

        if probe.backend_name == "windows_isolated_desktop":
            from opengui.backends.displays.win32desktop import Win32DesktopManager

            return Win32DesktopManager(
                width=self._gui_config.display_width,
                height=self._gui_config.display_height,
            )

        raise RuntimeError(f"Unsupported isolated backend: {probe.backend_name}")

    def _resolve_probe_target_app_class(
        self,
        backend: str | None,
        target_app_class: str | None,
        *,
        sys_platform: str | None = None,
    ) -> str | None:
        if not self._gui_config.background:
            return None
        if (sys_platform or sys.platform) != "win32":
            return None
        if (backend or self._gui_config.backend) != "local":
            return None
        return target_app_class or "classic-win32"

    def _get_skill_library(
        self,
        platform: str,
        *,
        embedding_signature: str | None = None,
    ) -> Any:
        if platform not in self._skill_libraries:
            from opengui.skills.library import SkillLibrary

            self._skill_libraries[platform] = SkillLibrary(
                store_dir=get_gui_skill_store_root(self._workspace),
                embedding_provider=self._embedding_adapter,
                merge_llm=self._llm_adapter,
                embedding_signature=embedding_signature,
            )
        return self._skill_libraries[platform]

    def _refresh_cached_skill_stores(self) -> None:
        for cached in getattr(self, "_skill_libraries", {}).values():
            refresh_if_stale = getattr(cached, "refresh_if_stale", None)
            if callable(refresh_if_stale):
                refresh_if_stale()

    def _make_run_dir(self) -> Path:
        runs_root = self._workspace / self._gui_config.artifacts_dir
        while True:
            run_dir = runs_root / datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
            try:
                run_dir.mkdir(parents=True, exist_ok=False)
                return run_dir
            except FileExistsError:
                continue

    @staticmethod
    def _resolve_trace_path(recorder_path: Path | None, agent_trace_path: str | None) -> Path | None:
        if recorder_path is not None and recorder_path.exists():
            return recorder_path

        if not agent_trace_path:
            return None

        candidate = Path(agent_trace_path)
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            matches = sorted(candidate.glob("**/*.jsonl"))
            if matches:
                return matches[0]
        return None

    @staticmethod
    def _background_json_failure(summary: str) -> str:
        return json.dumps(
            {
                "success": False,
                "summary": summary,
                "trace_path": None,
                "steps_taken": 0,
                "error": None,
            },
            ensure_ascii=False,
        )

    def _build_intervention_handler(self, active_backend: Any, task: str) -> InterventionHandler:
        return _GuiToolInterventionHandler(active_backend=active_backend, task=task)

    async def shutdown(self) -> None:
        shutdown = getattr(self._backend, "shutdown", None)
        if callable(shutdown):
            await shutdown()

    async def _shutdown_android_backend(self, backend: Any) -> None:
        if getattr(backend, "platform", None) != "android":
            return
        shutdown = getattr(backend, "shutdown", None)
        if not callable(shutdown):
            return
        result = shutdown()
        if inspect.isawaitable(result):
            await result


class _GuiToolInterventionHandler:
    def __init__(self, *, active_backend: Any, task: str) -> None:
        self._active_backend = active_backend
        self._task = task

    async def request_intervention(
        self,
        request: InterventionRequest,
    ) -> InterventionResolution:
        payload = {
            "task": self._task,
            "reason": request.reason,
            "target": self._resolve_target(request),
        }
        scrubbed = self._scrub_payload(payload)
        logger.warning(
            "gui intervention requested: task=%s reason=%s target=%s",
            scrubbed["task"],
            scrubbed["reason"],
            json.dumps(scrubbed["target"], ensure_ascii=False, sort_keys=True),
        )
        return InterventionResolution(
            resume_confirmed=False,
            note="resume_not_confirmed",
        )

    def _resolve_target(self, request: InterventionRequest) -> dict[str, Any]:
        target = dict(request.target)
        get_target = getattr(self._active_backend, "get_intervention_target", None)
        if callable(get_target):
            backend_target = get_target() or {}
            if isinstance(backend_target, dict):
                target.update(backend_target)
        return {
            key: value
            for key, value in target.items()
            if key in _SAFE_INTERVENTION_TARGET_KEYS
        }

    @staticmethod
    def _scrub_payload(payload: dict[str, Any]) -> dict[str, Any]:
        return GuiAgent._scrub_for_log(payload)
