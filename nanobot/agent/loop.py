"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import re
import os
import time
from contextlib import AsyncExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.capabilities import CapabilityCatalogBuilder, PlanningContext
from nanobot.agent.context import ContextBuilder
from nanobot.agent.execution_policy import (
    ExecutionDecision,
    PolicyNodeContext,
    PolicyTraceWriter,
    RuleExecutionPolicy,
)
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.planning_memory import PlanningMemoryHintExtractor
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.command import CommandContext, CommandRouter, register_builtin_commands
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, GuiConfig, WebSearchConfig
    from nanobot.cron.service import CronService


# ---------------------------------------------------------------------------
# Complexity assessment tool used by _needs_planning to decide whether a
# task should be decomposed via TaskPlanner before execution.
# ---------------------------------------------------------------------------

_COMPLEXITY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "assess_complexity",
        "description": (
            "Determine if a task requires GUI operations that need multi-step planning."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "needs_planning": {
                    "type": "boolean",
                    "description": (
                        "True ONLY if the task requires GUI operations - screen taps, "
                        "app navigation, interacting with device UI elements, opening "
                        "or switching between apps on a device screen. "
                        "False for tasks that can be completed with shell commands, "
                        "file operations, web searches, API calls, or any non-GUI tool. "
                        "Pure tool/shell tasks NEVER need planning."
                    ),
                }
            },
            "required": ["needs_planning"],
        },
    },
}


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
        gui_config: "GuiConfig | None" = None,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.subagent_provider, self.subagent_model = self._resolve_subagent_runtime(
            provider=provider,
            default_model=self.model,
        )
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self._start_time = time.time()
        self._last_usage: dict[str, int] = {}
        self._gui_config = gui_config
        self._session_retry_count: dict[str, int] = {}
        self._execution_policy = RuleExecutionPolicy.from_env()
        self._policy_trace_writer = (
            PolicyTraceWriter(workspace) if self._execution_policy.enabled else None
        )
        if self._execution_policy.enabled and self._policy_trace_writer is not None:
            logger.info("Execution policy enabled; traces -> {}", self._policy_trace_writer.path)

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=self.subagent_provider,
            workspace=workspace,
            bus=bus,
            model=self.subagent_model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        # NANOBOT_MAX_CONCURRENT_REQUESTS: <=0 means unlimited; default 3.
        _max = int(os.environ.get("NANOBOT_MAX_CONCURRENT_REQUESTS", "3"))
        self._concurrency_gate: asyncio.Semaphore | None = (
            asyncio.Semaphore(_max) if _max > 0 else None
        )
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
            max_completion_tokens=provider.generation.max_tokens,
        )
        self._register_default_tools()
        self.commands = CommandRouter()
        register_builtin_commands(self.commands)

    @staticmethod
    def _env_first_non_empty(*names: str) -> str | None:
        for name in names:
            value = os.environ.get(name)
            if not value:
                continue
            value = value.strip()
            if value:
                return value
        return None

    def _resolve_subagent_runtime(
        self,
        *,
        provider: LLMProvider,
        default_model: str,
    ) -> tuple[LLMProvider, str]:
        """Resolve optional subagent-only model/provider overrides from env vars."""
        subagent_model = self._env_first_non_empty(
            "NANOBOT_SUBAGENT_MODEL",
            "OPENAI_MODEL",
        ) or default_model
        subagent_api_base = self._env_first_non_empty(
            "NANOBOT_SUBAGENT_API_BASE",
            "OPENAI_API_BASE",
        )
        subagent_api_key = self._env_first_non_empty(
            "NANOBOT_SUBAGENT_API_KEY",
            "OPENAI_API_KEY",
        ) or "dummy"

        if not subagent_api_base:
            return provider, subagent_model

        from nanobot.providers.custom_provider import CustomProvider

        subagent_provider = CustomProvider(
            api_key=subagent_api_key,
            api_base=subagent_api_base,
            default_model=subagent_model,
        )
        if hasattr(provider, "generation"):
            subagent_provider.generation = provider.generation
        logger.info(
            "Subagent runtime override enabled: model={} api_base={}",
            subagent_model,
            subagent_api_base,
        )
        return subagent_provider, subagent_model

    def _build_policy_node_context(self, *, task: str, session_key: str, history_len: int) -> PolicyNodeContext:
        available_tools = tuple(sorted(self.tools.tool_names))
        retry_count = self._session_retry_count.get(session_key, 0)
        return PolicyNodeContext(
            task=task,
            has_gui=self._gui_config is not None,
            available_tools=available_tools,
            history_len=history_len,
            retry_count=retry_count,
            tool_error_signal=retry_count > 0,
            mismatch_signal=False,
        )

    def _record_policy_decision(
        self,
        *,
        session_key: str,
        msg: InboundMessage,
        node_context: PolicyNodeContext,
        decision: ExecutionDecision,
    ) -> None:
        if self._policy_trace_writer is None:
            return
        try:
            self._policy_trace_writer.append(
                session_key=session_key,
                channel=msg.channel,
                chat_id=msg.chat_id,
                node_context=node_context,
                decision=decision,
            )
        except Exception:
            logger.warning("Failed to append execution-policy trace", exc_info=True)

    @staticmethod
    def _output_has_failure_signal(output: str | None) -> bool:
        if not output:
            return False
        lowered = output.lower()
        failure_markers = (
            "error",
            "failed",
            "unable",
            "exception",
            "timeout",
            "timed out",
            "permission denied",
            "not found",
        )
        return any(marker in lowered for marker in failure_markers)

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        if self.exec_config.enable:
            self.tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))
        if self._gui_config is not None:
            from nanobot.agent.tools.gui import GuiSubagentTool

            self.tools.register(
                GuiSubagentTool(
                    gui_config=self._gui_config,
                    provider=self.subagent_provider,
                    model=self.subagent_model,
                    workspace=self.workspace,
                )
            )

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think></think> blocks that some models embed in content."""
        if not text:
            return None
        from nanobot.utils.helpers import strip_think
        return strip_think(text) or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}...")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    @staticmethod
    def _format_plan_tree(node: Any, *, indent: int = 0) -> str:
        """Render a plan tree into a human-readable indented outline."""
        prefix = "  " * indent
        node_type = getattr(node, "node_type", "unknown")
        if node_type == "atom":
            capability = getattr(node, "capability", "unknown")
            instruction = getattr(node, "instruction", "")
            route_id = getattr(node, "route_id", None)
            route_reason = getattr(node, "route_reason", "")
            fallback_route_ids = tuple(getattr(node, "fallback_route_ids", ()) or ())

            line = f"{prefix}- {str(capability).upper()}: {instruction}"
            if route_id:
                line += f" via {route_id}"

            lines = [line]
            if route_reason:
                lines.append(f"{prefix}  why: {route_reason}")
            for fallback_route_id in fallback_route_ids:
                lines.append(f"{prefix}  fallback -> {fallback_route_id}")
            return "\n".join(lines)

        header = f"{prefix}{str(node_type).upper()}"
        children = getattr(node, "children", ()) or ()
        if not children:
            return header
        rendered_children = [
            AgentLoop._format_plan_tree(child, indent=indent + 1)
            for child in children
        ]
        return "\n".join([header, *rendered_children])

    @classmethod
    def _build_plan_preview(cls, tree: Any) -> str:
        """Render a user-facing plan preview message for the current channel."""
        return "执行计划预览：\n```text\n" + cls._format_plan_tree(tree) + "\n```"

    @staticmethod
    def _load_gui_memory_for_planner() -> str:
        """Load os_guide, app_guide, and icon_guide entries from the opengui MemoryStore.

        Returns a formatted string of guide entries for planner consumption, or an empty
        string when the memory directory does not exist or opengui is unavailable.
        Guide entries (not policy) are surfaced here so the planner can refine GUI task
        instructions with device and app navigation knowledge.
        """
        from nanobot.agent.tools.gui import DEFAULT_OPENGUI_MEMORY_DIR

        if not DEFAULT_OPENGUI_MEMORY_DIR.exists():
            return ""
        try:
            from opengui.memory.store import MemoryStore as GuiMemoryStore
            from opengui.memory.types import MemoryType

            gui_store = GuiMemoryStore(DEFAULT_OPENGUI_MEMORY_DIR)
            guide_entries = []
            for memory_type in (MemoryType.OS_GUIDE, MemoryType.APP_GUIDE, MemoryType.ICON_GUIDE):
                guide_entries.extend(gui_store.list_all(memory_type=memory_type))
            if not guide_entries:
                return ""
            lines: list[str] = []
            for entry in guide_entries:
                tag = entry.memory_type.value.upper()
                prefix = f"[{tag}]"
                if entry.app:
                    prefix += f" ({entry.app})"
                lines.append(f"- {prefix} {entry.content}")
            return "\n".join(lines)
        except Exception:
            logger.warning("Failed to load GUI guide memory for planner", exc_info=True)
            return ""

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        channel: str = "cli",
        chat_id: str = "direct",
        message_id: str | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop.

        *on_stream*: called with each content delta during streaming.
        *on_stream_end(resuming)*: called when a streaming session finishes.
        ``resuming=True`` means tool calls follow (spinner should restart);
        ``resuming=False`` means this is the final response.
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        # Wrap on_stream with stateful think-tag filter so downstream
        # consumers (CLI, channels) never see <think> blocks.
        _raw_stream = on_stream
        _stream_buf = ""

        async def _filtered_stream(delta: str) -> None:
            nonlocal _stream_buf
            from nanobot.utils.helpers import strip_think
            prev_clean = strip_think(_stream_buf)
            _stream_buf += delta
            new_clean = strip_think(_stream_buf)
            incremental = new_clean[len(prev_clean):]
            if incremental and _raw_stream:
                await _raw_stream(incremental)

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            if on_stream:
                response = await self.provider.chat_stream_with_retry(
                    messages=messages,
                    tools=tool_defs,
                    model=self.model,
                    on_content_delta=_filtered_stream,
                )
            else:
                response = await self.provider.chat_with_retry(
                    messages=messages,
                    tools=tool_defs,
                    model=self.model,
                )

            usage = response.usage or {}
            self._last_usage = {
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            }

            if response.has_tool_calls:
                if on_stream and on_stream_end:
                    await on_stream_end(resuming=True)
                    _stream_buf = ""

                if on_progress:
                    if not on_stream:
                        thought = self._strip_think(response.content)
                        if thought:
                            await on_progress(thought)
                    tool_hint = self._tool_hint(response.tool_calls)
                    tool_hint = self._strip_think(tool_hint)
                    await on_progress(tool_hint, tool_hint=True)

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                immediate_final: str | None = None
                for tc in response.tool_calls:
                    tools_used.append(tc.name)
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tc.name, args_str[:200])

                # Re-bind tool context right before execution so that
                # concurrent sessions don't clobber each other's routing.
                self._set_tool_context(channel, chat_id, message_id)

                # Execute all tool calls concurrently - the LLM batches
                # independent calls in a single response on purpose.
                # return_exceptions=True ensures all results are collected
                # even if one tool is cancelled or raises BaseException.
                results = await asyncio.gather(*(
                    self.tools.execute(tc.name, tc.arguments)
                    for tc in response.tool_calls
                ), return_exceptions=True)

                for tool_call, result in zip(response.tool_calls, results):
                    if isinstance(result, BaseException):
                        result = f"Error: {type(result).__name__}: {result}"
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    if len(response.tool_calls) == 1:
                        immediate_final = self._maybe_finalize_successful_gui_task(
                            tool_call.name,
                            result,
                        )

                if immediate_final is not None:
                    messages = self.context.add_assistant_message(messages, immediate_final)
                    final_content = immediate_final
                    break
            else:
                if on_stream and on_stream_end:
                    await on_stream_end(resuming=False)
                    _stream_buf = ""

                clean = self._strip_think(response.content)
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    @staticmethod
    def _maybe_finalize_successful_gui_task(tool_name: str, result: Any) -> str | None:
        if tool_name != "gui_task" or not isinstance(result, str):
            return None

        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict) or not payload.get("success"):
            return None

        candidate = payload.get("model_summary") or payload.get("summary") or "GUI task completed successfully."
        if not isinstance(candidate, str):
            candidate = str(candidate)
        candidate = candidate.strip()
        if not candidate:
            return "GUI task completed successfully."
        if candidate[-1] not in ".!?。！？":
            candidate += "。"
        return candidate

    async def _needs_planning(self, task: str) -> bool:
        """One LLM call to assess whether a task warrants multi-step decomposition.

        Uses the ``assess_complexity`` tool to force a structured Boolean response.
        Returns ``False`` on any parsing failure (safe default - never blocks
        execution on gate ambiguity).
        """
        direct_tools_summary = "; ".join(
            f"{d['function']['name']}: {d['function'].get('description', '')}"
            for d in self.tools.get_definitions()
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a task complexity assessor for a device automation agent. "
                    "Determine if the user's task requires GUI operations - interacting "
                    "with a device screen (tapping, swiping, typing into app UI, navigating "
                    "between apps, reading screen content). "
                    "ONLY return True when GUI interaction is needed.\n\n"
                    f"The agent has these direct tools (no planning needed): {direct_tools_summary}.\n"
                    "If the task can be accomplished entirely with these tools (shell commands, "
                    "file I/O, web search, web fetch), return False. "
                    "Planning is ONLY for tasks that require controlling a device screen."
                ),
            },
            {"role": "user", "content": f"Task: {task}"},
        ]
        response = await self.provider.chat_with_retry(
            messages=messages,
            tools=[_COMPLEXITY_TOOL],
            model=self.model,
        )
        if response.tool_calls:
            args = response.tool_calls[0].arguments
            if isinstance(args, str):
                import json as _json
                try:
                    args = _json.loads(args)
                except Exception:
                    return False
            return bool(args.get("needs_planning", False))
        return False  # safe default: no tool call -> treat as simple

    async def _plan_and_execute(
        self,
        task: str,
        *,
        channel: str | None = None,
        chat_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Decompose *task* via TaskPlanner and dispatch via TreeRouter.

        Lazy-imports TaskPlanner and TreeRouter to keep loop.py import overhead
        low (both modules are only needed when planning is actually triggered).

        A thin ``_GuiDispatchAdapter`` bridges GuiSubagentTool's ``execute()``
        interface (returns a JSON string) to the ``run()`` interface that
        ``TreeRouter._run_gui`` expects (returns an object with ``.success``,
        ``.summary``, ``.error``, ``.trace_path``).
        """
        from nanobot.agent.planner import TaskPlanner
        from nanobot.agent.router import RouterContext, TreeRouter

        class _GuiDispatchAdapter:
            """Bridges GuiSubagentTool.execute() to TreeRouter._run_gui's expected interface."""

            def __init__(self, tool: Any) -> None:
                self._tool = tool

            async def run(self, instruction: str, max_retries: int = 1) -> Any:
                import json as _json
                from dataclasses import dataclass

                result_json = await self._tool.execute(task=instruction)
                data = _json.loads(result_json)

                @dataclass
                class _GuiResult:
                    success: bool
                    summary: str
                    error: str | None
                    trace_path: str | None

                return _GuiResult(
                    success=data.get("success", False),
                    summary=data.get("summary", ""),
                    error=data.get("error"),
                    trace_path=data.get("trace_path"),
                )

        planner = TaskPlanner(llm=self.provider)
        raw_gui_tool = self.tools.get("gui_task")
        gui_backend = self._gui_config.backend if self._gui_config is not None else "local"
        catalog = CapabilityCatalogBuilder().build(
            tool_registry=self.tools,
            gui_available=raw_gui_tool is not None,
            exec_enabled=self.exec_config.enable,
            gui_backend=gui_backend,
        )
        memory_hints = PlanningMemoryHintExtractor(self.workspace).build(
            task=task,
            catalog=catalog,
        )
        gui_memory_context = self._load_gui_memory_for_planner()
        # Derive the stable planner route_id from the active backend.
        # "local" maps to "gui.desktop" (not "gui.local"); "dry-run" falls back to "gui.desktop".
        active_gui_route = f"gui.{gui_backend}" if gui_backend in ("adb", "desktop") else "gui.desktop"
        planning_context = PlanningContext(
            catalog=catalog,
            memory_hints=memory_hints,
            gui_memory_context=gui_memory_context,
            active_gui_route=active_gui_route,
        )
        tree = await planner.plan(task, planning_context=planning_context)
        logger.info("Decomposed plan:\n{}", self._format_plan_tree(tree))
        logger.debug("Decomposed plan (raw): {}", tree.to_dict())
        if channel is not None and chat_id is not None:
            preview_meta = dict(metadata or {})
            preview_meta["_progress"] = True
            preview_meta["_plan_preview"] = True
            await self.bus.publish_outbound(OutboundMessage(
                channel=channel,
                chat_id=chat_id,
                content=self._build_plan_preview(tree),
                metadata=preview_meta,
            ))

        gui_agent = _GuiDispatchAdapter(raw_gui_tool) if raw_gui_tool is not None else None

        ctx = RouterContext(
            task=task,
            gui_agent=gui_agent,
            tool_registry=self.tools,
            mcp_client=self.tools,  # MCP tools are also registered in the registry
        )

        router = TreeRouter(planner=planner, max_replans=2)
        result = await router.execute(tree, ctx)

        output: str
        if result.output:
            output = result.output
        elif result.success:
            output = "Done."
        else:
            output = result.error or "Task failed."

        return output, ["task_planner", "tree_router"], []

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                # Preserve real task cancellation so shutdown can complete cleanly.
                # Only ignore non-task CancelledError signals that may leak from integrations.
                if not self._running or asyncio.current_task().cancelling():
                    raise
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            raw = msg.content.strip()
            if self.commands.is_priority(raw):
                ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, loop=self)
                result = await self.commands.dispatch_priority(ctx)
                if result:
                    await self.bus.publish_outbound(result)
                continue
            task = asyncio.create_task(self._dispatch(msg))
            self._active_tasks.setdefault(msg.session_key, []).append(task)
            task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message: per-session serial, cross-session concurrent."""
        lock = self._session_locks.setdefault(msg.session_key, asyncio.Lock())
        gate = self._concurrency_gate or nullcontext()
        async with lock, gate:
            try:
                on_stream = on_stream_end = None
                if msg.metadata.get("_wants_stream"):
                    async def on_stream(delta: str) -> None:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content=delta, metadata={"_stream_delta": True},
                        ))

                    async def on_stream_end(*, resuming: bool = False) -> None:
                        await self.bus.publish_outbound(OutboundMessage(
                            channel=msg.channel, chat_id=msg.chat_id,
                            content="", metadata={"_stream_end": True, "_resuming": resuming},
                        ))

                response = await self._process_message(
                    msg, on_stream=on_stream, on_stream_end=on_stream_end,
                )
                if response is not None:
                    await self.bus.publish_outbound(response)
                else:
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role=current_role,
            )
            final_content, _, all_msgs = await self._run_agent_loop(
                messages, channel=channel, chat_id=chat_id,
                message_id=msg.metadata.get("message_id"),
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        raw = msg.content.strip()
        ctx = CommandContext(msg=msg, session=session, key=key, raw=raw, loop=self)
        if result := await self.commands.dispatch(ctx):
            return result

        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )
        task_text = msg.content.strip()
        policy_node: PolicyNodeContext | None = None
        policy_decision: ExecutionDecision | None = None
        if self._execution_policy.enabled and task_text:
            policy_node = self._build_policy_node_context(
                task=task_text,
                session_key=key,
                history_len=len(history),
            )
            policy_decision = self._execution_policy.decide(policy_node)
            self._record_policy_decision(
                session_key=key,
                msg=msg,
                node_context=policy_node,
                decision=policy_decision,
            )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        # Complexity gate: evaluate whether the task warrants multi-step
        # decomposition.  Gate is skipped for short messages (already handled
        # above via slash-command logic or trivially short content) and when no
        # GUI config is present (planning currently relies on GUI capability).
        use_planning = False
        if policy_decision is not None and policy_decision.mode in {"slow", "replan"}:
            use_planning = True
        elif policy_decision is not None and policy_decision.route in {"gui", "hybrid"} and len(task_text) >= 20:
            use_planning = True
        elif self._gui_config is not None and len(task_text) >= 20:
            try:
                use_planning = await self._needs_planning(task_text)
            except Exception:
                logger.debug(
                    "Complexity gate raised unexpectedly; falling back to direct agent loop"
                )

        if use_planning:
            final_content, _tools_used_plan, _ = await self._plan_and_execute(
                task_text,
                channel=msg.channel,
                chat_id=msg.chat_id,
                metadata=msg.metadata,
            )
            # Reconstruct a minimal messages list for _save_turn / session history.
            all_msgs = initial_messages + [
                {"role": "assistant", "content": final_content}
            ]
        else:
            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages,
                on_progress=on_progress or _bus_progress,
                on_stream=on_stream,
                on_stream_end=on_stream_end,
                channel=msg.channel, chat_id=msg.chat_id,
                message_id=msg.metadata.get("message_id"),
            )

        policy_recovered = False
        if (
            policy_decision is not None
            and policy_decision.mode in {"verify", "replan"}
            and not use_planning
            and self._output_has_failure_signal(final_content)
        ):
            try:
                recovered_content, _tools_used_plan, _ = await self._plan_and_execute(
                    task_text,
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    metadata=msg.metadata,
                )
                if recovered_content:
                    final_content = recovered_content
                    all_msgs = initial_messages + [{"role": "assistant", "content": final_content}]
                    policy_recovered = True
            except Exception:
                logger.debug(
                    "Policy recovery planning failed; keeping direct-loop output",
                    exc_info=True,
                )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        if self._output_has_failure_signal(final_content):
            self._session_retry_count[key] = self._session_retry_count.get(key, 0) + 1
        else:
            self._session_retry_count[key] = 0

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)

        meta = dict(msg.metadata or {})
        if on_stream is not None and not use_planning:
            meta["_streamed"] = True
        if policy_decision is not None:
            meta["_policy_route"] = policy_decision.route
            meta["_policy_mode"] = policy_decision.mode
            meta["_policy_risk"] = policy_decision.risk_level
            meta["_policy_source"] = policy_decision.source
            if policy_recovered:
                meta["_policy_recovered"] = True
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=meta,
        )

    @staticmethod
    def _image_placeholder(block: dict[str, Any]) -> dict[str, str]:
        """Convert an inline image block into a compact text placeholder."""
        path = (block.get("_meta") or {}).get("path", "")
        return {"type": "text", "text": f"[image: {path}]" if path else "[image]"}

    def _sanitize_persisted_blocks(
        self,
        content: list[dict[str, Any]],
        *,
        truncate_text: bool = False,
        drop_runtime: bool = False,
    ) -> list[dict[str, Any]]:
        """Strip volatile multimodal payloads before writing session history."""
        filtered: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                filtered.append(block)
                continue

            if (
                drop_runtime
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                and block["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
            ):
                continue

            if (
                block.get("type") == "image_url"
                and block.get("image_url", {}).get("url", "").startswith("data:image/")
            ):
                filtered.append(self._image_placeholder(block))
                continue

            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text = block["text"]
                if truncate_text and len(text) > self._TOOL_RESULT_MAX_CHARS:
                    text = text[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                filtered.append({**block, "text": text})
                continue

            filtered.append(block)

        return filtered

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages - they poison session context
            if role == "tool":
                if isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                    entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
                elif isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, truncate_text=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = self._sanitize_persisted_blocks(content, drop_runtime=True)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a message directly and return the outbound payload."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        return await self._process_message(
            msg, session_key=session_key, on_progress=on_progress,
            on_stream=on_stream, on_stream_end=on_stream_end,
        )

