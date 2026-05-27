"""Regression tests for GUI memory split: policy entries to GUI agent.

Verifies:
- GuiSubagentTool._load_policy_context returns formatted policy text
- GuiAgent uses policy_context directly (no search) when provided
- GuiAgent falls back to memory_retriever when policy_context is None
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_gui_tool_load_policy_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_policy_context returns formatted policy lines from the memory store."""
    from opengui.memory.store import MemoryStore
    from opengui.memory.types import MemoryEntry, MemoryType

    # Build a real MemoryStore with known policy entries
    store_dir = tmp_path / "memory"
    store_dir.mkdir()
    store = MemoryStore(store_dir)
    store.add(
        MemoryEntry(
            entry_id="pol-001",
            memory_type=MemoryType.POLICY,
            platform="android",
            content="Never accept calls without user permission.",
        )
    )
    store.add(
        MemoryEntry(
            entry_id="pol-002",
            memory_type=MemoryType.POLICY,
            platform="android",
            content="Do not send messages to unknown contacts.",
        )
    )
    store.save()

    # Monkeypatch DEFAULT_OPENGUI_MEMORY_DIR inside the gui tool module
    import nanobot.agent.tools.gui as gui_module

    monkeypatch.setattr(gui_module, "DEFAULT_OPENGUI_MEMORY_DIR", store_dir)

    # Call _load_policy_context as an unbound method (self is unused in this method)
    result = gui_module.GuiSubagentTool._load_policy_context(object())

    assert result is not None, "Expected non-None policy context"
    assert "Never accept calls without user permission." in result
    assert "Do not send messages to unknown contacts." in result
    # Each entry is prefixed with "- "
    for line in result.splitlines():
        assert line.startswith("- "), f"Expected '- ' prefix, got: {line!r}"


# ---------------------------------------------------------------------------
def test_gui_tool_loads_policy_and_platform_os_guides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opengui.memory.store import MemoryStore
    from opengui.memory.types import MemoryEntry, MemoryType

    store_dir = tmp_path / "memory"
    store = MemoryStore(store_dir)
    store.add(MemoryEntry(
        entry_id="policy-001",
        memory_type=MemoryType.POLICY,
        platform="ios",
        content="Do not approve sensitive permissions automatically.",
    ))
    store.add(MemoryEntry(
        entry_id="ios-os-001",
        memory_type=MemoryType.OS_GUIDE,
        platform="ios",
        content="iOS Settings search field is at the bottom on the Settings home page.",
    ))
    store.add(MemoryEntry(
        entry_id="android-os-001",
        memory_type=MemoryType.OS_GUIDE,
        platform="android",
        content="Android quick settings open from the top shade.",
    ))

    import nanobot.agent.tools.gui as gui_module

    monkeypatch.setattr(gui_module, "DEFAULT_OPENGUI_MEMORY_DIR", store_dir)

    context, memory_store = gui_module.GuiSubagentTool._load_memory_context_and_memory_store(
        object(),
        platform="ios",
    )

    assert memory_store is not None
    assert context is not None
    assert "[policy]" in context
    assert "Do not approve sensitive permissions automatically." in context
    assert "[os]" in context
    assert "iOS Settings search field is at the bottom" in context
    assert "Android quick settings" not in context


# ---------------------------------------------------------------------------
# Test 5: GuiAgent uses policy_context directly (no retriever search)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gui_agent_uses_policy_context_directly(tmp_path: Path) -> None:
    """GuiAgent._retrieve_memory returns policy_context directly without calling the retriever."""
    from opengui.agent import GuiAgent
    from opengui.backends.dry_run import DryRunBackend
    from opengui.trajectory.recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="test task")
    recorder.start()

    # Provide a mock retriever — it must NOT be called
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(side_effect=AssertionError("retriever.search must not be called"))

    agent = GuiAgent(
        llm=MagicMock(),
        backend=DryRunBackend(),
        trajectory_recorder=recorder,
        policy_context="test policy line",
        memory_retriever=mock_retriever,
    )

    result = await agent._retrieve_memory("any task")

    assert result == "test policy line", f"Expected direct policy context, got: {result!r}"
    mock_retriever.search.assert_not_called()


# ---------------------------------------------------------------------------
# Test 6: GuiAgent falls back to retriever when policy_context is None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gui_agent_falls_back_to_retriever_when_no_policy_context(tmp_path: Path) -> None:
    """GuiAgent._retrieve_memory uses memory_retriever search when policy_context is None."""
    from opengui.agent import GuiAgent
    from opengui.backends.dry_run import DryRunBackend
    from opengui.trajectory.recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder(output_dir=tmp_path / "traj", task="test task")
    recorder.start()

    # Provide a mock retriever that returns no results (simulates search with no hits)
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(return_value=[])

    agent = GuiAgent(
        llm=MagicMock(),
        backend=DryRunBackend(),
        trajectory_recorder=recorder,
        policy_context=None,
        memory_retriever=mock_retriever,
    )

    result = await agent._retrieve_memory("find wifi settings")

    # With no hits, result should be None and the retriever should have been called
    assert result is None, f"Expected None for no search hits, got: {result!r}"
    mock_retriever.search.assert_called()
