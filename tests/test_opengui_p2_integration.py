"""Phase 2 integration tests — OpenGUI skills, trajectory, and full flow.

Covers: AGENT-05, SKILL-08, TRAJ-03, TEST-05.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from opengui.agent import GuiAgent
from opengui.backends.dry_run import DryRunBackend
from opengui.interfaces import LLMResponse, ToolCall
from opengui.memory.retrieval import MemoryRetriever
from opengui.memory.store import MemoryStore
from opengui.memory.types import MemoryEntry, MemoryType
from opengui.skills.data import Skill, SkillStep
from opengui.skills.library import SkillLibrary
from opengui.trajectory.recorder import TrajectoryRecorder

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Deterministic bag-of-token embedder for semantic-ish test retrieval."""

    DIM = 32

    async def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                slot = int.from_bytes(digest[:4], "big") % self.DIM
                vecs[i, slot] += 1.0
            norm = float(np.linalg.norm(vecs[i]))
            if norm > 0:
                vecs[i] /= norm
        return vecs


class _RecordingLLM:
    """Mock LLM that returns scripted responses and records all calls."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, tool_choice=None) -> LLMResponse:
        self.calls.append(copy.deepcopy(messages))
        if not self._responses:
            raise AssertionError("No scripted responses left")
        return self._responses.pop(0)


class _AndroidDryRunBackend(DryRunBackend):
    @property
    def platform(self) -> str:
        return "android"


def _done_response(call_id: str = "tc_done") -> LLMResponse:
    return LLMResponse(
        content="Action: Task complete",
        tool_calls=[ToolCall(
            id=call_id, name="computer_use",
            arguments={"action_type": "done", "status": "success"},
        )],
    )


def _wait_response(call_id: str = "tc_wait") -> LLMResponse:
    return LLMResponse(
        content="Action: waiting",
        tool_calls=[ToolCall(
            id=call_id, name="computer_use",
            arguments={"action_type": "wait", "duration_ms": 1},
        )],
    )


def _make_recorder(tmp_path: Path, task: str = "test") -> TrajectoryRecorder:
    return TrajectoryRecorder(output_dir=tmp_path / "traj", task=task)


def _make_memory_entry(
    entry_id: str, content: str, memory_type: MemoryType = MemoryType.APP_GUIDE,
) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id, memory_type=memory_type,
        platform="android", content=content,
    )


# ---------------------------------------------------------------------------
# AGENT-04 / MEM-05: Memory injected into system prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_injected_into_system_prompt(tmp_path: Path) -> None:
    """GuiAgent.run() should inject memory entries into the system prompt."""
    store = MemoryStore(tmp_path / "mem")
    store.add(_make_memory_entry("m1", "Settings is the gear icon on home screen"))
    store.add(_make_memory_entry("pol1", "Always confirm before deleting", MemoryType.POLICY))
    store.save()

    retriever = MemoryRetriever(embedding_provider=_FakeEmbedder())
    await retriever.index(store.list_all())

    llm = _RecordingLLM([_done_response()])
    agent = GuiAgent(
        llm, DryRunBackend(),
        trajectory_recorder=_make_recorder(tmp_path, "Open Settings"),
        memory_retriever=retriever,
        artifacts_root=tmp_path / "runs", max_steps=1,
    )

    result = await agent.run("Open Settings", max_retries=1)
    assert result.success

    # System prompt (first message of first LLM call) should contain memory content
    system_msg = llm.calls[0][0]["content"]
    assert "Relevant Knowledge" in system_msg
    assert "Always confirm before deleting" in system_msg


# ---------------------------------------------------------------------------
# AGENT-05 / SKILL-08: Skill path chosen above threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_path_chosen_above_threshold(tmp_path: Path) -> None:
    """When a matching skill exists above threshold, GuiAgent should use SkillExecutor path."""
    embedder = _FakeEmbedder()
    lib = SkillLibrary(store_dir=tmp_path / "skills", embedding_provider=embedder)
    skill = Skill(
        skill_id="wifi-toggle", name="Toggle Wi-Fi",
        description="Toggle Wi-Fi in Settings", app="com.android.settings",
        platform="android",
        steps=(SkillStep(action_type="tap", target="Wi-Fi toggle"),),
    )
    lib.add(skill)
    await lib._rebuild_index()

    # Mock executor that returns success
    mock_executor = AsyncMock()
    mock_exec_result = AsyncMock()
    mock_exec_result.state.value = "succeeded"
    mock_executor.execute = AsyncMock(return_value=mock_exec_result)

    llm = _RecordingLLM([
        LLMResponse(content='{"applicable": true}'),
        _done_response(),
    ])
    recorder = _make_recorder(tmp_path, "Turn on Wi-Fi")
    agent = GuiAgent(
        llm, _AndroidDryRunBackend(),
        trajectory_recorder=recorder,
        skill_library=lib, skill_executor=mock_executor,
        skill_threshold=0.3,  # low threshold to ensure match
        artifacts_root=tmp_path / "runs", max_steps=1,
    )

    result = await agent.run("Toggle Wi-Fi", max_retries=1)
    assert result.success

    # Check trajectory has a SKILL phase change
    traj_path = recorder.path
    assert traj_path is not None and traj_path.exists()
    events = [json.loads(line) for line in traj_path.read_text().splitlines()]
    phase_changes = [e for e in events if e.get("type") == "phase_change"]
    skill_phases = [e for e in phase_changes if e.get("to_phase") == "skill"]
    assert len(skill_phases) >= 1


# ---------------------------------------------------------------------------
# AGENT-05: Free explore when no skill match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_free_explore_when_no_skill_match(tmp_path: Path) -> None:
    """When no skill matches, GuiAgent should use free exploration."""
    embedder = _FakeEmbedder()
    lib = SkillLibrary(store_dir=tmp_path / "skills", embedding_provider=embedder)
    # Add an unrelated skill
    lib.add(Skill(
        skill_id="unrelated", name="Send Email",
        description="Send an email via Gmail", app="com.google.android.gm",
        platform="android",
    ))
    await lib._rebuild_index()

    llm = _RecordingLLM([_done_response()])
    recorder = _make_recorder(tmp_path, "Open calculator")
    agent = GuiAgent(
        llm, DryRunBackend(),
        trajectory_recorder=recorder,
        skill_library=lib, skill_threshold=0.9,  # high threshold — no match
        artifacts_root=tmp_path / "runs", max_steps=1,
    )

    result = await agent.run("Open calculator", max_retries=1)
    assert result.success

    # No SKILL phase in trajectory
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    skill_phases = [
        e for e in events
        if e.get("type") == "phase_change" and e.get("to_phase") == "skill"
    ]
    assert len(skill_phases) == 0


# ---------------------------------------------------------------------------
# AGENT-06 / TRAJ-03: Trajectory recorded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_trajectory_recorded_on_run(tmp_path: Path) -> None:
    """Every agent run should produce a JSONL trajectory with metadata/step/result."""
    llm = _RecordingLLM([_wait_response("w1"), _done_response()])
    recorder = _make_recorder(tmp_path, "Open Settings")
    agent = GuiAgent(
        llm, DryRunBackend(),
        trajectory_recorder=recorder,
        artifacts_root=tmp_path / "runs", max_steps=5,
    )

    result = await agent.run("Open Settings", max_retries=1)
    assert result.success

    assert recorder.path is not None and recorder.path.exists()
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]

    types = [e["type"] for e in events]
    assert types[0] == "metadata"
    assert "step" in types
    assert types[-1] == "result"
    assert events[-1]["success"] is True


# ---------------------------------------------------------------------------
# TEST-05: Full flow with mock LLM + memory + skills
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_flow_with_mock_llm(tmp_path: Path) -> None:
    """Full flow: DryRunBackend + mock LLM + pre-seeded memory + skill → completion."""
    # Seed memory
    store = MemoryStore(tmp_path / "mem")
    store.add(_make_memory_entry("m1", "Settings is the gear icon"))
    store.add(_make_memory_entry("pol1", "Confirm before destructive actions", MemoryType.POLICY))
    store.save()

    retriever = MemoryRetriever(embedding_provider=_FakeEmbedder())
    await retriever.index(store.list_all())

    # Seed skill library
    embedder = _FakeEmbedder()
    lib = SkillLibrary(store_dir=tmp_path / "skills", embedding_provider=embedder)
    skill = Skill(
        skill_id="open-settings", name="Open Settings",
        description="Open the Settings app", app="com.android.settings",
        platform="android",
        steps=(SkillStep(action_type="open_app", target="com.android.settings"),),
    )
    lib.add(skill)
    await lib._rebuild_index()

    mock_executor = AsyncMock()
    mock_exec_result = AsyncMock()
    mock_exec_result.state.value = "succeeded"
    mock_executor.execute = AsyncMock(return_value=mock_exec_result)

    # GuiAgent LLM: returns done immediately (after skill execution)
    agent_llm = _RecordingLLM([_done_response()])
    recorder = _make_recorder(tmp_path, "Open Settings")

    agent = GuiAgent(
        agent_llm, DryRunBackend(),
        trajectory_recorder=recorder,
        memory_retriever=retriever, skill_library=lib,
        skill_executor=mock_executor, skill_threshold=0.3,
        artifacts_root=tmp_path / "runs", max_steps=3,
    )

    result = await agent.run("Open Settings", max_retries=1)
    assert result.success

    # Memory appeared in system prompt
    system_msg = agent_llm.calls[0][0]["content"]
    assert "Settings is the gear icon" in system_msg
    assert "Confirm before destructive actions" in system_msg

    # Trajectory file has correct events
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    types = [e["type"] for e in events]
    assert "metadata" in types
    assert "result" in types
