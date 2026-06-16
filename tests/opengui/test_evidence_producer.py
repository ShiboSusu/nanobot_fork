from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nanobot.agent.gui_task_schema import normalize_gui_task_request
from opengui.interfaces import LLMResponse


class _ProducerLLM:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self._step = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        del tools, tool_choice, model, max_tokens, reasoning_effort
        self.calls.append(messages)
        if self._step == 0:
            self._step += 1
            return LLMResponse(
                content='Thought: answer is visible\nAction: {"action_type":"done","status":"success"}'
            )
        return LLMResponse(content=json.dumps({"answer": "13800138000"}, ensure_ascii=False))


class _EvidenceBackend:
    platform = "android"

    async def preflight(self) -> None:
        return None

    async def list_apps(self) -> list[str]:
        return []

    async def observe(self, screenshot_path: Path, timeout: float = 5.0):
        from opengui.observation import Observation

        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot_path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        return Observation(
            screenshot_path=screenshot_path,
            screen_width=1,
            screen_height=1,
            foreground_app="contacts",
            platform="android",
            extra={"visible_text": ["联系人 小明", "电话 13800138000"]},
        )

    async def execute(self, action, timeout: float = 5.0) -> str:
        return "done"


@pytest.mark.asyncio
async def test_agent_produces_answer_candidates_from_latest_step_extra(tmp_path: Path) -> None:
    from opengui.agent import GuiAgent
    from opengui.trajectory.recorder import TrajectoryRecorder

    llm = _ProducerLLM()
    request = normalize_gui_task_request({"task": "打开通讯录，告诉我小明的电话"})
    agent = GuiAgent(
        llm=llm,
        backend=_EvidenceBackend(),
        trajectory_recorder=TrajectoryRecorder(tmp_path / "trajectory", task=request.task, platform="android"),
        artifacts_root=tmp_path,
        max_steps=1,
        evidence_request=request,
        s2_enabled=False,
        s2_trigger_on_done_missing_evidence=False,
    )

    result = await agent.run(request.task, max_retries=1)

    assert result.success is True
    assert result.answer_candidates == [
        {
            "key": "answer",
            "text": "13800138000",
            "type": "answer",
            "confidence": 1.0,
            "source": "llm_evidence_producer",
            "evidence_refs": ["latest_step"],
        }
    ]
    assert result.evidence["producer"] == "llm"
    assert "visible_text" in json.dumps(llm.calls[-1], ensure_ascii=False)
