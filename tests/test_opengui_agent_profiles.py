from __future__ import annotations

from opengui.agent_profiles import normalize_profile_response
from opengui.interfaces import LLMResponse


def test_qwen3vl_answer_preserves_text_in_done_action() -> None:
    response = LLMResponse(
        content=(
            "Thought: I found the requested number.\n"
            "Action: answer with the integer.\n"
            '<tool_call>{"name":"mobile_use","arguments":{"action":"answer","text":"28"}}</tool_call>'
        )
    )

    normalized = normalize_profile_response("qwen3vl", response)

    assert normalized.tool_calls is not None
    assert normalized.tool_calls[0].arguments == {
        "action_type": "done",
        "status": "success",
        "text": "28",
    }
