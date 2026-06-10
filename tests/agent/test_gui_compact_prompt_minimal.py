from nanobot.agent.gui_experiment_policy import (
    GUI_E2E_COMPACT_GUI_TASK_DESCRIPTION,
    GUI_E2E_COMPACT_MAIN_PROMPT,
    is_gui_e2e_compact_mode,
)


def test_compact_prompt_is_short_and_gui_focused() -> None:
    assert "call gui_task" in GUI_E2E_COMPACT_MAIN_PROMPT
    assert "Do not answer app-visible or account-specific information from memory" in GUI_E2E_COMPACT_MAIN_PROMPT
    assert "Do not expose raw JSON" in GUI_E2E_COMPACT_MAIN_PROMPT
    assert "trace_path" in GUI_E2E_COMPACT_MAIN_PROMPT
    assert "Sensitive actions" in GUI_E2E_COMPACT_MAIN_PROMPT
    assert "confirmation" in GUI_E2E_COMPACT_MAIN_PROMPT
    assert len(GUI_E2E_COMPACT_MAIN_PROMPT) < 1200


def test_compact_gui_task_description_is_short() -> None:
    assert "answer_candidates" in GUI_E2E_COMPACT_GUI_TASK_DESCRIPTION
    assert "needs_human_confirm" in GUI_E2E_COMPACT_GUI_TASK_DESCRIPTION
    assert len(GUI_E2E_COMPACT_GUI_TASK_DESCRIPTION) < 500


def test_compact_env_flag(monkeypatch) -> None:
    monkeypatch.delenv("NB_GUI_E2E_COMPACT", raising=False)
    assert is_gui_e2e_compact_mode() is False
    monkeypatch.setenv("NB_GUI_E2E_COMPACT", "1")
    assert is_gui_e2e_compact_mode() is True
