from __future__ import annotations

from nanobot.agent.runner import AgentRunner


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _names(tools: list[dict]) -> list[str | None]:
    return [AgentRunner._tool_name(tool) for tool in tools]


def test_default_tool_filter_keeps_original_tools(monkeypatch) -> None:
    monkeypatch.delenv("NB_GUI_E2E_COMPACT", raising=False)
    monkeypatch.delenv("NB_FULLSYSTEM_TOOL_FILTER", raising=False)
    tools = [_tool("gui_task"), _tool("exec"), _tool("web_search")]

    assert AgentRunner._filter_llm_tools(tools) == tools


def test_compact_tool_filter_keeps_only_gui_task(monkeypatch) -> None:
    monkeypatch.setenv("NB_GUI_E2E_COMPACT", "1")
    monkeypatch.setenv("NB_FULLSYSTEM_TOOL_FILTER", "1")
    tools = [_tool("gui_task"), _tool("exec"), _tool("web_search"), _tool("mcp_demo")]

    filtered = AgentRunner._filter_llm_tools(tools)

    assert _names(filtered or []) == ["gui_task"]


def test_fullsystem_tool_filter_removes_development_tools(monkeypatch) -> None:
    monkeypatch.delenv("NB_GUI_E2E_COMPACT", raising=False)
    monkeypatch.setenv("NB_FULLSYSTEM_TOOL_FILTER", "1")
    tools = [
        _tool("gui_task"),
        _tool("web_search"),
        _tool("web_fetch"),
        _tool("ask_user"),
        _tool("message"),
        _tool("human_confirm"),
        _tool("mcp_calendar_search"),
        _tool("my_remote_tool"),
        _tool("exec"),
        _tool("spawn"),
        _tool("read_file"),
        _tool("write_file"),
        _tool("edit_file"),
        _tool("list_dir"),
        _tool("glob"),
        _tool("grep"),
        _tool("notebook_edit"),
        _tool("cron"),
        _tool("custom_api_lookup"),
    ]

    filtered = AgentRunner._filter_llm_tools(tools)
    names = set(_names(filtered or []))

    assert {
        "gui_task",
        "web_search",
        "web_fetch",
        "ask_user",
        "message",
        "human_confirm",
        "mcp_calendar_search",
        "my_remote_tool",
        "custom_api_lookup",
    } <= names
    assert {
        "exec",
        "spawn",
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "glob",
        "grep",
        "notebook_edit",
        "cron",
    }.isdisjoint(names)
