"""
opengui.prompts.system
~~~~~~~~~~~~~~~~~~~~~~
System prompt templates for the GUI agent.
"""

from __future__ import annotations

import json
from typing import Any

from opengui.agent_profiles import (
    canonicalize_agent_profile,
    profile_tool_definition,
    prompt_contract_for_profile,
)

def build_system_prompt(
    *,
    platform: str = "unknown",
    coordinate_mode: str = "absolute",
    memory_context: str | None = None,
    skill_context: str | None = None,
    tool_definition: dict[str, Any] | None = None,
    installed_apps: list[str] | None = None,
    agent_profile: str = "default",
) -> str:
    """Build a Mobile-Agent-style system prompt while keeping native tool calls."""
    profile_name = canonicalize_agent_profile(agent_profile)
    prompt_contract = prompt_contract_for_profile(profile_name)
    tool_schema = json.dumps(
        tool_definition or profile_tool_definition(profile_name),
        ensure_ascii=False,
    )

    if coordinate_mode == "relative_999":
        coordinate_rules = (
            "- The screen uses a 1000x1000 relative coordinate grid.\n"
            "- For coordinate-based actions, use values in [0, 999] and set `relative=true`."
        )
    else:
        coordinate_rules = (
            "- Use absolute pixel coordinates based on the current screenshot and screen metadata.\n"
            "- Only use `relative=true` when the host explicitly requests relative coordinates."
        )

    sections = [
        "# Tools",
        "",
        "You may call one function to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
        tool_schema,
        "</tools>",
        "",
        "# Environment",
        "",
        "- You are operating on a GUI screen and can only act through the `computer_use` tool.",
        "- Some actions may take time to complete, so you may need to wait and observe again.",
        "- Use the latest screenshot as the source of truth.",
        "- Click the center of the intended UI element unless the task clearly requires an edge.",
        "- When opening apps from home screen, do not rely on icon color alone; prefer exact text labels or app search.",
        "- After opening an app, verify the foreground page belongs to the target app before continuing.",
        "- If the foreground app does not match the target app named by the task, prefer the `open` action for that target app, or return home and use app search. Do not repeatedly tap inside the wrong foreground app.",
        "- Treat user constraints as tiered constraints: key constraints (date/time/location/price cap/model) must be exact.",
        "- For size/quantity constraints, prefer exact match; small near-matches may be used only with explicit disclosure in the final response.",
        "- If only a far mismatch is available, do not substitute silently; ask for confirmation via request_intervention or report failure.",
        "- Do not call done(status=\"success\") unless key constraints are satisfied and any near-match is clearly disclosed.",
        coordinate_rules,
    ]

    if prompt_contract["environment"]:
        sections.extend([""])
        sections.extend(prompt_contract["environment"])

    if platform != "unknown":
        sections.extend(["", f"- Platform: {platform}"])

    if memory_context:
        sections.extend(["", "# Relevant Knowledge", "", memory_context])

    if installed_apps:
        if platform == "android":
            from opengui.skills.normalization import annotate_android_apps

            # annotate_android_apps returns only mapped apps in "DisplayName: pkg" format.
            # Extract just the display name portion for the prompt; package name lookup
            # happens via resolve_android_package() at execution time.
            annotated = annotate_android_apps(installed_apps)
            display_names = [entry.split(": ", 1)[0] for entry in annotated]
            if display_names:
                app_list = "\n".join(f"- {name}" for name in display_names)
                sections.extend([
                    "",
                    "# Installed Apps",
                    "",
                    "The following apps are available on this device:",
                    app_list,
                ])
        elif platform == "ios":
            from opengui.skills.normalization import annotate_ios_apps

            # annotate_ios_apps returns only mapped apps in "DisplayName: bundleId" format.
            # Extract just the display name portion for the prompt; bundle ID lookup
            # happens via resolve_ios_bundle() at execution time.
            annotated = annotate_ios_apps(installed_apps)
            display_names = [entry.split(": ", 1)[0] for entry in annotated]
            if display_names:
                app_list = "\n".join(f"- {name}" for name in display_names)
                sections.extend([
                    "",
                    "# Installed Apps",
                    "",
                    "The following apps are available on this iOS device:",
                    app_list,
                ])
        else:
            app_list = "\n".join(f"- {app}" for app in installed_apps)
            sections.extend([
                "",
                "# Installed Apps",
                "",
                "Use these app names for `open_app` and `close_app` actions:",
                app_list,
            ])

    if skill_context:
        sections.extend(["", "# Available Skills", "", skill_context])

    sections.extend([
        "",
        "# Response format",
        "",
        "Response format for every step:",
    ])
    sections.extend(prompt_contract["format"])
    sections.extend(["", "Rules:"])
    sections.extend(prompt_contract["rules"])

    return "\n".join(sections)
