from __future__ import annotations

from eval.phase0_observable_signal_pilot import Phase0Task, instruction_for_run


def test_runtime_signal_instruction_preserves_task_as_primary_instruction() -> None:
    task = Phase0Task(
        task_id="weather",
        instruction="Open Huawei Weather and answer the current temperature.",
        instruction_ch="",
        risk_level="U0",
    )

    instruction = instruction_for_run(task, runtime_signal_enabled=True)

    assert instruction.startswith("Task:\nOpen Huawei Weather")
    assert "Before the <tool_call> block" in instruction
    assert instruction.index("Task:\nOpen Huawei Weather") < instruction.index("Before the <tool_call> block")
