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

    assert instruction.startswith("Open Huawei Weather")
    assert "Every response is invalid unless it contains exactly one" in instruction
    assert instruction.index("Open Huawei Weather") < instruction.index("Every response is invalid")
    assert "Instruction: Task:" not in f"Instruction: {instruction}"
