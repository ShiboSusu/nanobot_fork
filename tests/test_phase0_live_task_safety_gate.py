from __future__ import annotations

import pytest

from eval.phase0_observable_signal_pilot import Phase0Task, ensure_task4b_safe


def task(task_id: str, instruction: str, risk_level: str = "U0") -> Phase0Task:
    return Phase0Task(
        task_id=task_id,
        instruction=instruction,
        instruction_ch=instruction,
        risk_level=risk_level,
    )


@pytest.mark.parametrize(
    ("task_id", "instruction", "expected_term"),
    [
        ("SetAlarmTask", "Set a weekend alarm for 8:25 a.m.", "alarm"),
        ("TakeSelfieTask", "Take a photo.", "photo"),
        ("AdjustBrightnessMaximumTask", "Set the brightness to the maximum level.", "brightness"),
        ("AdjustFontIconMaximumTask", "Increase the font size and icons on my phone.", "font"),
        ("ChangeWallpaperTask", "Change the wallpaper to a photo from the album.", "wallpaper"),
        ("OpenFlightModeTask", "Turn on device flight mode.", "flight mode"),
    ],
)
def test_live_task_safety_gate_blocks_local_device_mutations(
    task_id: str,
    instruction: str,
    expected_term: str,
) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        ensure_task4b_safe(task(task_id, instruction))

    message = str(excinfo.value).lower()
    assert "local side effect" in message
    assert expected_term in message


def test_live_task_safety_gate_allows_read_only_answer_task() -> None:
    ensure_task4b_safe(
        task(
            "ChromeSearchBeijingWeatherTask",
            "Use Chrome to search for Beijing highest temperature today. ONLY give a integer number.",
        )
    )
