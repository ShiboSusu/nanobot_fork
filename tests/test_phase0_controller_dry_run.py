from __future__ import annotations

import json
import sys

import pytest

from eval import phase0_controller_dry_run as dry_run


def verifier_record(task_id: str, decision: str | None = None) -> dict[str, object]:
    verifier = {"called": True, "decision": decision, "error": None}
    return {
        "task_id": task_id,
        "task_risk_level": "U0",
        "steps": [
            {
                "step_index": 1,
                "trigger_features": {
                    "risk": {"rule_based_step_risk_level": "U0", "step_predicted_risk_level": "U0"},
                    "execution_state": {},
                },
                "verifier": verifier,
            }
        ],
    }


def run_main(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    input_path,
    *extra_args: str,
) -> tuple[list[dict[str, object]], dict[str, object], str]:
    output_path = tmp_path / "phase0_controller_dry_run.jsonl"
    monkeypatch.setattr(dry_run, "git_ignores_path", lambda path: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "phase0_controller_dry_run.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary",
            *extra_args,
        ],
    )

    exit_code = dry_run.main()

    stdout = capsys.readouterr().out
    summary_line = next(line for line in stdout.splitlines() if line.startswith("summary: "))
    outputs = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 0
    return outputs, json.loads(summary_line.removeprefix("summary: ")), stdout


def test_input_last_routes_only_last_valid_record(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    records = [
        verifier_record("line1", "pass"),
        verifier_record("line2", "replan"),
        verifier_record("line3", "block"),
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    outputs, summary, stdout = run_main(tmp_path, monkeypatch, capsys, input_path, "--input-last", "1")

    assert [output["task_id"] for output in outputs] == ["line3"]
    assert outputs[0]["controller"]["route"] == "BLOCK"
    assert summary["total_routed_steps"] == 1
    assert summary["route_distribution"] == {"BLOCK": 1}
    assert summary["input_line_range"] == [3, 3]
    assert summary["input_filters"] == {"last": 1, "since_line": None}
    assert "wrote_records: 1" in stdout


def test_input_since_line_filters_by_physical_jsonl_line(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(verifier_record("line1", "pass")),
                "",
                json.dumps(verifier_record("line3", "replan")),
                json.dumps(verifier_record("line4", "block")),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    outputs, summary, _stdout = run_main(tmp_path, monkeypatch, capsys, input_path, "--input-since-line", "3")

    assert [output["task_id"] for output in outputs] == ["line3", "line4"]
    assert [output["controller"]["route"] for output in outputs] == ["SLOW", "BLOCK"]
    assert summary["route_distribution"] == {"BLOCK": 1, "SLOW": 1}
    assert summary["input_line_range"] == [3, 4]
    assert summary["input_filters"] == {"last": None, "since_line": 3}


def test_input_filters_apply_since_line_before_last(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0_s2_verifier_offline.jsonl"
    records = [
        verifier_record("line1", "pass"),
        verifier_record("line2", "replan"),
        verifier_record("line3", "block"),
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    outputs, summary, _stdout = run_main(
        tmp_path,
        monkeypatch,
        capsys,
        input_path,
        "--input-since-line",
        "1",
        "--input-last",
        "1",
    )

    assert [output["task_id"] for output in outputs] == ["line3"]
    assert outputs[0]["controller"]["route"] == "BLOCK"
    assert summary["input_line_range"] == [3, 3]
    assert summary["input_filters"] == {"last": 1, "since_line": 1}


def test_input_filter_cli_values_must_be_positive(capsys) -> None:
    parser = dry_run.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "input.jsonl", "--output", "output.jsonl", "--input-last", "0"])

    stderr = capsys.readouterr().err
    assert "positive integer" in stderr
