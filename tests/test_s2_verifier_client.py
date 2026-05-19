from __future__ import annotations

import json

import pytest

from eval import s2_verifier_client as s2


def test_load_record_at_line_reads_valid_dict_on_physical_line(tmp_path) -> None:
    input_path = tmp_path / "phase0.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "line-1", "task_risk_level": "U0"}),
                "{not-json",
                json.dumps({"task_id": "line-3", "task_risk_level": "U1"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    record = s2.load_record_at_line(input_path, 3)

    assert record == {"task_id": "line-3", "task_risk_level": "U1"}


def test_run_offline_smoke_rejects_invalid_record_line(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "line-1", "task_risk_level": "U0"}),
                "{not-json",
                json.dumps({"task_id": "line-3", "task_risk_level": "U1"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MA_INTRANET_URL", raising=False)
    monkeypatch.delenv("MA_TOKEN", raising=False)

    exit_code = s2.run_offline_smoke(input_path, latest=False, record_line=2, step_index=None, timeout_s=1.0, write_output=None)

    assert exit_code == 2
    assert "ERROR: invalid JSON on physical line 2" in capsys.readouterr().out


def test_run_offline_smoke_uses_record_line_without_env(tmp_path, monkeypatch, capsys) -> None:
    input_path = tmp_path / "phase0.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "line-1", "task_risk_level": "U0"}),
                json.dumps(
                    {
                        "task_id": "line-2",
                        "task_risk_level": "U1",
                        "steps": [{"step_index": 0, "action": {"action_type": "done"}}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("MA_INTRANET_URL", raising=False)
    monkeypatch.delenv("MA_TOKEN", raising=False)

    exit_code = s2.run_offline_smoke(input_path, latest=False, record_line=2, step_index=None, timeout_s=1.0, write_output=None)

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "selected_task_id: line-2" in stdout
    assert f"record_source: {input_path}:2" in stdout
    assert '"verifier_error": "missing_env:MA_INTRANET_URL,MA_TOKEN"' in stdout


def test_parser_rejects_latest_with_record_line(capsys) -> None:
    parser = s2.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--offline-smoke", "--input", "phase0.jsonl", "--latest", "--record-line", "3"])

    stderr = capsys.readouterr().err
    assert "not allowed with argument" in stderr
