from pathlib import Path
import re


def test_model_tunnel_ssh_runs_without_stdin() -> None:
    script = Path("scripts/nanobot-ios-start").read_text(encoding="utf-8")
    start = script.index("nohup ssh")
    end = script.index('"$MODELARTS_SSH_TARGET"', start)
    command = script[start:end]

    assert " -n " in command or command.startswith("nohup ssh -n ")


def test_model_tunnel_ssh_uses_short_keepalive_interval() -> None:
    script = Path("scripts/nanobot-ios-start").read_text(encoding="utf-8")
    match = re.search(r"-o ServerAliveInterval=(\d+)", script)

    assert match is not None
    assert int(match.group(1)) <= 10
