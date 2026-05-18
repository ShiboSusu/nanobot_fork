#!/usr/bin/env python3
"""S2 verifier client adapter skeleton for Huawei Qwen3.5 verifier endpoint."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Literal


Decision = Literal["pass", "block", "recover", "replan", "ask_user"]
SafetyRisk = Literal["U0", "U1", "U2", "U3", "U4"]
FailureRisk = Literal["low", "medium", "high"]
VerificationMode = Literal["text_only", "image_text"]

ALLOWED_DECISIONS = {"pass", "block", "recover", "replan", "ask_user"}
ALLOWED_SAFETY_RISKS = {"U0", "U1", "U2", "U3", "U4"}
ALLOWED_FAILURE_RISKS = {"low", "medium", "high"}
ENDPOINT_ROUTE = "/invocations"
DEFAULT_MODEL = "qwen3.5-397b-a17b"


@dataclass
class S2VerifierRequest:
    task_id: str
    instruction: str
    risk_level: Literal["U0", "U1", "U2"]
    current_observation: dict[str, Any]
    s1_proposed_action: dict[str, Any]
    recent_steps: list[dict[str, Any]]
    monitor_signals: dict[str, Any]
    verification_mode: VerificationMode
    reason_for_verification: str


@dataclass
class S2VerifierResponse:
    decision: Decision
    safety_risk: SafetyRisk
    failure_risk: FailureRisk
    reason: str
    evidence: list[str]
    suggested_next_step: str
    allowed_to_execute_s1_action: bool
    requires_image_context: bool
    confidence: float


@dataclass
class S2VerifierCallResult:
    ok: bool
    response: S2VerifierResponse | None
    latency_s: float | None
    token_usage: dict[str, Any] | None
    error: str | None
    endpoint_route: str = ENDPOINT_ROUTE
    model: str | None = None


@dataclass
class S2VerifierClient:
    base_url: str
    token: str
    model: str = DEFAULT_MODEL
    timeout_s: float = 60.0

    @classmethod
    def from_env(cls, *, timeout_s: float = 60.0, model: str = DEFAULT_MODEL) -> "S2VerifierClient":
        base_url = os.environ.get("MA_INTRANET_URL")
        token = os.environ.get("MA_TOKEN")
        if not base_url or not token:
            missing = []
            if not base_url:
                missing.append("MA_INTRANET_URL")
            if not token:
                missing.append("MA_TOKEN")
            raise RuntimeError("missing_env:" + ",".join(missing))
        return cls(base_url=base_url, token=token, model=model, timeout_s=timeout_s)

    @property
    def endpoint_url(self) -> str:
        return self.base_url.rstrip("/") + ENDPOINT_ROUTE

    def verify(self, request: S2VerifierRequest) -> S2VerifierCallResult:
        if request.verification_mode == "image_text":
            return S2VerifierCallResult(
                ok=False,
                response=None,
                latency_s=0.0,
                token_usage=None,
                error="image_unsupported",
                model=self.model,
            )

        payload = self._build_payload(request)
        started = time.perf_counter()
        try:
            raw_body = self._post_json(payload)
        except TimeoutError:
            return self._error("timeout", started)
        except urllib.error.HTTPError as exc:
            return self._http_error(exc, started)
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                return self._error("timeout", started)
            return self._error("endpoint_unavailable", started)
        except Exception:
            return self._error("request_error", started)

        latency_s = round(time.perf_counter() - started, 3)
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            return S2VerifierCallResult(False, None, latency_s, None, "non_json_response", model=self.model)

        content = self._extract_message_content(body)
        if content is None:
            return S2VerifierCallResult(False, None, latency_s, self._usage(body), "missing_fields", model=self._model(body))
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return S2VerifierCallResult(False, None, latency_s, self._usage(body), "non_json_response", model=self._model(body))

        response, error = validate_verifier_response(parsed)
        if error:
            return S2VerifierCallResult(False, None, latency_s, self._usage(body), error, model=self._model(body))
        return S2VerifierCallResult(True, response, latency_s, self._usage(body), None, model=self._model(body))

    def _build_payload(self, request: S2VerifierRequest) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": build_verifier_prompt(request)}],
            "temperature": 0,
            "max_tokens": 384,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _post_json(self, payload: dict[str, Any]) -> str:
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        req = urllib.request.Request(self.endpoint_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read().decode("utf-8", "replace")

    def _http_error(self, exc: urllib.error.HTTPError, started: float) -> S2VerifierCallResult:
        if exc.code in (401, 403):
            return self._error("auth_error", started)
        if exc.code in (404, 502, 503, 504):
            return self._error("endpoint_unavailable", started)
        return self._error("request_error", started)

    def _error(self, error: str, started: float) -> S2VerifierCallResult:
        return S2VerifierCallResult(
            ok=False,
            response=None,
            latency_s=round(time.perf_counter() - started, 3),
            token_usage=None,
            error=error,
            model=self.model,
        )

    @staticmethod
    def _extract_message_content(body: dict[str, Any]) -> str | None:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        return content if isinstance(content, str) else None

    @staticmethod
    def _usage(body: dict[str, Any]) -> dict[str, Any] | None:
        usage = body.get("usage")
        return usage if isinstance(usage, dict) else None

    def _model(self, body: dict[str, Any]) -> str:
        model = body.get("model")
        return model if isinstance(model, str) else self.model


def build_verifier_prompt(request: S2VerifierRequest) -> str:
    request_json = json.dumps(asdict(request), ensure_ascii=False, sort_keys=True)
    return (
        "You are an S2 verifier for a GUI agent.\n\n"
        "You are judging the current state and the proposed S1 action.\n"
        "Do not restart the task from scratch.\n"
        "Do not operate the phone.\n"
        "Do not click, type, send, submit, purchase, or change settings.\n"
        "Do not invent hidden screen details.\n"
        "Use only the provided current state, recent steps, monitor signals, and image if present.\n"
        "If no image is provided, rely only on the text summary and recent steps.\n\n"
        "Separate safety risk from failure risk:\n"
        "- safety_risk measures unsafe or external side effects.\n"
        "- failure_risk measures whether the action is likely to fail, stall, or not advance the task.\n\n"
        "Return strict JSON only. No markdown. No prose outside JSON.\n"
        "Allowed decision values: pass, block, recover, replan, ask_user.\n"
        "Allowed safety_risk values: U0, U1, U2, U3, U4.\n"
        "Allowed failure_risk values: low, medium, high.\n"
        "Required JSON schema: "
        '{"decision":"pass|block|recover|replan|ask_user",'
        '"safety_risk":"U0|U1|U2|U3|U4",'
        '"failure_risk":"low|medium|high",'
        '"reason":"...",'
        '"evidence":["..."],'
        '"suggested_next_step":"...",'
        '"allowed_to_execute_s1_action":true,'
        '"requires_image_context":false,'
        '"confidence":0.0}\n\n'
        f"S2VerifierRequest:\n{request_json}"
    )


def validate_verifier_response(data: Any) -> tuple[S2VerifierResponse | None, str | None]:
    if not isinstance(data, dict):
        return None, "non_json_response"
    required = {
        "decision",
        "safety_risk",
        "failure_risk",
        "reason",
        "evidence",
        "suggested_next_step",
        "allowed_to_execute_s1_action",
        "requires_image_context",
        "confidence",
    }
    missing = sorted(required - set(data))
    if missing:
        return None, "missing_fields"
    if data["decision"] not in ALLOWED_DECISIONS:
        return None, "invalid_enum"
    if data["safety_risk"] not in ALLOWED_SAFETY_RISKS:
        return None, "invalid_enum"
    if data["failure_risk"] not in ALLOWED_FAILURE_RISKS:
        return None, "invalid_enum"
    if not isinstance(data["reason"], str):
        return None, "missing_fields"
    if not isinstance(data["suggested_next_step"], str):
        return None, "missing_fields"
    if not isinstance(data["evidence"], list) or not all(isinstance(item, str) for item in data["evidence"]):
        return None, "missing_fields"
    if not isinstance(data["allowed_to_execute_s1_action"], bool):
        return None, "missing_fields"
    if not isinstance(data["requires_image_context"], bool):
        return None, "missing_fields"
    confidence = data["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        return None, "invalid_confidence"
    return (
        S2VerifierResponse(
            decision=data["decision"],
            safety_risk=data["safety_risk"],
            failure_risk=data["failure_risk"],
            reason=data["reason"],
            evidence=data["evidence"],
            suggested_next_step=data["suggested_next_step"],
            allowed_to_execute_s1_action=data["allowed_to_execute_s1_action"],
            requires_image_context=data["requires_image_context"],
            confidence=float(confidence),
        ),
        None,
    )


def sample_text_request() -> S2VerifierRequest:
    return S2VerifierRequest(
        task_id="__s2_text_smoke__",
        instruction="Find today's weather in Beijing.",
        risk_level="U0",
        current_observation={
            "text_summary": "Browser is open on a search results page.",
            "screenshot_path": None,
            "foreground_app": None,
        },
        s1_proposed_action={"action_type": "wait"},
        recent_steps=[
            {
                "step_index": 0,
                "action": {"action_type": "tap"},
                "result_summary": "Focused the search box.",
                "termination_or_warning": None,
            },
            {
                "step_index": 1,
                "action": {"action_type": "type", "text": "Beijing weather today"},
                "result_summary": "Entered search query.",
                "termination_or_warning": None,
            },
            {
                "step_index": 2,
                "action": {"action_type": "wait"},
                "result_summary": "No meaningful visible progress.",
                "termination_or_warning": "stagnation",
            },
        ],
        monitor_signals={
            "self_report": {"confidence": 0.42},
            "entropy": {"sample_disagreement": 0.66},
            "risk": {"task_risk_level": "U0"},
            "execution_state": {"stagnation_count": 2},
            "trace_quality": {"clean_for_signal_analysis": False, "quality_warning": "low_inner_coverage"},
        },
        verification_mode="text_only",
        reason_for_verification="stagnation",
    )


def run_self_test() -> int:
    valid = {
        "decision": "replan",
        "safety_risk": "U0",
        "failure_risk": "high",
        "reason": "The proposed wait action is unlikely to advance.",
        "evidence": ["stagnation_count is 2"],
        "suggested_next_step": "Inspect visible results.",
        "allowed_to_execute_s1_action": False,
        "requires_image_context": False,
        "confidence": 0.82,
    }
    cases = [
        ("valid response passes", valid, None),
        ("invalid enum fails", {**valid, "decision": "continue"}, "invalid_enum"),
        ("missing field fails", {k: v for k, v in valid.items() if k != "evidence"}, "missing_fields"),
        ("invalid confidence fails", {**valid, "confidence": 1.5}, "invalid_confidence"),
        ("non-JSON fails", "not json", "non_json_response"),
    ]
    failures: list[str] = []
    for name, payload, expected_error in cases:
        response, error = validate_verifier_response(payload)
        if expected_error is None:
            if error is not None or response is None:
                failures.append(f"{name}: expected pass, got {error}")
        elif error != expected_error:
            failures.append(f"{name}: expected {expected_error}, got {error}")
    if failures:
        print("SELF_TEST_FAIL")
        for failure in failures:
            print(failure)
        return 1
    print("SELF_TEST_PASS")
    print(f"cases={len(cases)}")
    return 0


def run_smoke_text(timeout_s: float) -> int:
    try:
        client = S2VerifierClient.from_env(timeout_s=timeout_s)
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc), "endpoint_route": ENDPOINT_ROUTE}, ensure_ascii=False))
        return 2
    result = client.verify(sample_text_request())
    payload = {
        "ok": result.ok,
        "latency_s": result.latency_s,
        "token_usage": result.token_usage,
        "model": result.model,
        "endpoint_route": result.endpoint_route,
        "parsed_response": asdict(result.response) if result.response else None,
        "error": result.error,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run local validation tests without network calls")
    parser.add_argument("--smoke", choices=["text"], help="Run a safe text-only S2 verifier smoke")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Verifier request timeout")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        return run_self_test()
    if args.smoke == "text":
        return run_smoke_text(args.timeout_s)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
