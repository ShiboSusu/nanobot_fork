from __future__ import annotations

from typing import Any


def extract_gui_evidence(
    *,
    request: Any,
    payload: dict[str, Any],
    blackboard: dict[str, str] | None = None,
    blackboard_meta: dict[str, dict[str, Any]] | None = None,
    latest_step: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = _existing_candidates(payload)
    if not candidates:
        candidates = _candidates_from_blackboard_meta(blackboard_meta)
    if not candidates:
        candidates = _candidates_from_blackboard(request, blackboard)
    visible_text = _visible_text_from_latest_step(latest_step)
    sources: list[str] = []
    if not candidates and _is_answer_required(request):
        candidates, sources = _extract_answer_candidates_from_text(
            request=request,
            payload=payload,
            visible_text=visible_text,
        )
    if candidates and not sources:
        sources = _candidate_sources(candidates)
    return {
        "answer_candidates": candidates,
        "evidence": {
            "visible_text": visible_text,
            "sources": sources,
        },
        "uncertainty": {
            "level": "low" if candidates or not _is_answer_required(request) else "high",
            "reasons": [] if candidates or not _is_answer_required(request) else ["no_answer_candidate_extracted"],
        },
    }


def apply_evidence_contract(
    *,
    request: Any,
    payload: dict[str, Any],
    blackboard: dict[str, str] | None = None,
    blackboard_meta: dict[str, dict[str, Any]] | None = None,
    latest_step: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(payload)
    extracted = extract_gui_evidence(
        request=request,
        payload=out,
        blackboard=blackboard,
        blackboard_meta=blackboard_meta,
        latest_step=latest_step,
    )
    out["answer_candidates"] = extracted["answer_candidates"]
    if not out.get("evidence"):
        out["evidence"] = extracted["evidence"]
    if not out.get("uncertainty"):
        out["uncertainty"] = extracted["uncertainty"]

    if _is_answer_required(request) and not out["answer_candidates"] and out.get("success") is True:
        out["success"] = False
        out["status"] = "partial"
        out["error"] = "missing_required_answer_evidence"
        out["uncertainty"] = {
            "level": "high",
            "reasons": ["no_answer_candidate_extracted"],
        }
    return out


def _existing_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("answer_candidates")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _candidates_from_blackboard_meta(
    blackboard_meta: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not isinstance(blackboard_meta, dict):
        return candidates
    for key, meta in blackboard_meta.items():
        if not isinstance(meta, dict):
            continue
        value = meta.get("value")
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        candidates.append(
            {
                "key": str(key),
                "text": text,
                "source": "blackboard",
                "confidence": meta.get("confidence"),
                "source_subtask": meta.get("source_subtask"),
                "evidence_refs": meta.get("evidence_refs", []),
            }
        )
    return candidates


def _candidates_from_blackboard(
    request: Any,
    blackboard: dict[str, str] | None,
) -> list[dict[str, Any]]:
    if not isinstance(blackboard, dict):
        return []
    required_key = _required_key(request)
    keys = [required_key] if required_key else list(blackboard)
    candidates: list[dict[str, Any]] = []
    for key in keys:
        value = blackboard.get(str(key))
        if value is None:
            continue
        text = str(value).strip()
        if text:
            candidates.append(
                {
                    "key": str(key),
                    "text": text,
                    "source": "blackboard",
                    "confidence": None,
                    "evidence_refs": [],
                }
            )
    return candidates


def _extract_answer_candidates_from_text(
    *,
    request: Any,
    payload: dict[str, Any],
    visible_text: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    search_items: list[tuple[str, str]] = []
    for key in ("model_summary", "state_summary", "summary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            search_items.append((key, value))
    for index, text in enumerate(visible_text):
        search_items.append((f"visible_text[{index}]", text))

    required_key = _required_key(request)
    if required_key:
        minimum_confidence = _minimum_confidence(request)
        for source, text in search_items:
            answer = _extract_generic_answer(text, minimum_confidence=minimum_confidence)
            if answer:
                return [
                    {
                        "key": required_key,
                        "text": answer,
                        "type": "answer",
                        "confidence": 0.5,
                        "source": source,
                        "evidence_refs": [source],
                    }
                ], [source]

    return [], []


def _extract_generic_answer(text: str, *, minimum_confidence: float) -> str | None:
    del text
    fallback_confidence = 0.5
    if fallback_confidence < minimum_confidence:
        return None
    return None


def _clean_answer_text(text: str) -> str:
    return " ".join(text.split()).strip(" \t\r\n\"'“”‘’。；;，,")


def _visible_text_from_latest_step(latest_step: dict[str, Any] | None) -> list[str]:
    if not isinstance(latest_step, dict):
        return []
    values: list[str] = []
    for key in ("visible_text", "screen_text", "text"):
        values.extend(_coerce_text_list(latest_step.get(key)))
    observation = latest_step.get("observation")
    if isinstance(observation, dict):
        for key in ("visible_text", "screen_text", "text"):
            values.extend(_coerce_text_list(observation.get(key)))
        values.extend(_recursive_text_values(observation.get("extra")))
    return values


def _recursive_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        values: list[str] = []
        for item in value.values():
            values.extend(_recursive_text_values(item))
        return values
    if isinstance(value, list | tuple):
        values: list[str] = []
        for item in value:
            values.extend(_recursive_text_values(item))
        return values
    return []


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list | tuple):
        items: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def _candidate_sources(candidates: list[dict[str, Any]]) -> list[str]:
    sources: list[str] = []
    for candidate in candidates:
        source = candidate.get("source")
        if source is not None:
            sources.append(str(source))
    return sources


def _requires_answer_candidates(request: Any) -> bool:
    value = _nested_get(request, "evidence_requirements", "answer_candidates_required")
    return bool(value)


def _is_answer_required(request: Any) -> bool:
    mode = _enum_value(_get_value(request, "output_mode"))
    task_type = _enum_value(_get_value(request, "task_type"))
    return (
        _requires_answer_candidates(request)
        or mode == "answer_required"
        or task_type in {"information_query", "mixed_query_and_action"}
    )


def _required_key(request: Any) -> str | None:
    value = _nested_get(request, "success_condition", "required_key")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _minimum_confidence(request: Any) -> float:
    value = _nested_get(request, "evidence_requirements", "minimum_confidence")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    return parsed if parsed > 0 else 0.75


def _nested_get(value: Any, first: str, second: str) -> Any:
    first_value = _get_value(value, first)
    return _get_value(first_value, second)


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)
