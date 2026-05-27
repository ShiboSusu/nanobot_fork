"""
opengui.action
==============
Action dataclass, parsing, validation, and coordinate resolution.

* ``Action`` is frozen/immutable — safe to share across threads.
* ``parse_action`` is the sole normalisation boundary between LLM JSON and
  the rest of the framework.
* For ``scroll`` actions, direction is stored in ``Action.text``.
* ``resolve_coordinate`` maps [0,999] relative or absolute pixels to device px.
"""

from __future__ import annotations

import dataclasses
import json
import typing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTION_TYPES: frozenset[str] = frozenset({
    "tap", "long_press", "double_tap", "drag", "swipe", "scroll",
    "input_text", "hotkey", "screenshot", "wait",
    "open_app", "close_app", "back", "home", "enter", "app_switch", "done",
    "request_intervention",
})

_ACTION_ALIASES: dict[str, str] = {
    "click": "tap",
    "type": "input_text",
    "key": "hotkey",
    "press": "hotkey",
    "long_click": "long_press",
    "double_click": "double_tap",
    "keyboard_enter": "enter",
    "return": "enter",
    "navigate_back": "back",
    "navigate_home": "home",
    "recents": "app_switch",
    "recent_apps": "app_switch",
}

_RELATIVE_GRID_MAX: int = 999

_XY_REQUIRED: frozenset[str] = frozenset(
    {"tap", "long_press", "double_tap", "drag", "swipe"},
)
_XY2_REQUIRED: frozenset[str] = frozenset({"drag", "swipe"})
_VALID_SCROLL_DIRECTIONS: frozenset[str] = frozenset(
    {"up", "down", "left", "right"},
)


class ActionError(Exception):
    """Raised when an LLM payload cannot be converted into a valid Action."""


@dataclasses.dataclass(frozen=True)
class Action:
    """An immutable, validated representation of a single GUI automation step.

    Coordinates are stored in their original unit (relative or absolute).
    Call :func:`resolve_coordinate` at execution time to convert to device px.
    """

    action_type: str
    x: float | None = None
    y: float | None = None
    x2: float | None = None
    y2: float | None = None
    text: str | None = None
    key: list[str] | None = None
    pixels: int | None = None
    duration_ms: int | None = None
    relative: bool = False
    status: str | None = None
    auto_enter: bool = True


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def resolve_coordinate(value: float, extent: int, *, relative: bool) -> int:
    """Convert a coordinate value to an absolute device pixel.

    Returns a clamped int in ``[0, extent - 1]``.
    """
    if relative:
        pixel = round(value / _RELATIVE_GRID_MAX * (extent - 1))
    else:
        pixel = round(value)
    return max(0, min(pixel, extent - 1))


def parse_action(payload: dict[str, typing.Any]) -> Action:
    """Convert a raw LLM JSON payload into a validated :class:`Action`.

    Handles type aliases, field coercion, direction storage, and validation.
    """
    if not isinstance(payload, dict):
        raise ActionError(f"Expected dict payload, got {type(payload).__name__!r}.")

    payload = _normalize_coordinate_payload(payload)

    # 1. Normalise action_type
    raw_type = payload.get("action_type") or payload.get("action") or ""
    if not raw_type:
        raise ActionError("Payload is missing the required 'action_type' field.")
    action_type = _ACTION_ALIASES.get(str(raw_type).strip().lower(), str(raw_type).strip().lower())
    if action_type not in VALID_ACTION_TYPES:
        raise ActionError(
            f"Unknown action type {raw_type!r}. "
            f"Valid: {', '.join(sorted(VALID_ACTION_TYPES))}."
        )

    # 2. Coordinates
    x = _optional_float(payload, "x", action_type)
    y = _optional_float(payload, "y", action_type)
    x2 = _optional_float(payload, "x2", action_type)
    y2 = _optional_float(payload, "y2", action_type)

    # 3. Text — for scroll, direction takes priority
    if action_type == "scroll":
        raw_dir = payload.get("direction") or payload.get("text")
        text: str | None = str(raw_dir).strip().lower() if raw_dir is not None else None
    else:
        text = _optional_str(payload, "text")

    # 4. Other fields
    key = _parse_key(payload.get("key"), action_type)
    pixels = _optional_int(payload, "pixels", action_type)
    duration_ms = _optional_int(payload, "duration_ms", action_type)
    relative = bool(payload.get("relative", False))
    if relative:
        x = _normalize_fractional_relative_coordinate(x)
        y = _normalize_fractional_relative_coordinate(y)
        x2 = _normalize_fractional_relative_coordinate(x2)
        y2 = _normalize_fractional_relative_coordinate(y2)
    status = _optional_str(payload, "status")
    auto_enter = bool(payload.get("auto_enter", True))

    # 5. Validate
    _validate(action_type=action_type, x=x, y=y, x2=x2, y2=y2,
              text=text, key=key, pixels=pixels)

    return Action(
        action_type=action_type, x=x, y=y, x2=x2, y2=y2,
        text=text, key=key, pixels=pixels, duration_ms=duration_ms,
        relative=relative, status=status, auto_enter=auto_enter,
    )


def describe_action(action: Action) -> str:
    """Return a concise, human-readable summary of *action*."""
    t = action.action_type

    if t in ("tap", "long_press", "double_tap"):
        coord = _fmt_coord(action)
        suffix = f" for {action.duration_ms} ms" if t == "long_press" and action.duration_ms else ""
        return f"{t.replace('_', ' ')} at {coord}{suffix}"
    if t in ("drag", "swipe"):
        return f"{t} from {_fmt_coord(action)} to ({action.x2}, {action.y2})"
    if t == "scroll":
        direction = action.text or "?"
        px = f" by {action.pixels} px" if action.pixels is not None else ""
        return f"scroll {direction}{px} at {_fmt_coord(action)}"
    if t == "input_text":
        preview = (action.text or "")[:40]
        if len(action.text or "") > 40:
            preview += "..."
        return f'type "{preview}"'
    if t == "hotkey":
        return f"hotkey {'+'.join(action.key or [])}"
    if t == "open_app":
        return f"open app {action.text!r}"
    if t == "close_app":
        return f"close app {action.text!r}"
    if t == "request_intervention":
        preview = (action.text or "").strip()[:40]
        if len((action.text or "").strip()) > 40:
            preview += "..."
        return f"request intervention: {preview}"
    if t == "screenshot":
        return "take screenshot"
    if t == "wait":
        return f"wait {action.duration_ms} ms" if action.duration_ms else "wait"
    if t == "back":
        return "press back"
    if t == "home":
        return "press home"
    if t == "enter":
        return "press enter"
    if t == "app_switch":
        return "open app switcher"
    if t == "done":
        return f"task done – {action.status}" if action.status else "task done"
    return t


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fmt_coord(action: Action) -> str:
    if action.x is None or action.y is None:
        return "center"
    if action.relative:
        return f"({action.x}/{_RELATIVE_GRID_MAX}, {action.y}/{_RELATIVE_GRID_MAX})"
    return f"({action.x}, {action.y})"


def _normalize_fractional_relative_coordinate(value: float | None) -> float | None:
    if value is None:
        return None
    if 0.0 < value < 1.0:
        return value * _RELATIVE_GRID_MAX
    return value


def _normalize_coordinate_payload(payload: dict[str, typing.Any]) -> dict[str, typing.Any]:
    normalized = dict(payload)
    _normalize_named_coordinates(normalized)
    _normalize_compact_path_coordinates(normalized)
    _normalize_coordinate_pair(normalized, "x", "y")
    _normalize_coordinate_pair(normalized, "x2", "y2")
    _normalize_endpoint_only_path_coordinates(normalized)
    return normalized


def _normalize_endpoint_only_path_coordinates(payload: dict[str, typing.Any]) -> None:
    """Recover swipe/drag payloads that provide only end-point coordinates.

    Some providers occasionally emit ``x2/y2`` without ``x/y`` for path actions.
    We synthesize a conservative start point to avoid parser hard-failures.
    """
    raw_type = payload.get("action_type") or payload.get("action") or ""
    action_type = _ACTION_ALIASES.get(str(raw_type).strip().lower(), str(raw_type).strip().lower())
    if action_type not in {"swipe", "drag"}:
        return
    if payload.get("x2") is None or payload.get("y2") is None:
        return
    if payload.get("x") is not None and payload.get("y") is not None:
        return

    try:
        end_x = float(payload["x2"])
        end_y = float(payload["y2"])
    except (TypeError, ValueError):
        return

    if payload.get("x") is None:
        payload["x"] = end_x
    if payload.get("y") is None:
        # Default to a medium path length in the opposite vertical direction.
        if end_y >= (_RELATIVE_GRID_MAX / 2):
            start_y = end_y - 250
        else:
            start_y = end_y + 250
        payload["y"] = max(0.0, min(float(_RELATIVE_GRID_MAX), start_y))


def _normalize_named_coordinates(payload: dict[str, typing.Any]) -> None:
    # General mobile-use style aliases.
    _normalize_named_coordinate_pair(payload, "coordinate", "x", "y")
    _normalize_named_coordinate_pair(payload, "start_coordinate", "x", "y")
    _normalize_named_coordinate_pair(payload, "coordinate2", "x2", "y2")
    _normalize_named_coordinate_pair(payload, "end_coordinate", "x2", "y2")

    # Accept compact forms like coordinate=[x, y, x2, y2] when all canonical
    # fields are absent.
    compact = _coerce_coordinate_sequence(payload.get("coordinate"))
    if compact is None or len(compact) != 4:
        return
    if any(payload.get(key) is not None for key in ("x", "y", "x2", "y2")):
        return
    payload["x"], payload["y"], payload["x2"], payload["y2"] = compact


def _normalize_named_coordinate_pair(
    payload: dict[str, typing.Any],
    source_key: str,
    primary_key: str,
    secondary_key: str,
) -> None:
    source = payload.get(source_key)
    items = _coerce_coordinate_sequence(source)
    if items is None:
        return
    if len(items) != 2:
        return
    if payload.get(primary_key) in (None, [], ()):
        payload[primary_key] = items[0]
    if payload.get(secondary_key) in (None, [], ()):
        payload[secondary_key] = items[1]


def _normalize_compact_path_coordinates(payload: dict[str, typing.Any]) -> None:
    primary_items = _coerce_coordinate_sequence(payload.get("x"))
    if primary_items is None or len(primary_items) != 4:
        return
    if any(payload.get(key) is not None for key in ("y", "x2", "y2")):
        return
    payload["x"], payload["y"], payload["x2"], payload["y2"] = primary_items


def _normalize_coordinate_pair(
    payload: dict[str, typing.Any],
    primary_key: str,
    secondary_key: str,
) -> None:
    primary = payload.get(primary_key)
    secondary = payload.get(secondary_key)

    primary_items = _coerce_coordinate_sequence(primary)
    secondary_items = _coerce_coordinate_sequence(secondary)

    if primary_items is not None:
        if len(primary_items) == 1:
            payload[primary_key] = primary_items[0]
        elif len(primary_items) == 2:
            payload[primary_key] = primary_items[0]
            if secondary in (None, [], ()):
                payload[secondary_key] = primary_items[1]

    if secondary_items is not None:
        if len(secondary_items) == 1:
            payload[secondary_key] = secondary_items[0]
        elif len(secondary_items) == 2:
            # Some providers emit duplicated scalar coordinates as two-item
            # lists (e.g. y=[957, 957]); collapse them to a single scalar.
            if secondary_items[0] == secondary_items[1]:
                payload[secondary_key] = secondary_items[0]
            # Also accept swapped pair payloads like y=[x, y] when x is absent.
            elif primary in (None, [], ()):
                payload[primary_key] = secondary_items[0]
                payload[secondary_key] = secondary_items[1]


def _coerce_coordinate_sequence(value: typing.Any) -> tuple[typing.Any, ...] | None:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, list):
                return tuple(parsed)
    return None


def _optional_float(payload: dict, key: str, action_type: str) -> float | None:
    v = payload.get(key)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError) as exc:
        raise ActionError(f"Action {action_type!r}: {key!r} must be numeric, got {v!r}.") from exc


def _optional_int(payload: dict, key: str, action_type: str) -> int | None:
    v = payload.get(key)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError) as exc:
        raise ActionError(f"Action {action_type!r}: {key!r} must be int, got {v!r}.") from exc


def _optional_str(payload: dict, key: str) -> str | None:
    v = payload.get(key)
    return str(v) if v is not None else None


def _parse_key(raw: typing.Any, action_type: str) -> list[str] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        if not all(isinstance(k, str) for k in raw):
            raise ActionError(f"Action {action_type!r}: 'key' list must contain only strings.")
        return [k.strip() for k in raw if k]
    if isinstance(raw, str):
        for sep in ("+", " ", ","):
            if sep in raw:
                return [k.strip() for k in raw.split(sep) if k.strip()]
        return [raw.strip()]
    raise ActionError(f"Action {action_type!r}: 'key' must be str or list, got {type(raw).__name__!r}.")


def _validate(
    *, action_type: str, x: float | None, y: float | None,
    x2: float | None, y2: float | None, text: str | None,
    key: list[str] | None, pixels: int | None,
) -> None:
    if action_type in _XY_REQUIRED and (x is None or y is None):
        raise ActionError(f"Action {action_type!r} requires both 'x' and 'y' coordinates.")
    if action_type in _XY2_REQUIRED and (x2 is None or y2 is None):
        raise ActionError(f"Action {action_type!r} requires 'x2' and 'y2' end-point coordinates.")
    if action_type in _XY2_REQUIRED and x == x2 and y == y2:
        raise ActionError(
            f"Action {action_type!r} start and end coordinates must differ."
        )
    if action_type == "hotkey" and not key:
        raise ActionError("Action 'hotkey' requires the 'key' field.")
    if action_type == "input_text" and text is None:
        raise ActionError("Action 'input_text' requires the 'text' field.")
    if action_type == "request_intervention" and (text is None or not text.strip()):
        raise ActionError("Action 'request_intervention' requires a non-empty 'text' field.")
    if action_type == "scroll":
        if pixels is None:
            raise ActionError("Action 'scroll' requires the 'pixels' field.")
        if (x is None) != (y is None):
            raise ActionError("Action 'scroll' requires both 'x' and 'y' when either is set.")
        if text is not None and text not in _VALID_SCROLL_DIRECTIONS:
            raise ActionError(
                f"Action 'scroll' direction {text!r} invalid. "
                f"Use: {', '.join(sorted(_VALID_SCROLL_DIRECTIONS))}."
            )
    if action_type in ("open_app", "close_app") and not text:
        raise ActionError(f"Action {action_type!r} requires the 'text' field (app name).")
