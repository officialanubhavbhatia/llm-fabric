"""JSON codec for tenant-scoped dataclass records.

Records stay typed in Python. Persistence sees a JSON object. Reconstruction
uses dataclass field annotations so a payload cannot quietly become a different
type than the store asked for.
"""

from __future__ import annotations

import types
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Union, get_args, get_origin, get_type_hints

from llm_fabric.errors import ConfigurationError


def encode(value: Any) -> Any:
    """Turn a record (and nested dataclasses) into JSON-friendly data."""
    if is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [encode(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [encode(item) for item in sorted(value, key=str)]
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot persist value of type {type(value).__name__}")


def decode(record_type: type[Any], payload: Any) -> Any:
    """Rebuild a record of `record_type` from encoded JSON."""
    return _decode(record_type, payload)


def _decode(hint: Any, payload: Any) -> Any:
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        if payload is None:
            return None
        for arg in get_args(hint):
            if arg is type(None):
                continue
            try:
                return _decode(arg, payload)
            except (TypeError, ValueError, ConfigurationError, AttributeError, KeyError):
                continue
        raise TypeError(f"cannot decode {payload!r} as {hint}")

    if origin in (tuple, list, set, frozenset):
        args = get_args(hint)
        item_hint = args[0] if args else Any
        if not isinstance(payload, list):
            raise TypeError(f"expected a list for {hint}")
        items = [_decode(item_hint, item) for item in payload]
        if origin is tuple:
            return tuple(items)
        if origin is frozenset:
            return frozenset(items)
        if origin is set:
            return set(items)
        return items

    if origin is dict:
        args = get_args(hint)
        value_hint = args[1] if len(args) > 1 else Any
        if not isinstance(payload, dict):
            raise TypeError("expected an object")
        return {key: _decode(value_hint, item) for key, item in payload.items()}

    if hint is Any:
        return payload

    if isinstance(hint, type) and issubclass(hint, Enum):
        return hint(payload)

    if is_dataclass(hint):
        if not isinstance(payload, dict):
            raise TypeError(f"expected an object for {hint}")
        hints = get_type_hints(hint)
        kwargs = {}
        for field in fields(hint):
            if field.name not in payload:
                continue
            kwargs[field.name] = _decode(hints.get(field.name, Any), payload[field.name])
        return hint(**kwargs)  # type: ignore[operator]

    return payload
