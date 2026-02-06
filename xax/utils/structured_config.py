"""Typed configuration helpers for dataclass-based config + CLI overrides."""

import dataclasses
import enum
from pathlib import Path
from types import UnionType
from typing import Any, TypeGuard, TypeVar, Union, cast, get_args, get_origin

import yaml

ConfigT = TypeVar("ConfigT")


class _MissingValue:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _MissingValue()


def is_missing_value(value: object) -> bool:
    return value is MISSING


def to_primitive(value: object, *, preserve_missing: bool = False) -> object:
    if is_missing_value(value):
        return MISSING if preserve_missing else None
    if dataclasses.is_dataclass(value):
        return {
            field.name: to_primitive(getattr(value, field.name), preserve_missing=preserve_missing)
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(key): to_primitive(item, preserve_missing=preserve_missing) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_primitive(item, preserve_missing=preserve_missing) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.name
    return value


def to_yaml_text(value: object, *, sort_keys: bool = True) -> str:
    payload = to_primitive(value)
    return yaml.safe_dump(payload, sort_keys=sort_keys, default_flow_style=False)


def load_yaml(path: str | Path) -> dict[str, object]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Config at {path} must be a mapping/object")
    return {str(key): item for key, item in payload.items()}


def save_yaml(path: str | Path, value: object, *, sort_keys: bool = True) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(to_yaml_text(value, sort_keys=sort_keys), encoding="utf-8")


def deep_merge(base: object, override: object) -> object:
    if isinstance(base, dict) and isinstance(override, dict):
        merged: dict[str, object] = {str(key): item for key, item in base.items()}
        for key, value in override.items():
            key_str = str(key)
            if key_str in merged:
                merged[key_str] = deep_merge(merged[key_str], value)
            else:
                merged[key_str] = value
        return merged
    return override


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_norm = value.strip().lower()
        if value_norm in ("1", "true", "yes", "y", "on"):
            return True
        if value_norm in ("0", "false", "no", "n", "off"):
            return False
    raise TypeError(f"Cannot coerce {value!r} to bool")


def _is_dataclass_type(annotation: object) -> TypeGuard[type[object]]:
    return isinstance(annotation, type) and dataclasses.is_dataclass(annotation)


def coerce_value(value: object, annotation: object) -> object:
    if is_missing_value(value):
        return MISSING
    if annotation in (Any, object):
        return value
    if annotation in (None, type(None)):  # noqa: E721
        if value is None:
            return None
        raise TypeError(f"Expected None, got {value!r}")

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        options = get_args(annotation)
        if value is None and type(None) in options:
            return None
        for option in options:
            if option is type(None):
                continue
            try:
                return coerce_value(value, option)
            except Exception:
                continue
        raise TypeError(f"Value {value!r} does not match union type {annotation!r}")

    if origin is list:
        if not isinstance(value, list):
            value = [value]
        (item_type,) = get_args(annotation)
        return [coerce_value(item, item_type) for item in value]
    if origin is tuple:
        tuple_args = get_args(annotation)
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Expected tuple-like value for {annotation!r}, got {type(value)!r}")
        items = list(value)
        if len(tuple_args) == 2 and tuple_args[1] is Ellipsis:
            return tuple(coerce_value(item, tuple_args[0]) for item in items)
        if len(tuple_args) != len(items):
            raise TypeError(f"Expected tuple length {len(tuple_args)}, got {len(items)}")
        return tuple(coerce_value(item, tuple_type) for item, tuple_type in zip(items, tuple_args, strict=True))
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict for {annotation!r}, got {type(value)!r}")
        key_type, value_type = get_args(annotation)
        return {coerce_value(key, key_type): coerce_value(item, value_type) for key, item in value.items()}

    if _is_dataclass_type(annotation):
        if isinstance(value, annotation):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"Expected mapping for dataclass {annotation!r}, got {type(value)!r}")
        return dataclass_from_mapping(annotation, {str(key): item for key, item in value.items()})

    if origin is not None:
        raise TypeError(f"Unsupported annotation: {annotation!r}")

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        if isinstance(value, annotation):
            return value
        if isinstance(value, str):
            try:
                return annotation[value]
            except KeyError:
                return annotation(value)
        return annotation(value)

    if annotation is bool:
        return _coerce_bool(value)
    if annotation is int:
        if not isinstance(value, (int, float, str, bytes, bytearray)):
            raise TypeError(f"Cannot coerce {type(value)!r} to int")
        return int(value)
    if annotation is float:
        if not isinstance(value, (int, float, str, bytes, bytearray)):
            raise TypeError(f"Cannot coerce {type(value)!r} to float")
        return float(value)
    if annotation is str:
        return str(value)
    if annotation is Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))
    if isinstance(annotation, type):
        if isinstance(value, annotation):
            return value
        return annotation(value)
    return value


def dataclass_from_mapping(config_type: type[ConfigT], payload: dict[str, object]) -> ConfigT:
    if not dataclasses.is_dataclass(config_type):
        raise TypeError(f"Expected dataclass type, got {config_type!r}")
    field_map = {field.name: field for field in dataclasses.fields(config_type)}
    unknown_keys = [key for key in payload if key not in field_map]
    if unknown_keys:
        raise ValueError(f"Unknown config keys for {config_type.__name__}: {unknown_keys}")

    kwargs: dict[str, object] = {}
    for field in dataclasses.fields(config_type):
        if field.name in payload:
            kwargs[field.name] = coerce_value(payload[field.name], field.type)
        elif field.default is not dataclasses.MISSING:
            kwargs[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            kwargs[field.name] = field.default_factory()
        else:
            raise ValueError(f"Missing required field: {field.name}")
    return config_type(**kwargs)


def _default_mapping(config_type: type[ConfigT]) -> dict[str, object]:
    if not dataclasses.is_dataclass(config_type):
        raise TypeError(f"Expected dataclass type, got {config_type!r}")
    instance = config_type()
    mapping = to_primitive(instance, preserve_missing=True)
    if not isinstance(mapping, dict):
        raise TypeError(f"Expected dict-like config payload, got {type(mapping)!r}")
    mapping_dict = cast(dict[object, object], mapping)
    return {str(key): item for key, item in mapping_dict.items()}


def apply_dotted_override(config_payload: dict[str, object], dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    if not parts:
        raise ValueError(f"Invalid override key: {dotted_key!r}")

    cursor: dict[str, object] = config_payload
    for part in parts[:-1]:
        child = cursor.get(part)
        if isinstance(child, dict):
            child_dict: dict[str, object] = {str(key): item for key, item in child.items()}
        else:
            child_dict = {}
        cursor[part] = child_dict
        cursor = child_dict
    cursor[parts[-1]] = value


def parse_override_token(token: str) -> tuple[str, object]:
    if "=" not in token:
        raise ValueError(f"Invalid override token {token!r}; expected key=value")
    dotted_key, value_str = token.split("=", maxsplit=1)
    dotted_key = dotted_key.strip()
    if not dotted_key:
        raise ValueError(f"Invalid override token {token!r}; empty key")
    value = yaml.safe_load(value_str)
    return dotted_key, value


def merge_config_sources(
    config_type: type[ConfigT],
    sources: list[object],
) -> ConfigT:
    merged_payload: dict[str, object] = _default_mapping(config_type)
    for source in sources:
        source_payload: dict[str, object]
        if isinstance(source, (str, Path)):
            source_payload = load_yaml(source)
        elif dataclasses.is_dataclass(source):
            source_data = to_primitive(source)
            if not isinstance(source_data, dict):
                raise TypeError(f"Invalid dataclass source payload type: {type(source_data)!r}")
            source_payload = {str(key): item for key, item in source_data.items()}
        elif isinstance(source, dict):
            source_payload = {str(key): item for key, item in source.items()}
        else:
            raise TypeError(f"Unsupported config source type: {type(source)!r}")
        merged = deep_merge(merged_payload, source_payload)
        if not isinstance(merged, dict):
            raise TypeError(f"Expected merged payload to be a mapping, got {type(merged)!r}")
        merged_payload = {str(key): item for key, item in merged.items()}
    return dataclass_from_mapping(config_type, merged_payload)


def parse_key_value_overrides(args: list[str]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for token in args:
        key, value = parse_override_token(token)
        apply_dotted_override(payload, key, value)
    return payload


def _annotation_to_name(annotation: object) -> str:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return " | ".join(_annotation_to_name(option) for option in get_args(annotation))
    if origin is list:
        (item_type,) = get_args(annotation)
        return f"list[{_annotation_to_name(item_type)}]"
    if origin is dict:
        key_type, value_type = get_args(annotation)
        return f"dict[{_annotation_to_name(key_type)}, {_annotation_to_name(value_type)}]"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def render_dataclass_help(config_type: type[object], *, prog: str) -> str:
    """Renders dotted-field help text from dataclass metadata."""
    if not dataclasses.is_dataclass(config_type):
        raise TypeError(f"Expected dataclass type, got {config_type!r}")

    lines = [
        f"Usage: {prog} [config.yaml ...] [key=value ...]",
        "",
        "Config fields:",
    ]

    def visit(type_obj: type[object], prefix: str = "") -> None:
        for field in dataclasses.fields(type_obj):
            field_path = field.name if not prefix else f"{prefix}.{field.name}"
            help_text = str(field.metadata.get("help", ""))
            default_display = ""
            if field.default is not dataclasses.MISSING:
                default_display = f" (default: {field.default!r})"
            annotation_name = _annotation_to_name(field.type)
            if _is_dataclass_type(field.type):
                lines.append(f"- `{field_path}`: {annotation_name}{default_display}")
                if help_text:
                    lines.append(f"  {help_text}")
                visit(field.type, field_path)
            else:
                lines.append(f"- `{field_path}`: {annotation_name}{default_display}")
                if help_text:
                    lines.append(f"  {help_text}")

    visit(config_type)
    return "\n".join(lines)
