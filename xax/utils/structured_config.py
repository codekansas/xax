"""Typed configuration helpers for dataclass-based config + CLI overrides."""

import copy
import dataclasses
import enum
import os
import re
import shutil
import sys
from pathlib import Path
from types import UnionType
from typing import Any, Callable, Literal, Mapping, TypeGuard, TypeVar, Union, cast, get_args, get_origin, overload

import yaml

from xax.utils.text import TextBlock, colored, render_text_blocks

ConfigT = TypeVar("ConfigT")
FieldT = TypeVar("FieldT")
_HELP_DEFAULT_METADATA_KEY = "_xax_help_default_value"


class _MissingValue:
    def __repr__(self) -> str:
        return "MISSING"


MISSING = _MissingValue()


class _UnsetValue:
    pass


_UNSET = _UnsetValue()


def is_missing_value(value: object) -> bool:
    return value is MISSING


@overload
def field(  # noqa: A001
    value: _MissingValue,
    *,
    static: bool = True,
    help: str | None = None,
    metadata: Mapping[str, object] | None = None,
    **metadata_items: object,
) -> Any: ...  # noqa: ANN401


@overload
def field(  # noqa: A001
    value: Callable[[], FieldT],
    *,
    static: bool = True,
    help: str | None = None,
    metadata: Mapping[str, object] | None = None,
    **metadata_items: object,
) -> FieldT: ...


@overload
def field(  # noqa: A001
    value: FieldT,
    *,
    static: bool = True,
    help: str | None = None,
    metadata: Mapping[str, object] | None = None,
    **metadata_items: object,
) -> FieldT: ...


@overload
def field(  # noqa: A001
    value: _UnsetValue = _UNSET,
    *,
    default_factory: Callable[[], FieldT],
    static: bool = True,
    help: str | None = None,
    metadata: Mapping[str, object] | None = None,
    **metadata_items: object,
) -> FieldT: ...


@overload
def field(  # noqa: A001
    value: _UnsetValue = _UNSET,
    *,
    default_factory: _UnsetValue = _UNSET,
    static: bool = True,
    help: str | None = None,
    metadata: Mapping[str, object] | None = None,
    **metadata_items: object,
) -> Any: ...  # noqa: ANN401


def field(  # noqa: A001
    value: FieldT | _UnsetValue = _UNSET,
    *,
    default_factory: Callable[[], FieldT] | _UnsetValue = _UNSET,
    static: bool = True,
    help: str | None = None,
    metadata: Mapping[str, object] | None = None,
    **metadata_items: object,
) -> FieldT | Any:
    """Creates a dataclass field with optional structured metadata.

    Args:
        value: Optional default value. If omitted, the field is required.
        default_factory: Optional default factory. Mutually exclusive with ``value``.
        static: Whether to mark this dataclass field as static metadata.
        help: Optional help string stored in field metadata.
        metadata: Optional metadata mapping.
        metadata_items: Additional metadata key-value pairs.

    Returns:
        A dataclass field descriptor.
    """
    if value is not _UNSET and default_factory is not _UNSET:
        raise ValueError("Cannot specify both `value` and `default_factory`")
    merged_metadata = dict(metadata or {})
    merged_metadata.update(metadata_items)
    merged_metadata["static"] = static
    if help is not None:
        merged_metadata["help"] = help

    if default_factory is not _UNSET:
        factory = cast(Callable[[], FieldT], default_factory)
        return dataclasses.field(default_factory=factory, metadata=merged_metadata)

    if value is _UNSET:
        return dataclasses.field(metadata=merged_metadata)

    if callable(value):
        callable_value = cast(Callable[[], FieldT], value)
        return dataclasses.field(default_factory=callable_value, metadata=merged_metadata)

    default_value_for_help = copy.deepcopy(value) if value.__class__.__hash__ is None else value
    merged_metadata.setdefault(_HELP_DEFAULT_METADATA_KEY, default_value_for_help)

    if value.__class__.__hash__ is None:
        default_value = cast(FieldT, value)
        return dataclasses.field(
            default_factory=lambda default_value=default_value: copy.deepcopy(default_value),
            metadata=merged_metadata,
        )
    return dataclasses.field(default=value, metadata=merged_metadata)


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


_INTERPOLATION_RE = re.compile(r"\$\{([^{}]+)\}")


def _lookup_path(payload: object, path_parts: tuple[str, ...]) -> object:
    cursor = payload
    for part in path_parts:
        if isinstance(cursor, dict):
            cursor_dict = {str(key): item for key, item in cursor.items()}
            if part not in cursor_dict:
                raise KeyError(f"Unknown interpolation key: {'.'.join(path_parts)}")
            cursor = cursor_dict[part]
            continue
        if isinstance(cursor, list):
            if not part.isdigit():
                raise KeyError(f"Interpolation index must be an integer: {part!r}")
            item_idx = int(part)
            if item_idx < 0 or item_idx >= len(cursor):
                raise KeyError(f"Interpolation index out of range: {item_idx}")
            cursor = cursor[item_idx]
            continue
        raise KeyError(f"Cannot interpolate into value of type {type(cursor)!r}")
    return cursor


def resolve_interpolations(payload: dict[str, object]) -> dict[str, object]:
    """Resolves OmegaConf-like interpolations in a payload.

    Supported forms:
    - ``${oc.env:VAR}``
    - ``${oc.env:VAR,default}``
    - ``${foo.bar}`` (dotted lookup in the same merged payload)
    """
    root_payload: dict[str, object] = {str(key): value for key, value in payload.items()}
    cache: dict[tuple[str, ...], object] = {}
    resolving: set[tuple[str, ...]] = set()

    def resolve_path(path_parts: tuple[str, ...]) -> object:
        if path_parts in cache:
            return cache[path_parts]
        if path_parts in resolving:
            path_str = ".".join(path_parts)
            raise ValueError(f"Circular interpolation detected at {path_str!r}")

        resolving.add(path_parts)
        raw_value = _lookup_path(root_payload, path_parts)
        resolved_value = resolve_value(raw_value, path_parts)
        cache[path_parts] = resolved_value
        resolving.remove(path_parts)
        return resolved_value

    def resolve_expr(expr: str, current_path: tuple[str, ...]) -> object:
        expr = expr.strip()
        if expr.startswith("oc.env:"):
            payload_expr = expr.removeprefix("oc.env:").strip()
            env_name, env_default = payload_expr.split(",", maxsplit=1) if "," in payload_expr else (payload_expr, None)
            env_name = env_name.strip()
            if not env_name:
                raise ValueError("Environment interpolation must specify a variable name")
            env_value = os.environ.get(env_name)
            if env_value is not None:
                return env_value
            if env_default is None:
                raise KeyError(f"Missing required environment variable: {env_name}")
            default_value = yaml.safe_load(env_default.strip())
            return resolve_value(default_value, current_path)

        ref_parts = tuple(part.strip() for part in expr.split(".") if part.strip())
        if not ref_parts:
            raise ValueError(f"Invalid interpolation expression: {expr!r}")
        return resolve_path(ref_parts)

    def resolve_string(value: str, current_path: tuple[str, ...]) -> object:
        matches = list(_INTERPOLATION_RE.finditer(value))
        if not matches:
            return value

        # Preserve non-string referenced values when the entire field is one interpolation token.
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            return resolve_expr(matches[0].group(1), current_path)

        resolved_parts: list[str] = []
        cursor = 0
        for match in matches:
            resolved_parts.append(value[cursor : match.start()])
            replacement = resolve_expr(match.group(1), current_path)
            resolved_parts.append(str(replacement))
            cursor = match.end()
        resolved_parts.append(value[cursor:])
        return "".join(resolved_parts)

    def resolve_value(value: object, current_path: tuple[str, ...]) -> object:
        if isinstance(value, dict):
            return {str(key): resolve_value(item, current_path + (str(key),)) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve_value(item, current_path + (str(item_idx),)) for item_idx, item in enumerate(value)]
        if isinstance(value, str):
            return resolve_string(value, current_path)
        return value

    resolved = resolve_value(root_payload, ())
    if not isinstance(resolved, dict):
        raise TypeError(f"Expected resolved payload to be a mapping, got {type(resolved)!r}")
    return {str(key): value for key, value in resolved.items()}


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
    if origin is Literal:
        options = get_args(annotation)
        for option in options:
            if value == option:
                return option
            try:
                coerced = coerce_value(value, type(option))
            except Exception:
                continue
            if coerced == option:
                return option
        raise TypeError(f"Value {value!r} does not match literal options {options!r}")

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
    merged_payload = resolve_interpolations(merged_payload)
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
    if origin is Literal:
        return f"Literal[{', '.join(repr(option) for option in get_args(annotation))}]"
    if origin is list:
        (item_type,) = get_args(annotation)
        return f"list[{_annotation_to_name(item_type)}]"
    if origin is dict:
        key_type, value_type = get_args(annotation)
        return f"dict[{_annotation_to_name(key_type)}, {_annotation_to_name(value_type)}]"
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _format_help_default(value: object) -> str:
    if isinstance(value, enum.Enum):
        return repr(value.value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def _can_render_help_default(value: object) -> bool:
    if dataclasses.is_dataclass(value):
        return False
    if callable(value):
        return False
    return True


def _column_widths_for_help(total_width: int) -> tuple[int, int, int, int]:
    usable_width = max(total_width, 96)
    field_width = max(20, min(42, int(usable_width * 0.30)))
    type_width = max(18, min(34, int(usable_width * 0.24)))
    default_width = max(12, min(28, int(usable_width * 0.16)))
    # 13 accounts for table separators and framing from render_text_blocks.
    description_width = max(24, usable_width - field_width - type_width - default_width - 13)
    return field_width, type_width, default_width, description_width


def render_dataclass_help(config_type: type[object], *, prog: str, use_color: bool | None = None) -> str:
    """Renders dotted-field help text from dataclass metadata."""
    if not dataclasses.is_dataclass(config_type):
        raise TypeError(f"Expected dataclass type, got {config_type!r}")

    use_color = sys.stdout.isatty() if use_color is None else use_color
    header_color = "light-cyan" if use_color else None
    field_color = "light-blue" if use_color else None
    type_color = "grey" if use_color else None

    rows: list[tuple[str, str, str, str]] = []

    def visit(type_obj: type[object], prefix: str = "") -> None:
        for field in dataclasses.fields(type_obj):
            field_path = field.name if not prefix else f"{prefix}.{field.name}"
            help_text = str(field.metadata.get("help", ""))
            default_display = "-"
            if field.default is not dataclasses.MISSING:
                default_display = _format_help_default(field.default)
            elif _HELP_DEFAULT_METADATA_KEY in field.metadata:
                metadata_default = field.metadata[_HELP_DEFAULT_METADATA_KEY]
                if _can_render_help_default(metadata_default):
                    default_display = _format_help_default(metadata_default)
            elif field.default_factory is not dataclasses.MISSING:
                # Show simple built-in container defaults; hide other factories.
                if field.default_factory in (list, dict, set, tuple, frozenset):
                    default_display = _format_help_default(field.default_factory())
            annotation_name = _annotation_to_name(field.type).replace("NoneType", "None")
            description = help_text if help_text else "-"
            rows.append((field_path, annotation_name, default_display, description))

            if _is_dataclass_type(field.type):
                visit(field.type, field_path)

    visit(config_type)
    rows.sort(key=lambda row: row[0])

    terminal_width = shutil.get_terminal_size(fallback=(120, 20)).columns
    field_width, type_width, default_width, description_width = _column_widths_for_help(terminal_width)

    table_rows: list[list[TextBlock]] = [
        [
            TextBlock("field", color=header_color, bold=True, width=field_width),
            TextBlock("type", color=header_color, bold=True, width=type_width),
            TextBlock("default", color=header_color, bold=True, width=default_width),
            TextBlock("description", color=header_color, bold=True, width=description_width),
        ]
    ]
    for field_path, annotation_name, default_display, description in rows:
        table_rows.append(
            [
                TextBlock(field_path, color=field_color, width=field_width),
                TextBlock(annotation_name, color=type_color, width=type_width),
                TextBlock(default_display, width=default_width),
                TextBlock(description, width=description_width),
            ]
        )

    usage_label = colored("Usage", "light-cyan", bold=True) if use_color else "Usage"
    config_label = colored("Config Fields", "light-cyan", bold=True) if use_color else "Config fields"
    table_rendered = render_text_blocks(table_rows, align_all_blocks=True)
    return "\n".join(
        [
            f"{usage_label}: {prog} [config.yaml ...] [key=value ...]",
            "",
            f"{config_label}:",
            table_rendered,
        ]
    )
