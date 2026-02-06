"""Typed CLI parsing helpers built on dataclass field metadata."""

import dataclasses
import sys
from dataclasses import MISSING as DATACLASS_MISSING
from types import UnionType
from typing import TypeVar, Union, cast, get_args, get_origin

from xax.utils.structured_config import coerce_value

ARGPARSE_DEST_METADATA_KEY = "cli_name"
CLI_SHORT_METADATA_KEY = "cli_short"
CLI_POSITIONAL_METADATA_KEY = "cli_positional"
CLI_HELP_METADATA_KEY = "help"

_ArgsT = TypeVar("_ArgsT")


@dataclasses.dataclass(frozen=True)
class _CliField:
    field_name: str
    field_type: object
    cli_name: str
    short_name: str | None
    positional: bool
    required: bool
    default: object
    is_bool: bool
    is_list: bool


def _is_bool_type(annotation: object) -> bool:
    if annotation is bool:
        return True
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        options = [option for option in get_args(annotation) if option is not type(None)]
        return len(options) == 1 and options[0] is bool
    return False


def _is_list_type(annotation: object) -> bool:
    origin = get_origin(annotation)
    return origin is list


def _list_item_annotation(annotation: object) -> object:
    origin = get_origin(annotation)
    if origin is list:
        return get_args(annotation)[0]
    return annotation


def _build_fields(args_type: type[_ArgsT]) -> list[_CliField]:
    if not dataclasses.is_dataclass(args_type):
        raise TypeError(f"Expected dataclass type, got {args_type!r}")

    fields: list[_CliField] = []
    for field in dataclasses.fields(args_type):
        cli_name = field.metadata.get(ARGPARSE_DEST_METADATA_KEY, field.name)
        if not isinstance(cli_name, str):
            raise TypeError(f"Invalid cli_name metadata for field {field.name}: {cli_name!r}")
        short_name_raw = field.metadata.get(CLI_SHORT_METADATA_KEY)
        short_name = None if short_name_raw is None else str(short_name_raw)
        positional = bool(field.metadata.get(CLI_POSITIONAL_METADATA_KEY, False))
        required = field.default is DATACLASS_MISSING and field.default_factory is DATACLASS_MISSING
        default = None
        if field.default is not DATACLASS_MISSING:
            default = field.default
        elif field.default_factory is not DATACLASS_MISSING:
            default = field.default_factory()

        fields.append(
            _CliField(
                field_name=field.name,
                field_type=field.type,
                cli_name=cli_name,
                short_name=short_name,
                positional=positional,
                required=required,
                default=default,
                is_bool=_is_bool_type(field.type),
                is_list=_is_list_type(field.type),
            )
        )
    return fields


def _long_option_name(field: _CliField) -> str:
    return f"--{field.cli_name.replace('_', '-')}"


def _consume_option_value(argv: list[str], idx: int, token: str) -> tuple[str, int]:
    if "=" in token:
        _, value = token.split("=", maxsplit=1)
        return value, idx
    next_idx = idx + 1
    if next_idx >= len(argv):
        raise ValueError(f"Missing value for option: {token}")
    return argv[next_idx], next_idx


def _build_value_map(fields: list[_CliField]) -> dict[str, object]:
    value_map: dict[str, object] = {}
    for field in fields:
        if field.is_list:
            value_map[field.field_name] = []
        elif field.default is not None or not field.required:
            value_map[field.field_name] = field.default
    return value_map


def _assign_positionals(
    positional_fields: list[_CliField],
    positional_tokens: list[str],
    value_map: dict[str, object],
) -> list[str]:
    token_idx = 0
    for field_idx, field in enumerate(positional_fields):
        is_last = field_idx == len(positional_fields) - 1
        if field.is_list and is_last:
            item_type = _list_item_annotation(field.field_type)
            remaining = [coerce_value(token, item_type) for token in positional_tokens[token_idx:]]
            value_map[field.field_name] = remaining
            token_idx = len(positional_tokens)
            continue

        if token_idx >= len(positional_tokens):
            if field.required:
                raise ValueError(f"Missing required positional argument: {field.cli_name}")
            continue

        value_map[field.field_name] = coerce_value(positional_tokens[token_idx], field.field_type)
        token_idx += 1

    return positional_tokens[token_idx:]


def parse_known_args_as(args_type: type[_ArgsT], argv: list[str]) -> tuple[_ArgsT, list[str]]:
    fields = _build_fields(args_type)
    value_map = _build_value_map(fields)
    option_map: dict[str, _CliField] = {}
    positional_fields: list[_CliField] = []
    for field in fields:
        if field.positional:
            positional_fields.append(field)
            continue
        option_map[_long_option_name(field)] = field
        if field.short_name is not None:
            option_map[f"-{field.short_name}"] = field

    unknown_tokens: list[str] = []
    positional_tokens: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == "--":
            positional_tokens.extend(argv[idx + 1 :])
            break

        option_token = token.split("=", maxsplit=1)[0] if token.startswith("--") and "=" in token else token
        if option_token in option_map:
            field = option_map[option_token]
            if field.is_bool:
                value_map[field.field_name] = True
            else:
                value_str, idx = _consume_option_value(argv, idx, token)
                if field.is_list:
                    values = value_map.get(field.field_name, [])
                    if not isinstance(values, list):
                        values = []
                    values_list = cast(list[object], values)
                    item_type = _list_item_annotation(field.field_type)
                    values_list.append(coerce_value(value_str, item_type))
                    value_map[field.field_name] = values_list
                else:
                    value_map[field.field_name] = coerce_value(value_str, field.field_type)
        elif token.startswith("--no-"):
            positive_name = f"--{token[5:]}"
            field = option_map.get(positive_name)
            if field is None or not field.is_bool:
                unknown_tokens.append(token)
            else:
                value_map[field.field_name] = False
        elif token.startswith("-"):
            unknown_tokens.append(token)
        else:
            positional_tokens.append(token)
        idx += 1

    unknown_positionals = _assign_positionals(positional_fields, positional_tokens, value_map)
    unknown_tokens.extend(unknown_positionals)

    missing_required = [field.field_name for field in fields if field.required and field.field_name not in value_map]
    if missing_required:
        raise ValueError(f"Missing required CLI argument(s): {missing_required}")

    parsed = args_type(**value_map)
    return parsed, unknown_tokens


def parse_args_as(args_type: type[_ArgsT], argv: list[str] | None = None) -> _ArgsT:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parsed, unknown = parse_known_args_as(args_type, argv_list)
    if unknown:
        raise ValueError(f"Unknown CLI argument(s): {unknown}")
    return parsed


def render_help_text(args_type: type[_ArgsT], *, prog: str, description: str | None = None) -> str:
    fields = _build_fields(args_type)
    field_map = {field.name: field for field in dataclasses.fields(args_type)}
    lines: list[str] = [f"Usage: {prog} [options]"]
    if description is not None:
        lines.extend(["", description])
    lines.extend(["", "Options:"])

    for field in fields:
        field_meta = field_map[field.field_name]
        help_text = str(field_meta.metadata.get(CLI_HELP_METADATA_KEY, ""))
        if field.positional:
            cli_label = f"<{field.cli_name}>"
        else:
            cli_label = _long_option_name(field)
            if field.short_name is not None:
                cli_label = f"-{field.short_name}, {cli_label}"
        default_suffix = ""
        if not field.required and field.default not in (None, [], False):
            default_suffix = f" (default: {field.default!r})"
        lines.append(f"  {cli_label:24s} {help_text}{default_suffix}".rstrip())

    return "\n".join(lines)
