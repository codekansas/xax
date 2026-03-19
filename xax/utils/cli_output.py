"""Helpers for clean, user-facing CLI output."""

import sys
import traceback
from dataclasses import dataclass
from typing import Sequence, TextIO

from xax.utils.text import (
    Color,
    TextBlock,
    colored,
    display_width,
    outlined,
    render_text_blocks,
    truncate_to_display_width,
    uncolored,
)

_LEVEL_COLORS: dict[str, Color] = {
    "status": "light-green",
    "info": "light-cyan",
    "warning": "light-yellow",
    "error": "light-red",
}
_TABLE_TITLE_COLOR: Color = "light-magenta"


@dataclass(frozen=True)
class CliOutput:
    """Formats and writes CLI messages with consistent, compact styling."""

    prefix: str | None = None
    use_color: bool = True

    def _format_prefix(self, level: str) -> str:
        parts: list[str] = []
        if self.prefix is not None:
            prefix_label = f"[{self.prefix}]"
            parts.append(colored(prefix_label, "magenta", bold=True) if self.use_color else prefix_label)
        if level != "info":
            level_label = level.upper()
            level_color = _LEVEL_COLORS[level]
            parts.append(colored(level_label, level_color, bold=True) if self.use_color else level_label)
        return " ".join(parts)

    def _format_message(self, message: str, *args: object) -> str:
        return message % args if args else message

    def _emit(self, level: str, message: str, *args: object, stream: TextIO | None = None) -> None:
        if stream is None:
            stream = sys.stderr if level == "error" else sys.stdout
        payload = self._format_message(message, *args)
        prefix = self._format_prefix(level)
        if prefix:
            stream.write(f"{prefix} {payload}\n")
        else:
            stream.write(f"{payload}\n")
        stream.flush()

    def status(self, message: str, *args: object) -> None:
        self._emit("status", message, *args)

    def info(self, message: str, *args: object) -> None:
        self._emit("info", message, *args)

    def warning(self, message: str, *args: object) -> None:
        self._emit("warning", message, *args)

    def error(self, message: str, *args: object) -> None:
        self._emit("error", message, *args)

    def exception(self, message: str, *args: object) -> None:
        self.error(message, *args)
        traceback.print_exc(file=sys.stderr)

    def section(self, title: str) -> None:
        if self.use_color:
            self.plain(outlined(title, inner="light-cyan", side="cyan", bold=True))
        else:
            self.plain(f"[{title}]")

    def kv(self, key: str, value: object, *, key_width: int = 16, indent: int = 2) -> None:
        key_label = key.ljust(key_width)
        self.info("%s%s: %s", " " * max(0, indent), key_label, value)

    def list_item(self, message: str, *, indent: int = 2, bullet: str = "-") -> None:
        self.info("%s%s %s", " " * max(0, indent), bullet, message)

    def plain(self, message: str, *args: object, stream: TextIO | None = None) -> None:
        if stream is None:
            stream = sys.stdout
        payload = self._format_message(message, *args)
        stream.write(f"{payload}\n")
        stream.flush()

    def table(
        self,
        *,
        title: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        header_color: Color = "light-cyan",
        title_color: Color = _TABLE_TITLE_COLOR,
    ) -> None:
        header_blocks = [
            TextBlock(header, color=header_color if self.use_color else None, bold=True)
            for header in headers
        ]
        body_blocks = [[TextBlock(cell) for cell in row] for row in rows]
        block_rows = [header_blocks, *body_blocks]
        rendered = render_text_blocks(block_rows, align_all_blocks=True)
        rendered_lines = rendered.splitlines()
        if rendered_lines:
            rendered_lines[0] = self._render_titled_table_top(
                rendered_lines[0],
                title=title,
                title_color=title_color,
            )
            rendered = "\n".join(rendered_lines)
        self.plain(rendered)
        self.plain("")

    def _render_titled_table_top(self, top_border: str, *, title: str, title_color: Color) -> str:
        plain_top_border = uncolored(top_border)
        if not plain_top_border.startswith("┌") or not plain_top_border.endswith("┐"):
            return top_border
        if display_width(plain_top_border) <= 3:
            return top_border

        inner_width = display_width(plain_top_border) - 2
        title_text = f" {title.strip()} "
        if display_width(title_text) > inner_width:
            max_title_width = inner_width - 2
            if max_title_width <= 0:
                return top_border
            raw_title = truncate_to_display_width(title.strip(), max_title_width, "...")
            title_text = f" {raw_title} "

        if display_width(title_text) > inner_width:
            return top_border

        title_width = display_width(title_text)
        left_width = (inner_width - title_width) // 2
        right_width = inner_width - title_width - left_width
        title_rendered = colored(title_text, title_color, bold=True) if self.use_color else title_text
        return f"┌{'─' * left_width}{title_rendered}{'─' * right_width}┐"


def get_cli_output(prefix: str | None = None) -> CliOutput:
    use_color = sys.stdout.isatty() and sys.stderr.isatty()
    return CliOutput(prefix=prefix, use_color=use_color)
