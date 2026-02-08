"""Tests for terminal text helpers."""

from xax.utils.cli_output import CliOutput
from xax.utils.text import TextBlock, display_width, render_text_blocks, truncate_to_display_width


def test_display_width_handles_wide_unicode_characters() -> None:
    assert display_width("abc") == 3
    assert display_width("🚀") == 2
    assert display_width("👨‍💻") == 2
    assert display_width("👍🏽") == 2
    assert display_width("你好") == 4


def test_render_text_blocks_aligns_rows_with_wide_characters() -> None:
    rendered = render_text_blocks(
        [
            [TextBlock("name"), TextBlock("status")],
            [TextBlock("plain"), TextBlock("running")],
            [TextBlock("emoji"), TextBlock("🚀 launch")],
            [TextBlock("zwj"), TextBlock("👨‍💻 coding")],
            [TextBlock("cjk"), TextBlock("你好 world")],
        ],
        align_all_blocks=True,
    )
    widths = {display_width(line) for line in rendered.splitlines()}
    assert len(widths) == 1


def test_titled_table_top_preserves_border_width_for_emoji_title() -> None:
    output = CliOutput(use_color=False)
    top_border = "┌────────────────────┐"
    rendered_top = output._render_titled_table_top(
        top_border,
        title="🚀 Queue Status 🚀",
        title_color="light-magenta",
    )
    assert display_width(rendered_top) == display_width(top_border)


def test_truncate_to_display_width_handles_emoji_sequences() -> None:
    truncated = truncate_to_display_width("👨‍💻 debugging", 8)
    assert display_width(truncated) <= 8
    assert truncated.endswith("...")
    assert "👨‍💻" in truncated
