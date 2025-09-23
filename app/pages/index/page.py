"""
Index page

Displays the content of /assets/index.md using rx.markdown.
See: https://reflex.dev/docs/library/typography/markdown/
"""

from pathlib import Path
import reflex as rx
from app.templates.template import template


def load_markdown(file_name: str) -> str:
    """Load markdown from the assets folder."""
    path = Path(f"assets/{file_name}.md")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "# File not found"


MD = load_markdown("index")


# pylint: disable=not-callable
# pyright: ignore[reportArgumentType]
# pyrefly: ignore[bad-argument-type]
@rx.page(route="/")
@template
def page():
    """The index page."""
    return rx.container(
        rx.markdown(MD, padding="1em", font_size="1.1em", width="80%"),
        margin="auto",
        padding="2em",
    )
