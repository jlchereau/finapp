"""Cache management page"""

from datetime import datetime
from typing import List
import reflex as rx
from ..templates.template import template


class State(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""
    log_dirs: List[str] = [
        "20250811",
        "20250814",
    ]
    log_data: List = [
        [datetime.now(), "info", "All going well", "StockPriceWorkflow", "cache.py", "get_cache"],
        [datetime.now(), "debug", "ticker is MSFT", "PlayerStatsWorkflow", "cache.py", "get_player_stats"],
    ]
    log_columns: List[str] = ["Time", "Level", "Message", "Context", "File", "Function"]

# pylint: disable=not-callable
@rx.page(route="/cache")  # pyright: ignore[reportArgumentType]
@template
def page():
    """The cache management page."""
    return rx.vstack(
        rx.heading("Cache Management", size="5"),
        rx.box(
            rx.hstack(
                rx.text("Cache: "),
                rx.select(
                    items=State.log_dirs,
                    placeholder="Select date...",
                ),
                rx.button(
                    rx.icon("trash")
                ),
                rx.spacer()
            )
        ),
        rx.box(
            rx.data_table(
                data=State.log_data,
                columns=State.log_columns,
            ),
        ),
        spacing="5",
    )
