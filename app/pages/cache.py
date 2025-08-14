"""Cache management page"""

from typing import List, Dict, Any
import reflex as rx
from ..lib.storage import DateBasedStorage
from ..templates.template import template


class State(rx.State):  # pylint: disable=inherit-non-class
    """The app state."""

    cache_dirs: List[str] = []
    selected_cache: str = ""
    log_data: List[Dict[str, Any]] = []
    log_columns: List[str] = [
        "timestamp",
        "level",
        "message",
        "context",
        "file",
        "function",
    ]
    show_delete_dialog: bool = False
    is_loading: bool = False

    @property
    def storage(self) -> DateBasedStorage:
        """Get storage instance."""
        return DateBasedStorage()

    def on_load(self):
        """Load cache directories on page load."""
        self.cache_dirs = self.storage.list_date_folders()
        if self.cache_dirs and not self.selected_cache:
            self.selected_cache = self.cache_dirs[0]  # Select most recent by default
            self.load_log_data()

    def select_cache(self, cache_dir: str):
        """Handle cache selection."""
        if cache_dir != self.selected_cache:
            self.selected_cache = cache_dir
            self.load_log_data()

    def load_log_data(self):
        """Load log data for selected cache."""
        if not self.selected_cache:
            self.log_data = []
            return

        self.is_loading = True
        log_entries = self.storage.get_log_data(self.selected_cache)
        self.log_data = log_entries
        self.is_loading = False

    def show_delete_confirmation(self):
        """Show delete confirmation dialog."""
        if self.selected_cache:
            self.show_delete_dialog = True

    def confirm_delete(self):
        """Confirm and execute cache deletion."""
        if not self.selected_cache:
            self.show_delete_dialog = False
            return

        success = self.storage.delete_date_folder(self.selected_cache)

        if success:
            # Refresh cache list and clear selection
            self.cache_dirs = self.storage.list_date_folders()
            self.selected_cache = ""
            self.log_data = []

        self.show_delete_dialog = False

    def cancel_delete(self):
        """Cancel delete operation."""
        self.show_delete_dialog = False


def cache_selector() -> rx.Component:
    """Cache selector component."""
    return rx.box(
        rx.hstack(
            rx.text("Cache: "),
            rx.select(
                items=State.cache_dirs,
                value=State.selected_cache,
                placeholder="Select cache date...",
                on_change=State.select_cache,
            ),
            rx.button(
                rx.icon("trash"),
                on_click=State.show_delete_confirmation,
                disabled=State.selected_cache == "",
                variant="soft",
                color="red",
                size="2",
            ),
            rx.spacer(),
        )
    )


def log_grid() -> rx.Component:
    """Log grid component."""
    return rx.cond(
        State.is_loading,
        rx.spinner(loading=True),
        rx.box(
            rx.data_table(
                data=State.log_data,
                columns=State.log_columns,
                pagination=True,
                search=True,
                sort=True,
                width="100%",
            ),
        ),
    )


def confirm_delete_dialog() -> rx.Component:
    """Delete confirmation dialog."""
    return (
        rx.alert_dialog.root(
            rx.alert_dialog.content(
                rx.alert_dialog.title("Delete Cache"),
                rx.alert_dialog.description(
                    f"Are you sure you want to delete the cache folder "
                    f"'{State.selected_cache}'? This action cannot be undone."
                ),
                rx.flex(
                    rx.alert_dialog.cancel(
                        rx.button(
                            "Cancel",
                            variant="soft",
                            on_click=State.cancel_delete,
                        )
                    ),
                    rx.alert_dialog.action(
                        rx.button(
                            "Delete",
                            variant="solid",
                            color="red",
                            on_click=State.confirm_delete,
                        )
                    ),
                    spacing="3",
                    justify="end",
                ),
            ),
            open=State.show_delete_dialog,
        ),
    )


# pylint: disable=not-callable
@rx.page(route="/cache", on_load=State.on_load)  # pyright: ignore[reportArgumentType]
@template
def page():
    """The cache management page."""
    return rx.vstack(
        rx.heading("Cache Management", size="6", margin_bottom="1rem"),
        cache_selector(),
        log_grid(),
        confirm_delete_dialog(),
        spacing="0",
    )
