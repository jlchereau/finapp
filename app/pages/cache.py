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
        "params",
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
            self.selected_cache = self.cache_dirs[-1]  # Select most recent by default
            self.load_log_data()

    def select_cache(self, cache_dir: str):
        """Handle cache selection."""
        if cache_dir != self.selected_cache:
            self.selected_cache = cache_dir
            self.load_log_data()
            return rx.toast.info(f"Switched to cache: {cache_dir}")

    def load_log_data(self):
        """Load log data for selected cache."""
        if not self.selected_cache:
            self.log_data = []
            return

        self.is_loading = True
        log_entries = self.storage.get_log_data(self.selected_cache)
        self.log_data = log_entries
        self.is_loading = False

        if not log_entries:
            return rx.toast.warning(
                f"No log data found for cache: {self.selected_cache}"
            )
        else:
            return rx.toast.info(f"Loaded {len(log_entries)} log entries")

    def show_delete_confirmation(self):
        """Show delete confirmation dialog."""
        if self.selected_cache:
            self.show_delete_dialog = True

    def confirm_delete(self):
        """Confirm and execute cache deletion."""
        if not self.selected_cache:
            self.show_delete_dialog = False
            return rx.toast.error("No cache selected for deletion")

        success = self.storage.delete_date_folder(self.selected_cache)
        deleted_cache = self.selected_cache

        if success:
            # Refresh cache list and clear selection
            self.cache_dirs = self.storage.list_date_folders()
            self.selected_cache = ""
            self.log_data = []
            self.show_delete_dialog = False
            return rx.toast.success(f"Successfully deleted cache: {deleted_cache}")
        else:
            self.show_delete_dialog = False
            return rx.toast.error(f"Failed to delete cache: {deleted_cache}")

    def cancel_delete(self):
        """Cancel delete operation."""
        self.show_delete_dialog = False


# pylint: disable=not-callable
@rx.page(route="/cache", on_load=State.on_load)  # pyright: ignore[reportArgumentType]
@template
def page():
    """The cache management page."""
    return rx.vstack(
        rx.heading("Cache Management", size="5"),
        rx.box(
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
                ),
                rx.spacer(),
            )
        ),
        rx.cond(
            State.is_loading,
            rx.spinner(loading=True),
            rx.box(
                rx.data_table(
                    data=State.log_data,
                    columns=State.log_columns,
                    pagination=True,
                    search=True,
                    sort=True,
                ),
            ),
        ),
        # Delete confirmation dialog
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
        spacing="5",
    )
