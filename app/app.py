"""
Main application.

Pages are loaded in __init__.py.

See:
    - https://reflex.dev/docs/advanced-onboarding/
      code-structure#example_big_app/example_big_app.py
    - https://reflex.dev/docs/advanced-onboarding/code-structure#top-level-package:-
"""

import traceback
import reflex as rx
from app.lib.logger import logger
from app.lib.exceptions import FinAppException


def handle_backend_exception(exception: Exception):
    """
    Global backend exception handler for the FinApp application.

    Logs all exceptions with full context and provides user-friendly error messages.

    Args:
        exception: The exception that occurred
    """
    if isinstance(exception, FinAppException):
        # Custom app exceptions with user-friendly messages
        logger.error(
            f"FinApp exception [{exception.error_id}]: {exception.message} | "
            f"Context: {exception.context}"
        )
        return rx.toast.error(
            f"{exception.user_message} (Error ID: {exception.error_id})",
            duration=10000,  # 10 seconds for errors
        )
    else:
        # Unexpected system exceptions
        error_id = str(hash(str(exception)))[-8:]
        logger.error(
            f"Unexpected exception [{error_id}]: {exception} | "
            f"Type: {type(exception).__name__} | Traceback: {traceback.format_exc()}"
        )

        return rx.toast.error(
            f"An unexpected error occurred. Please try again. (Error ID: {error_id})",
            duration=15000,  # 15 seconds for unexpected errors
        )


app = rx.App(  # pylint: disable=not-callable
    # see https://reflex.dev/docs/utility-methods/exception-handlers/
    backend_exception_handler=handle_backend_exception
)
