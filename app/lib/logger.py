"""
Custom CSV logger for FinApp that logs to date-based folders.

Logs are stored in data/YYYYMMDD/log.csv with the following columns:
- timestamp: Date and time of the log entry
- level: Log level (debug, info, warning, error)
- message: Concise message formatted for table display
- context: Workflow name from stack or 'app'
- file: File path relative to project root
- function: Function name with class context (ClassName:method_name)
- params: Function parameters as JSON string (when feasible)
- exception: Full exception details (when applicable)
"""

import csv
import inspect
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .storage import DateBasedStorage
from .settings import settings


class CSVLogger:
    """
    A custom logger that writes log entries to CSV files organized by date.
    """

    # Log level hierarchy for filtering
    LEVEL_HIERARCHY = {
        "debug": 0,
        "info": 1,
        "warning": 2,
        "error": 3,
    }

    def __init__(self, storage: Optional[DateBasedStorage] = None):
        """
        Initialize the CSV logger.

        Args:
            storage: DateBasedStorage instance. If None, creates default instance.
        """
        self.storage = storage or DateBasedStorage()
        self._lock = threading.Lock()

    def _should_log(self, level: str) -> bool:
        """
        Check if a log entry should be recorded based on the configured DEBUG_LEVEL.

        Args:
            level: The log level to check

        Returns:
            True if the log should be recorded, False otherwise
        """
        try:
            current_level_value = self.LEVEL_HIERARCHY.get(level.lower(), 0)
            configured_level_value = self.LEVEL_HIERARCHY.get(
                settings.DEBUG_LEVEL.lower(), 0
            )
            return current_level_value >= configured_level_value
        except (AttributeError, KeyError):
            # If there's any error, default to logging (fail-safe)
            return True

    def _get_log_file_path(self) -> Path:
        """Get the path to today's log file."""
        return self.storage.get_file_path("log.csv")

    def _format_message_for_display(self, message: Union[str, Exception]) -> str:
        """Format message for table display (concise, single-line)."""
        message_str = str(message)

        # Handle common exception patterns
        if "ValidationError:" in message_str:
            # Extract just the first validation error
            lines = message_str.split("\n")
            for line in lines:
                if "Field required" in line or "Input should be" in line:
                    # Extract field name and error type
                    if "Field required" in line:
                        field_match = line.split()[0] if line.strip() else "unknown"
                        return f"ValidationError: Field required for {field_match}"
                    else:
                        return f"ValidationError: {line.strip()}"
            return "ValidationError: Multiple validation errors"

        # Handle other multi-line exceptions
        if "\n" in message_str:
            lines = message_str.split("\n")
            first_line = lines[0].strip()
            if len(first_line) > 100:  # Truncate very long first lines
                return first_line[:100] + "..."
            return first_line

        # Truncate single-line messages if too long
        if len(message_str) > 150:
            return message_str[:150] + "..."

        return message_str

    def _get_class_name_from_frame(self, frame) -> Optional[str]:
        """Extract class name from frame if called from instance method."""
        try:
            # Check if 'self' exists in local variables
            if "self" in frame.f_locals:
                self_obj = frame.f_locals["self"]
                return self_obj.__class__.__name__
            return None
        except (AttributeError, KeyError):
            return None

    def _get_caller_info(self) -> Dict[str, Any]:
        """Extract caller information from the stack."""
        frame = inspect.currentframe()
        try:
            # Skip frames: current -> _log -> debug/info/warning/error -> actual caller
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back.f_back
            else:
                caller_frame = None

            if caller_frame is None:
                return {
                    "file": "unknown",
                    "function": "unknown",
                    "context": "app",
                    "params": "{}",
                }

            # Get file path relative to project root
            file_path = Path(caller_frame.f_code.co_filename)
            try:
                relative_path = file_path.relative_to(self.storage.base_path.parent)
            except ValueError:
                relative_path = file_path

            # Get function name with class context if available
            function_name = caller_frame.f_code.co_name
            class_name = self._get_class_name_from_frame(caller_frame)

            if class_name:
                function_name = f"{class_name}:{function_name}"

            # Determine context from stack
            context = self._determine_context()

            # Get function parameters
            params = self._get_function_params(caller_frame)

            return {
                "file": str(relative_path),
                "function": function_name,
                "context": context,
                "params": params,
            }
        finally:
            del frame

    def _determine_context(self) -> str:
        """Determine if the call is from a workflow or regular app code."""
        frame = inspect.currentframe()
        try:
            # Look through the call stack for workflow indicators
            current = frame
            while current:
                if current.f_code.co_filename:
                    file_path = Path(current.f_code.co_filename)
                    # Check if we're in a workflow directory or file
                    if (
                        "flows" in file_path.parts
                        or "workflow" in file_path.name.lower()
                    ):
                        return "workflow"
                current = current.f_back
            return "app"
        finally:
            del frame

    def _get_function_params(self, frame) -> str:
        """Extract function parameters from the frame."""
        try:
            # Get local variables
            local_vars = frame.f_locals.copy()

            # Remove 'self' and other non-parameter variables
            params = {}
            for key, value in local_vars.items():
                if key.startswith("_") or key in ("self", "cls"):
                    continue

                # Try to serialize the value
                try:
                    # Handle common types that can be JSON serialized
                    if isinstance(
                        value, (str, int, float, bool, list, dict, type(None))
                    ):
                        params[key] = value
                    else:
                        # For complex objects, just store the type
                        params[key] = f"<{type(value).__name__}>"
                except (TypeError, ValueError, AttributeError):
                    params[key] = f"<{type(value).__name__}>"

            return json.dumps(params, default=str)
        except Exception:  # pylint: disable=broad-exception-caught
            # Need to catch all errors to ensure logging never fails
            return "{}"

    def _write_log_entry(self, level: str, message: Union[str, Exception]) -> None:
        """Write a log entry to the CSV file."""
        with self._lock:
            log_file = self._get_log_file_path()
            caller_info = self._get_caller_info()

            # Format messages
            display_message = self._format_message_for_display(message)
            exception_details = str(message) if isinstance(message, Exception) else ""

            # Check if file exists to determine if we need headers
            file_exists = log_file.exists()

            with open(log_file, "a", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "timestamp",
                    "level",
                    "message",
                    "context",
                    "file",
                    "function",
                    "params",
                    "exception",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header if file is new
                if not file_exists:
                    writer.writeheader()

                # Write log entry
                writer.writerow(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "level": level,
                        "message": display_message,
                        "context": caller_info["context"],
                        "file": caller_info["file"],
                        "function": caller_info["function"],
                        "params": caller_info["params"],
                        "exception": exception_details,
                    }
                )

    def debug(self, message: Union[str, Exception]) -> None:
        """Log a debug message."""
        if self._should_log("debug"):
            self._write_log_entry("debug", str(message))

    def info(self, message: Union[str, Exception]) -> None:
        """Log an info message."""
        if self._should_log("info"):
            self._write_log_entry("info", str(message))

    def warning(self, message: Union[str, Exception]) -> None:
        """Log a warning message."""
        if self._should_log("warning"):
            self._write_log_entry("warning", str(message))

    def error(self, message: Union[str, Exception]) -> None:
        """Log an error message."""
        if self._should_log("error"):
            self._write_log_entry("error", str(message))


# Global logger instance
logger = CSVLogger()
