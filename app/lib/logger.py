"""
Custom CSV logger for FinApp that logs to date-based folders.

Logs are stored in data/YYYYMMDD/log.csv with the following columns:
- timestamp: Date and time of the log entry
- level: Log level (debug, info, warning, error)
- message: Log message or exception details
- context: Workflow name from stack or 'app'
- file: File path relative to project root
- function: Function name that called the logger
- params: Function parameters as JSON string (when feasible)
"""

import csv
import inspect
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .storage import DateBasedStorage


class CSVLogger:
    """
    A custom logger that writes log entries to CSV files organized by date.
    """

    def __init__(self, storage: Optional[DateBasedStorage] = None):
        """
        Initialize the CSV logger.

        Args:
            storage: DateBasedStorage instance. If None, creates default instance.
        """
        self.storage = storage or DateBasedStorage()
        self._lock = threading.Lock()

    def _get_log_file_path(self) -> Path:
        """Get the path to today's log file."""
        return self.storage.get_file_path("log.csv")

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

            # Get function name
            function_name = caller_frame.f_code.co_name

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
                except Exception:
                    params[key] = f"<{type(value).__name__}>"

            return json.dumps(params, default=str)
        except Exception:
            return "{}"

    def _write_log_entry(self, level: str, message: str) -> None:
        """Write a log entry to the CSV file."""
        with self._lock:
            log_file = self._get_log_file_path()
            caller_info = self._get_caller_info()

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
                        "message": message,
                        "context": caller_info["context"],
                        "file": caller_info["file"],
                        "function": caller_info["function"],
                        "params": caller_info["params"],
                    }
                )

    def debug(self, message: Union[str, Exception]) -> None:
        """Log a debug message."""
        self._write_log_entry("debug", str(message))

    def info(self, message: Union[str, Exception]) -> None:
        """Log an info message."""
        self._write_log_entry("info", str(message))

    def warning(self, message: Union[str, Exception]) -> None:
        """Log a warning message."""
        self._write_log_entry("warning", str(message))

    def error(self, message: Union[str, Exception]) -> None:
        """Log an error message."""
        self._write_log_entry("error", str(message))


# Global logger instance
logger = CSVLogger()
