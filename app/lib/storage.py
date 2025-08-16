"""
Unified storage utilities for date-based folder management.

Provides common functionality for creating and managing date-based
storage folders used across the application (cache, logs, etc.).
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from .settings import settings


class DateBasedStorage:
    """
    Utility class for managing date-based folder structure.

    Provides consistent methods for creating and accessing folders
    organized by date in YYYYMMDD format.
    """

    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize storage manager.

        Args:
            base_path: Base directory for storage. If None, uses
                PROVIDER_CACHE_ROOT setting.
        """
        if base_path is None:
            self.base_path = Path(settings.PROVIDER_CACHE_ROOT)
        else:
            self.base_path = Path(base_path)

    def get_date_folder(
        self, date_str: Optional[str] = None, create: bool = True
    ) -> Path:
        """
        Get path to date-based folder.

        Args:
            date_str: Date string in YYYYMMDD format. If None, uses current date.
            create: Whether to create the folder if it doesn't exist.

        Returns:
            Path to the date folder.
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        date_folder = self.base_path / date_str

        if create:
            date_folder.mkdir(parents=True, exist_ok=True)

        return date_folder

    def get_file_path(
        self, filename: str, date_str: Optional[str] = None, create_folder: bool = True
    ) -> Path:
        """
        Get full path to a file in a date-based folder.

        Args:
            filename: Name of the file.
            date_str: Date string in YYYYMMDD format. If None, uses current date.
            create_folder: Whether to create the date folder if it doesn't exist.

        Returns:
            Full path to the file.
        """
        date_folder = self.get_date_folder(date_str, create=create_folder)
        return date_folder / filename

    def list_date_folders(self) -> list[str]:
        """
        List all available date folders.

        Returns:
            List of date strings (YYYYMMDD) for existing folders.
        """
        if not self.base_path.exists():
            return []

        date_folders = []
        for folder in self.base_path.iterdir():
            if folder.is_dir() and len(folder.name) == 8 and folder.name.isdigit():
                date_folders.append(folder.name)

        return sorted(date_folders, reverse=True)

    def get_cache_paths(
        self, provider_type: str, query: str, date_str: Optional[str] = None
    ) -> tuple[Path, Path]:
        """
        Get cache file paths for a provider query (JSON and Parquet).

        Args:
            provider_type: Type of the data provider.
            query: Query string (will be sanitized).
            date_str: Date string in YYYYMMDD format. If None, uses current date.

        Returns:
            Tuple of (json_path, parquet_path).
        """
        # Sanitize query for filename
        sanitized_query = query.upper().strip() if isinstance(query, str) else "none"
        base_name = f"{provider_type}_{sanitized_query}"

        date_folder = self.get_date_folder(date_str, create=True)

        json_path = date_folder / f"{base_name}.json"
        parquet_path = date_folder / f"{base_name}.parquet"

        return json_path, parquet_path

    def cleanup_old_folders(self, keep_days: int = 30) -> list[str]:
        """
        Clean up old date folders.

        Args:
            keep_days: Number of days to keep (from current date).

        Returns:
            List of removed folder names.
        """
        if not self.base_path.exists():
            return []

        current_date = datetime.now()
        removed_folders = []

        for folder in self.base_path.iterdir():
            if not (
                folder.is_dir() and len(folder.name) == 8 and folder.name.isdigit()
            ):
                continue

            try:
                folder_date = datetime.strptime(folder.name, "%Y%m%d")
                days_old = (current_date - folder_date).days

                if days_old > keep_days:
                    import shutil

                    shutil.rmtree(folder)
                    removed_folders.append(folder.name)
            except ValueError:
                # Skip folders that don't match YYYYMMDD format
                continue

        return removed_folders

    def delete_date_folder(self, date_str: str) -> bool:
        """
        Delete a specific date folder.

        Args:
            date_str: Date string in YYYYMMDD format.

        Returns:
            True if deletion was successful, False otherwise.
        """
        if not date_str or len(date_str) != 8 or not date_str.isdigit():
            return False

        folder_path = self.base_path / date_str

        if not folder_path.exists() or not folder_path.is_dir():
            return False

        try:
            import shutil

            shutil.rmtree(folder_path)
            return True
        except Exception:
            return False

    def get_log_data(self, date_str: str) -> list[dict]:
        """
        Read log data from a specific date folder's log.csv file.

        Args:
            date_str: Date string in YYYYMMDD format.

        Returns:
            List of log entries as dictionaries, empty list if file doesn't exist.
        """
        if not date_str:
            return []

        log_file = self.get_file_path("log.csv", date_str, create_folder=False)

        if not log_file.exists():
            return []

        try:
            import csv

            log_data = []
            with open(log_file, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Parse timestamp for better display
                    try:
                        from datetime import datetime as dt

                        timestamp = dt.fromisoformat(row["timestamp"])
                        row["timestamp"] = timestamp.strftime("%H:%M:%S")
                    except (ValueError, KeyError):
                        pass

                    # Backward compatibility for old log format
                    # Old format had verbose messages, new format has concise messages
                    # If long message (>200 chars), it's old format - truncate it
                    if "message" in row and len(row["message"]) > 200:
                        # This looks like an old verbose message, truncate it
                        if "ValidationError:" in row["message"]:
                            row["message"] = (
                                "ValidationError: Multiple validation errors"
                            )
                        else:
                            row["message"] = row["message"][:150] + "..."

                    log_data.append(row)
            return list(reversed(log_data))
        except Exception:
            return []


# Convenience functions for backward compatibility and ease of use
def get_data_folder(date_str: Optional[str] = None, create: bool = True) -> Path:
    """
    Get path to data folder for a specific date.

    Args:
        date_str: Date string in YYYYMMDD format. If None, uses current date.
        create: Whether to create the folder if it doesn't exist.

    Returns:
        Path to the date folder.
    """
    storage = DateBasedStorage()
    return storage.get_date_folder(date_str, create)


def get_data_file_path(
    filename: str, date_str: Optional[str] = None, create_folder: bool = True
) -> Path:
    """
    Get full path to a file in a date-based data folder.

    Args:
        filename: Name of the file.
        date_str: Date string in YYYYMMDD format. If None, uses current date.
        create_folder: Whether to create the date folder if it doesn't exist.

    Returns:
        Full path to the file.
    """
    storage = DateBasedStorage()
    return storage.get_file_path(filename, date_str, create_folder)


def get_cache_file_paths(
    provider_type: str, query: str, date_str: Optional[str] = None
) -> tuple[Path, Path]:
    """
    Get cache file paths for a provider query (JSON and Parquet).

    Args:
        provider_type: Type of the data provider.
        query: Query string (will be sanitized).
        date_str: Date string in YYYYMMDD format. If None, uses current date.

    Returns:
        Tuple of (json_path, parquet_path).
    """
    storage = DateBasedStorage()
    return storage.get_cache_paths(provider_type, query, date_str)


# Global storage instance
default_storage = DateBasedStorage()
