"""
Unit tests for the storage utilities.
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import pytest

from app.lib.storage import (
    DateBasedStorage,
    get_data_folder,
    get_data_file_path,
    get_cache_file_paths,
)


class TestDateBasedStorage:
    """Test cases for DateBasedStorage class."""

    @pytest.fixture
    def temp_base_path(self):
        """Create a temporary base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def storage(self, temp_base_path):
        """Create a DateBasedStorage instance with temporary directory."""
        return DateBasedStorage(base_path=temp_base_path)

    def test_storage_initialization_with_base_path(self, temp_base_path):
        """Test storage initialization with explicit base path."""
        storage = DateBasedStorage(base_path=temp_base_path)
        assert storage.base_path == temp_base_path

    def test_storage_initialization_default(self):
        """Test storage initialization with default path."""
        # Clear environment variables to test actual default behavior
        # This ensures the test works in both local and CI environments
        with mock.patch.dict(os.environ, {}, clear=True):
            # Need to reload settings module to pick up cleared environment
            import importlib
            from app.lib import settings

            importlib.reload(settings)

            storage = DateBasedStorage()
            # Should auto-detect project root and use data subfolder
            assert (storage.base_path.parent / "rxconfig.py").exists()
            assert storage.base_path.name == "data"

    def test_get_date_folder_current_date(self, storage):
        """Test getting date folder for current date."""
        folder = storage.get_date_folder()
        expected_date = datetime.now().strftime("%Y%m%d")

        assert folder.name == expected_date
        assert folder.exists()
        assert folder.is_dir()

    def test_get_date_folder_specific_date(self, storage):
        """Test getting date folder for specific date."""
        test_date = "20231225"
        folder = storage.get_date_folder(test_date)

        assert folder.name == test_date
        assert folder.exists()
        assert folder.is_dir()

    def test_get_date_folder_no_create(self, storage):
        """Test getting date folder without creating it."""
        test_date = "20231225"
        folder = storage.get_date_folder(test_date, create=False)

        assert folder.name == test_date
        assert not folder.exists()

    def test_get_file_path_current_date(self, storage):
        """Test getting file path for current date."""
        filename = "test.csv"
        file_path = storage.get_file_path(filename)

        expected_date = datetime.now().strftime("%Y%m%d")
        assert file_path.name == filename
        assert file_path.parent.name == expected_date
        assert file_path.parent.exists()

    def test_get_file_path_specific_date(self, storage):
        """Test getting file path for specific date."""
        filename = "test.log"
        test_date = "20231225"
        file_path = storage.get_file_path(filename, test_date)

        assert file_path.name == filename
        assert file_path.parent.name == test_date
        assert file_path.parent.exists()

    def test_get_file_path_no_create_folder(self, storage):
        """Test getting file path without creating folder."""
        filename = "test.txt"
        test_date = "20231225"
        file_path = storage.get_file_path(filename, test_date, create_folder=False)

        assert file_path.name == filename
        assert file_path.parent.name == test_date
        assert not file_path.parent.exists()

    def test_list_date_folders_empty(self, storage):
        """Test listing date folders when none exist."""
        folders = storage.list_date_folders()
        assert folders == []

    def test_list_date_folders_with_data(self, storage):
        """Test listing date folders with existing data."""
        # Create some date folders
        test_dates = ["20231220", "20231225", "20240101"]
        for date in test_dates:
            storage.get_date_folder(date)

        # Create a non-date folder (should be ignored)
        (storage.base_path / "not_a_date").mkdir()

        # Create a file (should be ignored)
        (storage.base_path / "20231230.txt").touch()

        folders = storage.list_date_folders()
        assert folders == sorted(test_dates, reverse=True)

    def test_get_cache_paths(self, storage):
        """Test getting cache file paths."""
        provider_type = "yahoo"
        query = "AAPL"

        json_path, parquet_path = storage.get_cache_paths(provider_type, query)

        # Check paths are in correct date folder
        expected_date = datetime.now().strftime("%Y%m%d")
        assert json_path.parent.name == expected_date
        assert parquet_path.parent.name == expected_date

        # Check filenames
        expected_base = f"{provider_type}_{query.upper()}"
        assert json_path.name == f"{expected_base}.json"
        assert parquet_path.name == f"{expected_base}.parquet"

        # Check folder was created
        assert json_path.parent.exists()

    def test_get_cache_paths_specific_date(self, storage):
        """Test getting cache file paths for specific date."""
        provider_type = "yahoo"
        query = "MSFT"
        test_date = "20231225"

        json_path, parquet_path = storage.get_cache_paths(
            provider_type, query, test_date
        )

        assert json_path.parent.name == test_date
        assert parquet_path.parent.name == test_date

    def test_get_cache_paths_query_sanitization(self, storage):
        """Test query sanitization in cache paths."""
        provider_type = "yahoo"
        query = "  aapl  "  # lowercase with spaces

        json_path, parquet_path = storage.get_cache_paths(provider_type, query)

        expected_base = f"{provider_type}_AAPL"
        assert json_path.name == f"{expected_base}.json"
        assert parquet_path.name == f"{expected_base}.parquet"

    def test_get_cache_paths_none_query(self, storage):
        """Test cache paths with None query."""
        provider_type = "yahoo"
        query = None

        json_path, parquet_path = storage.get_cache_paths(provider_type, query)

        expected_base = f"{provider_type}_none"
        assert json_path.name == f"{expected_base}.json"
        assert parquet_path.name == f"{expected_base}.parquet"

    def test_cleanup_old_folders(self, storage):
        """Test cleaning up old date folders."""
        # Create folders of different ages
        today = datetime.now()
        old_date = today - timedelta(days=35)
        recent_date = today - timedelta(days=10)

        old_folder_name = old_date.strftime("%Y%m%d")
        recent_folder_name = recent_date.strftime("%Y%m%d")

        storage.get_date_folder(old_folder_name)
        storage.get_date_folder(recent_folder_name)

        # Create a non-date folder (should not be removed)
        non_date_folder = storage.base_path / "not_a_date"
        non_date_folder.mkdir()

        # Clean up folders older than 30 days
        removed = storage.cleanup_old_folders(keep_days=30)

        assert old_folder_name in removed
        assert recent_folder_name not in removed

        # Check folders exist/don't exist as expected
        assert not (storage.base_path / old_folder_name).exists()
        assert (storage.base_path / recent_folder_name).exists()
        assert non_date_folder.exists()  # Non-date folder should remain

    def test_cleanup_old_folders_no_base_path(self, temp_base_path):
        """Test cleanup when base path doesn't exist."""
        non_existent_path = temp_base_path / "nonexistent"
        storage = DateBasedStorage(base_path=non_existent_path)

        removed = storage.cleanup_old_folders()
        assert removed == []

    def test_cleanup_old_folders_invalid_date_format(self, storage):
        """Test cleanup with invalid date format folders."""
        # Create folder with invalid date format
        invalid_folder = storage.base_path / "invalid_date"
        invalid_folder.mkdir()

        # Create valid date folder
        valid_date = datetime.now().strftime("%Y%m%d")
        storage.get_date_folder(valid_date)

        removed = storage.cleanup_old_folders()

        # Invalid folder should remain
        assert invalid_folder.exists()
        assert valid_date not in removed

    def test_delete_date_folder_success(self, storage):
        """Test successful deletion of a date folder."""
        test_date = "20231225"

        # Create the folder with some content
        file_path = storage.get_file_path("test.txt", test_date)
        file_path.write_text("test content")
        assert file_path.exists()

        # Delete the folder
        success = storage.delete_date_folder(test_date)

        assert success is True
        assert not (storage.base_path / test_date).exists()

    def test_delete_date_folder_invalid_date(self, storage):
        """Test deletion with invalid date string."""
        invalid_dates = ["", "invalid", "12345", "123456789", "20231301"]

        for invalid_date in invalid_dates:
            success = storage.delete_date_folder(invalid_date)
            assert success is False

    def test_delete_date_folder_nonexistent(self, storage):
        """Test deletion of non-existent folder."""
        test_date = "20231225"

        # Ensure folder doesn't exist
        assert not (storage.base_path / test_date).exists()

        success = storage.delete_date_folder(test_date)
        assert success is False

    def test_delete_date_folder_permission_error(self, storage):
        """Test deletion when permission error occurs."""
        test_date = "20231225"

        # Create the folder
        storage.get_date_folder(test_date)

        # Mock shutil.rmtree to raise an exception
        with mock.patch("shutil.rmtree", side_effect=OSError("Permission denied")):
            success = storage.delete_date_folder(test_date)
            assert success is False

    def test_get_log_data_success(self, storage):
        """Test successful log data retrieval."""
        test_date = "20231225"

        # Create log.csv file with test data
        log_file = storage.get_file_path("log.csv", test_date)
        log_content = (
            "timestamp,level,message,context,file,function,params\n"
            "2023-12-25T10:00:00.123456,info,Test message 1,app,test.py,test_func,"
            '"{\\"param1\\": \\"value1\\"}"\n'
            "2023-12-25T10:01:00.654321,error,Test message 2,workflow,other.py,"
            'other_func,"{\\"param2\\": \\"value2\\"}"\n'
        )
        log_file.write_text(log_content)

        # Get log data
        log_data = storage.get_log_data(test_date)

        assert len(log_data) == 2
        # Should be reversed - most recent first
        assert log_data[0]["level"] == "error"
        assert log_data[0]["message"] == "Test message 2"
        assert log_data[0]["timestamp"] == "10:01:00"
        assert log_data[1]["level"] == "info"
        assert log_data[1]["message"] == "Test message 1"
        assert (
            log_data[1]["timestamp"] == "10:00:00"
        )  # Should be formatted as time only

    def test_get_log_data_empty_date(self, storage):
        """Test log data retrieval with empty date."""
        log_data = storage.get_log_data("")
        assert log_data == []

    def test_get_log_data_nonexistent_file(self, storage):
        """Test log data retrieval when log file doesn't exist."""
        test_date = "20231225"

        log_data = storage.get_log_data(test_date)
        assert log_data == []

    def test_get_log_data_invalid_csv(self, storage):
        """Test log data retrieval with invalid CSV content."""
        test_date = "20231225"

        # Create CSV file with incorrect headers
        log_file = storage.get_file_path("log.csv", test_date)
        log_file.write_text("wrong,headers\nsome,data")

        log_data = storage.get_log_data(test_date)
        # Should still parse but without expected timestamp formatting
        assert len(log_data) == 1
        assert "timestamp" not in log_data[0]

    def test_get_log_data_malformed_timestamp(self, storage):
        """Test log data with malformed timestamp."""
        test_date = "20231225"

        # Create log.csv with malformed timestamp
        log_file = storage.get_file_path("log.csv", test_date)
        log_content = """timestamp,level,message,context,file,function,params
invalid-timestamp,info,Test message,app,test.py,test_func,"{}"
"""
        log_file.write_text(log_content)

        log_data = storage.get_log_data(test_date)

        assert len(log_data) == 1
        assert (
            log_data[0]["timestamp"] == "invalid-timestamp"
        )  # Should remain unchanged


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @pytest.fixture
    def mock_storage(self):
        """Mock the default storage instance."""
        with mock.patch("app.lib.storage.DateBasedStorage") as mock_class:
            mock_instance = mock.Mock()
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_get_data_folder(self, mock_storage):
        """Test get_data_folder convenience function."""
        test_date = "20231225"
        expected_path = Path("/test/path")
        mock_storage.get_date_folder.return_value = expected_path

        result = get_data_folder(test_date, create=False)

        mock_storage.get_date_folder.assert_called_once_with(test_date, False)
        assert result == expected_path

    def test_get_data_file_path(self, mock_storage):
        """Test get_data_file_path convenience function."""
        filename = "test.csv"
        test_date = "20231225"
        expected_path = Path("/test/path/test.csv")
        mock_storage.get_file_path.return_value = expected_path

        result = get_data_file_path(filename, test_date, create_folder=False)

        mock_storage.get_file_path.assert_called_once_with(filename, test_date, False)
        assert result == expected_path

    def test_get_cache_file_paths(self, mock_storage):
        """Test get_cache_file_paths convenience function."""
        provider_type = "yahoo"
        query = "AAPL"
        test_date = "20231225"
        expected_paths = (Path("/test/aapl.json"), Path("/test/aapl.parquet"))
        mock_storage.get_cache_paths.return_value = expected_paths

        result = get_cache_file_paths(provider_type, query, test_date)

        mock_storage.get_cache_paths.assert_called_once_with(
            provider_type, query, test_date
        )
        assert result == expected_paths


class TestStorageIntegration:
    """Integration tests for storage functionality."""

    @pytest.fixture
    def temp_project_root(self):
        """Create a temporary project structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            # Create rxconfig.py to simulate project root
            (project_root / "rxconfig.py").touch()
            yield project_root

    def test_storage_with_temp_directory(self, temp_project_root):
        """Test storage using a temporary directory structure."""
        # Create temp subfolder for testing
        temp_data = temp_project_root / "temp"
        temp_data.mkdir()

        storage = DateBasedStorage(base_path=temp_data)

        # Test file creation
        test_file = storage.get_file_path("test.log")
        test_file.write_text("test content")

        assert test_file.exists()
        assert test_file.read_text() == "test content"

        # Test date folder structure
        date_folders = storage.list_date_folders()
        assert len(date_folders) == 1
        assert date_folders[0] == datetime.now().strftime("%Y%m%d")

    def test_multiple_files_same_date(self, temp_project_root):
        """Test creating multiple files in the same date folder."""
        temp_data = temp_project_root / "temp"
        storage = DateBasedStorage(base_path=temp_data)

        # Create multiple files
        files = ["log.csv", "cache.json", "data.parquet"]
        test_date = "20231225"

        for filename in files:
            file_path = storage.get_file_path(filename, test_date)
            file_path.write_text(f"content for {filename}")

        # Check all files exist in the same date folder
        date_folder = storage.get_date_folder(test_date, create=False)
        existing_files = [f.name for f in date_folder.iterdir() if f.is_file()]

        assert set(existing_files) == set(files)

    def test_cleanup_integration(self, temp_project_root):
        """Test cleanup functionality with real folders."""
        temp_data = temp_project_root / "temp"
        storage = DateBasedStorage(base_path=temp_data)

        # Create folders with different dates
        dates = [
            (datetime.now() - timedelta(days=40)).strftime(
                "%Y%m%d"
            ),  # Should be removed
            (datetime.now() - timedelta(days=20)).strftime("%Y%m%d"),  # Should remain
            datetime.now().strftime("%Y%m%d"),  # Should remain
        ]

        # Create folders and add files
        for date in dates:
            file_path = storage.get_file_path(f"test_{date}.txt", date)
            file_path.write_text(f"data for {date}")

        # Verify all folders exist
        assert len(storage.list_date_folders()) == 3

        # Clean up old folders
        removed = storage.cleanup_old_folders(keep_days=30)

        # Check results
        remaining_folders = storage.list_date_folders()
        assert len(removed) == 1
        assert len(remaining_folders) == 2
        assert dates[0] in removed
        assert dates[0] not in remaining_folders
