"""
Unit tests for the CSV logger.
"""

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from app.lib.logger import CSVLogger
from app.lib.storage import DateBasedStorage


class TestCSVLogger:
    """Test cases for CSVLogger class."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary storage directory in the workspace temp folder."""
        # Use the project's temp directory to avoid polluting data or OS temp
        project_root = Path(__file__).resolve().parent.parent.parent
        temp_dir = project_root / "temp" / "test_logger"
        temp_dir.mkdir(parents=True, exist_ok=True)
        yield temp_dir
        # Cleanup after test
        import shutil

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    @pytest.fixture
    def logger(self, temp_storage_path):
        """Create a CSVLogger instance with temporary storage."""
        storage = DateBasedStorage(base_path=temp_storage_path)
        return CSVLogger(storage=storage)

    def test_logger_initialization(self, temp_storage_path):
        """Test logger initialization."""
        storage = DateBasedStorage(base_path=temp_storage_path)
        logger = CSVLogger(storage=storage)
        assert logger.storage.base_path == temp_storage_path
        assert hasattr(logger, "_lock")
        assert isinstance(logger._lock, type(threading.Lock()))

    def test_auto_detect_project_root(self):
        """Test auto-detection of project root."""
        logger = CSVLogger()
        # Should find the actual project root with rxconfig.py and use data subfolder
        assert (logger.storage.base_path.parent / "rxconfig.py").exists()
        assert logger.storage.base_path.name == "data"

    def test_get_log_file_path(self, logger):
        """Test log file path generation."""
        log_path = logger._get_log_file_path()
        date_str = datetime.now().strftime("%Y%m%d")
        expected_path = logger.storage.base_path / date_str / "log.csv"
        assert log_path == expected_path
        assert log_path.parent.exists()

    def test_determine_context_app(self, logger):
        """Test context detection for app code."""
        context = logger._determine_context()
        assert context == "app"

    def test_determine_context_workflow(self, logger, temp_storage_path):
        """Test context detection for workflow code."""
        # Create a mock workflow file
        flows_dir = temp_storage_path.parent / "flows"
        flows_dir.mkdir(exist_ok=True)
        workflow_file = flows_dir / "test_workflow.py"
        workflow_file.write_text("# test workflow")

        # Mock the frame to simulate call from workflow
        with mock.patch("inspect.currentframe") as mock_frame:
            mock_frame_obj = mock.Mock()
            mock_frame_obj.f_code.co_filename = str(workflow_file)
            mock_frame_obj.f_back = None
            mock_frame.return_value = mock_frame_obj

            context = logger._determine_context()
            assert context == "workflow"

    def test_get_function_params(self, logger):
        """Test function parameter extraction."""

        def test_function(param1, param2="default", param3=42):
            # pylint: disable=unused-argument
            # Simulate a frame with local variables
            frame = mock.Mock()
            frame.f_locals = {
                "param1": "value1",
                "param2": "value2",
                "param3": 123,
                "self": "should_be_filtered",
                "_private": "should_be_filtered",
            }
            return logger._get_function_params(frame)

        result = test_function("test", "modified", 456)
        params = json.loads(result)
        assert params["param1"] == "value1"
        assert params["param2"] == "value2"
        assert params["param3"] == 123
        assert "self" not in params
        assert "_private" not in params

    def test_get_function_params_complex_objects(self, logger):
        """Test parameter extraction with complex objects."""
        frame = mock.Mock()
        frame.f_locals = {
            "simple_str": "test",
            "simple_int": 42,
            "simple_list": [1, 2, 3],
            "simple_dict": {"key": "value"},
            "complex_obj": object(),
            "none_value": None,
        }

        result = logger._get_function_params(frame)
        params = json.loads(result)

        assert params["simple_str"] == "test"
        assert params["simple_int"] == 42
        assert params["simple_list"] == [1, 2, 3]
        assert params["simple_dict"] == {"key": "value"}
        assert params["complex_obj"] == "<object>"
        assert params["none_value"] is None

    def test_get_function_params_exception_handling(self, logger):
        """Test parameter extraction with serialization errors."""
        with mock.patch("json.dumps", side_effect=Exception("JSON error")):
            frame = mock.Mock()
            frame.f_locals = {"param": "value"}
            result = logger._get_function_params(frame)
            assert result == "{}"

    def test_debug_logging(self, logger):
        """Test debug level logging."""
        logger.debug("Test debug message")

        log_file = logger._get_log_file_path()
        assert log_file.exists()

        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["level"] == "debug"
        assert rows[0]["message"] == "Test debug message"
        assert "timestamp" in rows[0]
        assert "context" in rows[0]
        assert "file" in rows[0]
        assert "function" in rows[0]
        assert "params" in rows[0]

    def test_info_logging(self, logger):
        """Test info level logging."""
        logger.info("Test info message")

        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        assert rows[0]["level"] == "info"
        assert rows[0]["message"] == "Test info message"

    def test_warning_logging(self, logger):
        """Test warning level logging."""
        logger.warning("Test warning message")

        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        assert rows[0]["level"] == "warning"
        assert rows[0]["message"] == "Test warning message"

    def test_error_logging(self, logger):
        """Test error level logging."""
        logger.error("Test error message")

        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        assert rows[0]["level"] == "error"
        assert rows[0]["message"] == "Test error message"

    def test_exception_logging(self, logger):
        """Test logging of exception objects."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.error(e)

        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        assert rows[0]["level"] == "error"
        assert "Test exception" in rows[0]["message"]

    def test_multiple_log_entries(self, logger):
        """Test multiple log entries are written correctly."""
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        assert len(rows) == 4
        assert rows[0]["level"] == "debug"
        assert rows[1]["level"] == "info"
        assert rows[2]["level"] == "warning"
        assert rows[3]["level"] == "error"

    def test_csv_headers(self, logger):
        """Test that CSV headers are written correctly."""
        logger.info("Test message")

        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)

        expected_headers = [
            "timestamp",
            "level",
            "message",
            "context",
            "file",
            "function",
            "params",
            "exception",
        ]
        assert headers == expected_headers

    def test_timestamp_format(self, logger):
        """Test timestamp format is ISO format."""
        before_log = datetime.now()
        logger.info("Test message")
        after_log = datetime.now()

        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            row = next(reader)

        # Parse the timestamp
        timestamp = datetime.fromisoformat(row["timestamp"])
        assert before_log <= timestamp <= after_log

    def test_thread_safety(self, logger):
        """Test thread safety of logging."""

        def log_messages(thread_id):
            for i in range(10):
                logger.info(f"Message from thread {thread_id}, iteration {i}")
                time.sleep(0.001)  # Small delay to encourage race conditions

        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_messages, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        # Should have 50 entries (5 threads * 10 messages each)
        assert len(rows) == 50

        # All entries should be 'info' level
        assert all(row["level"] == "info" for row in rows)

    def test_file_path_relative_to_project(self, logger, temp_storage_path):
        """Test that file paths are relative to project root."""
        # Create a subdirectory structure
        subdir = temp_storage_path.parent / "app" / "lib"
        subdir.mkdir(parents=True, exist_ok=True)
        test_file = subdir / "test_module.py"
        test_file.write_text("# test module")

        with mock.patch.object(logger, "_get_caller_info") as mock_caller_info:
            mock_caller_info.return_value = {
                "file": "app/lib/test_module.py",
                "function": "test_function",
                "context": "app",
                "params": "{}",
            }

            logger.info("Test message")

        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            row = next(reader)

        # File path should be relative to project root
        assert row["file"] == "app/lib/test_module.py"
        assert row["function"] == "test_function"

    def test_global_logger_instance(self):
        """Test that global logger instance is available."""
        from app.lib.logger import logger

        assert isinstance(logger, CSVLogger)
        assert hasattr(logger, "debug")
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")


class TestLoggerIntegration:
    """Integration tests for logger usage."""

    def test_logger_in_function_with_params(self):
        """Test logger captures function parameters correctly."""
        from app.lib.logger import logger

        def test_function(param1, param2=42, param3="default"):
            # pylint: disable=unused-argument
            logger.info("Function called")
            return param1 + param2

        result = test_function(10, param2=20, param3="modified")
        assert result == 30

        # Check the log file for the entry
        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        # Find the log entry from our function
        our_log = None
        for row in rows:
            if row["function"] == "test_function":
                our_log = row
                break

        assert our_log is not None
        params = json.loads(our_log["params"])
        assert "param1" in params
        assert "param2" in params
        assert "param3" in params


class TestDebugLevelFiltering:
    """Test cases for DEBUG_LEVEL filtering functionality."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary storage directory in the workspace temp folder."""
        project_root = Path(__file__).resolve().parent.parent.parent
        temp_dir = project_root / "temp" / "test_debug_level"
        temp_dir.mkdir(parents=True, exist_ok=True)
        yield temp_dir
        # Cleanup after test
        import shutil

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_debug_level_hierarchy(self, temp_storage_path):
        """Test that the level hierarchy is correctly defined."""
        from app.lib.storage import DateBasedStorage

        storage = DateBasedStorage(base_path=temp_storage_path)
        logger = CSVLogger(storage=storage)

        expected_hierarchy = {
            "debug": 0,
            "info": 1,
            "warning": 2,
            "error": 3,
        }
        assert logger.LEVEL_HIERARCHY == expected_hierarchy

    def test_should_log_method(self, temp_storage_path):
        """Test the _should_log method with different configurations."""
        from app.lib.storage import DateBasedStorage

        storage = DateBasedStorage(base_path=temp_storage_path)
        logger = CSVLogger(storage=storage)

        # Mock settings for different debug levels
        with mock.patch("app.lib.logger.settings") as mock_settings:
            # Test debug level (should log everything)
            mock_settings.DEBUG_LEVEL = "debug"
            assert logger._should_log("debug") is True
            assert logger._should_log("info") is True
            assert logger._should_log("warning") is True
            assert logger._should_log("error") is True

            # Test info level (should skip debug)
            mock_settings.DEBUG_LEVEL = "info"
            assert logger._should_log("debug") is False
            assert logger._should_log("info") is True
            assert logger._should_log("warning") is True
            assert logger._should_log("error") is True

            # Test warning level (should only log warning and error)
            mock_settings.DEBUG_LEVEL = "warning"
            assert logger._should_log("debug") is False
            assert logger._should_log("info") is False
            assert logger._should_log("warning") is True
            assert logger._should_log("error") is True

            # Test error level (should only log error)
            mock_settings.DEBUG_LEVEL = "error"
            assert logger._should_log("debug") is False
            assert logger._should_log("info") is False
            assert logger._should_log("warning") is False
            assert logger._should_log("error") is True

    def test_case_insensitive_debug_level(self, temp_storage_path):
        """Test that DEBUG_LEVEL comparison is case-insensitive."""
        from app.lib.storage import DateBasedStorage

        storage = DateBasedStorage(base_path=temp_storage_path)
        logger = CSVLogger(storage=storage)

        with mock.patch("app.lib.logger.settings") as mock_settings:
            # Test various case combinations
            for debug_level in ["INFO", "Info", "iNfO", "info"]:
                mock_settings.DEBUG_LEVEL = debug_level
                assert logger._should_log("debug") is False
                assert logger._should_log("info") is True
                assert logger._should_log("WARNING") is True
                assert logger._should_log("Error") is True

    def test_debug_level_filtering_integration(self, temp_storage_path):
        """Test that actual logging respects DEBUG_LEVEL settings."""
        from app.lib.storage import DateBasedStorage

        storage = DateBasedStorage(base_path=temp_storage_path)
        logger = CSVLogger(storage=storage)

        with mock.patch("app.lib.logger.settings") as mock_settings:
            # Test with info level
            mock_settings.DEBUG_LEVEL = "info"

            logger.debug("Debug message")  # Should not be logged
            logger.info("Info message")  # Should be logged
            logger.warning("Warning message")  # Should be logged
            logger.error("Error message")  # Should be logged

            log_file = logger._get_log_file_path()
            with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

            # Should have 3 entries (info, warning, error - no debug)
            assert len(rows) == 3
            levels = [row["level"] for row in rows]
            assert "debug" not in levels
            assert "info" in levels
            assert "warning" in levels
            assert "error" in levels

    def test_debug_level_warning_only(self, temp_storage_path):
        """Test DEBUG_LEVEL='warning' logs only warning and error."""
        from app.lib.storage import DateBasedStorage

        storage = DateBasedStorage(base_path=temp_storage_path)
        logger = CSVLogger(storage=storage)

        with mock.patch("app.lib.logger.settings") as mock_settings:
            mock_settings.DEBUG_LEVEL = "warning"

            logger.debug("Debug message")  # Should not be logged
            logger.info("Info message")  # Should not be logged
            logger.warning("Warning message")  # Should be logged
            logger.error("Error message")  # Should be logged

            log_file = logger._get_log_file_path()
            with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

            # Should have 2 entries (warning, error)
            assert len(rows) == 2
            levels = [row["level"] for row in rows]
            assert "debug" not in levels
            assert "info" not in levels
            assert "warning" in levels
            assert "error" in levels

    def test_debug_level_error_only(self, temp_storage_path):
        """Test DEBUG_LEVEL='error' logs only error messages."""
        from app.lib.storage import DateBasedStorage

        storage = DateBasedStorage(base_path=temp_storage_path)
        logger = CSVLogger(storage=storage)

        with mock.patch("app.lib.logger.settings") as mock_settings:
            mock_settings.DEBUG_LEVEL = "error"

            logger.debug("Debug message")  # Should not be logged
            logger.info("Info message")  # Should not be logged
            logger.warning("Warning message")  # Should not be logged
            logger.error("Error message")  # Should be logged

            log_file = logger._get_log_file_path()
            with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

            # Should have 1 entry (error only)
            assert len(rows) == 1
            assert rows[0]["level"] == "error"
            assert rows[0]["message"] == "Error message"

    def test_should_log_exception_handling(self, temp_storage_path):
        """Test that _should_log handles exceptions gracefully."""
        from app.lib.storage import DateBasedStorage

        storage = DateBasedStorage(base_path=temp_storage_path)
        logger = CSVLogger(storage=storage)

        # Test with invalid/missing settings
        with mock.patch(
            "app.lib.logger.settings", side_effect=Exception("Settings error")
        ):
            # Should default to logging (fail-safe)
            assert logger._should_log("debug") is True
            assert logger._should_log("info") is True
            assert logger._should_log("warning") is True
            assert logger._should_log("error") is True

    def test_default_debug_level_all_logs(self, temp_storage_path):
        """Test that default DEBUG_LEVEL='debug' logs all messages."""
        from app.lib.storage import DateBasedStorage

        storage = DateBasedStorage(base_path=temp_storage_path)
        logger = CSVLogger(storage=storage)

        # Use actual settings (should default to 'debug')
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        log_file = logger._get_log_file_path()
        with open(log_file, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        # Should have 4 entries (all levels)
        assert len(rows) == 4
        levels = [row["level"] for row in rows]
        assert "debug" in levels
        assert "info" in levels
        assert "warning" in levels
        assert "error" in levels
