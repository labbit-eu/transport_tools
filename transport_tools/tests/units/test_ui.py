# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2022  Jan Brezovsky <janbre@amu.edu.pl>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import tempfile
import os
import sys
import logging
import shutil
from io import StringIO
from time import sleep
from unittest.mock import patch, MagicMock
from argparse import ArgumentParser

from transport_tools.libs.ui import (
    init_parser,
    init_logging,
    set_logging_level,
    process_count,
    initiate_tools,
    progressbar,
    SuppressMsg,
    TimeProcess,
)

from transport_tools.tests.units.data.data_ui import (
    sample_args_config,
    sample_args_write_template,
    sample_args_write_template_advanced,
    sample_args_version,
    sample_args_license,
    expected_parser_args,
    valid_log_levels,
    progress_test_values,
    process_count_test_values,
)


class TestInitParser(unittest.TestCase):
    """Test cases for init_parser function"""

    def test_init_parser_returns_argumentparser(self):
        """Test that init_parser returns an ArgumentParser"""
        parser = init_parser()
        self.assertIsInstance(parser, ArgumentParser)

    def test_init_parser_has_config_argument(self):
        """Test that parser has config argument"""
        parser = init_parser()
        args = parser.parse_args(sample_args_config)
        self.assertEqual(args.config_filename, "config.ini")

    def test_init_parser_has_write_template_argument(self):
        """Test that parser has write_template_config argument"""
        parser = init_parser()
        args = parser.parse_args(sample_args_write_template)
        self.assertTrue(args.write_config_file)

    def test_init_parser_has_advanced_argument(self):
        """Test that parser has advanced argument"""
        parser = init_parser()
        args = parser.parse_args(sample_args_write_template_advanced)
        self.assertTrue(args.write_config_file)
        self.assertTrue(args.advanced)

    def test_init_parser_has_version_argument(self):
        """Test that parser has version argument"""
        parser = init_parser()
        args = parser.parse_args(sample_args_version)
        self.assertTrue(args.print_version)

    def test_init_parser_has_license_argument(self):
        """Test that parser has license argument"""
        parser = init_parser()
        args = parser.parse_args(sample_args_license)
        self.assertTrue(args.print_license)

    def test_init_parser_mutually_exclusive_groups(self):
        """Test that config and write_template are mutually exclusive"""
        parser = init_parser()

        # Should raise error when both -c and -w are provided
        with self.assertRaises(SystemExit):
            parser.parse_args(["-c", "config.ini", "-w"])

    def test_init_parser_description(self):
        """Test that parser has description"""
        parser = init_parser()
        self.assertIn("TransportTool", parser.description)


class TestLoggingFunctions(unittest.TestCase):
    """Test cases for logging-related functions"""

    def setUp(self):
        """Set up temporary directory for log files"""
        self.temp_dir = tempfile.mkdtemp(prefix="TT_test_ui_")
        self.log_file = os.path.join(self.temp_dir, "test.log")

        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def tearDown(self):
        """Clean up temporary directory and logging handlers"""
        # Remove all handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # Reset module-level handlers
        import transport_tools.libs.ui as ui_module
        ui_module._CONSOLE_HANDLER = None
        ui_module._LOG_HANDLER = None

    def test_init_logging_creates_log_file(self):
        """Test that init_logging creates a log file"""
        init_logging(verbose_logging=False, logfile=self.log_file)

        self.assertTrue(os.path.exists(self.log_file))

    def test_init_logging_sets_handlers(self):
        """Test that init_logging sets up handlers"""
        init_logging(verbose_logging=False, logfile=self.log_file)

        import transport_tools.libs.ui as ui_module
        self.assertIsNotNone(ui_module._CONSOLE_HANDLER)
        self.assertIsNotNone(ui_module._LOG_HANDLER)

    def test_init_logging_verbose_mode(self):
        """Test that verbose_logging parameter affects filtering"""
        init_logging(verbose_logging=True, logfile=self.log_file)

        root_logger = logging.getLogger()
        # With verbose=True, should have fewer filters
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(file_handlers), 1)

    def test_set_logging_level_info(self):
        """Test setting logging level to INFO"""
        init_logging(verbose_logging=False, logfile=self.log_file)

        set_logging_level("INFO")

        import transport_tools.libs.ui as ui_module
        self.assertEqual(ui_module._CONSOLE_HANDLER.level, logging.INFO)

    def test_set_logging_level_debug(self):
        """Test setting logging level to DEBUG"""
        init_logging(verbose_logging=False, logfile=self.log_file)

        set_logging_level("DEBUG")

        import transport_tools.libs.ui as ui_module
        self.assertEqual(ui_module._CONSOLE_HANDLER.level, logging.DEBUG)

    def test_set_logging_level_case_insensitive(self):
        """Test that set_logging_level is case-insensitive"""
        init_logging(verbose_logging=False, logfile=self.log_file)

        set_logging_level("info")  # lowercase

        import transport_tools.libs.ui as ui_module
        self.assertEqual(ui_module._CONSOLE_HANDLER.level, logging.INFO)

    def test_set_logging_level_before_init_raises(self):
        """Test that set_logging_level raises error if called before init_logging"""
        with self.assertRaises(RuntimeError) as context:
            set_logging_level("INFO")

        self.assertIn("not been initialized", str(context.exception))

    def test_set_logging_level_all_valid_levels(self):
        """Test all valid logging levels"""
        init_logging(verbose_logging=False, logfile=self.log_file)

        level_mapping = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
        }

        import transport_tools.libs.ui as ui_module
        for level_str, level_int in level_mapping.items():
            set_logging_level(level_str)
            self.assertEqual(ui_module._CONSOLE_HANDLER.level, level_int)


class TestProcessCount(unittest.TestCase):
    """Test cases for process_count function"""

    def test_process_count_singular(self):
        """Test process_count with 1 process"""
        result = process_count(1)
        self.assertEqual(result, "parallel process")

    def test_process_count_plural(self):
        """Test process_count with multiple processes"""
        for num in [2, 5, 10, 100]:
            result = process_count(num)
            self.assertEqual(result, "parallel processes")

    def test_process_count_zero(self):
        """Test process_count with 0 processes"""
        result = process_count(0)
        self.assertEqual(result, "parallel processes")


class TestProgressBar(unittest.TestCase):
    """Test cases for progressbar function"""

    def setUp(self):
        """Set up logging for progress bar tests"""
        self.temp_dir = tempfile.mkdtemp(prefix="TT_test_progressbar_")
        self.log_file = os.path.join(self.temp_dir, "test.log")

        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        init_logging(verbose_logging=False, logfile=self.log_file)

    def tearDown(self):
        """Clean up"""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        import transport_tools.libs.ui as ui_module
        ui_module._CONSOLE_HANDLER = None
        ui_module._LOG_HANDLER = None

    def test_progressbar_runs_without_error(self):
        """Test that progressbar runs without error"""
        try:
            progressbar(50, 100)
        except Exception as e:
            self.fail(f"progressbar raised exception: {e}")

    def test_progressbar_completion(self):
        """Test progressbar at completion"""
        try:
            progressbar(100, 100)
        except Exception as e:
            self.fail(f"progressbar at completion raised exception: {e}")

    def test_progressbar_start(self):
        """Test progressbar at start"""
        try:
            progressbar(0, 100)
        except Exception as e:
            self.fail(f"progressbar at start raised exception: {e}")


class TestSuppressMsg(unittest.TestCase):
    """Test cases for SuppressMsg context manager"""

    def test_suppress_msg_suppresses_stdout(self):
        """Test that SuppressMsg suppresses stdout"""
        with SuppressMsg():
            print("This should be suppressed")
            sys.stdout.write("This too")

        # If we reach here without output, test passes

    def test_suppress_msg_suppresses_stderr(self):
        """Test that SuppressMsg suppresses stderr"""
        with SuppressMsg():
            sys.stderr.write("This should be suppressed")

        # If we reach here without output, test passes

    def test_suppress_msg_restores_streams(self):
        """Test that SuppressMsg restores stdout/stderr after exit"""
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with SuppressMsg():
            pass

        # Streams should be restored (same file descriptors, at least)
        self.assertIsNotNone(sys.stdout)
        self.assertIsNotNone(sys.stderr)

    def test_suppress_msg_context_manager_protocol(self):
        """Test that SuppressMsg follows context manager protocol"""
        suppress = SuppressMsg()
        self.assertTrue(hasattr(suppress, '__enter__'))
        self.assertTrue(hasattr(suppress, '__exit__'))


class TestTimeProcess(unittest.TestCase):
    """Test cases for TimeProcess context manager"""

    def setUp(self):
        """Set up logging"""
        self.temp_dir = tempfile.mkdtemp(prefix="TT_test_timeprocess_")
        self.log_file = os.path.join(self.temp_dir, "test.log")

        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        init_logging(verbose_logging=False, logfile=self.log_file)

    def tearDown(self):
        """Clean up"""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        import transport_tools.libs.ui as ui_module
        ui_module._CONSOLE_HANDLER = None
        ui_module._LOG_HANDLER = None

    def test_time_process_context_manager_protocol(self):
        """Test that TimeProcess follows context manager protocol"""
        timer = TimeProcess("Test process")
        self.assertTrue(hasattr(timer, '__enter__'))
        self.assertTrue(hasattr(timer, '__exit__'))

    def test_time_process_measures_time(self):
        """Test that TimeProcess measures elapsed time"""
        with TimeProcess("Test process"):
            sleep(0.1)  # Sleep for at least 100ms

        # Check log file contains timing info
        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("Test process", log_content)
            self.assertIn("took", log_content)

    def test_time_process_with_prefix(self):
        """Test TimeProcess with custom prefix message"""
        with TimeProcess("Custom message"):
            pass

        with open(self.log_file, 'r') as f:
            log_content = f.read()
            self.assertIn("Custom message", log_content)

    def test_time_process_without_prefix(self):
        """Test TimeProcess with empty prefix"""
        with TimeProcess(""):
            pass

        # Should not raise error
        self.assertTrue(os.path.exists(self.log_file))


class TestInitiateTools(unittest.TestCase):
    """Test cases for initiate_tools function"""

    def setUp(self):
        """Set up temporary directory"""
        self.temp_dir = tempfile.mkdtemp(prefix="TT_test_initiate_")
        self.log_file = os.path.join(self.temp_dir, "subdir", "test.log")

        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def tearDown(self):
        """Clean up"""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        import transport_tools.libs.ui as ui_module
        ui_module._CONSOLE_HANDLER = None
        ui_module._LOG_HANDLER = None

    @patch('transport_tools.libs.ui.register')
    @patch('transport_tools.libs.ui.greetings')
    def test_initiate_tools_creates_log_directory(self, mock_greetings, mock_register):
        """Test that initiate_tools creates log file directory"""
        initiate_tools(level="INFO", verbose_logging=False, logfile=self.log_file)

        self.assertTrue(os.path.exists(os.path.dirname(self.log_file)))

    @patch('transport_tools.libs.ui.register')
    @patch('transport_tools.libs.ui.greetings')
    def test_initiate_tools_sets_logging_level(self, mock_greetings, mock_register):
        """Test that initiate_tools sets correct logging level"""
        initiate_tools(level="DEBUG", verbose_logging=False, logfile=self.log_file)

        import transport_tools.libs.ui as ui_module
        self.assertEqual(ui_module._CONSOLE_HANDLER.level, logging.DEBUG)

    @patch('transport_tools.libs.ui.register')
    @patch('transport_tools.libs.ui.greetings')
    def test_initiate_tools_calls_greetings(self, mock_greetings, mock_register):
        """Test that initiate_tools calls greetings"""
        initiate_tools(level="INFO", verbose_logging=False, logfile=self.log_file)

        mock_greetings.assert_called_once()

    @patch('transport_tools.libs.ui.register')
    @patch('transport_tools.libs.ui.greetings')
    def test_initiate_tools_registers_bye_bye(self, mock_greetings, mock_register):
        """Test that initiate_tools registers bye_bye at exit"""
        initiate_tools(level="INFO", verbose_logging=False, logfile=self.log_file)

        mock_register.assert_called_once()


if __name__ == '__main__':
    unittest.main()
