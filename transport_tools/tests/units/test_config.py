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
import shutil
from configparser import ConfigParser
from unittest.mock import patch, MagicMock
from logging import getLogger

from transport_tools.libs.config import AnalysisConfig

from transport_tools.tests.units.data.data_config import (
    minimal_config_content,
    advanced_config_content,
    old_parameters_no_changes,
    old_parameters_with_changes,
    expected_template_sections,
    expected_advanced_template_sections,
    sample_filter_names,
)


class TestAnalysisConfigInit(unittest.TestCase):
    """Test cases for AnalysisConfig initialization"""

    def setUp(self):
        """Set up temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp(prefix="TT_test_config_")

    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_without_file(self):
        """Test initialization without config file"""
        config = AnalysisConfig(file2load_from=None, logging=False)

        self.assertIsNone(config.source_file)
        self.assertIsInstance(config.calculations_settings, dict)
        self.assertIsInstance(config.output_settings, dict)
        self.assertIsInstance(config.input_paths, dict)
        self.assertIsInstance(config.output_paths, dict)
        self.assertIsInstance(config.advanced_settings, dict)

    def test_init_sets_default_sections(self):
        """Test that initialization sets default section dictionaries"""
        config = AnalysisConfig(file2load_from=None, logging=False)

        # Check key default parameters exist
        self.assertIn("caver_results_path", config.input_paths)
        self.assertIn("aquaduct_results_path", config.input_paths)
        self.assertIn("start_from_stage", config.calculations_settings)
        self.assertIn("output_path", config.output_settings)


class TestGetSetParameters(unittest.TestCase):
    """Test cases for get/set parameter methods"""

    def setUp(self):
        """Set up test configuration"""
        self.config = AnalysisConfig(file2load_from=None, logging=False)
        # Manually set parameters dict for testing
        self.config.parameters = {
            "test_param1": 100,
            "test_param2": "test_value",
            "test_param3": 3.14,
        }

    def test_get_parameter_existing(self):
        """Test getting existing parameter"""
        result = self.config.get_parameter("test_param1")
        self.assertEqual(result, 100)

    def test_get_parameter_string(self):
        """Test getting string parameter"""
        result = self.config.get_parameter("test_param2")
        self.assertEqual(result, "test_value")

    def test_get_parameter_float(self):
        """Test getting float parameter"""
        result = self.config.get_parameter("test_param3")
        self.assertAlmostEqual(result, 3.14)

    def test_get_parameter_nonexistent_raises(self):
        """Test that getting nonexistent parameter raises KeyError"""
        with self.assertRaises(KeyError):
            self.config.get_parameter("nonexistent_param")

    def test_set_parameter(self):
        """Test setting parameter value"""
        self.config.set_parameter("test_param1", 200)
        self.assertEqual(self.config.parameters["test_param1"], 200)

    def test_set_parameter_new(self):
        """Test setting new parameter"""
        self.config.set_parameter("new_param", "new_value")
        self.assertEqual(self.config.parameters["new_param"], "new_value")

    def test_set_parameter_updates_parameters_dict(self):
        """Test that set_parameter updates the parameters dictionary"""
        self.config.parameters = {"test_calc_param": 10}

        self.config.set_parameter("test_calc_param", 20)

        self.assertEqual(self.config.parameters["test_calc_param"], 20)

    def test_get_parameters_returns_dict(self):
        """Test that get_parameters returns dictionary"""
        result = self.config.get_parameters()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["test_param1"], 100)
        self.assertEqual(result["test_param2"], "test_value")


class TestGetFilters(unittest.TestCase):
    """Test cases for get_filters method"""

    def setUp(self):
        """Set up test configuration with filter parameters"""
        self.config = AnalysisConfig(file2load_from=None, logging=False)
        # Set up filter parameters
        self.config.parameters = {
            "min_length": 10.0,
            "max_length": 50.0,
            "min_bottleneck_radius": 1.5,
            "max_bottleneck_radius": 5.0,
            "min_curvature": 0.1,
            "max_curvature": 2.0,
            "min_sims_num": 2,
            "min_snapshots_num": 5,
            "min_avg_snapshots_num": 3,
            "min_total_events": 10,
            "min_entry_events": 5,
            "min_release_events": 5,
        }

    def test_get_filters_correct_length(self):
        """Test that get_filters returns correct number of filters"""
        result = self.config.get_filters()
        self.assertEqual(len(result), 12)  # 12 filter parameters

    def test_get_filters_correct_order(self):
        """Test that get_filters returns filters in correct order"""
        result = self.config.get_filters()

        expected = tuple([10.0, 50.0, 1.5, 5.0, 0.1, 2.0, 2, 5, 3.0, 10, 5, 5])
        self.assertTupleEqual(result, expected)

    def test_get_filters_with_negative_values(self):
        """Test get_filters with inactive filters (-1)"""
        for key in self.config.parameters.keys():
            self.config.parameters[key] = -1

        result = self.config.get_filters()
        self.assertTupleEqual(result, tuple([-1] * 12))


class TestReportUpdates(unittest.TestCase):
    """Test cases for report_updates method"""

    def setUp(self):
        """Set up test configuration"""
        self.config = AnalysisConfig(file2load_from=None, logging=False)
        self.config.source_file = "/test/config.ini"

        # Set up configuration sections
        self.config.calculations_settings = {
            "start_from_stage": 1,
            "stop_after_stage": 5,
            "snapshots_per_simulation": 100,
            "min_tunnel_radius4clustering": 3.0,
        }
        self.config.output_settings = {
            "output_path": "/path/to/output",
            "verbose_logging": False,
        }
        self.config.input_paths = {
            "caver_results_path": "/path/to/caver",
        }
        self.config.output_paths = {
            "profiles_csv_path": "/path/to/profiles",
        }
        self.config.advanced_settings = {
            "random_seed": 42,
        }
        self.config.advanced_settings_defaults = {
            "random_seed": 42,
        }

    @patch('transport_tools.libs.config.logger')
    def test_report_updates_no_changes(self, mock_logger):
        """Test report_updates when no parameters changed"""
        old_params = self.config.calculations_settings.copy()
        old_params.update(self.config.output_settings)

        self.config.report_updates(old_params)

        # Should still log the header message
        mock_logger.debug.assert_called_once()
        log_message = mock_logger.debug.call_args[0][0]
        self.assertIn("Job configuration UPDATED", log_message)

    @patch('transport_tools.libs.config.logger')
    def test_report_updates_with_changes(self, mock_logger):
        """Test report_updates logs changed parameters"""
        old_params = {
            "start_from_stage": 2,  # Different
            "stop_after_stage": 5,  # Same
            "output_path": "/old/path",  # Different
        }

        self.config.report_updates(old_params)

        mock_logger.debug.assert_called_once()
        log_message = mock_logger.debug.call_args[0][0]

        # Check that changed parameters are logged
        self.assertIn("start_from_stage", log_message)
        self.assertIn("output_path", log_message)
        # Should show new values
        self.assertIn("1", log_message)  # new start_from_stage value
        self.assertIn("/path/to/output", log_message)

    @patch('transport_tools.libs.config.logger')
    def test_report_updates_section_headers(self, mock_logger):
        """Test that section headers are included in log message"""
        old_params = {"start_from_stage": 2}

        self.config.report_updates(old_params)

        log_message = mock_logger.debug.call_args[0][0]
        self.assertIn("General calculations settings", log_message)

    @patch('transport_tools.libs.config.logger')
    def test_report_updates_new_parameters(self, mock_logger):
        """Test report_updates with parameters not in old_parameters"""
        old_params = {}  # Empty old parameters

        self.config.report_updates(old_params)

        mock_logger.debug.assert_called_once()
        log_message = mock_logger.debug.call_args[0][0]

        # All parameters should be logged as new
        self.assertIn("start_from_stage", log_message)
        self.assertIn("output_path", log_message)

    @patch('transport_tools.libs.config.logger')
    def test_report_updates_advanced_settings_changed(self, mock_logger):
        """Test reporting when advanced settings differ from defaults"""
        self.config.advanced_settings = {"random_seed": 99}
        self.config.advanced_settings_defaults = {"random_seed": 42}

        old_params = {"random_seed": 50}

        self.config.report_updates(old_params)

        log_message = mock_logger.debug.call_args[0][0]
        self.assertIn("Advanced settings", log_message)
        self.assertIn("random_seed", log_message)


class TestWriteTemplateFile(unittest.TestCase):
    """Test cases for write_template_file method"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="TT_test_config_template_")
        self.config = AnalysisConfig(file2load_from=None, logging=False)
        self.config.float_params = ["min_tunnel_radius4clustering", "event_min_distance"]

    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_write_template_file_creates_file(self):
        """Test that write_template_file creates a file"""
        template_path = os.path.join(self.temp_dir, "template.ini")

        self.config.write_template_file(template_path, advanced=False)

        self.assertTrue(os.path.exists(template_path))
        self.assertTrue(os.path.isfile(template_path))

    def test_write_template_file_valid_ini_format(self):
        """Test that generated template is valid INI format"""
        template_path = os.path.join(self.temp_dir, "template.ini")

        self.config.write_template_file(template_path, advanced=False)

        # Try to parse it with ConfigParser
        parser = ConfigParser()
        parser.read(template_path)

        # Check that expected sections exist
        sections = parser.sections()
        self.assertIn("INPUT_PATHS", sections)
        self.assertIn("CALCULATIONS_SETTINGS", sections)
        self.assertIn("OUTPUT_SETTINGS", sections)

    def test_write_template_file_without_advanced(self):
        """Test template file without advanced settings"""
        template_path = os.path.join(self.temp_dir, "template_basic.ini")

        self.config.write_template_file(template_path, advanced=False)

        parser = ConfigParser()
        parser.read(template_path)
        sections = parser.sections()

        self.assertNotIn("ADVANCED_SETTINGS", sections)

    def test_write_template_file_with_advanced(self):
        """Test template file with advanced settings"""
        template_path = os.path.join(self.temp_dir, "template_advanced.ini")

        self.config.write_template_file(template_path, advanced=True)

        parser = ConfigParser()
        parser.read(template_path)
        sections = parser.sections()

        self.assertIn("ADVANCED_SETTINGS", sections)

    def test_write_template_file_content_structure(self):
        """Test that template file has proper content structure"""
        template_path = os.path.join(self.temp_dir, "template.ini")

        self.config.write_template_file(template_path, advanced=False)

        with open(template_path, "r") as f:
            content = f.read()

        # Check for section headers
        self.assertIn("[INPUT_PATHS]", content)
        self.assertIn("[CALCULATIONS_SETTINGS]", content)
        self.assertIn("[OUTPUT_SETTINGS]", content)

        # Check for some key parameters
        self.assertIn("caver_results_path", content)
        self.assertIn("start_from_stage", content)
        self.assertIn("output_path", content)

    def test_write_template_file_float_formatting(self):
        """Test that float parameters are formatted correctly"""
        template_path = os.path.join(self.temp_dir, "template.ini")

        # Set a float parameter
        self.config.calculations_settings["min_tunnel_radius4clustering"] = 3.5

        self.config.write_template_file(template_path, advanced=False)

        with open(template_path, "r") as f:
            content = f.read()

        # Should be formatted with 2 decimal places
        self.assertIn("min_tunnel_radius4clustering = 3.50", content)

    def test_write_template_file_excludes_caver_relative(self):
        """Test that caver_relative_ parameters are excluded"""
        template_path = os.path.join(self.temp_dir, "template.ini")

        # Add a caver_relative_ parameter
        self.config.input_paths["caver_relative_profile_file"] = "some_file.txt"

        self.config.write_template_file(template_path, advanced=False)

        with open(template_path, "r") as f:
            content = f.read()

        # Should NOT appear in template
        self.assertNotIn("caver_relative_profile_file", content)

    def test_write_template_file_includes_comments(self):
        """Test that template includes commentary lines"""
        template_path = os.path.join(self.temp_dir, "template.ini")

        self.config.write_template_file(template_path, advanced=False)

        with open(template_path, "r") as f:
            content = f.read()

        # Check for comment lines
        self.assertIn("# CAVER results", content)
        self.assertIn("# AQUA-DUCT results", content)


class TestParameterValidation(unittest.TestCase):
    """Test cases for parameter validation methods"""

    def setUp(self):
        """Set up test configuration"""
        self.config = AnalysisConfig(file2load_from=None, logging=False)

    def test_test_parameter_sanity_valid_value(self):
        """Test parameter sanity check with valid value"""
        self.config.parameters = {"test_param": 5}

        # Should not raise exception
        try:
            self.config._test_parameter_sanity("test_param", min_val=0, max_val=10)
        except ValueError:
            self.fail("_test_parameter_sanity raised ValueError unexpectedly")

    def test_test_parameter_sanity_below_minimum(self):
        """Test parameter sanity check with value below minimum"""
        self.config.parameters = {"test_param": -5}

        with self.assertRaises(ValueError) as context:
            self.config._test_parameter_sanity("test_param", min_val=0, max_val=10)

        self.assertIn("must be within", str(context.exception))

    def test_test_parameter_sanity_above_maximum(self):
        """Test parameter sanity check with value above maximum"""
        self.config.parameters = {"test_param": 15}

        with self.assertRaises(ValueError) as context:
            self.config._test_parameter_sanity("test_param", min_val=0, max_val=10)

        self.assertIn("must be within", str(context.exception))

    def test_test_parameter_sanity_edge_values(self):
        """Test parameter sanity check with edge values"""
        self.config.parameters = {"test_param": 0}

        # Minimum edge should be valid
        self.config._test_parameter_sanity("test_param", min_val=0, max_val=10)

        # Maximum edge should be valid
        self.config.parameters = {"test_param": 10}
        self.config._test_parameter_sanity("test_param", min_val=0, max_val=10)


class TestStrMethod(unittest.TestCase):
    """Test cases for __str__ method"""

    def setUp(self):
        """Set up test configuration"""
        # __str__ requires parameters attribute which is only set when loading from file
        # We need to mock the full initialization
        self.config = AnalysisConfig(file2load_from=None, logging=False)
        # Manually set parameters attribute to simulate loaded config
        self.config.parameters = {
            "test_setting": 42,
            "output_path": "/test/output",
            "visualize_super_cluster_volumes": False,
            "visualize_transformed_tunnels": False,
            "visualize_transformed_transport_events": False,
            "visualize_exact_matching_outcomes": False,
            "trajectory_engine": "mdtraj",  # Required by __str__
            "distance_backend": "local",
            "msms": None,  # Required by __str__
        }
        self.config.source_file = "/test/config.ini"
        self.config.calculations_settings = {"test_setting": 42}
        self.config.output_settings = {"output_path": "/test/output"}

    def test_str_returns_string(self):
        """Test that __str__ returns a string"""
        result = str(self.config)
        self.assertIsInstance(result, str)

    def test_str_includes_version_info(self):
        """Test that __str__ includes version information"""
        result = str(self.config)
        self.assertIn("TransportTools library version", result)
        self.assertIn("python", result)

    def test_str_includes_dependency_versions(self):
        """Test that __str__ includes dependency version information"""
        result = str(self.config)

        # Check for dependency information
        self.assertIn("numpy", result)
        self.assertIn("scipy", result)
        self.assertIn("biopython", result)

    def test_str_includes_parameter_sections(self):
        """Test that __str__ includes parameter sections"""
        result = str(self.config)

        # Check for section headers
        self.assertIn("General calculations settings", result)
        self.assertIn("Output settings", result)


# ---------------------------------------------------------------------------
# New v0.9.7 parameters & behaviors
# ---------------------------------------------------------------------------

class TestAquaductNewParameterDefaults(unittest.TestCase):
    """Defaults for v0.9.7 parameters: aquaduct_traced_residues_filter,
    aquaduct_allow_empty_folders, and the _logged_empty_aquaduct_folders cache."""

    def test_aquaduct_traced_residues_filter_default_is_none(self):
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertIn("aquaduct_traced_residues_filter", config.calculations_settings)
        self.assertIsNone(config.calculations_settings["aquaduct_traced_residues_filter"])

    def test_aquaduct_allow_empty_folders_default_is_false(self):
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertIn("aquaduct_allow_empty_folders", config.calculations_settings)
        self.assertFalse(config.calculations_settings["aquaduct_allow_empty_folders"])

    def test_logged_empty_folders_is_set(self):
        """The warn-once cache for empty AQUA-DUCT folders must be a set instance."""
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertIsInstance(config._logged_empty_aquaduct_folders, set)
        self.assertEqual(len(config._logged_empty_aquaduct_folders), 0)


class TestAquaductTracedResiduesFilterParsing(unittest.TestCase):
    """Comma-separated string → list[str] parsing done in _autocomplete_parameters."""

    def _make_config_with_parameters(self, filter_value):
        """Build a config object with the minimum parameters needed to run
        _autocomplete_parameters() without triggering filesystem auto-detection."""
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.parameters = {
            "aquaduct_traced_residues_filter": filter_value,
            "stop_after_stage": 5,
            "num_cpus": 1,
            "pdb_reference_structure": "/tmp/ignored.pdb",
            "snapshots_per_simulation": 100,
            "perform_comparative_analysis": False,
            "comparative_groups_definition": None,
            "trajectory_path": None,
            "legacy_pymol_support": False,
            "slurm_folder": None,
            "clustering_folder": "/tmp/ignored_clustering",
        }
        return config

    def test_comma_separated_string_parsed_to_list(self):
        config = self._make_config_with_parameters("WAT,O2")
        config._autocomplete_parameters()
        self.assertListEqual(["WAT", "O2"],
                             config.parameters["aquaduct_traced_residues_filter"])

    def test_values_uppercased(self):
        config = self._make_config_with_parameters("wat, o2, U01")
        config._autocomplete_parameters()
        self.assertListEqual(["WAT", "O2", "U01"],
                             config.parameters["aquaduct_traced_residues_filter"])

    def test_whitespace_stripped(self):
        config = self._make_config_with_parameters("  WAT  ,  O2  ")
        config._autocomplete_parameters()
        self.assertListEqual(["WAT", "O2"],
                             config.parameters["aquaduct_traced_residues_filter"])

    def test_none_stays_none(self):
        config = self._make_config_with_parameters(None)
        config._autocomplete_parameters()
        self.assertIsNone(config.parameters["aquaduct_traced_residues_filter"])

    def test_empty_string_becomes_empty_list(self):
        """An empty/whitespace-only string parses to an empty list. Note: downstream
        _validate_parameter_values rejects this with a ValueError, but the parser itself
        produces []."""
        config = self._make_config_with_parameters("   ,  , ")
        config._autocomplete_parameters()
        self.assertListEqual([], config.parameters["aquaduct_traced_residues_filter"])


class TestAquaductResultsPathParsing(unittest.TestCase):
    """Single comma-separated string → list[str] of root paths done in _set_input_paths."""

    def _make_config(self, aquaduct_path_value):
        config = AnalysisConfig(file2load_from=None, logging=False)
        # Minimum to make _set_input_paths run past the aquaduct block without raising:
        config.input_paths["caver_results_relative_subfolder_path"] = "caver"
        config.input_paths["caver_relative_pdb_file"] = "stripped_system.pdb"
        config.input_paths["aquaduct_results_path"] = aquaduct_path_value
        return config

    def test_single_path_becomes_list(self):
        config = self._make_config("/path/to/aquaduct")
        config._set_input_paths()
        self.assertEqual(["/path/to/aquaduct"], config.input_paths["aquaduct_results_path"])

    def test_comma_separated_splits(self):
        config = self._make_config("/path/water,/path/ligand")
        config._set_input_paths()
        self.assertEqual(["/path/water", "/path/ligand"],
                         config.input_paths["aquaduct_results_path"])

    def test_empty_segments_stripped(self):
        config = self._make_config("/path/water, ,/path/ligand,")
        config._set_input_paths()
        self.assertEqual(["/path/water", "/path/ligand"],
                         config.input_paths["aquaduct_results_path"])

    def test_none_stays_none(self):
        config = self._make_config(None)
        config._set_input_paths()
        self.assertIsNone(config.input_paths["aquaduct_results_path"])


class TestGetInputFoldersAquaductDict(unittest.TestCase):
    """get_input_folders() now returns a dict mapping each aquaduct_results_path to its
    list of MD-simulation folder names. Includes coverage of aquaduct_allow_empty_folders
    pruning behavior."""

    def setUp(self):
        self.maxDiff = None
        self.tmp = tempfile.mkdtemp(prefix="TT_test_get_input_folders_")

    def tearDown(self):
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)

    def _make_aquaduct_root(self, name, populated_mds, empty_mds=()):
        """Build a temp AQUA-DUCT root with given populated and empty MD subfolders.
        Populated MDs contain dummy tar.gz and analysis txt files; empty ones don't."""
        root = os.path.join(self.tmp, name)
        os.makedirs(root)
        for md in populated_mds:
            d = os.path.join(root, md, "aquaduct_results")
            os.makedirs(d)
            open(os.path.join(d, "6_visualize_results.tar.gz"), "w").close()
            open(os.path.join(d, "5_analysis_results.txt"), "w").close()
        for md in empty_mds:
            os.makedirs(os.path.join(self.tmp, name, md))
        return root

    def _config_with_aquaduct(self, paths_list, allow_empty=False):
        """Construct an AnalysisConfig with parameters dict populated to exercise
        get_input_folders()."""
        caver_root = os.path.join(self.tmp, "caver_root")
        os.makedirs(caver_root, exist_ok=True)
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.parameters = {
            "caver_results_path": caver_root,
            "caver_results_folder_pattern": "md*",
            "trajectory_path": None,
            "trajectory_folder_pattern": "md*",
            "aquaduct_results_path": paths_list,
            "aquaduct_results_folder_pattern": "e*",
            "aquaduct_results_relative_tarfile": "aquaduct_results/6_visualize_results.tar.gz",
            "aquaduct_results_relative_summaryfile": "aquaduct_results/5_analysis_results.txt",
            "aquaduct_allow_empty_folders": allow_empty,
        }
        return config

    def test_none_aquaduct_path_returns_empty_dict(self):
        config = self._config_with_aquaduct(None)
        _, _, aquaduct = config.get_input_folders()
        self.assertEqual({}, aquaduct)

    def test_single_path_returns_dict_keyed_by_path(self):
        root = self._make_aquaduct_root("water", populated_mds=["e10s0_md1", "e10s14_md2"])
        config = self._config_with_aquaduct([root])
        _, _, aquaduct = config.get_input_folders()
        self.assertEqual({root: ["e10s0_md1", "e10s14_md2"]}, aquaduct)

    def test_allow_empty_folders_prunes_folder_without_required_files(self):
        """Folders missing the required tar/summary files must be dropped from the
        result when aquaduct_allow_empty_folders is True."""
        root = self._make_aquaduct_root("water",
                                         populated_mds=["e10s0_full"],
                                         empty_mds=["e10s14_empty"])
        config = self._config_with_aquaduct([root], allow_empty=True)
        _, _, aquaduct = config.get_input_folders()
        self.assertEqual({root: ["e10s0_full"]}, aquaduct)

    def test_allow_empty_folders_logs_warning_exactly_once(self):
        """Same missing folder must only warn once across multiple get_input_folders()
        calls (tracked by _logged_empty_aquaduct_folders)."""
        root = self._make_aquaduct_root("water",
                                         populated_mds=["e10s0_full"],
                                         empty_mds=["e10s14_empty"])
        config = self._config_with_aquaduct([root], allow_empty=True)
        with patch("transport_tools.libs.config.logger") as mock_logger:
            config.get_input_folders()
            config.get_input_folders()
            warnings = [call for call in mock_logger.warning.call_args_list
                        if "e10s14_empty" in str(call)]
            self.assertEqual(1, len(warnings),
                             "expected exactly one warning across two calls, got: {}".format(warnings))


if __name__ == '__main__':
    unittest.main()
