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

from transport_tools.libs.config import AnalysisConfig, ShardContext

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
            "relevant_tunnel_min_radius": 3.0,
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
        """Test report_updates emits nothing when no parameters changed"""
        old_params = {}
        old_params.update(self.config.calculations_settings)
        old_params.update(self.config.output_settings)
        old_params.update(self.config.input_paths)
        old_params.update(self.config.output_paths)
        old_params.update(self.config.advanced_settings)

        self.config.report_updates(old_params)

        # Nothing changed -> no report should be logged
        mock_logger.warning.assert_not_called()

    @patch('transport_tools.libs.config.logger')
    def test_report_updates_with_changes(self, mock_logger):
        """Test report_updates logs changed parameters at warning level"""
        old_params = {
            "start_from_stage": 2,  # Different
            "stop_after_stage": 5,  # Same
            "output_path": "/old/path",  # Different
        }

        self.config.report_updates(old_params)

        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]

        # Header and changed parameters are logged
        self.assertIn("Job configuration UPDATED", log_message)
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

        log_message = mock_logger.warning.call_args[0][0]
        self.assertIn("General calculations settings", log_message)

    @patch('transport_tools.libs.config.logger')
    def test_report_updates_new_parameters(self, mock_logger):
        """Test report_updates with parameters not in old_parameters"""
        old_params = {}  # Empty old parameters

        self.config.report_updates(old_params)

        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]

        # All parameters should be logged as new
        self.assertIn("start_from_stage", log_message)
        self.assertIn("output_path", log_message)

    @patch('transport_tools.libs.config.logger')
    def test_report_updates_advanced_settings_changed(self, mock_logger):
        """Test that advanced settings are reported when changed vs. previous run"""
        self.config.advanced_settings = {"random_seed": 99}
        self.config.advanced_settings_defaults = {"random_seed": 42}

        old_params = {"random_seed": 50}

        self.config.report_updates(old_params)

        log_message = mock_logger.warning.call_args[0][0]
        self.assertIn("Advanced settings", log_message)
        self.assertIn("random_seed", log_message)

    @patch('transport_tools.libs.config.logger')
    def test_report_updates_advanced_setting_reverted_to_default(self, mock_logger):
        """Advanced setting changed back to its default vs. previous run is still reported"""
        self.config.advanced_settings = {"random_seed": 42}
        self.config.advanced_settings_defaults = {"random_seed": 42}

        old_params = {"random_seed": 99}  # previous run used a non-default value

        self.config.report_updates(old_params)

        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]
        self.assertIn("Advanced settings", log_message)
        self.assertIn("random_seed", log_message)


class TestWriteTemplateFile(unittest.TestCase):
    """Test cases for write_template_file method"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="TT_test_config_template_")
        self.config = AnalysisConfig(file2load_from=None, logging=False)
        self.config.float_params = ["relevant_tunnel_min_radius", "event_min_distance"]

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
        self.config.calculations_settings["relevant_tunnel_min_radius"] = 3.5

        self.config.write_template_file(template_path, advanced=False)

        with open(template_path, "r") as f:
            content = f.read()

        # Should be formatted with 2 decimal places
        self.assertIn("relevant_tunnel_min_radius = 3.50", content)

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
            "compute_backend": "local",
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

    def test_slurm_keep_shard_results_default_is_false(self):
        # cleanup-on-success is the default; users must opt in to keep the SLURM cache
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertIn("slurm_keep_shard_results", config.calculations_settings)
        self.assertFalse(config.calculations_settings["slurm_keep_shard_results"])

    def test_slurm_keep_shard_results_is_boolean_param(self):
        # registered as a boolean so ConfigParser correctly coerces "true" / "false" strings
        # from the INI file (otherwise the launcher would compare strings and never clean up)
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertIn("slurm_keep_shard_results", config.boolean_params)

    def test_slurm_additional_parameters_default_is_none(self):
        # absent INI knob -> None; executor_params() then omits the slurm_additional_parameters
        # key entirely so submitit emits no extra #SBATCH headers
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertIn("slurm_additional_parameters", config.calculations_settings)
        self.assertIsNone(config.calculations_settings["slurm_additional_parameters"])

    def test_slurm_additional_parameters_is_dict_param(self):
        # registered in dict_params so the INI parser coerces comma-separated key=value
        # strings into a dict at load time (not silently kept as a string)
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertIn("slurm_additional_parameters", config.dict_params)

    def test_slurm_setup_is_list_param(self):
        # slurm_setup moved from multi-line list to comma-separated list; the parser must
        # still treat it as a list (not a plain str), otherwise executor_params() would
        # pass a string to submitit which expects List[str]
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertIn("slurm_setup", config.list_params)

    def test_slurm_srun_args_is_list_param(self):
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertIn("slurm_srun_args", config.list_params)

    def test_logged_empty_folders_is_set(self):
        """The warn-once cache for empty AQUA-DUCT folders must be a set instance."""
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertIsInstance(config._logged_empty_aquaduct_folders, set)
        self.assertEqual(len(config._logged_empty_aquaduct_folders), 0)


class TestSlurmIniParameterParsing(unittest.TestCase):
    """Comma-separated INI -> typed Python value coercion for the SLURM knobs:
    `slurm_setup` / `slurm_srun_args` (list_params) and `slurm_additional_parameters`
    (dict_params). Uses the real `_load_configuration` parser branch, not a mock, so a
    regression in the parsing path is caught here even if no other test happens to load
    these knobs from an INI."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="TT_test_slurm_ini_")
        self.addCleanup(shutil.rmtree, self._tmp, True)

    def _load(self, calc_body: str):
        """Write a minimal INI with the given CALCULATIONS_SETTINGS body and load it via
        the parser branch under test; return the populated AnalysisConfig.

        `_load_configuration` always calls `_set_input_paths()` and the autoactivation
        helper at the very end - both expect a fully-populated INI. We're only testing the
        SLURM-parameter parsing branch here, so we stub those side effects out and assert
        directly on `calculations_settings`.
        """

        ini_path = os.path.join(self._tmp, "test.ini")
        with open(ini_path, "w") as out:
            out.write("[CALCULATIONS_SETTINGS]\n")
            out.write(calc_body)
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.source_file = ini_path
        with patch.object(config, "_set_input_paths"), \
                patch.object(config, "_autoactivate_functionalities"):
            config._load_configuration(ini_path)
        return config

    def test_slurm_setup_list_parsed_from_comma_separated(self):
        cfg = self._load("slurm_setup = module load python, source ~/env/bin/activate\n")
        self.assertEqual(["module load python", "source ~/env/bin/activate"],
                         cfg.calculations_settings["slurm_setup"])

    def test_slurm_srun_args_list_parsed_from_comma_separated(self):
        cfg = self._load("slurm_srun_args = --cpu-bind=none, --mpi=pmix\n")
        self.assertEqual(["--cpu-bind=none", "--mpi=pmix"],
                         cfg.calculations_settings["slurm_srun_args"])

    def test_slurm_additional_parameters_dict_parsed_from_comma_separated(self):
        cfg = self._load("slurm_additional_parameters = qos=high, constraint=icelake, "
                         "gres=gpu:1\n")
        self.assertEqual({"qos": "high", "constraint": "icelake", "gres": "gpu:1"},
                         cfg.calculations_settings["slurm_additional_parameters"])

    def test_slurm_additional_parameters_allows_empty_value(self):
        # 'gres=' is a legitimate SBATCH header meaning "no GRES request"; the parser must
        # accept it and produce an empty-string value rather than choking
        cfg = self._load("slurm_additional_parameters = gres=\n")
        self.assertEqual({"gres": ""},
                         cfg.calculations_settings["slurm_additional_parameters"])

    def test_slurm_additional_parameters_rejects_missing_equals(self):
        # a typo without '=' is the most likely user error; the parser must raise at load
        # time so the error surfaces immediately rather than three stages in when submitit
        # sees a malformed sbatch header
        with self.assertRaises(ValueError) as cm:
            self._load("slurm_additional_parameters = qos=high, oops_no_equals_here\n")
        self.assertIn("key=value", str(cm.exception))
        self.assertIn("oops_no_equals_here", str(cm.exception))

    def test_empty_list_param_yields_empty_list(self):
        # explicit empty value -> None via the possibly_none_params path (slurm_setup is
        # already there for the existing default-None behaviour), so the consumer's
        # `if parameters["slurm_setup"]:` check still works correctly. Asserting on the
        # config attribute directly avoids depending on validation.
        cfg = self._load("slurm_setup = \n")
        self.assertIsNone(cfg.calculations_settings["slurm_setup"])

    def test_empty_dict_param_yields_none(self):
        # same possibly_none_params handling for slurm_additional_parameters
        cfg = self._load("slurm_additional_parameters = \n")
        self.assertIsNone(cfg.calculations_settings["slurm_additional_parameters"])


class TestAquaductTracedResiduesFilterParsing(unittest.TestCase):
    """Comma-separated string → list[str] parsing done in _autocomplete_parameters."""

    def _make_config_with_parameters(self, filter_value):
        """Build a config object with the minimum parameters needed to run
        _autocomplete_parameters() without triggering filesystem auto-detection."""
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.legacy_params = {}
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
            "slurm_root_folder": None,
            "internal_folder": "/tmp/ignored_internal",
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


class TestStageBackendResolution(unittest.TestCase):
    """resolve_stage_backend() + per-stage backend override + compute_backend fallback."""

    def _make_config(self, compute_backend="local", stage04_backend=None):
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.parameters = {
            "compute_backend": compute_backend,
            "stage04_backend": stage04_backend,
        }
        return config

    def test_falls_back_to_compute_backend_when_override_is_none(self):
        config = self._make_config(compute_backend="slurm", stage04_backend=None)
        self.assertEqual("slurm", config.resolve_stage_backend("stage04_backend"))

    def test_per_stage_override_wins_over_compute_backend(self):
        config = self._make_config(compute_backend="local", stage04_backend="slurm")
        self.assertEqual("slurm", config.resolve_stage_backend("stage04_backend"))

    def test_per_stage_override_is_normalised_to_lowercase(self):
        config = self._make_config(compute_backend="local", stage04_backend="SLURM")
        self.assertEqual("slurm", config.resolve_stage_backend("stage04_backend"))

    def test_unknown_knob_raises_key_error(self):
        config = self._make_config()
        with self.assertRaises(KeyError):
            config.resolve_stage_backend("stage99_unknown_backend")

    def test_stage02_knob_is_registered(self):
        # adding TunnelNetworksShardStage required registering its backend knob in
        # _stage_backend_keys; resolve_stage_backend must recognise it
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.parameters = {"compute_backend": "local", "stage02_backend": "slurm"}
        self.assertEqual("slurm", config.resolve_stage_backend("stage02_backend"))

    def test_stage03_knob_is_registered(self):
        # adding TunnelLayeringShardStage required registering its backend knob in
        # _stage_backend_keys; resolve_stage_backend must recognise it
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.parameters = {"compute_backend": "local", "stage03_backend": "slurm"}
        self.assertEqual("slurm", config.resolve_stage_backend("stage03_backend"))

    def test_stage08_knob_is_registered(self):
        # adding AquaductLayeringShardStage required registering its backend knob in
        # _stage_backend_keys; resolve_stage_backend must recognise it
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.parameters = {"compute_backend": "local", "stage08_backend": "slurm"}
        self.assertEqual("slurm", config.resolve_stage_backend("stage08_backend"))

    def test_stage07_knob_is_registered(self):
        # adding AquaductNetworksShardStage required registering its backend knob in
        # _stage_backend_keys; resolve_stage_backend must recognise it
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.parameters = {"compute_backend": "local", "stage07_backend": "slurm"}
        self.assertEqual("slurm", config.resolve_stage_backend("stage07_backend"))

    def test_stage09_knob_is_registered(self):
        # adding EventAssignmentShardStage required registering its backend knob in
        # _stage_backend_keys; resolve_stage_backend must recognise it
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.parameters = {"compute_backend": "local", "stage09_backend": "slurm"}
        self.assertEqual("slurm", config.resolve_stage_backend("stage09_backend"))


class TestSlurmRootFolderDefault(unittest.TestCase):
    """slurm_root_folder defaults to <internal_folder>/_slurm when unset."""

    def test_autocompletes_under_internal_folder(self):
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.legacy_params = {}
        config.parameters = {
            "stop_after_stage": 5,
            "num_cpus": 1,
            "pdb_reference_structure": "/tmp/ignored.pdb",
            "snapshots_per_simulation": 100,
            "perform_comparative_analysis": False,
            "comparative_groups_definition": None,
            "trajectory_path": None,
            "legacy_pymol_support": False,
            "aquaduct_traced_residues_filter": None,
            "slurm_root_folder": None,
            "internal_folder": "/tmp/results/_internal",
            "clustering_folder": "/tmp/ignored_clustering",
        }
        config._autocomplete_parameters()
        self.assertEqual("/tmp/results/_internal/_slurm",
                         config.parameters["slurm_root_folder"])

    def test_explicit_value_is_respected(self):
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.legacy_params = {}
        config.parameters = {
            "stop_after_stage": 5,
            "num_cpus": 1,
            "pdb_reference_structure": "/tmp/ignored.pdb",
            "snapshots_per_simulation": 100,
            "perform_comparative_analysis": False,
            "comparative_groups_definition": None,
            "trajectory_path": None,
            "legacy_pymol_support": False,
            "aquaduct_traced_residues_filter": None,
            "slurm_root_folder": "/scratch/run42/_slurm",
            "internal_folder": "/tmp/results/_internal",
            "clustering_folder": "/tmp/ignored_clustering",
        }
        config._autocomplete_parameters()
        self.assertEqual("/scratch/run42/_slurm",
                         config.parameters["slurm_root_folder"])


class TestNumCpusLocalValidation(unittest.TestCase):
    """The whole num_cpus validation block is gated by the execution context (ShardContext).

    On a normal/launcher run (is_shard=False, the default) an over-subscribed num_cpus is
    rejected. SLURM shards rebuild the config on a compute node whose CPU count is unrelated
    to the launcher's - and where num_cpus is inert (the inner pool is sized from
    slurm_cpus_per_task) - so they pass ShardContext(is_shard=True) to skip the block entirely
    and avoid spuriously aborting (see slurm._rebuild_mol_system)."""

    def _make_config(self, context):
        """Assemble a fully-populated, otherwise-valid parameters dict from the shipped
        defaults, overriding only the handful that default to None, then set num_cpus well
        above this machine's reported core count. get_input_folders() (the filesystem-walking
        tail of _validate_parameter_values) is stubbed so the test stays hermetic."""

        config = AnalysisConfig(file2load_from=None, logging=False, context=context)
        config.parameters = {}
        for section in (config.calculations_settings, config.output_settings,
                        config.input_paths, config.output_paths,
                        config.advanced_settings, config.internal_settings):
            config.parameters.update(section)
        config.parameters["num_cpus"] = os.cpu_count() + 16
        config.parameters["snapshots_per_simulation"] = 100
        config.parameters["stop_after_stage"] = 5
        return config

    def test_oversubscribed_num_cpus_rejected_by_default(self):
        config = self._make_config(context=ShardContext(is_shard=False))
        with patch.object(config, "get_input_folders", return_value=({}, {}, {})):
            with self.assertRaises(ValueError) as cm:
                config._validate_parameter_values()
        self.assertIn("num_cpus", str(cm.exception))

    def test_oversubscribed_num_cpus_allowed_in_shard_context(self):
        config = self._make_config(context=ShardContext(is_shard=True))
        with patch.object(config, "get_input_folders", return_value=({}, {}, {})):
            # must not raise: num_cpus is inert in a shard, so the whole block is skipped
            config._validate_parameter_values()

    def test_context_defaults_to_launcher(self):
        config = AnalysisConfig(file2load_from=None, logging=False)
        self.assertFalse(config.context.is_shard)


def _populated_config(active_stages=(1, 10), is_shard=False, **overrides):
    """Build a fileless config with a fully-populated parameters dict (every section's defaults),
    a pinned active-stage interval and execution context, then apply per-test overrides. Mirrors
    the assembly used by TestNumCpusLocalValidation so the stage-scoped methods can be invoked in
    isolation without touching the filesystem."""

    config = AnalysisConfig(file2load_from=None, logging=False,
                            context=ShardContext(is_shard=is_shard))
    config._active_stages = active_stages
    config.parameters = {}
    for section in (config.calculations_settings, config.output_settings,
                    config.input_paths, config.output_paths,
                    config.advanced_settings, config.internal_settings):
        config.parameters.update(section)
    config.parameters.update(overrides)
    return config


class TestRunsStagePredicate(unittest.TestCase):
    """_runs_stage(*stages) is True iff any given stage falls inside the active [lo, hi] interval."""

    def _config(self, active_stages):
        config = AnalysisConfig(file2load_from=None, logging=False)
        config._active_stages = active_stages
        return config

    def test_full_run_matches_every_stage(self):
        config = self._config((1, 10))
        for stage in range(1, 11):
            self.assertTrue(config._runs_stage(stage))
        self.assertTrue(config._runs_stage(1, 2))
        self.assertTrue(config._runs_stage(7))

    def test_resume_interval_excludes_earlier_stages(self):
        config = self._config((5, 10))
        self.assertFalse(config._runs_stage(1))
        self.assertFalse(config._runs_stage(1, 2))
        self.assertTrue(config._runs_stage(5))
        self.assertTrue(config._runs_stage(2, 9))  # 9 is in range even though 2 is not

    def test_single_stage_shard_interval(self):
        config = self._config((4, 4))
        self.assertTrue(config._runs_stage(4))
        self.assertTrue(config._runs_stage(2, 4))
        self.assertFalse(config._runs_stage(1))
        self.assertFalse(config._runs_stage(2, 7))

    def test_any_semantics(self):
        config = self._config((3, 3))
        self.assertTrue(config._runs_stage(1, 3, 9))
        self.assertFalse(config._runs_stage(1, 2, 9))


class TestResolveActiveStages(unittest.TestCase):
    """_resolve_active_stages derives (lo, hi) from start/stop unless an explicit override is given."""

    def test_derives_from_start_and_stop(self):
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.calculations_settings["start_from_stage"] = 3
        config.calculations_settings["stop_after_stage"] = 6
        config._active_stages_override = None
        config._resolve_active_stages()
        self.assertEqual((3, 6), config._active_stages)

    def test_unset_stop_defaults_to_last_stage(self):
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.calculations_settings["start_from_stage"] = 1
        config.calculations_settings["stop_after_stage"] = None
        config._active_stages_override = None
        config._resolve_active_stages()
        self.assertEqual((1, 10), config._active_stages)

    def test_explicit_override_wins(self):
        config = AnalysisConfig(file2load_from=None, logging=False)
        config.calculations_settings["start_from_stage"] = 1
        config.calculations_settings["stop_after_stage"] = 10
        config._active_stages_override = (4, 4)
        config._resolve_active_stages()
        self.assertEqual((4, 4), config._active_stages)


class TestAutocompleteScoping(unittest.TestCase):
    """The expensive autodetect setups are scoped: reference-structure detection (CAVER glob) and
    snapshot detection (reads every trajectory) only fire when actually needed for the active
    stages and execution context."""

    def _run(self, active_stages, is_shard):
        config = _populated_config(active_stages=active_stages, is_shard=is_shard,
                                   num_cpus=2, pdb_reference_structure=None,
                                   snapshots_per_simulation=None, trajectory_path="/traj")
        with patch.object(config, "_detect_set_pdb_reference_structure") as pdb_mock, \
                patch.object(config, "_detect_set_num_snapshots") as snap_mock:
            config._autocomplete_parameters()
        return pdb_mock.called, snap_mock.called

    def test_launcher_full_run_detects_both(self):
        pdb, snap = self._run((1, 10), is_shard=False)
        self.assertTrue(pdb)
        self.assertTrue(snap)

    def test_launcher_resume_detects_neither(self):
        pdb, snap = self._run((5, 10), is_shard=False)
        self.assertFalse(pdb)
        self.assertFalse(snap)

    def test_launcher_initial_partial_run_still_detects(self):
        # stage 1 present => initial fresh run => both values are resolved and baked into the checkpoint
        pdb, snap = self._run((1, 4), is_shard=False)
        self.assertTrue(pdb)
        self.assertTrue(snap)

    def test_shard_alignment_stage_detects_reference_only(self):
        for stage in (2, 7):
            pdb, snap = self._run((stage, stage), is_shard=True)
            self.assertTrue(pdb, "stage %d shard should self-derive the reference" % stage)
            self.assertFalse(snap, "no shard consumes snapshots_per_simulation")

    def test_shard_non_alignment_stage_detects_neither(self):
        for stage in (3, 4, 8, 9):
            pdb, snap = self._run((stage, stage), is_shard=True)
            self.assertFalse(pdb, "stage %d shard does not consume the reference" % stage)
            self.assertFalse(snap)

    def test_snapshot_detection_skipped_does_not_raise_without_trajectory(self):
        # resume scope with neither snapshots nor trajectory access must not raise (value comes from
        # the checkpoint), whereas the initial run would still demand it
        config = _populated_config(active_stages=(5, 10), num_cpus=2,
                                   pdb_reference_structure="/ref.pdb",
                                   snapshots_per_simulation=None, trajectory_path=None)
        config._autocomplete_parameters()  # must not raise

    def test_initial_run_without_snapshots_or_trajectory_raises(self):
        config = _populated_config(active_stages=(1, 10), num_cpus=2,
                                   pdb_reference_structure="/ref.pdb",
                                   snapshots_per_simulation=None, trajectory_path=None)
        with self.assertRaises(RuntimeError):
            config._autocomplete_parameters()


class TestInputDataScoping(unittest.TestCase):
    """_test_input_data only walks the input trees for stages that will actually run; the
    get_input_folders() glob is not even invoked when no in-range stage consumes the data."""

    def _config(self, active_stages, is_shard=False, **overrides):
        params = dict(
            caver_results_path="/caver", caver_results_folder_pattern="*",
            caver_results_relative_subfolder_path="sub", pdb_reference_structure=None,
            trajectory_path="/traj", trajectory_folder_pattern="*",
            topology_relative_file="top.pdb", trajectory_relative_file="traj.xtc",
            aquaduct_results_path=["/aq"], aquaduct_results_folder_pattern="*",
            aquaduct_results_relative_tarfile="t.tar.gz", aquaduct_results_relative_summaryfile="s.txt",
            process_bottleneck_residues=False, visualize_comparative_super_cluster_volumes=False,
        )
        params.update(overrides)
        return _populated_config(active_stages=active_stages, is_shard=is_shard, **params)

    def _run(self, config):
        folders = (["md1"], ["md1"], {"/aq": ["md1"]})
        with patch.object(config, "get_input_folders", return_value=folders) as gif, \
                patch.object(config, "_test_input_files", return_value="") as tif:
            config._test_input_data()
        return gif, tif

    def test_full_run_walks_all_trees(self):
        gif, tif = self._run(self._config((1, 10)))
        self.assertTrue(gif.called)
        self.assertTrue(tif.called)

    def test_resume_without_consuming_stage_skips_glob(self):
        # stages 5-6 consume no input tree (CAVER 1-2, trajectory 2/9, AQUA-DUCT 7)
        gif, tif = self._run(self._config((5, 6)))
        self.assertFalse(gif.called)
        self.assertFalse(tif.called)

    def test_mid_pipeline_resume_skips_glob(self):
        gif, tif = self._run(self._config((3, 4)))
        self.assertFalse(gif.called)
        self.assertFalse(tif.called)

    def test_distance_shard_skips_glob(self):
        gif, tif = self._run(self._config((4, 4), is_shard=True))
        self.assertFalse(gif.called)
        self.assertFalse(tif.called)

    def test_tunnel_network_shard_walks_caver_and_trajectory(self):
        gif, tif = self._run(self._config((2, 2), is_shard=True))
        self.assertTrue(gif.called)
        # caver (stage 1-2) and trajectory (stage 2) sweeps run; AQUA-DUCT (stage 7) does not
        tested_roots = {call.kwargs.get("root_folder", call.args[2] if len(call.args) > 2 else None)
                        for call in tif.call_args_list}
        self.assertIn("/caver", tested_roots)
        self.assertIn("/traj", tested_roots)
        self.assertNotIn("/aq", tested_roots)

    def test_aquaduct_shard_walks_only_aquaduct(self):
        gif, tif = self._run(self._config((7, 7), is_shard=True))
        self.assertTrue(gif.called)
        tested_roots = {call.kwargs.get("root_folder", call.args[2] if len(call.args) > 2 else None)
                        for call in tif.call_args_list}
        self.assertIn("/aq", tested_roots)
        self.assertNotIn("/caver", tested_roots)

    def test_resume_tolerates_missing_reference_structure(self):
        # stage-5+ rerun after the input folder was moved/renamed: the reference is not read by the
        # active stages, so its now-stale path must not abort the run
        config = self._config((5, 10), aquaduct_results_path=None,
                              pdb_reference_structure="/gone/ref.pdb")
        self._run(config)  # must not raise

    def test_alignment_stage_still_requires_reference_structure(self):
        config = self._config((1, 10), pdb_reference_structure="/gone/ref.pdb")
        with self.assertRaises(RuntimeError):
            self._run(config)


class TestValidateLazyInputFolders(unittest.TestCase):
    """The comparative-group and exact-matching coverage checks in _validate_parameter_values walk
    the input folders; get_input_folders() is only invoked when an in-range stage needs the result."""

    def _config(self, active_stages, **overrides):
        params = dict(num_cpus=2, snapshots_per_simulation=100, stop_after_stage=10,
                      start_from_stage=active_stages[0])
        params.update(overrides)
        return _populated_config(active_stages=active_stages, **params)

    def test_event_stage_in_range_triggers_glob(self):
        config = self._config((1, 10), aquaduct_results_path=["/aq"])
        with patch.object(config, "get_input_folders",
                          return_value=(["md1"], ["md1"], {"/aq": ["md1"]})) as gif:
            config._validate_parameter_values()
        self.assertTrue(gif.called)

    def test_tunnels_only_run_skips_glob(self):
        config = self._config((1, 6), aquaduct_results_path=["/aq"])
        with patch.object(config, "get_input_folders",
                          return_value=(["md1"], ["md1"], {"/aq": ["md1"]})) as gif:
            config._validate_parameter_values()
        self.assertFalse(gif.called)

    def test_no_aquaduct_no_comparative_skips_glob(self):
        config = self._config((1, 10), aquaduct_results_path=None,
                              perform_comparative_analysis=False)
        with patch.object(config, "get_input_folders",
                          return_value=([], [], {})) as gif:
            config._validate_parameter_values()
        self.assertFalse(gif.called)


class TestPreserveAutodetectedOnResume(unittest.TestCase):
    """TransportProcesses.update_configuration preserves auto-detected setup values from the previous
    run when the freshly scoped config left them unset (a resume skips re-detecting them), and lets a
    genuinely changed value win."""

    def _apply(self, old_params, new_overrides, active_stages=(1, 10),
               input_folders=([], [], {}), preset=None):
        from types import SimpleNamespace
        from transport_tools.libs.tools import TransportProcesses

        new_config = _populated_config(active_stages=active_stages)
        new_config.parameters.update(new_overrides)
        attrs = dict(parameters=dict(old_params),
                     _outlier_transport_events=SimpleNamespace(parameters={}),
                     _super_clusters={},
                     caver_input_folders=[], traj_input_folders=[],
                     aquaduct_input_folders=[], aquaduct_md_to_roots={})
        if preset:
            attrs.update(preset)
        fake_self = SimpleNamespace(**attrs)
        with patch.object(new_config, "get_input_folders", return_value=input_folders) as gif:
            TransportProcesses.update_configuration(fake_self, new_config)
        fake_self._get_input_folders_mock = gif
        return fake_self

    def test_unset_values_are_preserved(self):
        old = {"snapshots_per_simulation": 100, "pdb_reference_structure": "/ref.pdb", "num_cpus": 8}
        fake_self = self._apply(old, {"snapshots_per_simulation": None,
                                      "pdb_reference_structure": None, "num_cpus": None})
        self.assertEqual(100, fake_self.parameters["snapshots_per_simulation"])
        self.assertEqual("/ref.pdb", fake_self.parameters["pdb_reference_structure"])
        self.assertEqual(8, fake_self.parameters["num_cpus"])
        self.assertEqual("/ref.pdb", fake_self.reference_pdb_file)

    def test_changed_value_wins(self):
        old = {"snapshots_per_simulation": 100, "pdb_reference_structure": "/ref.pdb", "num_cpus": 8}
        fake_self = self._apply(old, {"snapshots_per_simulation": 200,
                                      "pdb_reference_structure": "/new.pdb", "num_cpus": None})
        self.assertEqual(200, fake_self.parameters["snapshots_per_simulation"])
        self.assertEqual("/new.pdb", fake_self.parameters["pdb_reference_structure"])
        self.assertEqual(8, fake_self.parameters["num_cpus"])  # unset => preserved

    def test_input_folders_kept_from_checkpoint_on_resume_scope(self):
        # stages 5-6 re-read no input tree: the processed MD set stays as the checkpoint recorded it,
        # even if the source folders have moved (get_input_folders is not even called)
        old = {"snapshots_per_simulation": 100, "pdb_reference_structure": "/ref.pdb", "num_cpus": 8}
        fake_self = self._apply(old, {}, active_stages=(5, 6),
                                input_folders=(["new"], ["new"], {"/aq": ["new"]}),
                                preset={"caver_input_folders": ["md1", "md2", "md3"]})
        self.assertEqual(["md1", "md2", "md3"], fake_self.caver_input_folders)
        self.assertFalse(fake_self._get_input_folders_mock.called)

    def test_input_folders_refreshed_when_input_stage_runs(self):
        # a run that includes CAVER parsing (stage 2) re-globs the CAVER folders, honouring a changed
        # caver_results_path/pattern in the config
        old = {"snapshots_per_simulation": 100, "pdb_reference_structure": "/ref.pdb", "num_cpus": 8}
        fake_self = self._apply(old, {}, active_stages=(2, 10),
                                input_folders=(["a", "b"], ["a"], {"/aq": ["a"]}),
                                preset={"caver_input_folders": ["stale"]})
        self.assertEqual(["a", "b"], fake_self.caver_input_folders)


if __name__ == '__main__':
    unittest.main()
