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

__version__ = '0.9.8'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import unittest
import tempfile
import shutil
from unittest.mock import patch
from transport_tools.libs.config import *
from transport_tools.libs.utils import set_paths_from_package_root, prep_test_config

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        from transport_tools.libs.config import AnalysisConfig
        self.root = set_paths_from_package_root("tests", "data")
        self.out_path = set_paths_from_package_root("tests", "test_results", "TestConfig")
        os.makedirs(self.out_path, exist_ok=True)
        prep_test_config(self.root, self.out_path)
        self.config = AnalysisConfig(os.path.join(self.out_path, "config.ini"), logging=False)

    def tearDown(self):
        from shutil import rmtree
        rmtree(self.out_path)


    def test__get_filenames_by_pattern(self):

        results = self.config._get_caver_filenames()
        self.assertEqual("stripped_system.000501.pdb", results)

    def test_get_filters(self):
        results = self.config.get_filters()
        self.assertTupleEqual(tuple([-1, -1, -1, -1, -1, -1, -1, -1, -1.0, -1, -1, -1]), results)

    def test_get_input_folders(self):
        caver_folders, traj_folders, aquaduct_folders = self.config.get_input_folders()
        aquaduct_root = self.config.parameters["aquaduct_results_path"][0]
        self.assertEqual(['md1'], caver_folders)
        self.assertEqual(['md1'], traj_folders)
        self.assertEqual({aquaduct_root: ['md1']}, aquaduct_folders)

    def test_get_reference_pdb_file(self):
        results = self.config.get_reference_pdb_file()
        self.assertEqual(os.path.join(self.root, "simulations", "md1", "caver", "data", "stripped_system.000501.pdb"),
                         results)

    def test_set_parameter(self):
        saved_params = {'internal_folder': 'new_path/_internal',
                        'data_folder': 'new_path/data',
                        'visualization_folder': 'new_path/visualization',
                        'statistics_folder': 'new_path/statistics',
                        'checkpoints_folder': 'new_path/_internal/checkpoints',
                        'logfile_path': 'new_path/transport_tools.log',
                        'transformation_folder': 'new_path/_internal/transformations',
                        'clustering_folder': 'new_path/_internal/clustering',
                        'orig_caver_network_data_path': 'new_path/_internal/network_data/caver',
                        'orig_aquaduct_network_data_path': 'new_path/_internal/network_data/aquaduct',
                        'layered_caver_network_data_path': 'new_path/_internal/layered_data/caver',
                        'layered_aquaduct_network_data_path': 'new_path/_internal/layered_data/aquaduct',
                        'super_cluster_profiles_folder': 'new_path/_internal/super_cluster_initial_profiles',
                        'super_cluster_path_set_folder': 'new_path/_internal/super_cluster_pathsets',
                        'distance_matrix_csv_file': 'new_path/data/clustering/tunnel_clusters_distance_matrix.csv',
                        'super_cluster_csv_folder': 'new_path/data/super_clusters/CSV_profiles',
                        'super_cluster_details_folder': 'new_path/data/super_clusters/details',
                        'super_cluster_bottleneck_folder': 'new_path/data/super_clusters/bottlenecks',
                        'exact_matching_details_folder': 'new_path/data/exact_matching_analysis',
                        'orig_caver_vis_path': 'new_path/visualization/sources/network_data/caver',
                        'orig_aquaduct_vis_path': 'new_path/visualization/sources/network_data/aquaduct',
                        'layered_caver_vis_path': 'new_path/visualization/sources/layered_data/caver',
                        'layered_aquaduct_vis_path': 'new_path/visualization/sources/layered_data/aquaduct',
                        'super_cluster_vis_path': 'new_path/visualization/sources/super_cluster_CGOs',
                        'exact_matching_vis_path': 'new_path/visualization/exact_matching_analysis',
        }

        self.config.set_parameter("output_path", "new_path")
        results = self.config.get_parameters()

        for folder, path in saved_params.items():
            self.assertTrue(path, results[folder])


class TestAquaductAllowEmptyFoldersIntegration(unittest.TestCase):
    """
    Integration test for the aquaduct_allow_empty_folders parameter.
    Builds a substitute AQUA-DUCT root with symlinks to populated mixed_events folders
    plus a real empty folder (matching the e* pattern), then exercises AnalysisConfig
    end-to-end:
    - allow_empty=False: AnalysisConfig.__init__ must raise FileNotFoundError
    - allow_empty=True: __init__ succeeds; empty folder pruned; warning logged once
    """

    def setUp(self):
        from transport_tools.libs.config import AnalysisConfig
        self.maxDiff = None
        self.AnalysisConfig = AnalysisConfig
        self.root = set_paths_from_package_root("tests", "data")
        self.sims_orig = os.path.join(self.root, "simulations", "mixed_events")
        self.tmp = tempfile.mkdtemp(prefix="TT_allow_empty_int_")

        # Build a substitute aquaduct "water" root with symlinks to real populated
        # folders + a real empty folder
        self.aquaduct_water_dir = os.path.join(self.tmp, "water")
        os.makedirs(self.aquaduct_water_dir)
        for md in ("e10s0_seed43_f371m1821", "e10s14_e4s14f222m327_f1997m1236"):
            os.symlink(os.path.join(self.sims_orig, "water", md),
                       os.path.join(self.aquaduct_water_dir, md))
        os.makedirs(os.path.join(self.aquaduct_water_dir, "e10s99_empty"))

    def tearDown(self):
        if os.path.exists(self.tmp):
            shutil.rmtree(self.tmp)

    def _write_config(self, allow_empty: bool) -> str:
        """Write a tmp_config2.ini variant with overridden paths and the
        aquaduct_allow_empty_folders flag injected into the CALCULATIONS_SETTINGS
        section. Returns the path to the resulting config file."""
        in_config_file = os.path.join(self.root, "tmp_config2.ini")
        out_config_file = os.path.join(self.tmp, "config_{}.ini".format(allow_empty))

        caver_path = os.path.join(self.sims_orig, "caver")
        aqua_paths = "{},{}".format(self.aquaduct_water_dir,
                                     os.path.join(self.sims_orig, "ligand"))
        out_path = os.path.join(self.tmp, "out_{}".format(allow_empty))

        with open(in_config_file) as in_stream, open(out_config_file, "w") as out_stream:
            for line in in_stream:
                stripped = line.lstrip()
                if stripped.startswith("caver_results_path"):
                    line = "caver_results_path = {}\n".format(caver_path)
                elif stripped.startswith("aquaduct_results_path"):
                    line = "aquaduct_results_path = {}\n".format(aqua_paths)
                elif stripped.startswith("output_path"):
                    line = "output_path = {}\n".format(out_path)
                elif stripped.startswith("aquaduct_traced_residues_filter"):
                    # inject the allow_empty flag just after this existing setting
                    out_stream.write(line)
                    out_stream.write("aquaduct_allow_empty_folders = {}\n".format(allow_empty))
                    continue
                out_stream.write(line)
        return out_config_file

    def test_allow_empty_false_raises_filenotfound(self):
        """Default (allow_empty=False) must reject configurations that contain an
        AQUA-DUCT folder missing the required tar/summary files."""
        config_path = self._write_config(False)
        with self.assertRaises(FileNotFoundError):
            self.AnalysisConfig(config_path, logging=False)

    def test_allow_empty_true_prunes_empty_folder(self):
        """allow_empty=True succeeds and the empty folder is absent from the result of
        get_input_folders()."""
        config_path = self._write_config(True)
        config = self.AnalysisConfig(config_path, logging=False)
        _, _, aquaduct = config.get_input_folders()
        # Empty folder must not be present under any aquaduct root
        for root_path, md_labels in aquaduct.items():
            self.assertNotIn("e10s99_empty", md_labels,
                             "expected e10s99_empty to be pruned, found in {}".format(root_path))
        # The populated folders must still be present
        all_mds = [md for mds in aquaduct.values() for md in mds]
        self.assertIn("e10s0_seed43_f371m1821", all_mds)
        self.assertIn("e10s14_e4s14f222m327_f1997m1236", all_mds)

    def test_allow_empty_true_logs_warning_only_once(self):
        """The empty-folder warning must fire exactly once across init + any number of
        explicit get_input_folders() calls, thanks to _logged_empty_aquaduct_folders."""
        config_path = self._write_config(True)
        # Patch BEFORE init so we capture the warning emitted during _test_input_data
        with patch("transport_tools.libs.config.logger") as mock_logger:
            config = self.AnalysisConfig(config_path, logging=False)
            # __init__ already triggered one get_input_folders() call inside
            # _test_input_data; subsequent explicit calls must not re-warn
            config.get_input_folders()
            config.get_input_folders()
            empty_warnings = [c for c in mock_logger.warning.call_args_list
                              if "e10s99_empty" in str(c)]
            self.assertEqual(1, len(empty_warnings),
                             "expected exactly one warning for the empty folder; got {}".format(empty_warnings))


if __name__ == '__main__':
    unittest.main()
