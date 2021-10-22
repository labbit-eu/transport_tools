# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2021  Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej
# Surpeta, Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal  <janbre@amu.edu.pl>
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

__version__ = '0.9.0'
__author__ = 'Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta, ' \
             'Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal'
__mail__ = 'janbre@amu.edu.pl'

import unittest
from transport_tools.libs.config import *


def set_paths(*args):
    from transport_tools.libs.utils import splitall
    cwd = os.getcwd()
    all_parts = splitall(cwd)
    if "transport_tools" not in all_parts:
        raise RuntimeError("Must be executed from the 'transport_tools' folder")
    root_index = all_parts.index("transport_tools")
    root = os.path.join(*all_parts[:root_index + 1], *args)

    return root


def prep_config(root: str):
    in_config_file = os.path.join(root, "tmp_config.ini")
    out_config_file = os.path.join(root, "config.ini")
    update_parameters = ["caver_results_path", "aquaduct_results_path", "trajectory_path"]
    with open(in_config_file) as in_stream, open(out_config_file, "w") as out_stream:
        for line in in_stream.readlines():
            for param in update_parameters:
                if param in line:
                    line = "{} = {}\n".format(param, os.path.join(root, "simulations"))
            out_stream.write(line)


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        from transport_tools.libs.config import AnalysisConfig
        self.results = "test_results"
        self.root = set_paths("tests", "data")
        prep_config(self.root)
        self.config = AnalysisConfig(os.path.join(self.root, "config.ini"), logging=False)
        self.saved_data = os.path.join(self.root, "saved_outputs")

    def test__get_filenames_by_pattern(self):

        results = self.config._get_caver_filenames()
        self.assertEqual("stripped_system.000501.pdb", results)

    def test_get_filters(self):
        results = self.config.get_filters()
        self.assertListEqual([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], results)

    def test_get_input_folders(self):
        results = self.config.get_input_folders()
        self.assertSequenceEqual((['md1'], ['md1'], ['md1']), results)

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



if __name__ == '__main__':
    unittest.main()
