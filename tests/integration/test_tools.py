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
import os


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


class TestTransportProcesses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transport_tools.libs.config import AnalysisConfig
        cls.maxDiff = None
        cls.results = "test_results"
        cls.root = set_paths("tests", "data")
        prep_config(cls.root)
        cls.config = AnalysisConfig(os.path.join(cls.root, "config.ini"), logging=False)
        print(cls.config)
        cls.config.set_parameter("output_path", set_paths("tests", "test_results", "TestTransportProcesses"))
        cls.out_path = cls.config.get_parameter("output_path")
        os.makedirs(os.path.join(cls.out_path, "temp"), exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        from shutil import rmtree

        os.remove(os.path.join(cls.root, "config.ini"))
        rmtree(cls.out_path)

    def setUp(self):
        self.saved_data = os.path.join(TestTransportProcesses.root, "saved_outputs")
        self.out_path = TestTransportProcesses.out_path

    def _get_dumpfile(self, stage: int) -> str:
        return os.path.join(self.out_path, "temp", "mol_system_{}.dump".format(stage))

    def _compare_files(self, out_file: str, res_file: str,):
        import pickle
        import gzip
        import numpy as np
        from sys import maxsize
        np.set_printoptions(threshold=maxsize)

        res_lines = out_lines = None
        out_mat = res_mat = None
        if res_file.endswith(".dump.gz"):
            with gzip.open(res_file, 'rb') as res_in, gzip.open(out_file, 'rb') as out_in:
                res_lines = pickle.load(res_in)
                out_lines = pickle.load(out_in)
        elif ".dump" in res_file:
            with open(res_file, "rb") as res_in, open(out_file, "rb") as out_in:
                res_lines = pickle.load(res_in)
                out_lines = pickle.load(out_in)
        elif ".gz" in res_file:
            with gzip.open(res_file, 'r') as res_in, gzip.open(out_file, 'r') as out_in:
                res_lines = res_in.readlines()
                out_lines = out_in.readlines()
        elif ".npy" in res_file:
            res_mat = np.load(res_file)
            out_mat = np.load(out_file)
        else:
            with open(res_file, "r") as res_in, open(out_file, "r") as out_in:
                res_lines = res_in.readlines()
                out_lines = out_in.readlines()

        if res_lines is not None:
            if isinstance(res_lines, np.ndarray):
                self.assertTrue(np.allclose(out_lines, res_lines, atol=1e-7),
                                msg="In files '{}' and '{}':".format(out_file, res_file))
            else:
                self.assertTrue(len(res_lines) == len(out_lines),
                                msg="Different length of files '{}' and '{}':".format(out_file, res_file))
                for res_line, out_line in zip(res_lines, out_lines):
                    try:
                        if ".pdb" in res_file and "REMARK   1 CREATED WITH MDTraj" in res_line:
                            continue
                    except TypeError:
                        if ".pdb" in res_file and b"REMARK   1 CREATED WITH MDTraj" in res_line:
                            continue

                    if isinstance(res_line, list) or isinstance(res_line, tuple):
                        self.assertTrue(len(res_line) == len(out_line),
                                        msg="Different length of lists {} and {}\n "
                                            "in files '{}' and '{}':".format(res_line, out_line, out_file, res_file))
                        for res_item, out_item in zip(res_line, out_line):
                            try:
                                self.assertAlmostEqual(float(out_item), float(res_item),
                                                       msg="In files '{}' and '{}':".format(out_file, res_file))
                            except (ValueError, TypeError):
                                self.assertEqual(out_item, res_item, msg="In files '{}' and '{}':".format(out_file,
                                                                                                          res_file))

                    elif ".csv" in res_file:
                        for out_item, res_item in zip(out_line.split(","), res_line.split(",")):
                            try:
                                self.assertAlmostEqual(float(out_item), float(res_item))
                            except ValueError:
                                self.assertEqual(out_item, res_item)
                    else:
                        try:
                            self.assertAlmostEqual(float(out_line), float(res_line),
                                                   msg="In files '{}' and '{}':".format(out_file, res_file))
                        except (ValueError, TypeError):
                            self.assertEqual(out_line, res_line, msg="In files '{}' and '{}':".format(out_file,
                                                                                                      res_file))

        else:
            self.assertTrue(np.allclose(out_mat, res_mat, atol=1e-7),
                            msg="In files '{}' and '{}':".format(out_file, res_file))

    def _compare_folders(self, saved_outputs_dir: str, results_dir: str, pattern: str = ".+"):
        from re import search
        out_files = list()
        results_files = list()

        for _file in sorted(os.listdir(results_dir)):
            if not search(pattern, _file):
                continue
            results_files.append(_file)

        for _file in sorted(os.listdir(saved_outputs_dir)):
            if not search(pattern, _file):
                continue
            out_files.append(_file)

        self.assertEqual(out_files, results_files, msg="In folders '{}' and '{}':".format(saved_outputs_dir,
                                                                                          results_dir))

        for res_file, out_file in zip(results_files, out_files):
            res_file = os.path.join(results_dir, res_file)
            out_file = os.path.join(saved_outputs_dir, out_file)
            if os.path.isfile(res_file) and os.path.isfile(out_file):
                self._compare_files(out_file, res_file)

    def test_01compute_transformations(self):
        from transport_tools.libs.tools import save_checkpoint, TransportProcesses
        mol_system = TransportProcesses(TestTransportProcesses.config)
        mol_system.compute_transformations()
        self._compare_folders(os.path.join(self.saved_data, "_internal", "transformations"),
                              os.path.join(self.out_path, "_internal", "transformations"))
        self._compare_folders(os.path.join(self.saved_data, "_internal", "transformations", "aquaduct"),
                              os.path.join(self.out_path, "_internal", "transformations", "aquaduct"))
        self._compare_folders(os.path.join(self.saved_data, "_internal", "transformations", "caver"),
                              os.path.join(self.out_path, "_internal", "transformations", "caver"))
        save_checkpoint(mol_system, self._get_dumpfile(1), overwrite=True)

    def test_02process_tunnel_networks(self):
        from transport_tools.libs.tools import load_checkpoint, save_checkpoint

        mol_system = load_checkpoint(self._get_dumpfile(1))
        mol_system.process_tunnel_networks()
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"))
        save_checkpoint(mol_system, self._get_dumpfile(2), overwrite=True)

    def test_03create_layered_description4tunnel_networks(self):
        from transport_tools.libs.tools import load_checkpoint, save_checkpoint

        mol_system = load_checkpoint(self._get_dumpfile(2))
        mol_system.create_layered_description4tunnel_networks()
        self._compare_folders(os.path.join(self.saved_data, "_internal", "layered_data", "caver"),
                              os.path.join(self.out_path, "_internal", "layered_data", "caver"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1",
                                           "nodes"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1",
                                           "nodes"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1",
                                           "paths"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1",
                                           "paths"))
        save_checkpoint(mol_system, self._get_dumpfile(3), overwrite=True)

    def test_04merge_tunnel_clusters2super_clusters(self):
        from transport_tools.libs.tools import load_checkpoint, save_checkpoint

        mol_system = load_checkpoint(self._get_dumpfile(3))
        mol_system.compute_tunnel_clusters_distances()
        mol_system.merge_tunnel_clusters2super_clusters()
        self._compare_folders(os.path.join(self.saved_data, "_internal", "clustering"),
                              os.path.join(self.out_path, "_internal", "clustering"))
        self._compare_folders(os.path.join(self.saved_data, "data", "clustering"),
                              os.path.join(self.out_path, "data", "clustering"))
        save_checkpoint(mol_system, self._get_dumpfile(4), overwrite=True)

    def test_05create_super_cluster_profiles(self):
        from transport_tools.libs.tools import load_checkpoint, save_checkpoint

        mol_system = load_checkpoint(self._get_dumpfile(4))
        mol_system.create_super_cluster_profiles()
        self._compare_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "initial_super_cluster_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "initial_super_cluster_details.txt"))
        self._compare_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "initial"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "initial"))
        self._compare_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "initial"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "initial"))
        save_checkpoint(mol_system, self._get_dumpfile(5), overwrite=True)

    def test_06generate_super_cluster_summary(self):
        from transport_tools.libs.tools import load_checkpoint, save_checkpoint

        mol_system = load_checkpoint(self._get_dumpfile(5))
        mol_system.generate_super_cluster_summary(out_filename="1-initial_tunnels_summary.txt")
        self._compare_files(os.path.join(self.saved_data, "statistics", "1-initial_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "1-initial_tunnels_summary.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary.txt"))

        self._compare_files(os.path.join(self.saved_data, "statistics",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"))

        save_checkpoint(mol_system, self._get_dumpfile(6), overwrite=True)

    def test_07save_super_clusters_visualization(self):
        from transport_tools.libs.tools import load_checkpoint, save_checkpoint

        mol_system = load_checkpoint(self._get_dumpfile(6))
        mol_system.save_super_clusters_visualization(script_name="visualize_tunnels.py")
        self._compare_files(os.path.join(self.saved_data, "visualization", "visualize_tunnels.py"),
                            os.path.join(self.out_path, "visualization", "visualize_tunnels.py"))
        self._compare_files(os.path.join(self.saved_data, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels.py"),
                            os.path.join(self.out_path, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels.py"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "super_cluster_CGOs"),
                              os.path.join(self.out_path, "visualization", "sources", "super_cluster_CGOs"),
                              r'.+pathset_1\.dump')
        save_checkpoint(mol_system, self._get_dumpfile(7), overwrite=True)

    def test_08filter_super_cluster_profiles(self):
        from transport_tools.libs.tools import load_checkpoint, save_checkpoint

        mol_system = load_checkpoint(self._get_dumpfile(7))
        mol_system.filter_super_cluster_profiles(min_length=5, min_bottleneck_radius=1, min_avg_snapshots_num=-1,
                                                 min_sims_num=-1)
        mol_system.generate_super_cluster_summary(out_filename="2-filtered_tunnels_summary.txt")
        mol_system.save_super_clusters_visualization(script_name="visualize_tunnels_filtered.py")
        self._compare_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details1.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details1.txt"))

        self._compare_files(os.path.join(self.saved_data, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels_filtered.py"),
                            os.path.join(self.out_path, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels_filtered.py"))
        self._compare_files(os.path.join(self.saved_data, "visualization", "visualize_tunnels_filtered.py"),
                            os.path.join(self.out_path, "visualization", "visualize_tunnels_filtered.py"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "super_cluster_CGOs"),
                              os.path.join(self.out_path, "visualization", "sources", "super_cluster_CGOs"),
                              r'.+pathset_2\.dump')

        self._compare_files(os.path.join(self.saved_data, "statistics", "2-filtered_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "2-filtered_tunnels_summary.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary.txt"))

        self._compare_files(os.path.join(self.saved_data, "statistics",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"))

        self._compare_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "filtered01"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "filtered01"))
        self._compare_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "filtered01"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "filtered01"))
        save_checkpoint(mol_system, self._get_dumpfile(8), overwrite=True)

    def test_09process_aquaduct_networks(self):
        from transport_tools.libs.tools import load_checkpoint, save_checkpoint

        mol_system = load_checkpoint(self._get_dumpfile(8))
        mol_system.process_aquaduct_networks()
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"))

        save_checkpoint(mol_system, self._get_dumpfile(9), overwrite=True)

    def test_10create_layered_description4aquaduct_networks(self):
        from transport_tools.libs.tools import load_checkpoint, save_checkpoint

        mol_system = load_checkpoint(self._get_dumpfile(9))
        mol_system.create_layered_description4aquaduct_networks()
        self._compare_folders(os.path.join(self.saved_data, "_internal", "layered_data", "aquaduct"),
                              os.path.join(self.out_path, "_internal", "layered_data", "aquaduct"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "nodes"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1",
                                           "nodes"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "paths"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1",
                                           "paths"))

        save_checkpoint(mol_system, self._get_dumpfile(10), overwrite=True)

    def test_11assign_transport_events(self):
        from transport_tools.libs.tools import load_checkpoint, save_checkpoint

        mol_system = load_checkpoint(self._get_dumpfile(10))
        mol_system.assign_transport_events()
        mol_system.save_super_clusters_visualization(script_name="visualize_events.py")
        mol_system.generate_super_cluster_summary(out_filename="3-initial_events_summary.txt")
        mol_system.filter_super_cluster_profiles(min_length=5, min_avg_snapshots_num=-1, min_sims_num=1,
                                                 min_total_events=1)
        mol_system.save_super_clusters_visualization(script_name="visualize_events_filtered.py")
        mol_system.generate_super_cluster_summary(out_filename="4-filtered_events_summary.txt")
        save_checkpoint(mol_system, self._get_dumpfile(11), overwrite=True)

        self._compare_files(os.path.join(self.saved_data, "visualization", "visualize_events.py"),
                            os.path.join(self.out_path, "visualization", "visualize_events.py"))
        self._compare_files(os.path.join(self.saved_data, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events.py"),
                            os.path.join(self.out_path, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events.py"))
        self._compare_files(os.path.join(self.saved_data, "visualization", "visualize_events_filtered.py"),
                            os.path.join(self.out_path, "visualization", "visualize_events_filtered.py"))
        self._compare_files(os.path.join(self.saved_data, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events_filtered.py"),
                            os.path.join(self.out_path, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events_filtered.py"))

        self._compare_files(os.path.join(self.saved_data, "statistics", "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "3-initial_events_summary.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "3-initial_events_summary.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "4-filtered_events_summary.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary.txt"))

        self._compare_files(os.path.join(self.saved_data, "statistics",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "3-initial_events_summary_bottleneck_residues.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "3-initial_events_summary_bottleneck_residues.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary_bottleneck_residues.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "4-filtered_events_summary_bottleneck_residues.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary_bottleneck_residues.txt"))
        self._compare_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary_bottleneck_residues.txt"))

        self._compare_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "initial_super_cluster_events_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "initial_super_cluster_events_details.txt"))
        self._compare_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "outlier_transport_events_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "outlier_transport_events_details.txt"))
        self._compare_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details2.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details2.txt"))
        self._compare_folders(os.path.join(self.saved_data, "data", "exact_matching_analysis", "md1"),
                              os.path.join(self.out_path, "data", "exact_matching_analysis", "md1"))

        self._compare_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "1_entry_sc2"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "1_entry_sc2"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "1_release_sc3"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "1_release_sc3"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "1_release_sc2"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "1_release_sc2"))

        self._compare_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "filtered02"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "filtered02"))
        self._compare_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "filtered02"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "filtered02"))

    def test_12custom_analyses(self):
        from transport_tools.libs.tools import load_checkpoint, define_filters
        from numpy import save

        mol_system = load_checkpoint(self._get_dumpfile(11))
        super_cluster_id = 1
        filters = define_filters(min_length=10, min_bottleneck_radius=1.4)

        visualization_output_folder = os.path.join(self.out_path, "customs", "static")
        mol_system.show_tunnels_passing_filter(super_cluster_id, filters, visualization_output_folder,
                                               start_snapshot=1, end_snapshot=100, trajectory=False)
        self._compare_folders(os.path.join(self.saved_data, "customs", "static"),
                              os.path.join(self.out_path, "customs", "static"))

        visualization_output_folder = os.path.join(self.out_path, "customs", "dynamics")
        md_labels = ["md1"]
        mol_system.show_tunnels_passing_filter(super_cluster_id, filters, visualization_output_folder,
                                               md_labels=md_labels, start_snapshot=1, end_snapshot=100, trajectory=True)
        self._compare_folders(os.path.join(self.saved_data, "customs", "dynamics"),
                              os.path.join(self.out_path, "customs", "dynamics"))

        filters = define_filters(min_length=10)
        super_cluster_id = None
        parameter = "bottleneck_radius"
        dataset = mol_system.get_property_time_evolution_data(parameter, active_filters=filters, sc_id=super_cluster_id)
        self.assertListEqual([27, 13, 11, 10, 26, 12, 25, 9, 24, 23, 3, 4, 17, 22, 21, 20, 2, 19, 6, 18, 8, 16, 15, 14,
                              7, 1, 5], [*dataset.keys()])
        save(os.path.join(self.out_path, "customs", "time_evolution_dataset.npy"), dataset[1]["md1"])
        self._compare_files(os.path.join(self.saved_data, "customs", "time_evolution_dataset.npy"),
                            os.path.join(self.out_path, "customs", "time_evolution_dataset.npy"))


if __name__ == "__main__":
    unittest.main()
