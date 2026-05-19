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

__version__ = '0.9.7'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import unittest
import os
import pytest
from transport_tools.libs.utils import set_paths_from_package_root, prep_test_config, compare_test_folders, compare_test_files
from transport_tools.libs.tools import load_checkpoint, define_filters, TransportProcesses, save_checkpoint

class TestTransportProcesses(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _pytest_info(self, request):
        """Capture pytest request for failure detection"""
        self._request = request

    @classmethod
    def setUpClass(cls):
        from transport_tools.libs.config import AnalysisConfig

        cls._test_failed = False # Initialize flag to track failures
        cls.maxDiff = None
        cls.results = "test_results"
        cls.root = set_paths_from_package_root("tests", "data")
        cls.out_path = set_paths_from_package_root("tests", "test_results", "TestTransportProcesses")
        os.makedirs(cls.out_path, exist_ok=True)
        prep_test_config(cls.root, cls.out_path)
        
        cls.config = AnalysisConfig(os.path.join(cls.out_path, "config.ini"), logging=False)
        print(cls.config)
        cls.config.set_parameter("output_path", cls.out_path)
        os.makedirs(os.path.join(cls.out_path, "temp"), exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        from shutil import rmtree
        # Check if any test failed and keep results
        if cls._test_failed:
            return

        # Config file is now in the output directory, so it's removed with rmtree
        rmtree(cls.out_path)

    def tearDown(self):
        """
        Called after each test method - track if any test failed
        """
        # Check if any test has failed
        # Works with both unittest and pytest
        if hasattr(self, '_outcome'):
            # For unittest (Python 3.11+)
            if hasattr(self._outcome, 'result'):  # type: ignore[attr-defined]
                result_errors = getattr(self._outcome.result, 'errors', [])  # type: ignore[attr-defined]
                result_failures = getattr(self._outcome.result, 'failures', [])  # type: ignore[attr-defined]
                if result_errors or result_failures:
                    self.__class__._test_failed = True

        # For pytest - check using pytest's request fixture if available
        # This is set by pytest when running tests
        if hasattr(self, '_request') and hasattr(self._request, 'node'):
            # Check if this test or any previous test in the session has failed
            if self._request.session.testsfailed > 0:
                self.__class__._test_failed = True

    def setUp(self):
        self.saved_data = os.path.join(TestTransportProcesses.root, "saved_outputs")
        self.out_path = TestTransportProcesses.out_path

    def _get_dumpfile(self, stage: int) -> str:
        return os.path.join(self.out_path, "temp", "mol_system_{}.dump".format(stage))

    def test_01compute_transformations(self):
        mol_system = TransportProcesses(TestTransportProcesses.config)
        mol_system.compute_transformations()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "transformations"),
                              os.path.join(self.out_path, "_internal", "transformations"), self)
        compare_test_folders(os.path.join(self.saved_data, "_internal", "transformations", "aquaduct"),
                              os.path.join(self.out_path, "_internal", "transformations", "aquaduct"), self)
        compare_test_folders(os.path.join(self.saved_data, "_internal", "transformations", "caver"),
                              os.path.join(self.out_path, "_internal", "transformations", "caver"), self)
        save_checkpoint(mol_system, self._get_dumpfile(1), overwrite=True)

    def test_02process_tunnel_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(1))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.process_tunnel_networks()
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"), self)
        save_checkpoint(mol_system, self._get_dumpfile(2), overwrite=True)

    def test_03create_layered_description4tunnel_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(2))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.create_layered_description4tunnel_networks()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "layered_data", "caver"),
                              os.path.join(self.out_path, "_internal", "layered_data", "caver"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1",
                                           "nodes"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1",
                                           "nodes"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1",
                                           "paths"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1",
                                           "paths"), self)
        save_checkpoint(mol_system, self._get_dumpfile(3), overwrite=True)

    def test_04distance_shard_consistency(self):
        """SLURM-shard distance calculation must reproduce the local distance matrix exactly"""
        import numpy as np

        try:
            mol_system = load_checkpoint(self._get_dumpfile(3))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        from scipy.spatial.distance import squareform

        cluster_specifications, path_sets, _ = mol_system._assemble_cluster_path_sets(load_path_sets=True)
        local_condensed = mol_system._compute_intercluster_distances(cluster_specifications, path_sets)

        # rebuild the matrix from sharded calculations, exactly as the SLURM backend would
        num_shards = 3
        num_clusters = len(cluster_specifications)
        sharded_matrix = np.full((num_clusters, num_clusters), np.inf)
        for shard_id in range(num_shards):
            for id1, id2, distance in mol_system.compute_distance_shard(num_shards, shard_id):
                sharded_matrix[int(id1), int(id2)] = distance
        for cls1 in range(num_clusters):
            sharded_matrix[cls1, cls1] = 0
            for cls2 in range(cls1 + 1, num_clusters):
                sharded_matrix[cls2, cls1] = sharded_matrix[cls1, cls2]

        self.assertFalse(np.isinf(sharded_matrix).any())
        self.assertTrue(np.array_equal(local_condensed, squareform(sharded_matrix, checks=False)))

    def test_04merge_tunnel_clusters2super_clusters(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(3))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.compute_tunnel_clusters_distances()
        mol_system.merge_tunnel_clusters2super_clusters()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "clustering"),
                              os.path.join(self.out_path, "_internal", "clustering"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "clustering"),
                              os.path.join(self.out_path, "data", "clustering"), self)
        save_checkpoint(mol_system, self._get_dumpfile(4), overwrite=True)

    def test_04merge_tunnel_clusters2super_clusters_slurm(self):
        """Same checks as test_04merge..., but the stage-4 distances are computed via the SLURM
        backend (submitit). Runs only when a SLURM environment with submitit is detected."""
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(3))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        # the SLURM shards rebuild the analysis from the config file, so it must carry an absolute
        # output_path matching this test run (the shared test config uses a relative one) and the
        # same slurm_folder as the launcher - kept outside the clustering folder compared below
        slurm_folder = os.path.join(self.out_path, "temp", "slurm_shards")
        slurm_config = os.path.join(self.out_path, "config_slurm.ini")
        with open(os.path.join(self.out_path, "config.ini")) as in_stream, \
                open(slurm_config, "w") as out_stream:
            for line in in_stream:
                if line.startswith("output_path"):
                    line = "output_path = {}\n".format(self.out_path)
                out_stream.write(line)
                if line.strip() == "[CALCULATIONS_SETTINGS]":
                    out_stream.write("slurm_folder = {}\n".format(slurm_folder))
        mol_system.config_file = slurm_config

        mol_system.parameters["distance_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 3
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_folder"] = slurm_folder

        mol_system.compute_tunnel_clusters_distances()
        mol_system.merge_tunnel_clusters2super_clusters()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "clustering"),
                              os.path.join(self.out_path, "_internal", "clustering"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "clustering"),
                              os.path.join(self.out_path, "data", "clustering"), self)

    def test_05create_super_cluster_profiles(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(4))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.create_super_cluster_profiles()
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "initial_super_cluster_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "initial_super_cluster_details.txt"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "initial"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "initial"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "initial"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "initial"), self)
        save_checkpoint(mol_system, self._get_dumpfile(5), overwrite=True)

    def test_06generate_super_cluster_summary(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(5))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.generate_super_cluster_summary(out_filename="1-initial_tunnels_summary.txt")
        compare_test_files(os.path.join(self.saved_data, "statistics", "1-initial_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "1-initial_tunnels_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "statistics",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"), self)

        save_checkpoint(mol_system, self._get_dumpfile(6), overwrite=True)

    def test_07save_super_clusters_visualization(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(6))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.save_super_clusters_visualization(script_name="visualize_tunnels.py")
        compare_test_files(os.path.join(self.saved_data, "visualization", "visualize_tunnels.py"),
                            os.path.join(self.out_path, "visualization", "visualize_tunnels.py"), self)
        compare_test_files(os.path.join(self.saved_data, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels.py"),
                            os.path.join(self.out_path, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels.py"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "super_cluster_CGOs"),
                              os.path.join(self.out_path, "visualization", "sources", "super_cluster_CGOs"), self,
                              pattern=r'.+pathset_1\.dump')
        save_checkpoint(mol_system, self._get_dumpfile(7), overwrite=True)

    def test_08filter_super_cluster_profiles(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(7))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.filter_super_cluster_profiles(min_length=5, min_bottleneck_radius=1, min_avg_snapshots_num=-1,
                                                 min_sims_num=-1)
        mol_system.generate_super_cluster_summary(out_filename="2-filtered_tunnels_summary.txt")
        mol_system.save_super_clusters_visualization(script_name="visualize_tunnels_filtered.py")
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details1.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details1.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels_filtered.py"),
                            os.path.join(self.out_path, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels_filtered.py"), self)
        compare_test_files(os.path.join(self.saved_data, "visualization", "visualize_tunnels_filtered.py"),
                            os.path.join(self.out_path, "visualization", "visualize_tunnels_filtered.py"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "super_cluster_CGOs"),
                              os.path.join(self.out_path, "visualization", "sources", "super_cluster_CGOs"), self,
                              pattern=r'.+pathset_2\.dump')

        compare_test_files(os.path.join(self.saved_data, "statistics", "2-filtered_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "2-filtered_tunnels_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "statistics",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"), self)

        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "filtered01"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "filtered01"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "filtered01"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "filtered01"), self)
        save_checkpoint(mol_system, self._get_dumpfile(8), overwrite=True)

    def test_09process_aquaduct_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(8))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.process_aquaduct_networks()
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"), self)

        save_checkpoint(mol_system, self._get_dumpfile(9), overwrite=True)

    def test_10create_layered_description4aquaduct_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(9))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.create_layered_description4aquaduct_networks()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "layered_data", "aquaduct"),
                              os.path.join(self.out_path, "_internal", "layered_data", "aquaduct"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "nodes"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1",
                                           "nodes"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "paths"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1",
                                           "paths"), self)

        save_checkpoint(mol_system, self._get_dumpfile(10), overwrite=True)

    def test_11assign_transport_events(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(10))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.assign_transport_events()
        mol_system.save_super_clusters_visualization(script_name="visualize_events.py")
        mol_system.generate_super_cluster_summary(out_filename="3-initial_events_summary.txt")
        mol_system.filter_super_cluster_profiles(min_length=5, min_avg_snapshots_num=-1, min_sims_num=1,
                                                 min_total_events=1)
        mol_system.save_super_clusters_visualization(script_name="visualize_events_filtered.py")
        mol_system.generate_super_cluster_summary(out_filename="4-filtered_events_summary.txt")

        compare_test_files(os.path.join(self.saved_data, "visualization", "visualize_events.py"),
                            os.path.join(self.out_path, "visualization", "visualize_events.py"),self)
        compare_test_files(os.path.join(self.saved_data, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events.py"),
                            os.path.join(self.out_path, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events.py"), self)
        compare_test_files(os.path.join(self.saved_data, "visualization", "visualize_events_filtered.py"),
                            os.path.join(self.out_path, "visualization", "visualize_events_filtered.py"),self)
        compare_test_files(os.path.join(self.saved_data, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events_filtered.py"),
                            os.path.join(self.out_path, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events_filtered.py"), self)

        compare_test_files(os.path.join(self.saved_data, "statistics", "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "3-initial_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "3-initial_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "4-filtered_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "statistics",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "3-initial_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "3-initial_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "4-filtered_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary_bottleneck_residues.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "initial_super_cluster_events_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "initial_super_cluster_events_details.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "outlier_transport_events_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "outlier_transport_events_details.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details2.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details2.txt"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "exact_matching_analysis", "md1"),
                              os.path.join(self.out_path, "data", "exact_matching_analysis", "md1"), self)

        compare_test_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_entry_sc2"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_entry_sc2"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc3"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc3"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc2"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc2"), self)

        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "filtered02"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "filtered02"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "filtered02"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "filtered02"), self)

        save_checkpoint(mol_system, self._get_dumpfile(11), overwrite=True)

    def test_12custom_analyses(self):
        from numpy import save

        try:
            mol_system = load_checkpoint(self._get_dumpfile(11))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        super_cluster_id = 1
        filters = define_filters(min_length=10, min_bottleneck_radius=1.4)

        visualization_output_folder = os.path.join(self.out_path, "customs", "static")
        mol_system.show_tunnels_passing_filter(super_cluster_id, filters, visualization_output_folder,
                                               start_snapshot=1, end_snapshot=100, trajectory=False)
        compare_test_folders(os.path.join(self.saved_data, "customs", "static"),
                              os.path.join(self.out_path, "customs", "static"), self)

        visualization_output_folder = os.path.join(self.out_path, "customs", "dynamics")
        md_labels = ["md1"]
        mol_system.show_tunnels_passing_filter(super_cluster_id, filters, visualization_output_folder,
                                               md_labels=md_labels, start_snapshot=1, end_snapshot=100, trajectory=True)
        compare_test_folders(os.path.join(self.saved_data, "customs", "dynamics"),
                              os.path.join(self.out_path, "customs", "dynamics"), self)

        filters = define_filters(min_length=10)
        super_cluster_id = None
        parameter = "bottleneck_radius"
        dataset = mol_system.get_property_time_evolution_data(parameter, active_filters=filters, sc_id=super_cluster_id)
        self.assertListEqual([28, 14, 12, 10, 27, 13, 26, 8, 25, 24, 3, 4, 18, 23, 22, 21, 2, 20, 6, 19, 9, 17, 16, 15, 7, 11, 1, 5],
                              [*dataset.keys()])
        save(os.path.join(self.out_path, "customs", "time_evolution_dataset.npy"), dataset[1]["md1"])
        compare_test_files(os.path.join(self.saved_data, "customs", "time_evolution_dataset.npy"),
                            os.path.join(self.out_path, "customs", "time_evolution_dataset.npy"), self)


if __name__ == "__main__":
    unittest.main()
