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
import os
import pytest
from transport_tools.libs.utils import (set_paths_from_package_root, prep_test_config,
                                         compare_test_folders, compare_test_files)
from transport_tools.libs.tools import load_checkpoint, TransportProcesses, save_checkpoint


class TestTransportProcessesMixedEvents(unittest.TestCase):
    """
    End-to-end pipeline test on the mixed_events fixture exercising:
    - multi-path AQUA-DUCT inputs (water + ligand)
    - per-residue event tracking (WAT + U01)
    - per-residue summary columns and PyMOL object names
    - comparative analysis (group1: e10s0_*, group2: e10s14_*)
    """

    @pytest.fixture(autouse=True)
    def _pytest_info(self, request):
        self._request = request

    @classmethod
    def setUpClass(cls):
        from transport_tools.libs.config import AnalysisConfig

        cls._test_failed = False
        cls.maxDiff = None
        cls.root = set_paths_from_package_root("tests", "data")
        cls.out_path = set_paths_from_package_root("tests", "test_results",
                                                    "TestTransportProcessesMixedEvents")
        os.makedirs(cls.out_path, exist_ok=True)

        sims = os.path.join(cls.root, "simulations", "mixed_events")
        param_overrides = {
            "caver_results_path": os.path.join(sims, "caver"),
            "aquaduct_results_path": "{},{}".format(
                os.path.join(sims, "water"), os.path.join(sims, "ligand")
            ),
        }
        prep_test_config(cls.root, cls.out_path,
                         source_filename="tmp_config2.ini",
                         param_overrides=param_overrides)

        cls.config = AnalysisConfig(os.path.join(cls.out_path, "config.ini"), logging=False)
        cls.config.set_parameter("output_path", cls.out_path)
        os.makedirs(os.path.join(cls.out_path, "temp"), exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        from shutil import rmtree
        if cls._test_failed:
            return
        rmtree(cls.out_path)

    def tearDown(self):
        if hasattr(self, '_outcome'):
            if hasattr(self._outcome, 'result'):  # type: ignore[attr-defined]
                errors = getattr(self._outcome.result, 'errors', [])  # type: ignore[attr-defined]
                failures = getattr(self._outcome.result, 'failures', [])  # type: ignore[attr-defined]
                if errors or failures:
                    self.__class__._test_failed = True
        if hasattr(self, '_request') and hasattr(self._request, 'node'):
            if self._request.session.testsfailed > 0:
                self.__class__._test_failed = True

    def setUp(self):
        self.saved_data = os.path.join(TestTransportProcessesMixedEvents.root,
                                       "saved_outputs_mixed_events")
        self.out_path = TestTransportProcessesMixedEvents.out_path

    def _get_dumpfile(self, stage: int) -> str:
        return os.path.join(self.out_path, "temp", "mol_system_{}.dump".format(stage))

    # ------------------------------------------------------------------
    # Stage 01 — transformations
    # Exercises: multi-path AQUA-DUCT discovery for both water and ligand
    # ------------------------------------------------------------------
    def test_01compute_transformations(self):
        mol_system = TransportProcesses(TestTransportProcessesMixedEvents.config)
        mol_system.compute_transformations()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "transformations", "aquaduct"),
                             os.path.join(self.out_path, "_internal", "transformations", "aquaduct"), self)
        compare_test_folders(os.path.join(self.saved_data, "_internal", "transformations", "caver"),
                             os.path.join(self.out_path, "_internal", "transformations", "caver"), self)
        save_checkpoint(mol_system, self._get_dumpfile(1), overwrite=True)

    # ------------------------------------------------------------------
    # Stage 02 — caver networks
    # ------------------------------------------------------------------
    def test_02process_tunnel_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(1))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.process_tunnel_networks()
        for md in ("e10s0_seed43_f371m1821", "e10s14_e4s14f222m327_f1997m1236"):
            compare_test_folders(
                os.path.join(self.saved_data, "visualization", "sources", "network_data", "caver", md),
                os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", md), self)
        save_checkpoint(mol_system, self._get_dumpfile(2), overwrite=True)

    # ------------------------------------------------------------------
    # Stage 03 — layered caver
    # ------------------------------------------------------------------
    def test_03create_layered_description4tunnel_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(2))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.create_layered_description4tunnel_networks()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "layered_data", "caver"),
                             os.path.join(self.out_path, "_internal", "layered_data", "caver"), self)
        save_checkpoint(mol_system, self._get_dumpfile(3), overwrite=True)

    # ------------------------------------------------------------------
    # Stage 04 — clustering
    # ------------------------------------------------------------------
    def test_04merge_tunnel_clusters2super_clusters(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(3))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.compute_tunnel_clusters_distances()
        mol_system.merge_tunnel_clusters2super_clusters()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "clustering"),
                             os.path.join(self.out_path, "_internal", "clustering"), self)
        save_checkpoint(mol_system, self._get_dumpfile(4), overwrite=True)

    # ------------------------------------------------------------------
    # Stage 05 — initial SC profiles
    # ------------------------------------------------------------------
    def test_05create_super_cluster_profiles(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(4))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.create_super_cluster_profiles()
        compare_test_files(
            os.path.join(self.saved_data, "data", "super_clusters", "details",
                         "initial_super_cluster_details.txt"),
            os.path.join(self.out_path, "data", "super_clusters", "details",
                         "initial_super_cluster_details.txt"), self)
        compare_test_folders(
            os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "initial"),
            os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "initial"), self)
        save_checkpoint(mol_system, self._get_dumpfile(5), overwrite=True)

    # ------------------------------------------------------------------
    # Stage 06 — initial tunnel summary (compared per group too)
    # ------------------------------------------------------------------
    def test_06generate_super_cluster_summary(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(5))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.generate_super_cluster_summary(out_filename="1-initial_tunnels_statistics.txt")
        for rel in (
            os.path.join("statistics", "1-initial_tunnels_statistics.txt"),
            os.path.join("statistics", "comparative_analysis", "1-initial_tunnels_statistics.txt"),
            os.path.join("statistics", "comparative_analysis", "group1",
                         "1-initial_tunnels_statistics.txt"),
            os.path.join("statistics", "comparative_analysis", "group2",
                         "1-initial_tunnels_statistics.txt"),
        ):
            compare_test_files(os.path.join(self.saved_data, rel),
                               os.path.join(self.out_path, rel), self)
        save_checkpoint(mol_system, self._get_dumpfile(6), overwrite=True)

    # ------------------------------------------------------------------
    # Stage 07 — tunnel visualization (per group)
    # ------------------------------------------------------------------
    def test_07save_super_clusters_visualization(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(6))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.save_super_clusters_visualization(script_name="1-visualize_initial_tunnels.py")
        for rel in (
            os.path.join("visualization", "1-visualize_initial_tunnels.py"),
            os.path.join("visualization", "comparative_analysis", "group1",
                         "1-visualize_initial_tunnels.py"),
            os.path.join("visualization", "comparative_analysis", "group2",
                         "1-visualize_initial_tunnels.py"),
        ):
            compare_test_files(os.path.join(self.saved_data, rel),
                               os.path.join(self.out_path, rel), self)
        save_checkpoint(mol_system, self._get_dumpfile(7), overwrite=True)

    # ------------------------------------------------------------------
    # Stage 08 — filter 1 + filtered summary/viz
    # ------------------------------------------------------------------
    def test_08filter_super_cluster_profiles(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(7))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.filter_super_cluster_profiles(min_sims_num=-1, min_snapshots_num=-1)
        mol_system.generate_super_cluster_summary(out_filename="2-filtered_tunnels_statistics.txt")
        mol_system.save_super_clusters_visualization(script_name="2-visualize_filtered_tunnels.py")

        compare_test_files(
            os.path.join(self.saved_data, "data", "super_clusters", "details",
                         "filtered_super_cluster_details1.txt"),
            os.path.join(self.out_path, "data", "super_clusters", "details",
                         "filtered_super_cluster_details1.txt"), self)

        for rel in (
            os.path.join("statistics", "2-filtered_tunnels_statistics.txt"),
            os.path.join("statistics", "comparative_analysis", "2-filtered_tunnels_statistics.txt"),
            os.path.join("statistics", "comparative_analysis", "group1",
                         "2-filtered_tunnels_statistics.txt"),
            os.path.join("statistics", "comparative_analysis", "group2",
                         "2-filtered_tunnels_statistics.txt"),
            os.path.join("visualization", "2-visualize_filtered_tunnels.py"),
            os.path.join("visualization", "comparative_analysis", "group1",
                         "2-visualize_filtered_tunnels.py"),
            os.path.join("visualization", "comparative_analysis", "group2",
                         "2-visualize_filtered_tunnels.py"),
        ):
            compare_test_files(os.path.join(self.saved_data, rel),
                               os.path.join(self.out_path, rel), self)

        compare_test_folders(
            os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "filtered01"),
            os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "filtered01"), self)
        save_checkpoint(mol_system, self._get_dumpfile(8), overwrite=True)

    # ------------------------------------------------------------------
    # Stage 09 — process AQUA-DUCT networks (CORE NEW: merges water + ligand)
    # ------------------------------------------------------------------
    def test_09process_aquaduct_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(8))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.process_aquaduct_networks()
        for md in ("e10s0_seed43_f371m1821", "e10s14_e4s14f222m327_f1997m1236"):
            compare_test_folders(
                os.path.join(self.saved_data, "visualization", "sources", "network_data",
                             "aquaduct", md),
                os.path.join(self.out_path, "visualization", "sources", "network_data",
                             "aquaduct", md), self)
        save_checkpoint(mol_system, self._get_dumpfile(9), overwrite=True)

    # ------------------------------------------------------------------
    # Stage 10 — layered AQUA-DUCT
    # ------------------------------------------------------------------
    def test_10create_layered_description4aquaduct_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(9))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.create_layered_description4aquaduct_networks()
        compare_test_folders(
            os.path.join(self.saved_data, "_internal", "layered_data", "aquaduct"),
            os.path.join(self.out_path, "_internal", "layered_data", "aquaduct"), self)
        save_checkpoint(mol_system, self._get_dumpfile(10), overwrite=True)

    # ------------------------------------------------------------------
    # Stage 11 — assign events (CORE NEW: per-residue tracking, multi-residue summaries)
    # ------------------------------------------------------------------
    def test_11assign_transport_events(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(10))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.assign_transport_events()
        mol_system.save_super_clusters_visualization(script_name="3-visualize_initial_events.py")
        mol_system.generate_super_cluster_summary(out_filename="3-initial_events_statistics.txt")
        mol_system.filter_super_cluster_profiles(min_sims_num=-1, min_snapshots_num=-1)
        mol_system.save_super_clusters_visualization(script_name="4-visualize_filtered_events.py")
        mol_system.generate_super_cluster_summary(out_filename="4-filtered_events_statistics.txt")

        # Per-residue breakdowns in detail files
        for fname in ("initial_super_cluster_events_details.txt",
                      "outlier_transport_events_details.txt",
                      "filtered_super_cluster_details2.txt"):
            compare_test_files(
                os.path.join(self.saved_data, "data", "super_clusters", "details", fname),
                os.path.join(self.out_path, "data", "super_clusters", "details", fname), self)

        # Per-residue statistics columns (overall + comparative_analysis + per group)
        for stem in ("3-initial_events_statistics", "4-filtered_events_statistics"):
            for rel in (
                os.path.join("statistics", "{}.txt".format(stem)),
                os.path.join("statistics", "comparative_analysis", "{}.txt".format(stem)),
                os.path.join("statistics", "comparative_analysis", "group1", "{}.txt".format(stem)),
                os.path.join("statistics", "comparative_analysis", "group2", "{}.txt".format(stem)),
            ):
                compare_test_files(os.path.join(self.saved_data, rel),
                                   os.path.join(self.out_path, rel), self)

        # PyMOL visualization scripts with per-residue colors / object names
        for stem in ("3-visualize_initial_events", "4-visualize_filtered_events"):
            for rel in (
                os.path.join("visualization", "{}.py".format(stem)),
                os.path.join("visualization", "comparative_analysis", "group1", "{}.py".format(stem)),
                os.path.join("visualization", "comparative_analysis", "group2", "{}.py".format(stem)),
            ):
                compare_test_files(os.path.join(self.saved_data, rel),
                                   os.path.join(self.out_path, rel), self)

        save_checkpoint(mol_system, self._get_dumpfile(11), overwrite=True)


class TestAquaductTracedResiduesFilterIntegration(unittest.TestCase):
    """
    Integration test for the aquaduct_traced_residues_filter parameter.
    With filter=['WAT'], U01 (ligand) paths must be skipped during
    read_raw_paths_data, leaving orig_entities devoid of U01 events.
    """

    @classmethod
    def setUpClass(cls):
        from transport_tools.libs.config import AnalysisConfig

        cls.maxDiff = None
        cls.root = set_paths_from_package_root("tests", "data")
        cls.out_path = set_paths_from_package_root(
            "tests", "test_results", "TestAquaductTracedResiduesFilterIntegration")
        os.makedirs(cls.out_path, exist_ok=True)

        sims = os.path.join(cls.root, "simulations", "mixed_events")
        param_overrides = {
            "caver_results_path": os.path.join(sims, "caver"),
            "aquaduct_results_path": "{},{}".format(
                os.path.join(sims, "water"), os.path.join(sims, "ligand")
            ),
        }
        prep_test_config(cls.root, cls.out_path,
                         source_filename="tmp_config2.ini",
                         param_overrides=param_overrides)

        cls.config = AnalysisConfig(os.path.join(cls.out_path, "config.ini"), logging=False)
        cls.config.set_parameter("output_path", cls.out_path)
        # Filter to WAT only — U01 ligand events must be skipped
        cls.config.set_parameter("aquaduct_traced_residues_filter", ["WAT"])

    @classmethod
    def tearDownClass(cls):
        from shutil import rmtree
        if os.path.exists(cls.out_path):
            rmtree(cls.out_path)

    def test_filter_excludes_u01_events_from_aquaduct_network(self):
        """End-to-end: with filter=['WAT'], the AquaductNetwork dumps written by
        process_aquaduct_networks must contain only WAT-traced paths."""
        import pickle

        mol_system = TransportProcesses(self.config)
        mol_system.compute_transformations()
        mol_system.process_aquaduct_networks()

        for md in ("e10s0_seed43_f371m1821", "e10s14_e4s14f222m327_f1997m1236"):
            dump = os.path.join(self.out_path, "_internal", "network_data", "aquaduct",
                                "{}_aqua.dump".format(md))
            self.assertTrue(os.path.exists(dump),
                            "expected aquaduct network dump at {}".format(dump))
            with open(dump, "rb") as f:
                orig_entities = pickle.load(f)  # AquaductNetwork.save_orig_network dumps the list directly

            self.assertGreater(len(orig_entities), 0,
                               "WAT events should still be present for {}".format(md))
            resnames = {path.traced_residue[0] for path in orig_entities}
            self.assertEqual({"WAT"}, resnames,
                             "expected only WAT in {} orig_entities, got {}".format(md, resnames))


if __name__ == "__main__":
    unittest.main()
