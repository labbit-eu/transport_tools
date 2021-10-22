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


class TestTunnelNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transport_tools.libs.config import AnalysisConfig
        from transport_tools.libs.networks import TunnelNetwork

        cls.maxDiff = None
        cls.root = set_paths("tests", "data")
        prep_config(cls.root)
        cls.config = AnalysisConfig(os.path.join(cls.root, "config.ini"), logging=False)
        cls.config.set_parameter("output_path", set_paths("tests", "test_results", "TestTunnelNetwork"))
        cls.out_path = cls.config.get_parameter("output_path")
        os.makedirs(cls.out_path, exist_ok=True)
        cls.config.set_parameter("transformation_folder", os.path.join(cls.root, "saved_outputs",
                                                                       "_internal", "transformations"))

        cls.net = TunnelNetwork(cls.config.get_parameters(), "md1")
        cls.net.read_tunnels_data()
        for cluster in cls.net.get_clusters4layering():
            cls_id, md_label, layered_path_set = cluster.create_layered_cluster()
            cls.net.add_layered_entity(cls_id, layered_path_set)

    @classmethod
    def tearDownClass(cls):
        from shutil import rmtree

        os.remove(os.path.join(cls.root, "config.ini"))
        rmtree(cls.out_path)

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
                    if ".pdb" in res_file and "REMARK   1 CREATED WITH MDTraj" in res_line:
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

    def _compare_folders(self, saved_outputs_dir: str, results_dir: str, ):
        results_files = sorted(os.listdir(results_dir))
        out_files = sorted(os.listdir(saved_outputs_dir))
        self.assertEqual(out_files, results_files, msg="In folders '{}' and '{}':".format(saved_outputs_dir,
                                                                                          results_dir))

        for res_file, out_file in zip(results_files, out_files):
            res_file = os.path.join(results_dir, res_file)
            out_file = os.path.join(saved_outputs_dir, out_file)
            if os.path.isfile(res_file) and os.path.isfile(out_file):
                self._compare_files(out_file, res_file)

    def setUp(self):
        self.saved_data = os.path.join(TestTunnelNetwork.root, "saved_outputs")
        self.parameters = TestTunnelNetwork.config.get_parameters()
        self.network = TestTunnelNetwork.net

    def test_get_cluster(self):
        for i in range(1, 50):
            self.assertEqual(self.network.get_cluster(i).cluster_id, i)
        self.assertRaises(ValueError, self.network.get_cluster, 0)
        self.assertRaises(ValueError, self.network.get_cluster, 51)

    def test_cluster_exists(self):
        for i in range(1, 51):
            self.assertTrue(self.network.cluster_exists(i))
        self.assertRaises(ValueError, self.network.cluster_exists, 0)
        self.assertFalse(self.network.cluster_exists(51))

    def test_save_orig_network(self):
        from transport_tools.libs.networks import TunnelNetwork

        self.network.save_orig_network()
        new_net = TunnelNetwork(self.parameters, "md1")
        new_net.load_orig_network()

        saved_net = TunnelNetwork(self.parameters, "md1")
        saved_net.orig_dump_file = os.path.join(self.saved_data, "_internal", "network_data", "caver", "md1_caver.dump")
        saved_net.load_orig_network()

        self.assertEqual(len(saved_net.orig_entities), len(new_net.orig_entities))
        for new_cluster, saved_cluster in zip(new_net.orig_entities, saved_net.orig_entities):
            self.assertTrue(new_cluster.is_same(saved_cluster))

    def test_save_orig_network_visualization(self):
        self.network.save_orig_network_visualization()
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "caver", "md1"),
                              self.network.orig_viz_path)

    def test_layering(self):
        self.assertEqual(50, len(self.network.get_clusters4layering()))
        entity_ids = list()
        for cluster in self.network.get_clusters4layering():
            entity_ids.append(cluster.cluster_id)
        self.assertTrue(self.network.is_layering_complete(entity_ids))

    def test_save_layered_network(self):
        from transport_tools.libs.networks import TunnelNetwork

        self.network.save_layered_network()

        new_net = TunnelNetwork(self.parameters, "md1")
        new_net.load_layered_network()

        saved_net = TunnelNetwork(self.parameters, "md1")
        saved_net.layered_dump_file = os.path.join(self.saved_data, "_internal", "layered_data", "caver",
                                                   "md1_layered_paths.dump")
        saved_net.load_layered_network()

        self.assertSetEqual(set(saved_net.layered_entities.keys()), set(new_net.layered_entities.keys()))
        for entity_label in saved_net.layered_entities.keys():
            new_pathset = new_net.layered_entities[entity_label]
            saved_pathset = saved_net.layered_entities[entity_label]
            self.assertTrue(new_pathset.is_same(saved_pathset))

    def test_save_layered_visualization(self):
        self.network.save_layered_visualization(save_pdb_files=True)
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver",
                                           "md1"),
                              self.network.layered_viz_path)
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver",  "md1",
                                           "nodes"),
                              os.path.join(self.network.layered_viz_path, "nodes"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver",  "md1",
                                           "paths"),
                              os.path.join(self.network.layered_viz_path, "paths"))


class TestAquaductNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transport_tools.libs.config import AnalysisConfig
        from transport_tools.libs.networks import AquaductNetwork

        cls.maxDiff = None
        cls.root = set_paths("tests", "data")
        prep_config(cls.root)
        cls.config = AnalysisConfig(os.path.join(cls.root, "config.ini"), logging=False)
        cls.config.set_parameter("output_path", set_paths("tests", "test_results", "TestAquaductNetwork"))
        cls.out_path = cls.config.get_parameter("output_path")
        os.makedirs(cls.out_path, exist_ok=True)
        cls.config.set_parameter("transformation_folder", os.path.join(cls.root, "saved_outputs",
                                                                       "_internal", "transformations"))

        cls.net = AquaductNetwork(cls.config.get_parameters(), "md1")
        cls.net.read_raw_paths_data()

        for event in cls.net.get_events4layering():
            event_id, md_label, layered_path_set = event.create_layered_event()
            cls.net.add_layered_entity(event_id, layered_path_set)

    @classmethod
    def tearDownClass(cls):
        from shutil import rmtree

        os.remove(os.path.join(cls.root, "config.ini"))
        rmtree(cls.out_path)
        cls.net.clean_tempfile()

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
                    if ".pdb" in res_file and "REMARK   1 CREATED WITH MDTraj" in res_line:
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

    def _compare_folders(self, saved_outputs_dir: str, results_dir: str, ):
        results_files = sorted(os.listdir(results_dir))
        out_files = sorted(os.listdir(saved_outputs_dir))
        self.assertEqual(out_files, results_files, msg="In folders '{}' and '{}':".format(saved_outputs_dir,
                                                                                          results_dir))

        for res_file, out_file in zip(results_files, out_files):
            res_file = os.path.join(results_dir, res_file)
            out_file = os.path.join(saved_outputs_dir, out_file)
            if os.path.isfile(res_file) and os.path.isfile(out_file):
                self._compare_files(out_file, res_file)

    def setUp(self):
        self.saved_data = os.path.join(TestAquaductNetwork.root, "saved_outputs")
        self.parameters = TestAquaductNetwork.config.get_parameters()
        self.network = TestAquaductNetwork.net

    def test_get_tempfile(self):
        self.network.get_pdb_file()
        self._compare_files(os.path.join(self.saved_data, "tmpm36nzt5t.pdb"), self.network.pdb_file)

    def test_get_events4layering(self):
        self.assertEqual(5, len(self.network.get_events4layering()))

    def test_save_orig_network(self):
        from transport_tools.libs.networks import AquaductNetwork

        self.network.save_orig_network()

        new_net = AquaductNetwork(self.parameters, "md1", load_only=True)
        new_net.load_orig_network()

        saved_net = AquaductNetwork(self.parameters, "md1", load_only=True)
        saved_net.orig_dump_file = os.path.join(self.saved_data, "_internal", "network_data", "aquaduct",
                                                "md1_aqua.dump")
        saved_net.load_orig_network()

        self.assertEqual(len(saved_net.orig_entities), len(new_net.orig_entities))
        for new_path, saved_path in zip(new_net.orig_entities, saved_net.orig_entities):
            self.assertTrue(new_path.is_same(saved_path))

    def test_save_layered_network(self):
        from transport_tools.libs.networks import AquaductNetwork

        self.network.save_layered_network()
        new_net = AquaductNetwork(self.parameters, "md1", load_only=True)
        new_net.load_layered_network()

        saved_net = AquaductNetwork(self.parameters, "md1", load_only=True)
        saved_net.layered_dump_file = os.path.join(self.saved_data, "_internal", "layered_data", "aquaduct",
                                                   "md1_layered_paths.dump")
        saved_net.load_layered_network()
        self.assertSetEqual(set(saved_net.layered_entities.keys()), set(new_net.layered_entities.keys()))
        for entity_label in saved_net.layered_entities.keys():
            new_pathset = new_net.layered_entities[entity_label]
            saved_pathset = saved_net.layered_entities[entity_label]
            self.assertTrue(new_pathset.is_same(saved_pathset))

    def test_save_orig_network_visualization(self):
        self.network.save_orig_network_visualization()
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"),
                              self.network.orig_viz_path)

    def test_save_layered_visualization(self):
        self.network.save_layered_visualization(save_pdb_files=True)
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"),
                              self.network.layered_viz_path)
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "nodes"),
                              os.path.join(self.network.layered_viz_path, "nodes"))
        self._compare_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "paths"),
                              os.path.join(self.network.layered_viz_path, "paths"))


class TestSuperCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transport_tools.libs.config import AnalysisConfig
        from shutil import copytree

        cls.maxDiff = None
        cls.root = set_paths("tests", "data")
        prep_config(cls.root)
        cls.config = AnalysisConfig(os.path.join(cls.root, "config.ini"), logging=False)
        cls.config.set_parameter("output_path", set_paths("tests", "test_results", "TestSuperCluster"))
        cls.out_path = cls.config.get_parameter("output_path")
        cls.config.set_parameter("transformation_folder", os.path.join(cls.root, "saved_outputs",
                                                                       "_internal", "transformations"))

        os.makedirs(cls.out_path, exist_ok=True)
        copytree(os.path.join(cls.root, "saved_outputs", "_internal", "super_cluster_profiles"),
                 os.path.join(cls.out_path, "_internal", "super_cluster_profiles"), dirs_exist_ok=True)
        copytree(os.path.join(cls.root, "saved_outputs", "_internal", "network_data", "caver"),
                 os.path.join(cls.out_path, "_internal", "network_data", "caver"), dirs_exist_ok=True)
        copytree(os.path.join(cls.root, "saved_outputs", "_internal", "layered_data", "caver"),
                 os.path.join(cls.out_path, "_internal", "layered_data", "caver"), dirs_exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        from shutil import rmtree

        os.remove(os.path.join(cls.root, "config.ini"))
        rmtree(cls.out_path)

    def _compare_files(self, out_file: str, res_file: str,):
        import gzip
        import pickle

        if res_file.endswith(".dump.gz"):
            with gzip.open(res_file, 'rb') as res_in, gzip.open(out_file, 'rb') as out_in:
                res_lines = pickle.load(res_in)
                out_lines = pickle.load(out_in)
        elif ".gz" in res_file:
            with gzip.open(res_file, 'r') as res_in, gzip.open(out_file, 'r') as out_in:
                res_lines = res_in.readlines()
                out_lines = out_in.readlines()
        else:
            with open(res_file, "r") as res_in, open(out_file, "r") as out_in:
                res_lines = res_in.readlines()
                out_lines = out_in.readlines()

        self.assertTrue(len(res_lines) == len(out_lines),
                        msg="Different length of files '{}' and '{}':".format(out_file, res_file))
        for res_line, out_line in zip(res_lines, out_lines):
            if ".pdb" in res_file and "REMARK   1 CREATED WITH MDTraj" in res_line:
                continue
            self.assertEqual(out_line, res_line, msg="In files '{}' and '{}':".format(out_file, res_file))

    def _compare_folders(self, saved_outputs_dir: str, results_dir: str, ):
        results_files = sorted(os.listdir(results_dir))
        out_files = sorted(os.listdir(saved_outputs_dir))
        self.assertEqual(out_files, results_files, msg="In folders '{}' and '{}':".format(saved_outputs_dir,
                                                                                          results_dir))

        for res_file, out_file in zip(results_files, out_files):
            res_file = os.path.join(results_dir, res_file)
            out_file = os.path.join(saved_outputs_dir, out_file)
            if os.path.isfile(res_file) and os.path.isfile(out_file):
                self._compare_files(out_file, res_file)

    def setUp(self):
        from transport_tools.libs.networks import TunnelNetwork, SuperCluster

        self.saved_data = os.path.join(TestSuperCluster.root, "saved_outputs")
        self.out_path = TestSuperCluster.out_path
        self.parameters = TestSuperCluster.config.get_parameters()
        sc_defs = [(('md1', 50), 27), (('md1', 49), 13), (('md1', 48), 11), (('md1', 47), 10), (('md1', 46), 11),
                   (('md1', 45), 13), (('md1', 44), 26), (('md1', 43), 12), (('md1', 42), 25), (('md1', 41), 9),
                   (('md1', 40), 24), (('md1', 39), 23), (('md1', 38), 3), (('md1', 37), 4), (('md1', 36), 17),
                   (('md1', 35), 22), (('md1', 34), 21), (('md1', 33), 10), (('md1', 32), 4), (('md1', 31), 20),
                   (('md1', 30), 2), (('md1', 29), 19), (('md1', 28), 17), (('md1', 27), 6), (('md1', 26), 18),
                   (('md1', 25), 8), (('md1', 24), 16), (('md1', 23), 15), (('md1', 22), 13), (('md1', 21), 14),
                   (('md1', 20), 3), (('md1', 19), 11), (('md1', 18), 12), (('md1', 17), 7), (('md1', 16), 10),
                   (('md1', 15), 8), (('md1', 14), 8), (('md1', 13), 1), (('md1', 12), 7), (('md1', 11), 6),
                   (('md1', 10), 7), (('md1', 9), 9), (('md1', 8), 2), (('md1', 7), 6), (('md1', 6), 5),
                   (('md1', 5), 1), (('md1', 4), 4), (('md1', 3), 3), (('md1', 2), 2), (('md1', 1), 1)]

        path_sets = dict()
        tunnel_network = TunnelNetwork(self.parameters, "md1")
        tunnel_network.load_layered_network()

        for cls_id, layered_path_set in tunnel_network.layered_entities.items():
            cluster_specification = ("md1", cls_id)
            path_sets[cluster_specification] = layered_path_set

        self.super_clusters = dict()
        for cluster_specification, sc_id in sc_defs:
            md_label, cls_id = cluster_specification
            if sc_id in self.super_clusters.keys():
                self.super_clusters[sc_id].add_caver_cluster(md_label, int(cls_id), path_sets[cluster_specification])
            else:
                self.super_clusters[sc_id] = SuperCluster(sc_id, self.parameters, 1)
                self.super_clusters[sc_id].add_caver_cluster(md_label, int(cls_id), path_sets[cluster_specification])

        self.super_clusters[1].add_transport_event("md1", str(1), "release", ('WAT:326', (908, 965)))
        self.super_clusters[1].add_transport_event("md1", str(1), "entry", ('WAT:326', (144, 893)))
        self.super_clusters[1].add_transport_event("md1", str(4), "entry", ('WAT:326', (880, 902)))

    def _prioritize(self):
        from transport_tools.libs.networks import define_filters
        sc_order = list()
        for sc_id, super_cluster in self.super_clusters.items():
            super_cluster.set_properties(super_cluster.process_cluster_profile()[1])
            if super_cluster.has_passed_filter(consider_transport_events=True, active_filters=define_filters()):
                sc_order.append((super_cluster.properties["overall"]["priority"], sc_id))

        sc_order = [sc_id[1] for sc_id in sorted(sc_order, reverse=True)]

        for new_id, old_id in enumerate(sc_order):
            self.super_clusters[old_id].prioritized_sc_id = new_id + 1

    def test__get_csv_file(self):
        self.assertEqual(os.path.join(set_paths("tests"), "test_results", "TestSuperCluster", "data", "super_clusters",
                                      "CSV_profiles", "initial", "super_cluster_05.csv"),
                         self.super_clusters[5]._get_csv_file())

    def test__get_bottleneck_file(self):
        self.assertEqual(os.path.join(set_paths("tests"), "test_results", "TestSuperCluster", "data", "super_clusters",
                                      "bottlenecks", "initial", "super_cluster_05.csv"),
                         self.super_clusters[5]._get_bottleneck_file())

    def test_report_details(self):
        self._prioritize()
        self.assertEqual("""Supercluster ID 1

Details on tunnel network:
Number of MD simulations = 1
Number of tunnel clusters = 3
Tunnel clusters:
from md1: 1, 5, 13, 

Details on transport events:
Number of MD simulations = 1
Number of entry events = 2
Number of release events = 1
entry: (from Simulation: AQUA-DUCT ID, (Resname:Residue), start_frame->end_frame; ... )
from md1: 1, (WAT:326), 144->893; 4, (WAT:326), 880->902; 
release: (from Simulation: AQUA-DUCT ID, (Resname:Residue), start_frame->end_frame; ... )
from md1: 1, (WAT:326), 908->965; 
""", self.super_clusters[1].report_details(True))

        self.assertEqual("""Supercluster ID 23

Details on tunnel network:
Number of MD simulations = 1
Number of tunnel clusters = 1
Tunnel clusters:
from md1: 39, 

Details on transport events:
Number of MD simulations = 0
Number of entry events = 0
Number of release events = 0
""", self.super_clusters[23].report_details(True))

    def test_compute_space_descriptors(self):
        import numpy as np
        sc_id, avg_direction = self.super_clusters[1].compute_space_descriptors()
        self.super_clusters[1].load_path_sets()
        self.assertSequenceEqual((48, 7), self.super_clusters[1].path_sets["overall"].nodes_data.shape)
        self.assertTrue(np.allclose(avg_direction, np.array([-2.49484359, -7.03691757, -6.94671987]), atol=1e-7))
        sc_id, avg_direction = self.super_clusters[23].compute_space_descriptors()
        self.super_clusters[23].load_path_sets()
        self.assertSequenceEqual((33, 7), self.super_clusters[23].path_sets["overall"].nodes_data.shape)
        self.assertTrue(np.allclose(avg_direction, np.array([15.43776866, 6.38202966, 21.02110355]), atol=1e-7))

    def test_has_passed_filter(self):
        from transport_tools.libs.networks import define_filters

        self._prioritize()
        self.assertTrue(self.super_clusters[1].has_passed_filter(False))
        self.assertTrue(self.super_clusters[5].has_passed_filter(False))
        self.assertRaises(RuntimeError, self.super_clusters[5].has_passed_filter, True)

        self.assertTrue(self.super_clusters[1].has_passed_filter(True, define_filters(min_total_events=1)))
        self.assertFalse(self.super_clusters[5].has_passed_filter(True, define_filters(min_total_events=1)))
        self.assertTrue(self.super_clusters[5].has_passed_filter(False))

    def test_get_labels(self):
        self.assertListEqual(['md1'], self.super_clusters[1].get_md_labels())
        self.assertListEqual([13, 5, 1], self.super_clusters[1].get_caver_cluster_ids4md_label("md1"))
        self.assertListEqual(['md1_13', 'md1_5', 'md1_1'], self.super_clusters[1].get_caver_clusters_full_labels())
        self.assertListEqual(['md1'], self.super_clusters[23].get_md_labels())
        self.assertListEqual([39], self.super_clusters[23].get_caver_cluster_ids4md_label("md1"))
        self.assertListEqual(['md1_39'], self.super_clusters[23].get_caver_clusters_full_labels())

    def test_is_directionally_aligned(self):
        import numpy as np

        sc_id, avg_direction = self.super_clusters[23].compute_space_descriptors()
        self.super_clusters[23].load_path_sets()
        self.super_clusters[23].avg_direction = avg_direction
        self.assertFalse(self.super_clusters[23].is_directionally_aligned(np.array([-1, 1, -1])))
        self.assertTrue(self.super_clusters[23].is_directionally_aligned(np.array([2, -0.3, 1])))
        self.assertTrue(self.super_clusters[23].is_directionally_aligned(np.array([1, -1, 1])))
        self.assertFalse(self.super_clusters[23].is_directionally_aligned(np.array([6, 21, -15])))

    def test_compute_distance2transport_event(self):
        from transport_tools.libs.networks import AquaductNetwork

        saved_net = AquaductNetwork(self.parameters, "md1", load_only=True)
        saved_net.layered_dump_file = os.path.join(self.saved_data, "_internal", "layered_data", "aquaduct",
                                                   "md1_layered_paths.dump")
        saved_net.load_layered_network()

        sc_id, avg_direction = self.super_clusters[2].compute_space_descriptors()
        self.super_clusters[2].load_path_sets()
        self.super_clusters[2].avg_direction = avg_direction

        sc_id, avg_direction = self.super_clusters[1].compute_space_descriptors()
        self.super_clusters[1].load_path_sets()
        self.super_clusters[1].avg_direction = avg_direction

        outputs = {
            "1_release": ((0.8571428571428571, 0.3333333333333333), (0.0, 1.0)),
            "1_entry": ((0.8571428571428571, 0.5), (0.0, 1.0)),
            "4_release": ((0.7142857142857143, 0.3333333333333333), (0.0, 1.0)),
            "4_entry": ((0.7142857142857143, 0.3333333333333333), (0.0, 1.0)),
            "6_entry": ((0.7272727272727273, 0.3333333333333333), (0.0, 1.0))
        }

        for event in saved_net.layered_entities.values():
            results = self.super_clusters[2].compute_distance2transport_event(event)
            self.assertAlmostEqual(outputs[event.entity_label][0][0], results[0])
            self.assertAlmostEqual(outputs[event.entity_label][0][1], results[1])

            results = self.super_clusters[1].compute_distance2transport_event(event)
            self.assertAlmostEqual(outputs[event.entity_label][1][0], results[0])
            self.assertAlmostEqual(outputs[event.entity_label][1][1], results[1])

    def test_prepare_visualization(self):
        self._prioritize()
        self.super_clusters[1].compute_space_descriptors()
        self.super_clusters[1].load_path_sets()
        self.assertListEqual(["with gzip.open('sources/super_cluster_CGOs/SC01_overall_pathset.dump.gz', 'rb') as in_stream:\n",
                              "    pathset = pickle.load(in_stream)\n",
                              "cmd.load_cgo(pathset, 'cluster_001')\n",
                              "cmd.set('cgo_line_width', 5, 'cluster_001')\n\n",
                              "events = ['sources/layered_data/aquaduct/md1/paths/wat_1_entry_pathset.dump.gz',\n'sources/layered_data/aquaduct/md1/paths/wat_4_entry_pathset.dump.gz']\n",
                              "for event in events:\n",
                              "    with gzip.open(event, 'rb') as in_stream:\n",
                              "        pathset = pickle.load(in_stream)\n",
                              "        for path in pathset:\n",
                              "            path[3:6] = [0.0, 0.0, 1.0]\n",
                              "            cmd.load_cgo(path, 'entry_001')\n",
                              "cmd.set('cgo_line_width', 2, 'entry_001')\n\n",
                              "events = ['sources/layered_data/aquaduct/md1/paths/wat_1_release_pathset.dump.gz']\n",
                              "for event in events:\n",
                              "    with gzip.open(event, 'rb') as in_stream:\n",
                              "        pathset = pickle.load(in_stream)\n",
                              "        for path in pathset:\n",
                              "            path[3:6] = [0.0, 0.0, 1.0]\n",
                              "            cmd.load_cgo(path, 'release_001')\n",
                              "cmd.set('cgo_line_width', 2, 'release_001')\n\n"],
                             self.super_clusters[1].prepare_visualization()[0])

    def test_get_summary_line(self):
        self._prioritize()
        self.assertListEqual(['1', '1', '991', '991.0', '1.077', '0.179', '1.720', '14.358', '1.465', '1.258', '0.095',
                              '0.50773', '0.07843', '0.50316'], self.super_clusters[1].get_summary_line_data(False))
        self.assertListEqual(['1', '1', '991', '991.0', '1.077', '0.179', '1.720', '14.358', '1.465', '1.258', '0.095',
                              '0.50773', '0.07843', '0.50316', '3', '2', '1'],
                             self.super_clusters[1].get_summary_line_data(True))

    def test_process_cluster_profile(self):
        for super_cluster in self.super_clusters.values():
            super_cluster.process_cluster_profile()
        self._compare_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "initial"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "initial"))
        self._compare_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "initial"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "initial"))

    def test_filter_super_cluster(self):
        from transport_tools.libs.networks import define_filters
        os.makedirs(os.path.join(self.out_path, "_internal", "super_cluster_filtered_profiles"), exist_ok=True)
        self.assertEqual({'md1': [1, 13, 5]}, self.super_clusters[1].filter_super_cluster(False,
                                                                                          define_filters(), 1)[3])
        self.assertEqual({'md1': [39]}, self.super_clusters[23].filter_super_cluster(False, define_filters(), 1)[3])
        self.assertEqual({'md1': [1, 13]},
                         self.super_clusters[1].filter_super_cluster(False,
                                                                     define_filters(min_bottleneck_radius=1.4), 1)[3])
        self.assertEqual({}, self.super_clusters[23].filter_super_cluster(False,
                                                                          define_filters(min_bottleneck_radius=.9),
                                                                          1)[3])
        self.assertEqual({'md1': [1]},
                         self.super_clusters[1].filter_super_cluster(False,
                                                                     define_filters(min_bottleneck_radius=1.7), 1)[3])
        self.assertEqual({}, self.super_clusters[23].filter_super_cluster(False,
                                                                          define_filters(min_bottleneck_radius=1.4),
                                                                          1)[3])

        self.assertDictEqual({'md1': {'146': 0.9495459132189707, '142': 0.958627648839556, '173': 0.963673057517659,
                                      '141': 0.8668012108980827, '169': 0.9263370332996973, '138': 0.8113017154389506,
                                      '172': 0.7739656912209889, '165': 0.2795156407669021, '242': 0.6347124117053481,
                                      '270': 0.14127144298688193, '269': 0.22401614530776992,
                                      '170': 0.27547931382441976, '145': 0.7507568113017155, '149': 0.29868819374369326,
                                      '168': 0.14127144298688193, '139': 0.19576185671039353,
                                      '103': 0.026236125126135216, '38': 0.02320887991927346,
                                      '206': 0.01917255297679112, '243': 0.026236125126135216,
                                      '268': 0.04641775983854692, '241': 0.01917255297679112,
                                      '240': 0.008072653884964682, '140': 0.009081735620585268,
                                      '143': 0.013118062563067608, '104': 0.010090817356205853,
                                      '164': 0.018163471241170535, '39': 0.013118062563067608,
                                      '174': 0.0020181634712411706, '203': 0.007063572149344097,
                                      '144': 0.0030272452068617556, '127': 0.0010090817356205853,
                                      '202': 0.0010090817356205853, '199': 0.0010090817356205853,
                                      '159': 0.0010090817356205853},
                              'overall': {'146': 0.9495459132189707, '142': 0.958627648839556, '173': 0.963673057517659,
                                          '141': 0.8668012108980827, '169': 0.9263370332996973,
                                          '138': 0.8113017154389506, '172': 0.7739656912209889,
                                          '165': 0.2795156407669021, '242': 0.6347124117053481,
                                          '270': 0.14127144298688193, '269': 0.22401614530776992,
                                          '170': 0.27547931382441976, '145': 0.7507568113017155,
                                          '149': 0.29868819374369326, '168': 0.14127144298688193,
                                          '139': 0.19576185671039353, '103': 0.026236125126135216,
                                          '38': 0.02320887991927346, '206': 0.01917255297679112,
                                          '243': 0.026236125126135216, '268': 0.04641775983854692,
                                          '241': 0.01917255297679112, '240': 0.008072653884964682,
                                          '140': 0.009081735620585268, '143': 0.013118062563067608,
                                          '104': 0.010090817356205853, '164': 0.018163471241170535,
                                          '39': 0.013118062563067608, '174': 0.0020181634712411706,
                                          '203': 0.007063572149344097, '144': 0.0030272452068617556,
                                          '127': 0.0010090817356205853, '202': 0.0010090817356205853,
                                          '199': 0.0010090817356205853, '159': 0.0010090817356205853}},
                             self.super_clusters[1].filter_super_cluster(False, define_filters(), 1)[2])

        self.assertDictEqual({'overall': {'63': 0.44285714285714284, '85': 0.37142857142857144, '61': 0.1,
                                          '35': 0.12857142857142856, '19': 0.5, '88': 0.45714285714285713,
                                          '33': 0.02857142857142857, '84': 0.42857142857142855,
                                          '14': 0.42857142857142855, '64': 0.34285714285714286,
                                          '87': 0.35714285714285715, '91': 0.12857142857142856,
                                          '12': 0.17142857142857143, '89': 0.08571428571428572,
                                          '92': 0.07142857142857142, '62': 0.07142857142857142,
                                          '20': 0.05714285714285714, '18': 0.05714285714285714,
                                          '69': 0.014285714285714285, '81': 0.1, '66': 0.37142857142857144,
                                          '108': 0.12857142857142856, '202': 0.35714285714285715,
                                          '67': 0.02857142857142857, '65': 0.05714285714285714,
                                          '38': 0.34285714285714286, '104': 0.37142857142857144,
                                          '165': 0.11428571428571428, '103': 0.22857142857142856,
                                          '36': 0.12857142857142856, '37': 0.24285714285714285,
                                          '203': 0.14285714285714285, '13': 0.05714285714285714,
                                          '105': 0.14285714285714285, '78': 0.05714285714285714,
                                          '106': 0.05714285714285714, '206': 0.014285714285714285,
                                          '39': 0.05714285714285714, '199': 0.07142857142857142,
                                          '198': 0.04285714285714286, '83': 0.07142857142857142,
                                          '15': 0.1, '86': 0.02857142857142857, '107': 0.014285714285714285,
                                          '34': 0.014285714285714285, '115': 0.014285714285714285,
                                          '82': 0.014285714285714285, '112': 0.014285714285714285,
                                          '111': 0.014285714285714285, '17': 0.07142857142857142,
                                          '102': 0.014285714285714285, '16': 0.014285714285714285},
                              'md1': {'63': 0.44285714285714284, '85': 0.37142857142857144, '61': 0.1,
                                      '35': 0.12857142857142856, '19': 0.5, '88': 0.45714285714285713,
                                      '33': 0.02857142857142857, '84': 0.42857142857142855, '14': 0.42857142857142855,
                                      '64': 0.34285714285714286, '87': 0.35714285714285715, '91': 0.12857142857142856,
                                      '12': 0.17142857142857143, '89': 0.08571428571428572, '92': 0.07142857142857142,
                                      '62': 0.07142857142857142, '20': 0.05714285714285714, '18': 0.05714285714285714,
                                      '69': 0.014285714285714285, '81': 0.1, '66': 0.37142857142857144,
                                      '108': 0.12857142857142856, '202': 0.35714285714285715, '67': 0.02857142857142857,
                                      '65': 0.05714285714285714, '38': 0.34285714285714286, '104': 0.37142857142857144,
                                      '165': 0.11428571428571428, '103': 0.22857142857142856, '36': 0.12857142857142856,
                                      '37': 0.24285714285714285, '203': 0.14285714285714285, '13': 0.05714285714285714,
                                      '105': 0.14285714285714285, '78': 0.05714285714285714, '106': 0.05714285714285714,
                                      '206': 0.014285714285714285, '39': 0.05714285714285714,
                                      '199': 0.07142857142857142, '198': 0.04285714285714286, '83': 0.07142857142857142,
                                      '15': 0.1, '86': 0.02857142857142857, '107': 0.014285714285714285,
                                      '34': 0.014285714285714285, '115': 0.014285714285714285,
                                      '82': 0.014285714285714285, '112': 0.014285714285714285,
                                      '111': 0.014285714285714285, '17': 0.07142857142857142,
                                      '102': 0.014285714285714285, '16': 0.014285714285714285}},
                             self.super_clusters[23].filter_super_cluster(False, define_filters(), 1)[2])

        self.assertDictEqual({'md1': {'138': 0.7692307692307693, '269': 0.3333333333333333, '146': 0.9487179487179487,
                                      '243': 0.20512820512820512, '242': 0.5641025641025641, '103': 0.20512820512820512,
                                      '206': 0.1794871794871795, '165': 0.6923076923076923, '270': 0.3333333333333333,
                                      '104': 0.1282051282051282, '38': 0.15384615384615385, '142': 0.7948717948717948,
                                      '169': 0.8205128205128205, '141': 0.6923076923076923, '173': 0.8717948717948718,
                                      '145': 0.7435897435897436, '172': 0.6666666666666666, '149': 0.6410256410256411,
                                      '168': 0.2564102564102564, '203': 0.05128205128205128, '164': 0.05128205128205128,
                                      '170': 0.38461538461538464, '139': 0.10256410256410256, '39': 0.05128205128205128,
                                      '143': 0.07692307692307693, '268': 0.02564102564102564,
                                      '127': 0.02564102564102564},
                              'overall': {'138': 0.7692307692307693, '269': 0.3333333333333333,
                                          '146': 0.9487179487179487, '243': 0.20512820512820512,
                                          '242': 0.5641025641025641, '103': 0.20512820512820512,
                                          '206': 0.1794871794871795, '165': 0.6923076923076923,
                                          '270': 0.3333333333333333, '104': 0.1282051282051282,
                                          '38': 0.15384615384615385, '142': 0.7948717948717948,
                                          '169': 0.8205128205128205, '141': 0.6923076923076923,
                                          '173': 0.8717948717948718, '145': 0.7435897435897436,
                                          '172': 0.6666666666666666, '149': 0.6410256410256411,
                                          '168': 0.2564102564102564, '203': 0.05128205128205128,
                                          '164': 0.05128205128205128, '170': 0.38461538461538464,
                                          '139': 0.10256410256410256, '39': 0.05128205128205128,
                                          '143': 0.07692307692307693, '268': 0.02564102564102564,
                                          '127': 0.02564102564102564}},
                             self.super_clusters[1].filter_super_cluster(False,
                                                                         define_filters(min_bottleneck_radius=1.4),
                                                                         1)[2])
        self.assertDictEqual({'overall': {}},
                             self.super_clusters[23].filter_super_cluster(False,
                                                                          define_filters(min_bottleneck_radius=.9),
                                                                          1)[2])

    def test_get_property_time_evolution_data(self):
        from transport_tools.libs.networks import define_filters

        data = self.super_clusters[1].get_property_time_evolution_data("bottleneck_radius", define_filters())["md1"]
        self.assertListEqual([0.8958575417014984, 0.8824133711991726, 0.7997563299293802, 1.198631201226353],
                             data[:4].tolist())


if __name__ == '__main__':
    unittest.main()
