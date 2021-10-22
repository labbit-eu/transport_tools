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
import numpy as np
from sys import maxsize

np.set_printoptions(threshold=maxsize)


class TestTunnel(unittest.TestCase):
    def setUp(self):
        from transport_tools.libs.networks import Tunnel
        from transport_tools.tests.units.data.data_networks import datasection0, datasection1, datasection2, \
            datasection3

        self.maxDiff = None
        parameters = {
            "min_tunnel_radius4clustering": 0,
            "min_tunnel_length4clustering": 5,
            "max_tunnel_curvature4clustering": 999,
            "tunnel_properties_quantile": 0.9,
            "md_label": "",
            "layer_thickness": 1.5
        }

        self.transform_mat = np.array([[-0.068, -0.539,  0.838,  -0.139],
                                       [-0.423,  0.777,  0.465, -28.982],
                                       [-0.903, -0.323, -0.282,  60.986],
                                       [0.0, 0.0, 0.0, 1.0]]
                                      )

        self.tun1 = Tunnel(parameters, self.transform_mat)
        self.tun1.fill_data(datasection0)
        self.tun2 = Tunnel(parameters, self.transform_mat)
        self.tun2.fill_data(datasection1)
        self.tun3 = Tunnel(parameters, self.transform_mat)
        self.tun3.fill_data(datasection2)
        self.tun4 = Tunnel(parameters, self.transform_mat)
        self.tun4.fill_data(datasection3)

    def test_create_from_data_section(self):
        from transport_tools.tests.units.data.data_networks import test_create_from_data_section_out

        self.assertEqual(test_create_from_data_section_out, str(self.tun1))

    def test_get_center_line(self):
        from transport_tools.tests.units.data.data_networks import test_get_center_line_out
        self.assertTrue(np.allclose(test_get_center_line_out, self.tun1.get_center_line(), atol=1e-7))

    def test_get_snapshot_id(self):
        self.assertEqual(1, self.tun1.get_snapshot_id())
        with self.assertRaises(RuntimeError):
            self.assertRaises(ValueError, self.tun1.get_snapshot_id, id_position=2)
            self.assertRaises(IndexError, self.tun1.get_snapshot_id, delimiter="_")
        self.assertEqual(2, self.tun2.get_snapshot_id())
        self.assertEqual(3590, self.tun3.get_snapshot_id())
        self.assertEqual(9881, self.tun4.get_snapshot_id())

    def test_get_points_data(self):
        from transport_tools.tests.units.data.data_networks import test_tunnel_get_points_data_out
        self.assertTrue(np.allclose(test_tunnel_get_points_data_out, self.tun1.get_points_data(), atol=1e-7))

    def test_get_csv_lines(self):
        from transport_tools.tests.units.data.data_networks import test_get_csv_lines_out

        for test_line, res_line in zip(test_get_csv_lines_out.split("\n"), self.tun1.get_csv_lines("md_label").split("\n")):
            for test_item, res_item in zip(test_line.split(","), res_line.split(",")):

                try:
                    self.assertAlmostEqual(float(test_item), float(res_item))
                except ValueError:
                    self.assertEqual(test_item, res_item)

    def test_is_same(self):
        self.assertFalse(self.tun1.is_same(self.tun2))
        self.assertTrue(self.tun1.is_same(self.tun1))
        self.assertFalse(self.tun3.is_same(self.tun2))
        self.assertFalse(self.tun3.is_same(self.tun4))
        self.assertTrue(self.tun4.is_same(self.tun4))

    def test_has_better_throughput(self):
        self.assertFalse(self.tun1.has_better_throughput(self.tun1))
        self.assertFalse(self.tun1.has_better_throughput(self.tun2))
        self.assertTrue(self.tun2.has_better_throughput(self.tun3))
        self.assertTrue(self.tun2.has_better_throughput(self.tun4))

    def test_get_parameters(self):
        self.assertSequenceEqual((13.935477752845646, 1.1096497696890928, 1.19923106015622, 0.5662269402609552),
                                 self.tun1.get_parameters())
        self.assertSequenceEqual((14.155197558747808, 1.594852937295866, 1.1566240802231509, 0.6443086380845451),
                                 self.tun2.get_parameters())

        self.assertSequenceEqual((14.37838512501541, 1.173287696427179, 1.1248349566005307, 0.5671650786843087),
                                 self.tun3.get_parameters())
        self.assertSequenceEqual((14.637354970685315, 1.5270199148154504, 1.2361080659231054, 0.6184593557197385),
                                 self.tun4.get_parameters())

    def test_does_tunnel_pass_filters(self):
        in_vals = [
            (0, 999, 0, 999, 0, 999), (14, 999, 0, 999, 0, 999), (0, 9, 0, 999, 0, 999),
            (0, 999, 2, 999, 0, 999), (0, 999, 0, 1.5, 0, 999), (0, 999, 0, 1, 0, 999),
            (0, 999, 0, 999, 0, 999), (0, 999, 0, 999, 1, 999), (0, 999, 0, 999, 2, 999),
            (0, 999, 0, 999, 0, 1)
        ]
        out_vals = [True, False, False, False, True, False, True, True, False, False]
        results = list()
        for in_val in in_vals:
            active_filters = {
                "length": (in_val[0], in_val[1]),
                "radius": (in_val[2], in_val[3]),
                "curvature": (in_val[4], in_val[5]),
                "min_sims_num": 1,
                "min_avg_snapshots_num": 1,
                "min_avg_water_events": -1,
                "min_avg_entry_events": -1,
                "min_avg_release_events": -1
            }
            results.append(self.tun1.does_tunnel_pass_filters(active_filters))

        self.assertListEqual(results, out_vals)

    def test_get_closest_sphere2coords(self):
        from transport_tools.tests.units.data.data_networks import test_get_closest_sphere2coords1, \
            test_get_closest_sphere2coords2, test_get_closest_sphere2coords3

        min_distance, closest_sphere = self.tun1.get_closest_sphere2coords([-9.47, -32.61, 55.17])
        self.assertAlmostEqual(test_get_closest_sphere2coords1[0], min_distance)
        self.assertTrue(np.allclose(test_get_closest_sphere2coords1[1], closest_sphere, atol=1e-7))

        min_distance, closest_sphere = self.tun1.get_closest_sphere2coords([-5.47, -35.61, 65.17])
        self.assertAlmostEqual(test_get_closest_sphere2coords2[0], min_distance)
        self.assertTrue(np.allclose(test_get_closest_sphere2coords2[1], closest_sphere, atol=1e-7))

        min_distance, closest_sphere = self.tun1.get_closest_sphere2coords([-5.47, -32.61, 55.17])
        self.assertAlmostEqual(test_get_closest_sphere2coords3[0], min_distance)
        self.assertTrue(np.allclose(test_get_closest_sphere2coords3[1], closest_sphere, atol=1e-7))

    def test_get_pdb_file_format(self):
        from transport_tools.tests.units.data.data_networks import test_get_pdb_file_format_out

        self.assertListEqual(test_get_pdb_file_format_out, self.tun1.get_pdb_file_format())


class TestTunnelCluster(unittest.TestCase):
    def setUp(self):
        from transport_tools.libs.networks import TunnelCluster, Tunnel
        from transport_tools.tests.units.data.data_networks import datasection0, datasection1, datasection2, \
            datasection3
        self.maxDiff = None

        self.parameters = {
            "min_tunnel_radius4clustering": 0,
            "min_tunnel_length4clustering": 5,
            "max_tunnel_curvature4clustering": 999,
            "tunnel_properties_quantile": 0.9,
            "layer_thickness": 1.5,
            "md_label": "e10s1_e9s3p0f1600",
            "snapshot_id_position": 1,
            "snapshot_delimiter": ".",
            "output_path": "",
            "layered_caver_vis_path": "",
            "orig_caver_vis_rel_path": "",
            "visualize_layered_clusters": False,
            "snapshots_per_simulation": 10000,
            "sp_radius": 0.5
        }

        self.transform_mat = np.array([[-0.068, -0.539,  0.838,  -0.139],
                                       [-0.423,  0.777,  0.465, -28.982],
                                       [-0.903, -0.323, -0.282,  60.986],
                                       [0.0, 0.0, 0.0, 1.0]]
                                      )
        self.sp = np.array([[42.789, 42.178, 30.813]])

        self.cluster = TunnelCluster(1, self.parameters, self.transform_mat, self.sp)
        tun = Tunnel(self.parameters, self.transform_mat)
        tun.fill_data(datasection0)
        self.cluster.add_tunnel(tun)
        tun = Tunnel(self.parameters, self.transform_mat)
        tun.fill_data(datasection1)
        self.cluster.add_tunnel(tun)
        tun = Tunnel(self.parameters, self.transform_mat)
        tun.fill_data(datasection2)
        self.cluster.add_tunnel(tun)
        tun = Tunnel(self.parameters, self.transform_mat)
        tun.fill_data(datasection3)
        self.cluster.add_tunnel(tun)

    def test_count_tunnels(self):
        self.assertEqual(4, self.cluster.count_tunnels())

    def test_is_same(self):
        from transport_tools.libs.networks import TunnelCluster, Tunnel
        from transport_tools.tests.units.data.data_networks import datasection0, datasection3

        tmp_cluster = TunnelCluster(2, self.parameters, self.transform_mat, self.sp)
        tun = Tunnel(self.parameters, self.transform_mat)
        tun.fill_data(datasection0)
        tmp_cluster.add_tunnel(tun)
        tun = Tunnel(self.parameters, self.transform_mat)
        tun.fill_data(datasection3)
        tmp_cluster.add_tunnel(tun)

        self.assertFalse(tmp_cluster.is_same(self.cluster))
        self.assertTrue(tmp_cluster.is_same(tmp_cluster))
        self.assertTrue(self.cluster.is_same(self.cluster))
        self.assertFalse(self.cluster.is_same(tmp_cluster))

    def test_get_subcluster(self):
        # three types: snap_ids, active_filters, and both
        active_filters = {
            "length": (0, 14.5),
            "radius": (0, 5),
            "curvature": (1, 10),
            "min_sims_num": 1,
            "min_avg_snapshots_num": 1,
            "min_avg_water_events": -1,
            "min_avg_entry_events": -1,
            "min_avg_release_events": -1
        }

        subcluster = self.cluster.get_subcluster(active_filters=active_filters)
        self.assertListEqual([1, 2, 3590], [*subcluster.tunnels.keys()])
        subcluster = self.cluster.get_subcluster(snap_ids=[1, 2, 5, 7])
        self.assertListEqual([1, 2], [*subcluster.tunnels.keys()])
        subcluster = self.cluster.get_subcluster(snap_ids=[1, 3590], active_filters=active_filters)
        self.assertListEqual([1, 3590], [*subcluster.tunnels.keys()])
        subcluster = self.cluster.get_subcluster(snap_ids=[3590], active_filters=active_filters)
        self.assertListEqual([3590], [*subcluster.tunnels.keys()])
        subcluster = self.cluster.get_subcluster(snap_ids=[9881, 9882], active_filters=active_filters)
        self.assertListEqual([], [*subcluster.tunnels.keys()])

    def test_get_property(self):
        active_filters = {
            "length": (0, 20),
            "radius": (0, 5),
            "curvature": (1, 10),
            "min_sims_num": 1,
            "min_avg_snapshots_num": 1,
            "min_avg_water_events": -1,
            "min_avg_entry_events": -1,
            "min_avg_release_events": -1
        }

        result = self.cluster.get_property("length", active_filters)
        self.assertTrue(np.allclose(np.array([13.93547775, 14.15519756, 14.37838513, 14.63735497]), result, atol=1e-7))
        result = self.cluster.get_property("bottleneck_radius", active_filters)
        self.assertTrue(np.allclose(np.array([1.10964977, 1.59485294, 1.1732877, 1.52701991]), result, atol=1e-7))

    def test_remove_tunnel(self):
        self.cluster.remove_tunnel(1)
        self.assertEqual(3, self.cluster.count_tunnels())

    def test_count_valid_tunnels(self):
        self.assertEqual(4, self.cluster.count_valid_tunnels())


class TestTransportEvent(unittest.TestCase):
    def setUp(self):
        from transport_tools.libs.geometry import Point
        from transport_tools.libs.networks import TransportEvent

        self.maxDiff = None
        parameters = {
            "tunnel_properties_quantile": 0.9,
            "layer_thickness": 1.5,
            "md_label": "e10s1_e9s3p0f1600",
            "output_path": "",
            "layered_aquaduct_vis_path": "",
            "orig_aquaduct_vis_path": "",
            "visualize_layered_events": False,
            "sp_radius": 0.5
        }

        self.transform_mat = np.array([[-0.068, -0.539,  0.838,  -0.139],
                                       [-0.423,  0.777,  0.465, -28.982],
                                       [-0.903, -0.323, -0.282,  60.986],
                                       [0.0, 0.0, 0.0, 1.0]]
                                      )

        self.inside = TransportEvent("inside", "raw_paths_1", parameters, "e10s1_e9s3p0f1600",
                                     ('WAT', 588, (9206, 9209), (9210, 9224)), self.transform_mat)
        self.inside.add_point(Point([-1.13, -5.07, -3.74], 6.41, 0.00))
        self.inside.add_point(Point([-1.56, -4.67, -3.70], 6.16, 0.00))
        self.inside.add_point(Point([-2.20, -6.91, -4.58], 8.58, 0.00))

        self.release = TransportEvent("release", "raw_paths_1", parameters, "e10s1_e9s3p0f1600",
                                      ('WAT', 588, (9206, 9209), (9210, 9224)), self.transform_mat)
        self.release.add_point(Point([-2.84, -9.14, -5.46], 11.02, 0.00))
        self.release.add_point(Point([-3.33, -9.83, -5.21], 11.61, 0.00))
        self.release.add_point(Point([-4.63, -8.80, -6.05], 11.64, 0.00))
        self.release.add_point(Point([-0.63, -8.21, -8.69], 11.98, 0.00))
        self.release.add_point(Point([0.68, -8.53, -10.63], 13.65, 0.00))
        self.release.add_point(Point([-0.43, -9.16, -9.71], 13.36, 0.00))
        self.release.add_point(Point([-0.39, -8.36, -8.65], 12.04, 0.00))
        self.release.add_point(Point([-0.10, -9.09, -9.46], 13.12, 0.00))
        self.release.add_point(Point([-2.83, -8.50, -5.15], 10.33, 0.00))
        self.release.add_point(Point([-2.61, -9.27, -5.67], 11.17, 0.00))
        self.release.add_point(Point([-0.42, -8.79, -9.33], 12.83, 0.00))
        self.release.add_point(Point([0.61, -10.84, -9.57], 14.47, 0.00))
        self.release.add_point(Point([2.21, -10.26, -11.00], 15.20, 0.00))
        self.release.add_point(Point([1.17, -10.71, -10.25], 14.87, 0.00))

        self.entry = TransportEvent("entry", "raw_paths_1", parameters, "e10s1_e9s3p0f1600",
                                    ('WAT', 588, (9206, 9209), (9210, 9224)), self.transform_mat)
        self.entry.add_point(Point([-4.18, -10.17, -6.83], 12.95, 0.00))
        self.entry.add_point(Point([-3.79, -9.25, -6.62], 11.99, 0.00))
        self.entry.add_point(Point([-1.00, -4.31, -5.51], 7.06, 0.00))
        self.entry.add_point(Point([-1.07, -4.69, -4.62], 6.67, 0.00))

    def test_get_min_distance(self):
        self.assertAlmostEqual(6.16, self.inside.get_min_distance())
        self.assertAlmostEqual(10.33, self.release.get_min_distance())
        self.assertAlmostEqual(6.67, self.entry.get_min_distance())

    def test_get_points_data(self):
        from transport_tools.tests.units.data.data_networks import test_event_get_points_data_out1, \
            test_event_get_points_data_out2, test_event_get_points_data_out3

        self.assertTrue(np.allclose(test_event_get_points_data_out1, self.inside.get_points_data(), atol=1e-7))
        self.assertTrue(np.allclose(test_event_get_points_data_out2, self.release.get_points_data(), atol=1e-7))
        self.assertTrue(np.allclose(test_event_get_points_data_out3, self.entry.get_points_data(), atol=1e-7))

    def test_get_min_distance2starting_point(self):
        self.assertAlmostEqual(6.158936596, self.inside.get_min_distance2starting_point())
        self.assertAlmostEqual(10.333508600, self.release.get_min_distance2starting_point())
        self.assertAlmostEqual(6.669737626, self.entry.get_min_distance2starting_point())

    def test_create_layered_event(self):
        from transport_tools.tests.units.data.data_networks import test_create_layered_event_pathset1, \
            test_create_layered_event_pathset2, test_create_layered_event_pathset3

        entity_label, md_label, layered_pathset = self.inside.create_layered_event()
        self.assertEqual("1_inside", entity_label)
        self.assertEqual("e10s1_e9s3p0f1600", md_label)
        self.assertEqual(test_create_layered_event_pathset1, str(layered_pathset))

        entity_label, md_label, layered_pathset = self.release.create_layered_event()
        self.assertEqual("1_release", entity_label)
        self.assertEqual("e10s1_e9s3p0f1600", md_label)
        self.assertEqual(test_create_layered_event_pathset2, str(layered_pathset))

        entity_label, md_label, layered_pathset = self.entry.create_layered_event()
        self.assertEqual("1_entry", entity_label)
        self.assertEqual("e10s1_e9s3p0f1600", md_label)
        self.assertEqual(test_create_layered_event_pathset3, str(layered_pathset))


class TestAquaductPath(unittest.TestCase):
    def setUp(self):
        from transport_tools.libs.networks import AquaductPath
        self.maxDiff = None
        parameters = {
            "event_min_distance": 6.0,
            "aqauduct_ligand_effective_radius": 1.0
        }
        self.transform_mat = np.array([[-0.068, -0.539,  0.838,  -0.139],
                                       [-0.423,  0.777,  0.465, -28.982],
                                       [-0.903, -0.323, -0.282,  60.986],
                                       [0.0, 0.0, 0.0, 1.0]]
                                      )

        self.path1 = AquaductPath("raw_paths_1", parameters, ('WAT', 588, (9206, 9209), (9210, 9224)),
                                  self.transform_mat,  "e10s1_e9s3p0f1600")
        self.path2 = AquaductPath("raw_paths_9", parameters, ('WAT', 8612, (7770, 7782), (7785, 7789)),
                                  self.transform_mat,  "e10s1_e9s3p0f1600", )
        self.path = AquaductPath("raw_path", parameters, ('WAT', 1, (1, 2), (2, 3)), self.transform_mat, "md_label")

    def test__parse_aquaduct_path(self):
        from transport_tools.tests.units.data.data_networks import test__parse_aquaduct_path_out1, \
            test__parse_aquaduct_path_out2, test__parse_aquaduct_path_out3, test__parse_aquaduct_path_out4, \
            test__parse_aquaduct_path_out5, test__parse_aquaduct_path_out6, cgo1, cgo2

        events = self.path1._parse_aquaduct_path(cgo1)
        self.assertEqual(3, len(events))
        self.assertEqual("entry", events[0].type)
        self.assertEqual("inside", events[1].type)
        self.assertEqual("release", events[2].type)
        self.assertTrue(np.allclose(test__parse_aquaduct_path_out1, events[0].get_points_data(), atol=1e-7))
        self.assertTrue(np.allclose(test__parse_aquaduct_path_out2, events[1].get_points_data(), atol=1e-7))
        self.assertTrue(np.allclose(test__parse_aquaduct_path_out3, events[2].get_points_data(), atol=1e-7))

        events = self.path2._parse_aquaduct_path(cgo2)
        self.assertEqual(3, len(events))
        self.assertEqual("entry", events[0].type)
        self.assertEqual("inside", events[1].type)
        self.assertEqual("release", events[2].type)
        self.assertTrue(np.allclose(test__parse_aquaduct_path_out4, events[0].get_points_data(), atol=1e-7))
        self.assertTrue(np.allclose(test__parse_aquaduct_path_out5, events[1].get_points_data(), atol=1e-7))
        self.assertTrue(np.allclose(test__parse_aquaduct_path_out6, events[2].get_points_data(), atol=1e-7))

    def test__get_continuously_closer_points(self):
        from transport_tools.tests.units.data.data_networks import list_of_points1, list_of_points2, list_of_points3

        include_points = self.path._get_continuously_closer_points(list_of_points1, 4.754082296583211)
        self.assertEqual(1, len(include_points))
        self.assertTrue(np.allclose(np.array([[0.3, -0.81, -0.55, 1.03, 0.0]]), include_points[0].data))

        include_points = self.path._get_continuously_closer_points(list_of_points2, 8.345964657716086)
        self.assertEqual(2, len(include_points))
        self.assertTrue(np.allclose(np.array([[-2.78, -4.36, -4.04, 6.56, 0.0]]), include_points[0].data))
        self.assertTrue(np.allclose(np.array([[-1.84, -2.34, -3.85, 4.87, 0.0]]), include_points[1].data))

        include_points = self.path._get_continuously_closer_points(list_of_points3, 1.6693291082253607)
        self.assertEqual(2, len(include_points))
        self.assertTrue(np.allclose(np.array([[0.06, -0.04, -0.06, 0.1, 0.0]]), include_points[0].data))
        self.assertTrue(np.allclose(np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]), include_points[1].data))

    def test__get_points_from_intersection_between_starting_and_border_points(self):
        from transport_tools.tests.units.data.data_networks import list_of_points1

        a, b = self.path._get_points_from_intersection_between_starting_and_border_points(list_of_points1,
                                                                                          list_of_points1[0])
        self.assertFalse(a)
        self.assertEqual(0, len(b))

        a, b = self.path._get_points_from_intersection_between_starting_and_border_points(list_of_points1,
                                                                                          list_of_points1[-1])
        self.assertFalse(a)
        self.assertEqual(4, len(b))
        self.assertTrue(np.allclose(np.array([[0.3, -0.81, -0.55, 1.03, 0.0]]), b[0].data))
        self.assertTrue(np.allclose(np.array([[-0.43, -2.07, -1.27, 2.47, 0.0]]), b[1].data))
        self.assertTrue(np.allclose(np.array([[-0.78, -4.81, -3.23, 5.85, 0.0]]), b[2].data))
        self.assertTrue(np.allclose(np.array([[-1.84, -2.34, -3.85, 4.87, 0.0]]), b[3].data))

        self.path.parameters["aqauduct_ligand_effective_radius"] = 1.5
        a, b = self.path._get_points_from_intersection_between_starting_and_border_points(list_of_points1,
                                                                                          list_of_points1[0])
        self.assertTrue(a)
        self.assertEqual(0, len(b))

    def test__find_overlapping_path2starting_point(self):
        from transport_tools.libs.geometry import Point
        from transport_tools.tests.units.data.data_networks import list_of_points1, list_of_points4

        outlist = self.path._find_overlapping_path2starting_point(list_of_points1[:2],
                                                                  Point([-1.84, -2.34, -3.85], 4.87, 0.00))
        self.assertEqual(0, len(outlist))

        outlist = self.path._find_overlapping_path2starting_point([], Point([-1.84, -2.34, -3.85], 4.87, 0.00))
        self.assertEqual(0, len(outlist))

        outlist = self.path._find_overlapping_path2starting_point(list_of_points4,
                                                                  Point([0.40, -2.78, 4.85], 5.61, 0.00))
        self.assertEqual(1, len(outlist))
        self.assertTrue(np.allclose(np.array([[0.09, -2.88, 4.79, 5.59, 0.0]]), outlist[0].data))

    def test__get_path(self):
        from transport_tools.tests.units.data.data_networks import list_of_points1, list_of_points4

        path_points = self.path._get_path({0: 'BP', 1: 'BP', 2: 'BP', 'SP': 2}, 1, list_of_points4)
        self.assertEqual(1, len(path_points))
        self.assertTrue(np.allclose(np.array([[0.09, -2.88, 4.79, 5.59, 0.0]]), path_points[0].data))

        path_points = self.path._get_path({0: 'BP', 1: 'BP', 2: 'BP', 'SP': 2}, 'SP', list_of_points4)
        self.assertEqual(2, len(path_points))
        self.assertTrue(np.allclose(np.array([[0.67, -2.40, 4.93, 5.52, 0.00]]), path_points[0].data))
        self.assertTrue(np.allclose(np.array([[0.00, 0.00, 0.00, 0.00, 0.00]]), path_points[1].data))

        path_points = self.path._get_path({}, 'BP', list_of_points1[:2])
        self.assertEqual(0, len(path_points))

    def test__get_distance2starting_point(self):
        from transport_tools.tests.units.data.data_networks import list_of_points4

        self.assertAlmostEqual(5.49, self.path._get_distance2starting_point(0, list_of_points4))
        self.assertAlmostEqual(5.59, self.path._get_distance2starting_point(1, list_of_points4))
        self.assertAlmostEqual(5.52, self.path._get_distance2starting_point(2, list_of_points4))
        self.assertAlmostEqual(0, self.path._get_distance2starting_point('SP', list_of_points4))

    def test__get_cheapest_point(self):
        from transport_tools.tests.units.data.data_networks import full_scores

        self.assertEqual(1, self.path._get_cheapest_point({1}, full_scores))
        self.assertEqual(0, self.path._get_cheapest_point({0, 1, 2}, full_scores))
        self.assertEqual(2, self.path._get_cheapest_point({1, 2}, full_scores))
        self.assertEqual('BP', self.path._get_cheapest_point({'BP'}, full_scores))
        self.assertEqual('SP', self.path._get_cheapest_point({'SP'}, full_scores))

    def test__compute_network_of_overlapping_points(self):
        from transport_tools.libs.geometry import Point
        from transport_tools.tests.units.data.data_networks import list_of_points1, list_of_points4

        network = self.path._compute_network_of_overlapping_points(list_of_points1[:2],
                                                                   Point([-1.84, -2.34, -3.85], 4.87, 0.00))
        self.assertDictEqual({'BP': set(), 'SP': set(), 0: {1}, 1: {0}}, network)
        network = self.path._compute_network_of_overlapping_points(list_of_points4,
                                                                   Point([0.40, -2.78, 4.85], 5.61, 0.00))
        self.assertDictEqual({'BP': {0, 1, 2}, 'SP': set(), 0: {1, 2, 'BP'}, 1: {0, 2, 'BP'}, 2: {0, 1, 'BP'}}, network)

        self.path.parameters["aqauduct_ligand_effective_radius"] = 1.5
        network = self.path._compute_network_of_overlapping_points(list_of_points1[:2],
                                                                   Point([-1.84, -2.34, -3.85], 4.87, 0.00))
        self.assertDictEqual({'BP': {1}, 'SP': {0}, 0: {1, 'SP'}, 1: {0, 'BP'}}, network)


class TestHelpers(unittest.TestCase):
    def test_define_filters(self):
        from transport_tools.libs.networks import define_filters
        from transport_tools.tests.units.data.data_networks import filter_out1, filter_out2

        self.assertDictEqual(filter_out1, define_filters())
        self.assertDictEqual(filter_out2, define_filters(min_bottleneck_radius=2, min_avg_snapshots_num=500,
                                                         max_length=20, min_length=10, max_curvature=5))
        self.assertRaises(ValueError, define_filters, min_curvature=5, max_curvature=2)


if __name__ == "__main__":
    unittest.main()
