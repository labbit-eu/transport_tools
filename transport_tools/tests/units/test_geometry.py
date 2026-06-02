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
import numpy as np
from sys import maxsize

np.set_printoptions(threshold=maxsize)


class TestPoint(unittest.TestCase):
    def setUp(self):
        from transport_tools.libs.geometry import Point

        self.point1 = Point([1, 1, 1])
        self.point2 = Point([2, 2, 2])
        self.point3 = Point([0, 0, 0])
        self.point4 = Point([-1, 0, 0])

    def test_str(self):
        self.assertEqual("(x, y, z, dist, radius) = (1.00, 1.00, 1.00, -1.00, 0.00)", str(self.point1))

    def test_distance2point(self):
        self.assertAlmostEqual(1.7320508075688772, self.point1.distance2point(self.point2))
        self.assertAlmostEqual(1.7320508075688772, self.point1.distance2point(self.point3))
        self.assertEqual(0, self.point1.distance2point(self.point1))
        self.assertEqual(1, self.point3.distance2point(self.point4))

    def test_convert2viz_atom(self):
        self.assertEqual("HETATM    0  UNK UNK T   0       1.000   1.000   1.000\n",
                         str(self.point1.convert2viz_atom(0, 0)))
        self.assertEqual("HETATM    0  UNK AKA T   0       2.000   2.000   2.000\n",
                         str(self.point2.convert2viz_atom(0, 0, "AKA")))


class TestPointMatrix(unittest.TestCase):
    def setUp(self):
        from transport_tools.libs.geometry import PointMatrix
        from transport_tools.tests.units.data.data_geometry import test_matrix_input1, test_matrix_input2

        self.matrix0 = PointMatrix(np.array([]))
        self.matrix1 = PointMatrix(test_matrix_input1)
        self.matrix2 = PointMatrix(test_matrix_input2)

    def test_is_empty(self):
        self.assertTrue(self.matrix0.is_empty())
        self.assertFalse(self.matrix1.is_empty())
        self.assertFalse(self.matrix2.is_empty())

    def test_get_functions(self):
        self.assertRaises(IndexError, self.matrix0.get_num_columns)
        self.assertEqual(7, self.matrix1.get_num_columns())
        self.assertEqual(0, self.matrix0.get_num_points())
        self.assertEqual(2, self.matrix1.get_num_points())
        self.assertEqual(3, self.matrix2.get_num_points())
        self.assertTrue(np.allclose(self.matrix1.get_radii(), np.array([1.47279732, 1.84011172]), atol=1e-3))
        self.assertTrue(np.all(self.matrix2.get_start_points_indexing() == np.array([True, False, False])))
        self.assertTrue(np.all(self.matrix1.get_end_points_indexing() == np.array([True, True])))
        self.assertTrue(np.all(self.matrix2.get_end_points_indexing() == np.array([False, False, False])))
        self.assertTrue(np.all(self.matrix1.get_tunnels_ids() == np.array([5, 9])))
        self.assertTrue(np.all(self.matrix2.get_tunnels_ids() == np.array([11, 11, 11])))
        self.assertTrue(np.all(self.matrix1.get_points_ids4tunnel(5) == np.array([94])))
        self.assertTrue(np.all(self.matrix2.get_points_ids4tunnel(11) == np.array([0, 4, 5])))

    def test_coords(self):
        new_coords1 = np.array([[-999, 999, -999], [-999, -999, 999]])
        new_coords2 = np.array([[999, 0, 999], [999, 999, 9099], [-1, 999, 999]])
        self.matrix1.alter_coords(new_coords1)
        self.assertTrue(np.allclose(self.matrix1.get_coords(), new_coords1, atol=1e-3))
        self.matrix2.alter_coords(new_coords2)
        self.assertTrue(np.allclose(self.matrix2.get_coords(), new_coords2, atol=1e-3))


class TestClusterInLayer(unittest.TestCase):
    def setUp(self):
        from transport_tools.libs.geometry import ClusterInLayer
        from transport_tools.tests.units.data.data_geometry import test_cluster_point_mat1, test_cluster_point_mat2, \
            test_cluster_point_mat3, test_cluster_point_mat4

        self.cluster1 = ClusterInLayer(test_cluster_point_mat1, 1.5, 0.9, False, 1, 0)
        self.cluster2 = ClusterInLayer(test_cluster_point_mat2, 1.5, 0.9, False, 1, 4)
        self.cluster3 = ClusterInLayer(test_cluster_point_mat3, 1.5, 0.9, True, 2, 11)
        self.cluster4 = ClusterInLayer(test_cluster_point_mat4, 1.5, 0.9, False, 2, 7)

    def test_get_node_label(self):
        self.assertEqual("0_1", self.cluster1.get_node_label())
        self.assertEqual("4_1", self.cluster2.get_node_label())
        self.assertEqual("11_2", self.cluster3.get_node_label())

    def test_is_representative(self):
        self.assertTrue(self.cluster1.is_representative())
        self.assertTrue(self.cluster2.is_representative())
        self.assertTrue(self.cluster3.is_representative())
        self.assertFalse(self.cluster4.is_representative())

    def test_merge_with_cluster(self):
        from transport_tools.tests.units.data.data_geometry import test_cluster_merged_coors
        self.assertFalse(self.cluster1.end_point)
        self.cluster1.merge_with_cluster(self.cluster3)
        self.assertTrue(np.allclose(self.cluster1.get_coords(), test_cluster_merged_coors, atol=1e-3))
        self.assertTrue(self.cluster1.end_point)
        self.assertEqual(6, self.cluster1.num_points)
        self.assertSetEqual({0, 1, 2}, self.cluster1.tunnel_ids)

    def test_compute_averages(self):
        self.assertTrue(np.allclose(self.cluster1.average, np.array([0.10043729397500001, 0.15234930799425,
                                                                     0.35513633175000003]), atol=1e-3))
        self.assertAlmostEqual(0.559016979, self.cluster1.rmsf)
        self.assertAlmostEqual(1.900842161, self.cluster1.radius)
        self.assertEqual(0, self.cluster1.num_end_points)
        self.assertTrue(self.cluster1.start_point)

        self.assertTrue(np.allclose(self.cluster3.average, np.array([11.809807613289406, -4.5125658372592135,
                                                                     11.09742654016791]), atol=1e-3))
        self.assertAlmostEqual(0.191486341, self.cluster3.rmsf)
        self.assertAlmostEqual(1.740502133, self.cluster3.radius)
        self.assertEqual(2, self.cluster3.num_end_points)
        self.assertFalse(self.cluster3.start_point)


class TestLayeredPathSet(unittest.TestCase):
    def setUp(self):
        from transport_tools.libs.geometry import LayeredPathSet
        from transport_tools.tests.units.data.data_geometry import test_layers_tun, test_layers_event

        self.params = {
            "use_cluster_spread": False,
            "clustering_max_num_rep_frag": 0,
            "directional_cutoff": 1.5707963267948966,
            "random_seed": 4,
            "sp_radius": 0.5,
            "calculate_exact_path_distances": False
        }

        self.node_path_tun = ['0_1', '1_1', '2_1', '3_1', '4_1', '5_1', '6_2', '7_1', '8_1', '8_3', '7_3', '6_3', '5_2',
                              '5_3', '6_1', '7_2', '7_4', '8_2', '9_1']

        self.node_path_event = ['0_0', '1_0', '4_0', '5_0', '7_0']
        self.pathset_tun = LayeredPathSet("Cluster_44", "e10s1_e9s3p0f1600", self.params,
                                          starting_point_coords=np.array([0., 0., 0.]))
        self.pathset_tun.add_node_path(self.node_path_tun, test_layers_tun)
        self.pathset_event = LayeredPathSet("2_release", "e10s1_e9s3p0f1600", self.params, starting_point_coords=None)
        self.pathset_event.add_node_path(self.node_path_event, test_layers_event)
        self.pathset_event.set_traced_event(('WAT', 1769, (908, 914), (2594, 2595)))
        self.merged_pathset = self.pathset_tun + self.pathset_tun
        self.merged_pathset.remove_duplicates()
        self.merged_pathset.compute_node_depths()

    def test_is_same(self):
        self.assertTrue(self.pathset_tun.is_same(self.pathset_tun))

    def test_str(self):
        from transport_tools.tests.units.data.data_geometry import test_pathset_tun_str, test_pathset_event_str, \
            test_merged_pathset_str

        self.assertEqual(test_pathset_tun_str, str(self.pathset_tun))
        self.assertEqual(test_pathset_event_str, str(self.pathset_event))
        self.assertEqual("('WAT:1769', (2594, 2595))", str(self.pathset_event.traced_event))
        self.assertEqual(test_merged_pathset_str, str(self.merged_pathset))

    def test__get_direction(self):
        self.assertTrue(np.allclose([-3.1848191599444857, 8.731479931193862, 10.643383638680215],
                                    self.pathset_tun._get_direction(), atol=1e-3))
        self.assertTrue(np.allclose([-3.0059155769984187, -9.277677638382618, -5.4319921074718565],
                                    self.pathset_event._get_direction(), atol=1e-3))

    def test__get_extended_labels(self):
        from transport_tools.tests.units.data.data_geometry import test_pathset_extended_labels4paths1, \
            test_pathset_extended_labels4paths2, test_pathset_extended_labels4paths3, \
            test_pathset_extended_labels4nodes1, test_pathset_extended_labels4nodes2, \
            test_pathset_extended_labels4nodes3

        self.assertListEqual(test_pathset_extended_labels4paths1, self.pathset_tun._get_extended_labels4paths())
        self.assertListEqual(test_pathset_extended_labels4paths2, self.pathset_event._get_extended_labels4paths())
        self.assertListEqual(test_pathset_extended_labels4paths3, self.merged_pathset._get_extended_labels4paths())
        self.assertListEqual(test_pathset_extended_labels4nodes1, self.pathset_tun._get_extended_labels4nodes())
        self.assertListEqual(test_pathset_extended_labels4nodes2, self.pathset_event._get_extended_labels4nodes())
        self.assertListEqual(test_pathset_extended_labels4nodes3, self.merged_pathset._get_extended_labels4nodes())

    def test__get_most_complete_shortest_path_from_ids(self):
        from transport_tools.tests.units.data.data_geometry import test_layers_event
        self.assertEqual(0, self.pathset_tun._get_most_complete_shortest_path_from_ids([0]))
        self.pathset_tun.add_node_path(self.node_path_event, test_layers_event)
        self.assertEqual(1, self.pathset_tun._get_most_complete_shortest_path_from_ids([0, 1]))

    def test__get_adjacent_nodes_data(self):
        from transport_tools.tests.units.data.data_geometry import test_pathset_adjacent_nodes1, \
            test_pathset_adjacent_nodes2, test_pathset_adjacent_nodes3, test_pathset_adjacent_nodes4, \
            test_pathset_adjacent_data1, test_pathset_adjacent_data2, test_pathset_adjacent_data3, \
            test_pathset_adjacent_data4

        query_node_data = np.array([4.59581293, 1.16044215, 7.02264766, 5., 0., 1.30898044, 0.82296819])
        nodes1, data1 = self.pathset_tun._get_adjacent_nodes_data(query_node_data, 8, 10)
        self.assertListEqual(test_pathset_adjacent_nodes1, nodes1.tolist())
        self.assertTrue(np.allclose(test_pathset_adjacent_data1, data1, atol=1e-3))

        nodes2, data2 = self.pathset_tun._get_adjacent_nodes_data(query_node_data, 5, 5)
        self.assertListEqual(test_pathset_adjacent_nodes2, nodes2.tolist())
        self.assertTrue(np.allclose(test_pathset_adjacent_data2, data2, atol=1e-3))

        query_node_data = np.array([4.59581293, 1.16044215, 7.02264766, 5., 1., 1.30898044, 0.82296819])
        nodes3, data3 = self.pathset_tun._get_adjacent_nodes_data(query_node_data, 8, 10)
        self.assertListEqual(test_pathset_adjacent_nodes3, nodes3.tolist())
        self.assertTrue(np.allclose(test_pathset_adjacent_data3, data3, atol=1e-3))

        query_node_data = np.array([11.381985419714198, -3.0926934136428694, 10.140161516790034, 10.0, 1.0,
                                    1.7704778497877127, 0.7225262721915131])
        nodes4, data4 = self.pathset_tun._get_adjacent_nodes_data(query_node_data, 15, 16)

        self.assertListEqual(test_pathset_adjacent_nodes4, nodes4.tolist())
        self.assertTrue(np.allclose(test_pathset_adjacent_data4, data4, atol=1e-3))

    def test__compute_distances(self):
        from transport_tools.tests.units.data.data_geometry import test_pathset_zipobject1, test_pathset_zipobject2, \
            test_pathset_zipobject3

        node_data = np.array([4.59581293, 1.16044215, 7.02264766, 5., 0., 1.30898044, 0.82296819])
        first_terminal_layer = self.merged_pathset._get_first_terminal_layer()

        results = self.merged_pathset._compute_distances(node_data, 9, first_terminal_layer, self.pathset_tun)
        for out_val, result in zip(test_pathset_zipobject1, results):
            self.assertEqual(out_val[0], result[0])
            for i, j in zip(out_val[1:], result[1:]):
                self.assertAlmostEqual(i, j)

        results = self.merged_pathset._compute_distances(node_data, 9, first_terminal_layer, self.pathset_event,
                                                         consider_rmsf=False)
        for out_val, result in zip(test_pathset_zipobject2, results):
            self.assertEqual(out_val[0], result[0])
            for i, j in zip(out_val[1:], result[1:]):
                self.assertAlmostEqual(i, j)

        results = self.merged_pathset._compute_distances(node_data, 9, first_terminal_layer, self.pathset_event,
                                                         consider_rmsf=True)
        for out_val, result in zip(test_pathset_zipobject3, results):
            self.assertEqual(out_val[0], result[0])
            for i, j in zip(out_val[1:], result[1:]):
                self.assertAlmostEqual(i, j)

    def test__get_first_terminal_layer(self):
        self.assertEqual(9.0, self.pathset_tun._get_first_terminal_layer())
        self.assertEqual(7.0, self.pathset_event._get_first_terminal_layer())
        self.assertEqual(9.0, self.merged_pathset._get_first_terminal_layer())

    def test__get_terminal_node_labels(self):
        self.assertEqual(np.array(['9_1']), self.pathset_tun._get_terminal_node_labels())
        self.assertEqual(np.array(['7_0']), self.pathset_event._get_terminal_node_labels())

    def test__get_path_fragments(self):
        from transport_tools.libs.geometry import LayeredPathSet
        from transport_tools.tests.units.data.data_geometry import test_pathset_fragments1, test_pathset_fragments2
        path = np.array(['SP', '0_1', '1_1', '2_1', '3_1', '4_1', '5_1', '6_2', '7_1', '8_1', '8_3', '7_3', '6_3',
                         '5_2', '5_3', '6_1', '7_2', '7_4', '8_2', '9_1'])

        out_fragments = self.pathset_tun._get_path_fragments(path, {'9_1'})
        self.assertListEqual(test_pathset_fragments1, out_fragments)
        out_fragments = self.pathset_tun._get_path_fragments(path, {'3_1', '9_1'})
        self.assertListEqual(test_pathset_fragments2, out_fragments)

        tmp_parms = self.params.copy()
        tmp_parms["clustering_max_num_rep_frag"] = 0
        pathset = LayeredPathSet("Cluster_X", "e10s1_e9s3p0f1600", tmp_parms,
                                 starting_point_coords=np.array([0., 0., 0.]))
        out_fragments = pathset._get_path_fragments(path, {'2_1', '3_1', '5_1', '8_1', '6_1', '9_1'})
        self.assertEqual(6, len(out_fragments))

        for i in range(1, 7):
            tmp_parms["clustering_max_num_rep_frag"] = i
            pathset = LayeredPathSet("Cluster_X", "e10s1_e9s3p0f1600", tmp_parms,
                                     starting_point_coords=np.array([0., 0., 0.]))
            out_fragments = pathset._get_path_fragments(path, {'2_1', '3_1', '5_1', '8_1', '6_1', '9_1'})
            self.assertEqual(i, len(out_fragments))

        tmp_parms["clustering_max_num_rep_frag"] = 90
        pathset = LayeredPathSet("Cluster_X", "e10s1_e9s3p0f1600", tmp_parms,
                                 starting_point_coords=np.array([0., 0., 0.]))
        out_fragments = pathset._get_path_fragments(path, {'2_1', '3_1', '5_1', '8_1', '6_1', '9_1'})
        self.assertEqual(6, len(out_fragments))

    def test_get_fragmented_paths(self):
        self.assertSequenceEqual((1, [[['SP', '0_1', '1_1', '2_1', '3_1', '4_1', '5_1', '6_2', '7_1', '8_1', '8_3',
                                        '7_3', '6_3', '5_2', '5_3', '6_1', '7_2', '7_4', '8_2', '9_1']]]),
                                 self.pathset_tun.get_fragmented_paths())

    def test__get_dist2closest_node(self):
        from transport_tools.tests.units.data.data_geometry import test_pathset_dist_mat, \
            test_pathset_dists2closest_node

        for i, label in enumerate(self.pathset_tun.node_labels):
            min_dist, min_label = self.pathset_tun._get_dist2closest_node(label, test_pathset_dist_mat)
            out_dist, out_label = test_pathset_dists2closest_node[i]
            self.assertEqual(out_label, min_label)
            self.assertAlmostEqual(out_dist, min_dist)

        self.assertRaises(KeyError, self.pathset_tun._get_dist2closest_node, "SP", test_pathset_dist_mat,
                          nodes_subset={"1_1"})
        min_dist, min_label = self.pathset_tun._get_dist2closest_node("0_1", test_pathset_dist_mat,
                                                                      nodes_subset={"1_1"})
        self.assertEqual("1_1", min_label)
        self.assertAlmostEqual(0, min_dist)

        min_dist, min_label = self.pathset_tun._get_dist2closest_node("3_1", test_pathset_dist_mat,
                                                                      nodes_subset={"4_1"})
        self.assertEqual("4_1", min_label)
        self.assertAlmostEqual(8.990838392, min_dist)

    def test_how_much_is_inside(self):
        from transport_tools.tests.units.data.data_geometry import test_layers_event
        from transport_tools.libs.geometry import LayeredPathSet

        buriedness, max_depth = self.pathset_event.how_much_is_inside(self.merged_pathset)
        self.assertAlmostEqual(0.4, buriedness)
        self.assertAlmostEqual(0.5, max_depth)

        tmp_parms = self.params.copy()
        tmp_parms["use_cluster_spread"] = True
        pathset_event = LayeredPathSet("2_release", "e10s1_e9s3p0f1600", tmp_parms, starting_point_coords=None)
        pathset_event.add_node_path(self.node_path_event, test_layers_event)
        pathset_event.set_traced_event(('WAT', 1769, (908, 914), (2594, 2595)))
        buriedness, max_depth = pathset_event.how_much_is_inside(self.merged_pathset)
        self.assertAlmostEqual(0.6, buriedness)
        self.assertAlmostEqual(0.5, max_depth)

    def test_avg_distance2path_set(self):
        from transport_tools.libs.geometry import LayeredPathSet
        from transport_tools.tests.units.data.data_geometry import test_pathset_event2_nodes_data, test_layers_tun

        pathset_event2 = LayeredPathSet("2_release", "e10s1_e9s3p0f1600", self.params, starting_point_coords=None)
        pathset_event2.node_paths = [np.array(['0_0', '1_0', '4_0', '5_0', '7_0'])]
        pathset_event2.nodes_data = test_pathset_event2_nodes_data
        pathset_event2.node_labels = ['0_0', '1_0', '4_0', '5_0', '7_0']

        self.assertAlmostEqual(999, self.pathset_tun.avg_distance2path_set(self.pathset_event))
        self.assertAlmostEqual(999, self.pathset_tun.avg_distance2path_set(pathset_event2))
        self.assertAlmostEqual(0.0, self.pathset_tun.avg_distance2path_set(self.pathset_tun))
        self.assertAlmostEqual(5.615857478, self.pathset_event.avg_distance2path_set(pathset_event2))

        tmp_parms = self.params.copy()
        tmp_parms["directional_cutoff"] = np.pi*2/3
        pathset_tun = LayeredPathSet("Cluster_44", "e10s1_e9s3p0f1600", tmp_parms,
                                     starting_point_coords=np.array([0., 0., 0.]))
        pathset_tun.add_node_path(self.node_path_tun, test_layers_tun)

        self.assertAlmostEqual(8.138099648, pathset_tun.avg_distance2path_set(pathset_event2, dist_type=0))
        self.assertAlmostEqual(6.843536853, pathset_tun.avg_distance2path_set(pathset_event2, dist_type=1))
        self.assertAlmostEqual(6.843536853, pathset_tun.avg_distance2path_set(pathset_event2, dist_type=2))
        self.assertAlmostEqual(999, pathset_tun.avg_distance2path_set(pathset_event2, distance_cutoff=2))

    def test_avg_distance2path_set_matches_reference(self):
        """The vectorized avg_distance2path_set must match the pure-Python reference implementation."""
        from transport_tools.libs.geometry import LayeredPathSet
        from transport_tools.tests.units.data.data_geometry import test_pathset_event2_nodes_data, test_layers_tun

        pathset_event2 = LayeredPathSet("2_release", "e10s1_e9s3p0f1600", self.params, starting_point_coords=None)
        pathset_event2.node_paths = [np.array(['0_0', '1_0', '4_0', '5_0', '7_0'])]
        pathset_event2.nodes_data = test_pathset_event2_nodes_data
        pathset_event2.node_labels = ['0_0', '1_0', '4_0', '5_0', '7_0']

        exact_parms = self.params.copy()
        exact_parms["calculate_exact_path_distances"] = True
        pathset_tun_exact = LayeredPathSet("Cluster_44", "e10s1_e9s3p0f1600", exact_parms,
                                           starting_point_coords=np.array([0., 0., 0.]))
        pathset_tun_exact.add_node_path(self.node_path_tun, test_layers_tun)

        # covers overlapping (self pairs), directionally misaligned, far-away (cutoff) and exact-distance cases
        pathsets = [self.pathset_tun, self.pathset_event, self.merged_pathset, pathset_event2, pathset_tun_exact]
        for query in pathsets:
            for other in pathsets:
                for dist_type in (0, 1, 2):
                    for cutoff in (999, 2):
                        reference = query._avg_distance2path_set_reference(other, cutoff, dist_type)
                        vectorized = query.avg_distance2path_set(other, cutoff, dist_type)
                        self.assertAlmostEqual(reference, vectorized, places=7,
                                               msg=f"{query} vs {other}, dist_type={dist_type}, cutoff={cutoff}")


class TestLayers(unittest.TestCase):
    def setUp(self):
        from transport_tools.libs.geometry import Layer4Tunnels
        params = {
            "tunnel_properties_quantile": 0.9,
            "random_seed": 4,
            "caver_foldername": "caver"
        }

        self.layer1 = Layer4Tunnels(0, 1.5, params, "Cluster_44", "e10s1_e9s3p0f1600")
        self.layer2 = Layer4Tunnels(4, 1.5, params, "Cluster_44", "e10s1_e9s3p0f1600")
        self.layer3 = Layer4Tunnels(9, 1.5, params, "Cluster_44", "e10s1_e9s3p0f1600")
        self.layer4 = Layer4Tunnels(7, 1.5, params, "Cluster_41", "e10s1_e9s3p0f1600")

    def test_cluster_data(self):
        from transport_tools.tests.units.data.data_geometry import test_layers_points_mat1, test_layers_points_mat2, \
            test_layers_points_mat3, test_layers_points_mat4

        self.layer1.cluster_data(test_layers_points_mat1)
        self.layer2.cluster_data(test_layers_points_mat2)
        self.layer3.cluster_data(test_layers_points_mat3)
        self.layer4.cluster_data(test_layers_points_mat4)

        self.assertEqual(4, self.layer1.num_points)
        self.assertEqual(6, self.layer2.num_points)
        self.assertEqual(5, self.layer3.num_points)
        self.assertEqual(74, self.layer4.num_points)

        self.assertSequenceEqual([0], [*self.layer1.clusters.keys()])
        self.assertSequenceEqual([0], [*self.layer2.clusters.keys()])
        self.assertSequenceEqual([0, 1], [*self.layer3.clusters.keys()])
        self.assertSequenceEqual([0, 1, 2], [*self.layer4.clusters.keys()])

        self.assertTrue(np.allclose(np.array([0.100437294,  0.152349308,  0.355136331]),
                                    self.layer1.clusters[0].average, atol=1e-3))
        self.assertTrue(np.allclose(np.array([1.634404038, -6.250796876, -0.300136259]),
                                    self.layer2.clusters[0].average, atol=1e-3))
        self.assertTrue(np.allclose(np.array([-3.025853592,  8.640077626, 10.519369731]),
                                    self.layer3.clusters[0].average, atol=1e-3))
        self.assertTrue(np.allclose(np.array([-3.820681430,  9.097089150, 11.139439270]),
                                    self.layer3.clusters[1].average, atol=1e-3))
        # Spatial sorting changed cluster IDs: sorted by z, y, x coordinates
        self.assertTrue(np.allclose(np.array([-0.834558695,  8.588890506,  6.174215682]),
                                    self.layer4.clusters[0].average, atol=1e-3))
        self.assertTrue(np.allclose(np.array([0.282124062,  1.284429191, 11.041600659]),
                                    self.layer4.clusters[1].average, atol=1e-3))
        self.assertTrue(np.allclose(np.array([4.711509259, -2.706210238,  8.762292707]),
                                    self.layer4.clusters[2].average, atol=1e-3))

    def test__cluster_data(self):
        from transport_tools.tests.units.data.data_geometry import test_layers_points_mat1, test_layers_points_mat2, \
            test_layers_points_mat3

        self.assertListEqual([1, 1, 1, 2], self.layer1._cluster_data(test_layers_points_mat1).tolist())
        self.assertListEqual([2, 2, 2, 1, 1, 1], self.layer2._cluster_data(test_layers_points_mat2).tolist())
        self.assertListEqual([2, 2, 1, 1, 3], self.layer3._cluster_data(test_layers_points_mat3).tolist())

    def test__out_filtering(self):
        from transport_tools.tests.units.data.data_geometry import test_layers_ouliers, test_layers_ouliers_point_coords
        self.assertListEqual(test_layers_ouliers, self.layer1._out_filtering(test_layers_ouliers_point_coords).tolist())


class TestLayeredRepresentation(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        from transport_tools.libs.geometry import LayeredRepresentationOfTunnels, LayeredRepresentationOfEvents
        from transport_tools.tests.units.data.data_geometry import test_layered_repre_tunnel_points_data1, \
            test_layered_repre_tunnel_points_data2, test_layered_repre_tunnel_points_data3, \
            test_layered_repre_tunnel_points_data4, test_layered_repre_tunnel_points_data5

        params = {
            "tunnel_properties_quantile": 0.9,
            "output_path": "",
            "layered_caver_vis_path": "",
            "layered_aquaduct_vis_path": "",
            "orig_caver_vis_path": "",
            "orig_aquaduct_vis_path": "",
            "layer_thickness": 1.5,
            "md_label": "e10s1_e9s3p0f1600",
            "sp_radius": 0.5,
            "caver_foldername": "caver"
        }

        self.transform_mat = np.array([[-0.068, -0.539,  0.838,  -0.139],
                                       [-0.423,  0.777,  0.465, -28.982],
                                       [-0.903, -0.323, -0.282,  60.986],
                                       [0.0, 0.0, 0.0, 1.0]]
                                      )
        self.sp = np.array([[42.789, 42.178, 30.813]])

        self.repre1 = LayeredRepresentationOfTunnels(params, "Cluster_34")
        self.repre1.points_mat = test_layered_repre_tunnel_points_data1
        self.repre2 = LayeredRepresentationOfTunnels(params, "Cluster_34")
        self.repre2.points_mat = test_layered_repre_tunnel_points_data2
        self.repre3 = LayeredRepresentationOfTunnels(params, "Cluster_34")
        self.repre3.points_mat = test_layered_repre_tunnel_points_data3
        self.repre4 = LayeredRepresentationOfTunnels(params, "Cluster_34")
        self.repre4.points_mat = test_layered_repre_tunnel_points_data4
        self.repre5 = LayeredRepresentationOfEvents(params, "2_release")
        self.repre5.points_mat = test_layered_repre_tunnel_points_data5

        self.merged_repre = LayeredRepresentationOfTunnels(params, "Cluster_34")
        self.merged_repre += self.repre1
        self.merged_repre += self.repre2
        self.merged_repre += self.repre3
        self.merged_repre += self.repre4

        self.repre1.split_points_to_layers()
        self.repre2.split_points_to_layers()
        self.repre3.split_points_to_layers()
        self.repre4.split_points_to_layers()
        self.repre5.split_points_to_layers()
        self.merged_repre.split_points_to_layers()

    def test__assign_entity_points2clusters(self):
        from transport_tools.tests.units.data.data_geometry import test_layered_repre_entity_points2clusters1, \
            test_layered_repre_entity_points2clusters2, test_layered_repre_entity_points2clusters3, \
            test_layered_repre_entity_points2clusters4, test_layered_repre_entity_points2clusters5, \
            test_layered_repre_entity_points2clusters6
        self.assertDictEqual(test_layered_repre_entity_points2clusters1,
                             self.repre1._assign_entity_points2clusters())
        self.assertDictEqual(test_layered_repre_entity_points2clusters2,
                             self.repre2._assign_entity_points2clusters())
        self.assertDictEqual(test_layered_repre_entity_points2clusters3,
                             self.repre3._assign_entity_points2clusters())
        self.assertDictEqual(test_layered_repre_entity_points2clusters4,
                             self.repre4._assign_entity_points2clusters())
        self.assertDictEqual(test_layered_repre_entity_points2clusters5,
                             self.repre5._assign_entity_points2clusters())
        self.assertDictEqual(test_layered_repre_entity_points2clusters6,
                             self.merged_repre._assign_entity_points2clusters())

    def test_find_representative_paths(self):
        from transport_tools.tests.units.data.data_geometry import test_layered_repre_str1, test_layered_repre_str2, \
            test_layered_repre_str3, test_layered_repre_str4, test_layered_repre_str5, test_layered_repre_str6
        self.assertEqual(test_layered_repre_str1, str(self.repre1.find_representative_paths(self.transform_mat,
                                                                                            self.sp)))
        self.assertEqual(test_layered_repre_str2, str(self.repre2.find_representative_paths(self.transform_mat,
                                                                                            self.sp)))
        self.assertEqual(test_layered_repre_str3, str(self.repre3.find_representative_paths(self.transform_mat,
                                                                                            self.sp)))
        self.assertEqual(test_layered_repre_str4, str(self.repre4.find_representative_paths(self.transform_mat,
                                                                                            self.sp)))
        self.assertEqual(test_layered_repre_str5, str(self.repre5.find_representative_paths(self.transform_mat, None)))
        self.assertEqual(test_layered_repre_str6, str(self.merged_repre.find_representative_paths(self.transform_mat,
                                                                                                  self.sp)))


class TestHelpers(unittest.TestCase):
    def test_get_coarse_grained_path(self):
        from transport_tools.libs.geometry import get_coarse_grained_path
        from transport_tools.tests.units.data.data_geometry import test_helpers_coarse_grained_paths, \
            test_helpers_point2cluster_map

        for path_id, out_val in zip(test_helpers_point2cluster_map.keys(), test_helpers_coarse_grained_paths):
            self.assertListEqual(get_coarse_grained_path(test_helpers_point2cluster_map, path_id), out_val)

    def test_get_redundant_path_ids(self):
        from transport_tools.libs.geometry import get_redundant_path_ids
        from transport_tools.tests.units.data.data_geometry import test_helpers_redundant_path_ids, \
            test_helpers_all_paths

        for all_paths, out_val in zip(test_helpers_all_paths, test_helpers_redundant_path_ids):
            self.assertSetEqual(get_redundant_path_ids(all_paths), out_val)

    def test_remove_loops_from_path(self):
        from transport_tools.libs.geometry import remove_loops_from_path
        from transport_tools.tests.units.data.data_geometry import test_helpers_direct_paths, \
            test_helpers_squeezed_paths

        for path, out_val in zip(test_helpers_squeezed_paths, test_helpers_direct_paths):
            self.assertListEqual(remove_loops_from_path(path), out_val)

    def test_assign_layer_from_distances(self):
        from transport_tools.libs.geometry import assign_layer_from_distances
        from transport_tools.tests.units.data.data_geometry import test_helpers_unique_ids_membership
        layer_thicknesses = [0.001, .5, 1, 1.5, 2, 5]
        distances = np.array([0.0, 0.57, 12.35, 1.73, 1, 2, 3, 3.5, 4, 5, 7.5, 1, 3, 5.5, 5, 10, 120, 0.3])

        self.assertRaises(ValueError, assign_layer_from_distances, distances, 0)
        for thickness, out_val in zip(layer_thicknesses, test_helpers_unique_ids_membership):
            self.assertTrue(np.allclose(assign_layer_from_distances(distances, thickness)[0], out_val[0], atol=1e-3))
            self.assertTrue(np.allclose(assign_layer_from_distances(distances, thickness)[1], out_val[1], atol=1e-3))

    def test_get_layer_id_from_distance(self):
        from transport_tools.libs.geometry import get_layer_id_from_distance
        from transport_tools.tests.units.data.data_geometry import test_helpers_layer_ids
        layer_thicknesses = [0.001, .5, 1, 1.5, 2, 5]
        distances = np.array([0.0, 0.57, 12.35, 1.73, 1, 2, 3, 3.5, 4, 5, 7.5, 1, 3, 5.5, 5, 10, 120, 0.3])

        self.assertRaises(ValueError, get_layer_id_from_distance, distances, 0)
        for thickness, out_val in zip(layer_thicknesses, test_helpers_layer_ids):
            self.assertTrue(np.allclose(get_layer_id_from_distance(distances, thickness), out_val, atol=1e-3))

    def test_cart2spherical(self):
        from transport_tools.libs.geometry import cart2spherical
        from transport_tools.tests.units.data.data_geometry import test_helpers_rthetaphis, test_helpers_xyzs

        for i, xyz in enumerate(test_helpers_xyzs):
            self.assertTrue(np.allclose(cart2spherical(xyz), test_helpers_rthetaphis[i], atol=1e-3))

    def test_vector_angle(self):
        from transport_tools.libs.geometry import vector_angle
        from transport_tools.tests.units.data.data_geometry import test_helpers_angles, test_helpers_xyzs

        for i in range(len(test_helpers_xyzs) - 1):
            self.assertEqual(vector_angle(test_helpers_xyzs[i], test_helpers_xyzs[i + 1]), test_helpers_angles[i])

    def test_read_starting_points(self):
        """Verify read_starting_points() correctly reads PDB file"""
        from transport_tools.libs.geometry import read_starting_points
        from transport_tools.libs.utils import set_paths_from_package_root
        import os

        # Use test data if available
        test_pdb = set_paths_from_package_root("tests", "data", "simulations", "md1", "caver", "data", "v_origins.pdb")

        if os.path.exists(test_pdb):
            result = read_starting_points(test_pdb)

            # Should return ndarray with shape (n_points, 4) - x, y, z, radius
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape[1], 4)
            self.assertGreater(result.shape[0], 0)
        else:
            self.skipTest("Test data not available")

    def test_read_starting_points_nonexistent_file(self):
        """Verify read_starting_points() raises error for nonexistent file"""
        from transport_tools.libs.geometry import read_starting_points

        with self.assertRaises(FileNotFoundError):
            read_starting_points("/nonexistent/file.pdb")

    def test_einsum_dist_same_point(self):
        """Verify einsum_dist() returns zero for same point"""
        from transport_tools.libs.geometry import einsum_dist

        point = np.array([[1.0, 2.0, 3.0]])
        result = einsum_dist(point, point)

        self.assertIsInstance(result, np.ndarray)
        self.assertAlmostEqual(result[0], 0.0)

    def test_einsum_dist_known_distance(self):
        """Verify einsum_dist() calculates correct distances"""
        from transport_tools.libs.geometry import einsum_dist

        # Points at distance 5 apart (3-4-5 triangle)
        point1 = np.array([[0.0, 0.0, 0.0]])
        point2 = np.array([[3.0, 4.0, 0.0]])

        result = einsum_dist(point1, point2)
        self.assertAlmostEqual(result[0], 5.0)

    def test_einsum_dist_multiple_points(self):
        """Verify einsum_dist() handles multiple point pairs"""
        from transport_tools.libs.geometry import einsum_dist

        points1 = np.array([[0.0, 0.0, 0.0],
                           [1.0, 1.0, 1.0]])
        points2 = np.array([[1.0, 0.0, 0.0],
                           [1.0, 1.0, 1.0]])

        result = einsum_dist(points1, points2)

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], 1.0)  # Distance from (0,0,0) to (1,0,0)
        self.assertAlmostEqual(result[1], 0.0)  # Distance from (1,1,1) to (1,1,1)

    def test_iter_pair_chunks(self):
        """Verify iter_pair_chunks() yields all upper-triangle index pairs in balanced chunks"""
        from transport_tools.libs.geometry import iter_pair_chunks

        num_clusters = 5
        expected_pairs = [(i, j) for i in range(num_clusters) for j in range(i + 1, num_clusters)]

        for chunk_size in (1, 3, 4, 100):
            chunks = list(iter_pair_chunks(num_clusters, chunk_size))
            # no chunk exceeds the requested size
            self.assertTrue(all(len(chunk) <= chunk_size for chunk in chunks))
            # all pairs are present exactly once, preserving the upper-triangle order
            flat_pairs = [pair for chunk in chunks for pair in chunk]
            self.assertListEqual(flat_pairs, expected_pairs)

        # nothing to do when there are not enough clusters to form a pair
        self.assertListEqual(list(iter_pair_chunks(1, 10)), [])

    def test_distance_worker_chunk(self):
        """Verify init_distance_worker()/calc_distance_chunk() match direct distance calculation"""
        from transport_tools.libs.geometry import LayeredPathSet, init_distance_worker, calc_distance_chunk
        from transport_tools.tests.units.data.data_geometry import test_layers_tun, test_layers_event

        params = {
            "use_cluster_spread": False,
            "clustering_max_num_rep_frag": 0,
            "directional_cutoff": 1.5707963267948966,
            "random_seed": 4,
            "sp_radius": 0.5,
            "calculate_exact_path_distances": False
        }
        node_path_tun = ['0_1', '1_1', '2_1', '3_1', '4_1', '5_1', '6_2', '7_1', '8_1', '8_3', '7_3', '6_3', '5_2',
                         '5_3', '6_1', '7_2', '7_4', '8_2', '9_1']
        node_path_event = ['0_0', '1_0', '4_0', '5_0', '7_0']

        pathset_tun = LayeredPathSet("Cluster_44", "e10s1_e9s3p0f1600", params,
                                     starting_point_coords=np.array([0., 0., 0.]))
        pathset_tun.add_node_path(node_path_tun, test_layers_tun)
        pathset_event = LayeredPathSet("2_release", "e10s1_e9s3p0f1600", params, starting_point_coords=None)
        pathset_event.add_node_path(node_path_event, test_layers_event)

        cluster_specifications = [("e10s1_e9s3p0f1600", 44), ("e10s1_e9s3p0f1600", 2)]
        path_sets = {cluster_specifications[0]: pathset_tun, cluster_specifications[1]: pathset_event}
        cutoff, precision = 999, 4

        init_distance_worker(path_sets, cluster_specifications, precision, cutoff)
        results = calc_distance_chunk([(0, 1), (0, 0)])

        # index pairs are echoed back unchanged so the matrix fill is order-independent
        self.assertListEqual([(r[0], r[1]) for r in results], [(0, 1), (0, 0)])
        self.assertAlmostEqual(results[0][2],
                               np.around(pathset_tun.avg_distance2path_set(pathset_event, cutoff), precision))
        self.assertAlmostEqual(results[1][2], 0.0)  # distance of a cluster to itself

    def test_iter_shard_pair_chunks(self):
        """Verify the shards together cover every upper-triangle pair exactly once"""
        from transport_tools.libs.geometry import iter_shard_pair_chunks, count_pairs_for_shard

        num_clusters, num_shards = 7, 3
        expected_pairs = [(i, j) for i in range(num_clusters) for j in range(i + 1, num_clusters)]

        collected = list()
        for shard_id in range(num_shards):
            shard_pairs = [pair for chunk in iter_shard_pair_chunks(num_clusters, num_shards, shard_id, 4)
                           for pair in chunk]
            # count_pairs_for_shard agrees with what the generator actually yields
            self.assertEqual(count_pairs_for_shard(num_clusters, num_shards, shard_id), len(shard_pairs))
            collected.extend(shard_pairs)

        # the shards partition the upper triangle - every pair is present exactly once
        self.assertListEqual(sorted(collected), sorted(expected_pairs))

        # a single shard covering everything equals the plain upper-triangle enumeration
        single_shard = [pair for chunk in iter_shard_pair_chunks(num_clusters, 1, 0, 100) for pair in chunk]
        self.assertListEqual(single_shard, expected_pairs)

    @staticmethod
    def _partition(labels):
        """Convert a per-point component-label array into the order-independent set of point groups"""
        groups = dict()
        for idx, lab in enumerate(labels):
            groups.setdefault(int(lab), set()).add(idx)
        return frozenset(frozenset(group) for group in groups.values())

    def test_subcutoff_connected_components(self):
        """Verify subcutoff_connected_components() partitions points by the sub-cutoff graph,
        is chunk-size invariant, keeps zero-distance edges connected, and handles edge cases"""
        from transport_tools.libs.geometry import subcutoff_connected_components
        from scipy.spatial.distance import squareform
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        far = 999.0
        # two tight groups {0,1,2} and {3,4} plus an isolated point 5; all cross distances are far
        dense = np.full((6, 6), far)
        np.fill_diagonal(dense, 0.0)
        dense[0, 1] = dense[1, 0] = 0.0   # legitimate zero-distance edge must stay connected
        dense[1, 2] = dense[2, 1] = 1.0
        dense[0, 2] = dense[2, 0] = 1.5
        dense[3, 4] = dense[4, 3] = 0.8
        condensed = squareform(dense)
        cutoff = 2.0

        # ground truth: components of the dense thresholded adjacency
        adjacency = (dense <= cutoff).astype(np.uint8)
        np.fill_diagonal(adjacency, 0)
        n_expected, labels_expected = connected_components(coo_matrix(adjacency), directed=False)

        n_comp, labels = subcutoff_connected_components(condensed, 6, cutoff)
        self.assertEqual(n_comp, n_expected)
        self.assertEqual(self._partition(labels), self._partition(labels_expected))
        self.assertEqual(self._partition(labels),
                         frozenset({frozenset({0, 1, 2}), frozenset({3, 4}), frozenset({5})}))

        # chunk_size is a pure performance knob - identical partition for any chunk size
        for chunk_size in (1, 2, 3, 7, 100):
            n_c, labels_c = subcutoff_connected_components(condensed, 6, cutoff, chunk_size=chunk_size)
            self.assertEqual(n_c, n_comp)
            self.assertEqual(self._partition(labels_c), self._partition(labels))

        # everything beyond the cutoff -> every point its own component
        n_far, labels_far = subcutoff_connected_components(np.full(condensed.size, far), 6, cutoff)
        self.assertEqual(n_far, 6)
        self.assertEqual(self._partition(labels_far), frozenset(frozenset({i}) for i in range(6)))

        # everything within the cutoff -> a single component
        n_near, labels_near = subcutoff_connected_components(np.zeros(condensed.size), 6, cutoff)
        self.assertEqual(n_near, 1)
        self.assertEqual(self._partition(labels_near), frozenset({frozenset(range(6))}))

        # trivial point counts
        self.assertEqual(subcutoff_connected_components(np.empty(0), 0, cutoff)[0], 0)
        self.assertEqual(subcutoff_connected_components(np.empty(0), 1, cutoff)[0], 1)

    def test_gather_dense_submatrix(self):
        """Verify gather_dense_submatrix() reconstructs the dense sub-block from condensed form"""
        from transport_tools.libs.geometry import gather_dense_submatrix
        from scipy.spatial.distance import squareform

        rng = np.random.default_rng(0)
        num_points = 8
        dense = rng.random((num_points, num_points))
        dense = dense + dense.T
        np.fill_diagonal(dense, 0.0)
        condensed = squareform(dense)

        members = np.array([1, 3, 4, 6])
        block = gather_dense_submatrix(condensed, num_points, members)
        np.testing.assert_allclose(block, dense[np.ix_(members, members)])
        np.testing.assert_allclose(block, block.T)               # symmetric
        self.assertTrue(np.all(np.diag(block) == 0.0))           # zero diagonal

        # member order defines the block order; the block stays correct when members is unsorted
        members_unsorted = np.array([6, 1, 4, 3])
        block_unsorted = gather_dense_submatrix(condensed, num_points, members_unsorted)
        np.testing.assert_allclose(block_unsorted, dense[np.ix_(members_unsorted, members_unsorted)])

        # 1-member and empty blocks
        self.assertEqual(gather_dense_submatrix(condensed, num_points, np.array([2])).shape, (1, 1))
        self.assertEqual(gather_dense_submatrix(condensed, num_points, np.array([], dtype=int)).shape, (0, 0))

        # row-striping is a pure perf/memory knob: the block is identical for any stripe size,
        # including a stripe height of 1 (one row materialised at a time) and a single full stripe
        for stripe_elements in (1, 3, num_points, 10 ** 9):
            np.testing.assert_array_equal(
                gather_dense_submatrix(condensed, num_points, members, stripe_elements=stripe_elements),
                dense[np.ix_(members, members)])

    def test_gather_condensed_subvector(self):
        """gather_condensed_subvector() equals squareform of the dense block, for any chunk size"""
        from transport_tools.libs.geometry import gather_dense_submatrix, gather_condensed_subvector
        from scipy.spatial.distance import squareform

        rng = np.random.default_rng(1)
        num_points = 12
        dense = rng.random((num_points, num_points))
        dense = dense + dense.T
        np.fill_diagonal(dense, 0.0)
        condensed = squareform(dense)

        for members in (np.array([1, 3, 4, 6, 9]), np.array([9, 1, 6, 3, 4]), np.array([0, 11])):
            reference = squareform(gather_dense_submatrix(condensed, num_points, members), checks=False)
            for chunk_size in (1, 2, 7, 10 ** 9):
                np.testing.assert_array_equal(
                    gather_condensed_subvector(condensed, num_points, members, chunk_size=chunk_size),
                    reference)

        # degenerate components produce an empty condensed vector
        self.assertEqual(gather_condensed_subvector(condensed, num_points, np.array([5])).shape, (0,))
        self.assertEqual(gather_condensed_subvector(condensed, num_points, np.array([], dtype=int)).shape, (0,))


if __name__ == '__main__':
    unittest.main()
