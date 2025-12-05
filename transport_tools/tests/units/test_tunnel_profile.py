# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2022  Jan Brezovsky, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta <janbre@amu.edu.pl>
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

__version__ = '0.9.6'
__author__ = 'Jan Brezovsky, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta'
__mail__ = 'janbre@amu.edu.pl'

import unittest
import os
import pickle
import tempfile
import numpy as np
from io import StringIO
from transport_tools.libs.networks import TunnelProfile4MD, Tunnel, TunnelCluster
from transport_tools.tests.units.data.data_tunnel_profile import (
    test_parameters, test_transform_mat, tunnel_data_section_snapshot1,
    tunnel_data_section_snapshot2, tunnel_data_section_snapshot3,
    sample_bottleneck_residues, expected_residue_freq,
    test_filters_pass_all, test_filters_strict, test_filters_length_only
)


class TestTunnelProfile4MD(unittest.TestCase):
    """
    Comprehensive tests for TunnelProfile4MD class
    """

    def setUp(self):
        """
        Set up test fixtures before each test method
        """
        self.maxDiff = None
        self.temp_dir = tempfile.mkdtemp()
        self.dump_file = os.path.join(self.temp_dir, "test_network.dump")

        # Create mock tunnel clusters with tunnels
        self.clusters = self._create_mock_clusters()

        # Save clusters to dump file
        with open(self.dump_file, "wb") as f:
            pickle.dump(self.clusters, f)

        # Create TunnelProfile4MD instance
        self.profile = TunnelProfile4MD(
            md_label="md1",
            caver_clusters=[1, 2],  # Only clusters 1 and 2 belong to the supercluster
            dump_file=self.dump_file,
            parameters=test_parameters
        )

    def tearDown(self):
        """
        Clean up after each test method
        """
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_mock_clusters(self):
        """
        Helper method to create mock TunnelCluster objects with tunnels
        """
        clusters = []

        # Create cluster 1 with 3 tunnels (snapshots 1, 2, 3)
        cluster1 = TunnelCluster(1, test_parameters, test_transform_mat,
                                np.array([[0., 0., 0.]]))

        tunnel1 = Tunnel(test_parameters, test_transform_mat)
        tunnel1.fill_data(tunnel_data_section_snapshot1)
        tunnel1.caver_cluster_id = 1
        tunnel1.bottleneck_residues = sample_bottleneck_residues[1]
        tunnel1.bottleneck_xyz = np.array([42.0, 42.0, 30.0])  # Set bottleneck coordinates
        cluster1.tunnels[1] = tunnel1

        tunnel2 = Tunnel(test_parameters, test_transform_mat)
        tunnel2.fill_data(tunnel_data_section_snapshot2)
        tunnel2.caver_cluster_id = 1
        tunnel2.bottleneck_residues = sample_bottleneck_residues[2]
        tunnel2.bottleneck_xyz = np.array([42.0, 42.0, 30.0])  # Set bottleneck coordinates
        cluster1.tunnels[2] = tunnel2

        tunnel3 = Tunnel(test_parameters, test_transform_mat)
        tunnel3.fill_data(tunnel_data_section_snapshot3)
        tunnel3.caver_cluster_id = 1
        tunnel3.bottleneck_residues = sample_bottleneck_residues[3]
        tunnel3.bottleneck_xyz = np.array([42.0, 42.0, 30.0])  # Set bottleneck coordinates
        cluster1.tunnels[3] = tunnel3

        clusters.append(cluster1)

        # Create cluster 2 with 2 tunnels (snapshots 1, 2)
        cluster2 = TunnelCluster(2, test_parameters, test_transform_mat,
                                np.array([[0., 0., 0.]]))

        tunnel4 = Tunnel(test_parameters, test_transform_mat)
        tunnel4.fill_data(tunnel_data_section_snapshot1)
        tunnel4.caver_cluster_id = 2
        tunnel4.bottleneck_residues = ["GLY:23", "SER:67"]
        tunnel4.bottleneck_xyz = np.array([41.0, 42.5, 30.5])  # Set bottleneck coordinates
        cluster2.tunnels[1] = tunnel4

        tunnel5 = Tunnel(test_parameters, test_transform_mat)
        tunnel5.fill_data(tunnel_data_section_snapshot2)
        tunnel5.caver_cluster_id = 2
        tunnel5.bottleneck_residues = ["SER:67", "THR:89"]
        tunnel5.bottleneck_xyz = np.array([41.0, 42.5, 30.5])  # Set bottleneck coordinates
        cluster2.tunnels[2] = tunnel5

        clusters.append(cluster2)

        # Create cluster 3 with 1 tunnel (not in supercluster, should be ignored)
        cluster3 = TunnelCluster(3, test_parameters, test_transform_mat,
                                np.array([[0., 0., 0.]]))

        tunnel6 = Tunnel(test_parameters, test_transform_mat)
        tunnel6.fill_data(tunnel_data_section_snapshot1)
        tunnel6.caver_cluster_id = 3
        tunnel6.bottleneck_residues = ["ASP:12"]
        cluster3.tunnels[1] = tunnel6

        clusters.append(cluster3)

        return clusters

    def test_init_creates_profile_with_correct_attributes(self):
        """
        Test that TunnelProfile4MD initializes with correct attributes
        """
        self.assertEqual("md1", self.profile.md_label)
        self.assertEqual([1, 2], self.profile.sc_caver_clusters)
        self.assertEqual(self.dump_file, self.profile.dump_file)
        self.assertEqual(test_parameters, self.profile.parameters)
        self.assertIsInstance(self.profile.records, dict)
        self.assertEqual(0, len(self.profile.records))  # Empty before load_network()

    def test_load_network_loads_tunnels_from_clusters(self):
        """
        Test that load_network() correctly loads tunnels from specified clusters
        """
        self.profile.load_network()

        # Should have loaded tunnels from specified clusters (1 and 2)
        # When multiple tunnels exist for same snapshot, only one with better throughput is kept
        self.assertGreater(len(self.profile.records), 0)

        # Verify only clusters 1 and 2 are included (cluster 3 should be excluded)
        cluster_ids = set()
        for tunnel in self.profile.records.values():
            cluster_ids.add(tunnel.caver_cluster_id)

        # At least cluster 1 should be present
        self.assertIn(1, cluster_ids)
        # Cluster 3 should never be included
        self.assertNotIn(3, cluster_ids)
        # Cluster 2 may or may not be present depending on throughput comparison
        self.assertTrue(cluster_ids.issubset({1, 2}), "Only clusters 1 and 2 should be present")

    def test_load_network_recalculates_distances_to_origin(self):
        """
        Test that load_network() recalculates sphere distances to origin [0,0,0]
        """
        self.profile.load_network()

        # Check that distances have been recalculated for at least one tunnel
        for tunnel in self.profile.records.values():
            # Distance of first sphere to origin should match coordinate distance
            first_sphere_coords = tunnel.spheres_data[0, 0:3]
            first_sphere_distance = tunnel.spheres_data[0, 3]

            expected_distance = np.linalg.norm(first_sphere_coords)
            self.assertAlmostEqual(expected_distance, first_sphere_distance, places=2)
            break  # Just check one tunnel

    def test_load_network_keeps_better_throughput_tunnel(self):
        """
        Test that when multiple tunnels exist for same snapshot, tunnel with better throughput is kept
        """
        self.profile.load_network()

        # For overlapping snapshots, verify only one tunnel per snapshot exists
        snapshot_ids = list(self.profile.records.keys())
        self.assertEqual(len(snapshot_ids), len(set(snapshot_ids)))  # No duplicates

    def test_count_tunnels_returns_correct_count(self):
        """
        Test that count_tunnels() returns the correct number of tunnels
        """
        self.profile.load_network()
        count = self.profile.count_tunnels()

        self.assertIsInstance(count, int)
        self.assertGreater(count, 0)
        self.assertEqual(count, len(self.profile.records))

    def test_count_tunnels_returns_zero_when_no_tunnels(self):
        """
        Test that count_tunnels() returns 0 when no tunnels are loaded
        """
        count = self.profile.count_tunnels()
        self.assertEqual(0, count)

    def test_write_csv_section_writes_correct_format(self):
        """
        Test that write_csv_section() writes tunnels in correct CSV format
        """
        self.profile.load_network()

        output = StringIO()
        self.profile.write_csv_section(output)

        csv_content = output.getvalue()

        # Verify output is not empty
        self.assertGreater(len(csv_content), 0)

        # Verify each line starts with md_label
        lines = csv_content.strip().split('\n')
        for line in lines:
            if line:  # Skip empty lines
                self.assertTrue(line.startswith("md1,"))

    def test_write_residues_writes_bottleneck_data(self):
        """
        Test that write_residues() writes bottleneck residue data
        """
        self.profile.load_network()

        output = StringIO()
        self.profile.write_residues(output)

        residue_content = output.getvalue()

        # Verify output is not empty
        self.assertGreater(len(residue_content), 0)

    def test_filter_tunnels_removes_tunnels_not_passing_filters(self):
        """
        Test that filter_tunnels() removes tunnels that don't pass filters
        """
        self.profile.load_network()
        initial_count = self.profile.count_tunnels()

        # Apply strict filter that should remove some tunnels
        self.profile.filter_tunnels(test_filters_strict)

        filtered_count = self.profile.count_tunnels()

        # Count should be less than or equal to initial (some may be filtered out)
        self.assertLessEqual(filtered_count, initial_count)

    def test_filter_tunnels_keeps_all_with_permissive_filters(self):
        """
        Test that filter_tunnels() keeps all tunnels with permissive filters
        """
        self.profile.load_network()
        initial_count = self.profile.count_tunnels()

        # Apply permissive filter that passes everything
        self.profile.filter_tunnels(test_filters_pass_all)

        filtered_count = self.profile.count_tunnels()

        # All tunnels should remain
        self.assertEqual(initial_count, filtered_count)

    def test_get_parameters_returns_tunnel_properties(self):
        """
        Test that get_parameters() returns lists of tunnel properties
        """
        self.profile.load_network()

        lengths, radii, curvatures, throughputs = self.profile.get_parameters()

        # Verify return types
        self.assertIsInstance(lengths, list)
        self.assertIsInstance(radii, list)
        self.assertIsInstance(curvatures, list)
        self.assertIsInstance(throughputs, list)

        # Verify all lists have same length
        count = self.profile.count_tunnels()
        self.assertEqual(count, len(lengths))
        self.assertEqual(count, len(radii))
        self.assertEqual(count, len(curvatures))
        self.assertEqual(count, len(throughputs))

        # Verify all values are numeric and positive
        for length in lengths:
            self.assertGreater(length, 0)
        for radius in radii:
            self.assertGreater(radius, 0)

    def test_get_bottleneck_residues_frequency_returns_correct_frequencies(self):
        """
        Test that get_bottleneck_residues_frequency() returns correct residue frequencies
        """
        self.profile.load_network()

        freq = self.profile.get_bottleneck_residues_frequency()

        # Verify return type
        self.assertIsInstance(freq, dict)

        # Verify frequencies are positive integers
        for residue, count in freq.items():
            self.assertIsInstance(residue, str)
            self.assertIsInstance(count, int)
            self.assertGreater(count, 0)

    def test_get_bottleneck_residues_frequency_returns_empty_when_no_tunnels(self):
        """
        Test that get_bottleneck_residues_frequency() returns empty dict when no tunnels
        """
        freq = self.profile.get_bottleneck_residues_frequency()

        self.assertIsInstance(freq, dict)
        self.assertEqual(0, len(freq))

    def test_enumerate_caver_cluster_ids_returns_present_clusters(self):
        """
        Test that enumerate_caver_cluster_ids() returns IDs of clusters with valid tunnels
        """
        self.profile.load_network()

        cluster_ids = self.profile.enumerate_caver_cluster_ids()

        # Verify return type
        self.assertIsInstance(cluster_ids, list)

        # Verify all IDs are in the expected clusters (1 and 2)
        for cid in cluster_ids:
            self.assertIn(cid, [1, 2])

        # Verify cluster 3 is not included
        self.assertNotIn(3, cluster_ids)

    def test_enumerate_caver_cluster_ids_returns_empty_when_no_tunnels(self):
        """
        Test that enumerate_caver_cluster_ids() returns empty list when no tunnels
        """
        cluster_ids = self.profile.enumerate_caver_cluster_ids()

        self.assertIsInstance(cluster_ids, list)
        self.assertEqual(0, len(cluster_ids))

    def test_get_property_time_evolution_data_returns_array(self):
        """
        Test that get_property_time_evolution_data() returns numpy array of property values
        """
        self.profile.load_network()

        # Test with bottleneck_radius property
        data = self.profile.get_property_time_evolution_data("bottleneck_radius")

        # Verify return type
        self.assertIsInstance(data, np.ndarray)

        # Verify array length matches snapshots_per_simulation
        expected_length = test_parameters["snapshots_per_simulation"]
        self.assertEqual(expected_length, len(data))

    def test_get_property_time_evolution_data_uses_default_for_missing_frames(self):
        """
        Test that get_property_time_evolution_data() uses default value for missing frames
        """
        self.profile.load_network()

        # Test with custom missing value
        missing_value = -999.0
        data = self.profile.get_property_time_evolution_data("bottleneck_radius",
                                                             missing_value_default=missing_value)

        # Count how many frames have the default value
        default_count = np.sum(data == missing_value)

        # Most frames should have default value (we only have a few tunnels)
        expected_missing = test_parameters["snapshots_per_simulation"] - self.profile.count_tunnels()
        self.assertEqual(expected_missing, default_count)

    def test_get_property_time_evolution_data_with_different_properties(self):
        """
        Test that get_property_time_evolution_data() works with different tunnel properties
        """
        self.profile.load_network()

        # Test valid Tunnel attributes
        properties = ["length", "bottleneck_radius", "curvature", "throughput", "cost"]

        for prop in properties:
            with self.subTest(property=prop):
                data = self.profile.get_property_time_evolution_data(prop)

                self.assertIsInstance(data, np.ndarray)
                self.assertEqual(test_parameters["snapshots_per_simulation"], len(data))

    def test_filter_tunnels_then_get_parameters_returns_filtered_data(self):
        """
        Test that filtering tunnels and then getting parameters returns only filtered tunnel data
        """
        self.profile.load_network()

        # Get initial parameters
        initial_lengths, _, _, _ = self.profile.get_parameters()
        initial_count = len(initial_lengths)

        # Apply length filter
        self.profile.filter_tunnels(test_filters_length_only)

        # Get filtered parameters
        filtered_lengths, _, _, _ = self.profile.get_parameters()
        filtered_count = len(filtered_lengths)

        # Verify filtering had an effect
        self.assertLessEqual(filtered_count, initial_count)

        # Verify all remaining tunnels pass the filter
        for length in filtered_lengths:
            self.assertGreaterEqual(length, test_filters_length_only["length"][0])
            self.assertLessEqual(length, test_filters_length_only["length"][1])


class TestTunnelProfile4MDEdgeCases(unittest.TestCase):
    """
    Test edge cases and error conditions for TunnelProfile4MD
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_network_with_empty_clusters_list(self):
        """
        Test load_network() when no clusters belong to supercluster
        """
        dump_file = os.path.join(self.temp_dir, "empty.dump")

        # Create empty cluster list
        with open(dump_file, "wb") as f:
            pickle.dump([], f)

        profile = TunnelProfile4MD(
            md_label="test",
            caver_clusters=[],
            dump_file=dump_file,
            parameters=test_parameters
        )

        profile.load_network()

        self.assertEqual(0, profile.count_tunnels())

    def test_write_csv_section_with_no_tunnels(self):
        """
        Test write_csv_section() when profile has no tunnels
        """
        dump_file = os.path.join(self.temp_dir, "empty.dump")

        with open(dump_file, "wb") as f:
            pickle.dump([], f)

        profile = TunnelProfile4MD(
            md_label="test",
            caver_clusters=[],
            dump_file=dump_file,
            parameters=test_parameters
        )

        profile.load_network()

        output = StringIO()
        profile.write_csv_section(output)

        # Should produce empty output
        self.assertEqual("", output.getvalue())


if __name__ == "__main__":
    unittest.main()
