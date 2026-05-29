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
import numpy as np


class TestClusteringDriverHelpers(unittest.TestCase):
    """Unit tests for the stage-5 connected-components partition-driver helpers used by both the
    agglomerative and HDBSCAN backends."""

    def test_components_by_min_index(self):
        """Component members are grouped, ascending within a group, and groups are ordered by their
        smallest member index regardless of the component ids assigned by connected_components."""
        from transport_tools.libs.tools import TransportProcesses

        component_labels = np.array([2, 0, 2, 1, 0])  # points 0..4
        groups = TransportProcesses._components_by_min_index(component_labels)
        self.assertEqual([group.tolist() for group in groups], [[0, 2], [1, 4], [3]])
        self.assertEqual([int(group[0]) for group in groups], [0, 1, 3])

        self.assertEqual(TransportProcesses._components_by_min_index(np.empty(0, dtype=np.intp)), [])

    def test_stitch_component_labels_agglomerative(self):
        """Distinct local labels map to fresh global labels; equal local labels stay grouped; the
        global label counter advances by the number of distinct clusters."""
        from transport_tools.libs.tools import TransportProcesses

        cluster_labels = np.full(6, -9, dtype=np.intp)
        next_label = TransportProcesses._stitch_component_labels(
            cluster_labels, np.array([0, 2, 4]), np.array([1, 1, 2]), 1)
        self.assertEqual(cluster_labels[0], cluster_labels[2])      # equal local -> same global
        self.assertNotEqual(cluster_labels[0], cluster_labels[4])   # distinct local -> distinct global
        self.assertEqual(next_label, 3)                             # two clusters consumed from 1

        next_label = TransportProcesses._stitch_component_labels(
            cluster_labels, np.array([1, 3]), np.array([5, 5]), next_label)
        self.assertEqual(cluster_labels[1], cluster_labels[3])
        self.assertEqual(next_label, 4)

        # labels of different components never collide, and untouched points keep their value
        self.assertNotEqual(cluster_labels[0], cluster_labels[1])
        self.assertEqual(cluster_labels[5], -9)

    def test_stitch_component_labels_hdbscan_noise(self):
        """Each HDBSCAN noise point (-1) becomes its own unique singleton label, while clustered
        points stay grouped - this is what preserves the every-cluster-maps invariant."""
        from transport_tools.libs.tools import TransportProcesses

        cluster_labels = np.full(5, -9, dtype=np.intp)
        local_labels = np.array([0, 0, -1, 1, -1])
        next_label = TransportProcesses._stitch_component_labels(
            cluster_labels, np.arange(5), local_labels, 1)

        self.assertEqual(cluster_labels[0], cluster_labels[1])        # cluster 0 grouped
        self.assertNotEqual(cluster_labels[0], cluster_labels[3])     # cluster 1 separate
        self.assertNotEqual(cluster_labels[2], cluster_labels[4])     # the two noise points are singletons
        # noise labels are distinct from clustered labels
        self.assertNotIn(cluster_labels[2], (cluster_labels[0], cluster_labels[3]))
        self.assertNotIn(cluster_labels[4], (cluster_labels[0], cluster_labels[3]))
        # 2 noise singletons + 2 clusters = 4 distinct labels, counter advanced by 4
        self.assertEqual(len(set(cluster_labels.tolist())), 4)
        self.assertEqual(next_label, 5)


if __name__ == '__main__':
    unittest.main()
