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

    def test_reassign_hdbscan_noise_within_cutoff_absorbs_near_noise(self):
        """Noise (-1) within the cutoff of an actual cluster is absorbed into the nearest cluster,
        including coincident (0 A) points; noise beyond the cutoff from every cluster stays -1."""
        from transport_tools.libs.tools import TransportProcesses

        # points: 0,1 -> cluster 0 ; 2 -> cluster 1 ; 3,4 -> noise
        local_labels = np.array([0, 0, 1, -1, -1])
        # point 3 is 0 A from point 0 (cluster 0); point 4 is 5 A from everything (true outlier)
        d = np.array([
            [0.0, 0.4, 3.0, 0.0, 5.0],
            [0.4, 0.0, 3.0, 0.3, 5.0],
            [3.0, 3.0, 0.0, 2.9, 5.0],
            [0.0, 0.3, 2.9, 0.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 0.0],
        ])
        out = TransportProcesses._reassign_hdbscan_noise_within_cutoff(local_labels, d, cutoff=1.5)
        self.assertEqual(out[3], 0)        # coincident noise folded into cluster 0
        self.assertEqual(out[4], -1)       # far noise stays a singleton
        # clustered points are never touched; input array is not mutated
        self.assertEqual(out[:3].tolist(), [0, 0, 1])
        self.assertEqual(local_labels[3], -1)

    def test_reassign_hdbscan_noise_within_cutoff_picks_nearer_cluster(self):
        """A noise point within the cutoff of two clusters joins the nearer one (min distance to any
        clustered member); ties break to the lowest index."""
        from transport_tools.libs.tools import TransportProcesses

        # points: 0 -> cluster 0 ; 1 -> cluster 1 ; 2 -> noise, closer to cluster 1
        local_labels = np.array([0, 1, -1])
        d = np.array([
            [0.0, 2.0, 0.9],
            [2.0, 0.0, 0.4],
            [0.9, 0.4, 0.0],
        ])
        out = TransportProcesses._reassign_hdbscan_noise_within_cutoff(local_labels, d, cutoff=1.0)
        self.assertEqual(out[2], 1)

    def test_reassign_hdbscan_noise_within_cutoff_noop_without_clusters(self):
        """With no clustered points (whole component is noise) or no noise at all, labels are
        returned unchanged - residual noise still becomes singletons downstream."""
        from transport_tools.libs.tools import TransportProcesses

        all_noise = np.array([-1, -1, -1])
        d = np.zeros((3, 3))
        self.assertEqual(
            TransportProcesses._reassign_hdbscan_noise_within_cutoff(all_noise, d, 1.0).tolist(),
            [-1, -1, -1])

        no_noise = np.array([0, 0, 1])
        self.assertEqual(
            TransportProcesses._reassign_hdbscan_noise_within_cutoff(no_noise, d, 1.0).tolist(),
            [0, 0, 1])


class TestOrphanConsolidation(unittest.TestCase):
    """Unit tests for the global, method-agnostic orphan->core supercluster consolidation pass."""

    @staticmethod
    def _condensed(dense):
        from scipy.spatial.distance import squareform
        return squareform(np.asarray(dense, dtype=np.float64), checks=False)

    def test_no_cores_leaves_labels_unchanged(self):
        """If no supercluster reaches core_min_size there is nothing to attach orphans to."""
        from transport_tools.libs.tools import TransportProcesses

        labels = np.array([0, 1, 2])           # three singletons, core_min_size 3 -> no cores
        cond = self._condensed([[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]])
        out = TransportProcesses._consolidate_orphans_to_cores(labels, cond, 3, 3, None)
        self.assertEqual(out.tolist(), [0, 1, 2])

    def test_orphans_join_nearest_core_uncapped(self):
        """With no cap every orphan joins its nearest core; core members are untouched."""
        from transport_tools.libs.tools import TransportProcesses

        # points 0,1,2 = coreA (label 10); 3,4,5 = coreB (label 20); point 6 = orphan (label 99)
        labels = np.array([10, 10, 10, 20, 20, 20, 99])
        dense = np.full((7, 7), 5.0); np.fill_diagonal(dense, 0.0)
        # make each core internally tight
        for grp in ([0, 1, 2], [3, 4, 5]):
            for a in grp:
                for b in grp:
                    dense[a, b] = 0.0 if a == b else 0.2
        dense[6] = dense[:, 6] = 5.0
        dense[6, 4] = dense[4, 6] = 0.3    # orphan closest to coreB member 4
        dense[6, 1] = dense[1, 6] = 2.0    # farther from coreA
        np.fill_diagonal(dense, 0.0)
        out = TransportProcesses._consolidate_orphans_to_cores(labels, self._condensed(dense), 7, 3, None)
        self.assertEqual(out[6], 20)                       # orphan absorbed into nearest core (coreB)
        self.assertEqual(out[:6].tolist(), [10, 10, 10, 20, 20, 20])  # cores unchanged

    def test_cap_keeps_far_orphan_isolated(self):
        """An orphan beyond assignment_cutoff from every core stays its own supercluster."""
        from transport_tools.libs.tools import TransportProcesses

        labels = np.array([10, 10, 10, 98, 99])           # core (0,1,2) + two orphans
        dense = np.full((5, 5), 9.0); np.fill_diagonal(dense, 0.0)
        for a in (0, 1, 2):
            for b in (0, 1, 2):
                dense[a, b] = 0.0 if a == b else 0.2
        dense[3, 0] = dense[0, 3] = 0.5                    # orphan 3 near the core
        dense[4, 0] = dense[0, 4] = 4.0                    # orphan 4 far from the core
        np.fill_diagonal(dense, 0.0)
        out = TransportProcesses._consolidate_orphans_to_cores(labels, self._condensed(dense), 5, 3, 1.0)
        self.assertEqual(out[3], 10)                       # within cap -> joined
        self.assertEqual(out[4], 99)                       # beyond cap -> stays singleton

    def test_single_pass_does_not_chain(self):
        """Orphans attach only to ORIGINAL cores, so a near-orphan that just joined a core does not
        pull a second, far orphan in after it."""
        from transport_tools.libs.tools import TransportProcesses

        # core 0,1,2 ; orphan 3 is 0.1 from the core ; orphan 4 is 0.1 from orphan 3 but 3.0 from core
        labels = np.array([10, 10, 10, 98, 99])
        dense = np.full((5, 5), 9.0); np.fill_diagonal(dense, 0.0)
        for a in (0, 1, 2):
            for b in (0, 1, 2):
                dense[a, b] = 0.0 if a == b else 0.2
        dense[3, 0] = dense[0, 3] = 0.1
        dense[4, 3] = dense[3, 4] = 0.1
        dense[4, 0] = dense[0, 4] = 3.0
        dense[4, 1] = dense[1, 4] = 3.0
        dense[4, 2] = dense[2, 4] = 3.0
        np.fill_diagonal(dense, 0.0)
        out = TransportProcesses._consolidate_orphans_to_cores(labels, self._condensed(dense), 5, 3, 1.0)
        self.assertEqual(out[3], 10)                       # joined the core
        self.assertEqual(out[4], 99)                       # NOT chained in via orphan 3


class TestDecoupledClusteringScales(unittest.TestCase):
    """Unit tests for decoupling the partition scale from the per-component clustering scale, and the
    HDBSCAN cluster-selection-epsilon / noise-merge scales from clustering_cutoff."""

    @staticmethod
    def _stub(**params):
        from transport_tools.libs.tools import TransportProcesses
        obj = TransportProcesses.__new__(TransportProcesses)
        obj.parameters = params
        return obj

    @staticmethod
    def _condensed(dense):
        from scipy.spatial.distance import squareform
        return squareform(np.asarray(dense, dtype=np.float64), checks=False)

    def test_partition_cutoff_backcompat_and_exact_for_agglomerative(self):
        """partition_cutoff=None reproduces the legacy single-scale call, and for agglomerative a
        larger partition_cutoff is exact - it yields the same flat clustering (just bigger blocks)."""
        # two tight pairs {0,1} and {2,3} (0.4 apart) with a 1.6 gap between the pairs
        dense = np.array([
            [0.0, 0.4, 1.6, 1.7],
            [0.4, 0.0, 1.7, 1.6],
            [1.6, 1.7, 0.0, 0.4],
            [1.7, 1.6, 0.4, 0.0],
        ])
        cond = self._condensed(dense)
        obj = self._stub()
        cut = 0.5
        legacy = obj._cluster_labels_partitioned(cond, 4, cut, "complete")
        explicit = obj._cluster_labels_partitioned(cond, 4, cut, "complete", partition_cutoff=cut)
        generous = obj._cluster_labels_partitioned(cond, 4, cut, "complete", partition_cutoff=5.0)

        def grouping(labels):
            return {frozenset(np.flatnonzero(labels == l).tolist()) for l in set(labels.tolist())}

        self.assertEqual(grouping(legacy), grouping(explicit))   # None == cutoff (back-compat)
        self.assertEqual(grouping(legacy), grouping(generous))   # larger partition is exact
        self.assertEqual(grouping(legacy), {frozenset({0, 1}), frozenset({2, 3})})

    def _capture_hdbscan_scales(self, params, cutoff):
        """Run _cluster_component_hdbscan with HDBSCAN and the noise-reassign step mocked out, so the
        test is deterministic and isolates exactly what Phase A changed: which scale reaches the
        HDBSCAN cluster_selection_epsilon and which reaches the noise-merge floor."""
        from unittest import mock
        from transport_tools.libs.tools import TransportProcesses

        captured = {}

        class FakeClusterer:
            def __init__(self, **kw):
                captured["epsilon"] = kw["cluster_selection_epsilon"]

            def fit_predict(self, _block):
                return np.array([0, -1], dtype=np.intp)  # one clustered point, one noise point

        def fake_reassign(local_labels, _block, floor):
            captured["floor"] = floor
            return local_labels

        dense = np.array([[0.0, 1.0], [1.0, 0.0]])  # 2x2 so it clears the min_cluster_size early-return
        obj = self._stub(hdbscan_min_cluster_size=2, hdbscan_min_samples=1,
                         hdbscan_cluster_selection_method="eom", **params)
        with mock.patch("hdbscan.HDBSCAN", FakeClusterer), \
                mock.patch.object(TransportProcesses, "_reassign_hdbscan_noise_within_cutoff",
                                  staticmethod(fake_reassign)):
            obj._cluster_component_hdbscan(dense, cutoff=cutoff)
        return captured

    def test_hdbscan_dedicated_scales_are_used(self):
        """hdbscan_cluster_selection_epsilon reaches HDBSCAN and hdbscan_noise_merge_cutoff reaches
        the noise-merge floor - each independent of clustering_cutoff."""
        captured = self._capture_hdbscan_scales(
            dict(hdbscan_cluster_selection_epsilon=0.7, hdbscan_noise_merge_cutoff=2.5), cutoff=9.9)
        self.assertEqual(captured["epsilon"], 0.7)
        self.assertEqual(captured["floor"], 2.5)

    def test_hdbscan_scales_fall_back_to_cutoff_when_none(self):
        """When the dedicated epsilon/noise params are None both inherit clustering_cutoff (legacy
        single-scale behavior)."""
        captured = self._capture_hdbscan_scales(
            dict(hdbscan_cluster_selection_epsilon=None, hdbscan_noise_merge_cutoff=None), cutoff=1.5)
        self.assertEqual(captured["epsilon"], 1.5)
        self.assertEqual(captured["floor"], 1.5)


if __name__ == '__main__':
    unittest.main()
