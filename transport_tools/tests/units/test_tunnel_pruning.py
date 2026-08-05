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

from transport_tools.libs.networks import Tunnel, TunnelCluster
from transport_tools.libs import tunnel_pruning
from transport_tools.tests.units.data.data_networks import datasection0, datasection1, datasection2, \
    datasection3

np.set_printoptions(threshold=100000)


def build_tunnels():
    """Build four real CAVER tunnels from the shared test data (one CAVER cluster, 4 snapshots)."""
    parameters = {
        "relevant_tunnel_min_radius": 0,
        "relevant_tunnel_min_length": 5,
        "relevant_tunnel_max_curvature": 999,
        "tunnel_properties_quantile": 0.9,
        "md_label": "test_md",
        "layer_thickness": 1.5,
    }
    tunnels = []
    for datasection in (datasection0, datasection1, datasection2, datasection3):
        tunnel = Tunnel(parameters, np.eye(4))
        tunnel.fill_data(datasection)
        tunnels.append(tunnel)
    return tunnels


class TestComputePerClusterCut(unittest.TestCase):
    def setUp(self):
        self.tunnels = build_tunnels()

    def test_produces_cut_for_cluster(self):
        cuts = tunnel_pruning.compute_per_cluster_cut(self.tunnels)
        self.assertEqual({1}, set(cuts.keys()))
        cut_distance, core_r, core_q3 = cuts[1]
        self.assertGreater(cut_distance, 0)
        self.assertGreater(core_r, 0)
        self.assertGreater(core_q3, 0)
        # cut must lie within the per-tunnel max-distance range
        max_dists = [t.spheres_data[-1, 3] for t in self.tunnels]
        self.assertLessEqual(cut_distance, max(max_dists))

    def test_peak_mode_also_finds_cut(self):
        cuts = tunnel_pruning.compute_per_cluster_cut(self.tunnels, mode="peak")
        self.assertEqual({1}, set(cuts.keys()))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            tunnel_pruning.compute_per_cluster_cut(self.tunnels, mode="bogus")

    def test_empty_input(self):
        self.assertEqual({}, tunnel_pruning.compute_per_cluster_cut([]))


class TestVerifyPostcut(unittest.TestCase):
    def setUp(self):
        self.tunnels = build_tunnels()
        self.cut_distance = tunnel_pruning.compute_per_cluster_cut(self.tunnels)[1][0]

    def test_only_extending_tunnels_reported(self):
        verdicts = tunnel_pruning.verify_postcut(self.tunnels, self.cut_distance)
        for tunnel in self.tunnels:
            extends = (tunnel.spheres_data[:, 3] >= self.cut_distance).any()
            key = (tunnel.snapshot, tunnel.caver_cluster_id, tunnel.tunnel_id)
            self.assertEqual(extends, key in verdicts)

    def test_categories_and_flags_consistent(self):
        verdicts = tunnel_pruning.verify_postcut(self.tunnels, self.cut_distance)
        for flags in verdicts.values():
            cat = flags["category"]
            if flags["inflating"] and flags["curved"]:
                self.assertEqual("both", cat)
            elif flags["inflating"]:
                self.assertEqual("inflating", cat)
            elif flags["curved"]:
                self.assertEqual("curved", cat)
            else:
                self.assertEqual("none", cat)


class TestTruncateTunnel(unittest.TestCase):
    def setUp(self):
        self.tunnels = build_tunnels()
        self.cut_distance = tunnel_pruning.compute_per_cluster_cut(self.tunnels)[1][0]
        self.offender = next(t for t in self.tunnels
                             if t.snapshot == "stripped_system.3590.pdb")

    def test_truncation_shortens_spheres_and_refreshes_attributes(self):
        n_before = len(self.offender.spheres_data)
        self.assertTrue(tunnel_pruning.truncate_tunnel(self.offender, self.cut_distance))
        n_after = len(self.offender.spheres_data)
        self.assertLess(n_after, n_before)
        self.assertEqual(n_after, len(self.offender.layer_membership))
        self.assertAlmostEqual(self.offender.length, self.offender.spheres_data[-1, 5])
        self.assertEqual(self.offender.filters_passed,
                         self.offender.bottleneck_radius >= 0
                         and self.offender.length >= 5.0
                         and self.offender.curvature <= 999)

    def test_far_cut_matches_reference_truncation(self):
        # with a cut beyond the tunnel's max distance the guarded cut clamps to the
        # last sphere and (when its radius >= water floor and the bottleneck is earlier)
        # drops just that last sphere - identical to the reference good_gardener pipeline
        far_cut = max(self.offender.spheres_data[:, 3]) + 100
        n_before = len(self.offender.spheres_data)
        truncated = tunnel_pruning.truncate_tunnel(self.offender, far_cut)
        n_after = len(self.offender.spheres_data)
        self.assertEqual(n_before - 1, n_after)
        self.assertEqual(truncated, n_after < n_before)
        self.assertEqual(n_after, len(self.offender.layer_membership))


class TestPruneTunnels(unittest.TestCase):
    def setUp(self):
        self.tunnels = build_tunnels()

    def test_offensive_truncated_and_others_untouched(self):
        n_before = {t.snapshot: len(t.spheres_data) for t in self.tunnels}
        stats = tunnel_pruning.prune_tunnels(self.tunnels)
        self.assertEqual(1, stats["n_clusters"])
        self.assertEqual(1, stats["n_clusters_with_cut"])
        self.assertEqual(1, stats["n_truncated"])
        self.assertGreaterEqual(stats["n_offensive"], 1)
        self.assertEqual(stats["n_offensive"], stats["n_inflated"] + stats["n_curved"] - stats["n_both"])

        for tunnel in self.tunnels:
            n_after = len(tunnel.spheres_data)
            if tunnel.snapshot == "stripped_system.3590.pdb":
                self.assertLess(n_after, n_before[tunnel.snapshot])
            else:
                self.assertEqual(n_after, n_before[tunnel.snapshot])
            self.assertEqual(n_after, len(tunnel.layer_membership))

    def test_consistency_with_good_gardener_reference(self):
        # the port must reproduce the reference pipeline's verdicts on the same cluster
        cuts = tunnel_pruning.compute_per_cluster_cut(self.tunnels)
        verdicts = tunnel_pruning.verify_postcut(self.tunnels, cuts[1][0])
        self.assertEqual("inflating",
                         verdicts[("stripped_system.3590.pdb", 1, 1)]["category"])
        self.assertEqual("none", verdicts[("stripped_system.2.pdb", 1, 1)]["category"])


class TestTunnelNetworkPruneIntegration(unittest.TestCase):
    """Exercise TunnelNetwork._prune_tunnels on a lightweight stand-in (mirrors
    TestValidateSnapshotSampling), verifying the parameter plumbing into prune_tunnels."""

    @staticmethod
    def _make_self():
        from types import SimpleNamespace
        tunnels = build_tunnels()
        cluster = TunnelCluster(1, tunnels[0].parameters, np.eye(4),
                                np.array([[0.0, 0.0, 0.0]]))
        for tunnel in tunnels:
            cluster.tunnels[tunnel.get_snapshot_id()] = tunnel
        parameters = tunnels[0].parameters.copy()
        parameters.update({
            "prune_tunnels": True,
            "prune_tunnels_mode": "first",
            "prune_tunnels_bin_size": 0.5,
            "prune_tunnels_survival_perc_low": 0.1,
            "prune_tunnels_survival_perc_high": 0.9,
            "prune_tunnels_core_fraction_low": 0.3,
            "prune_tunnels_core_fraction_high": 0.6,
            "prune_tunnels_min_joint_threshold": 0.01,
            "prune_tunnels_eff_thresh": 0.5,
            "prune_tunnels_slope_percentile": 90,
            "prune_tunnels_water_radius": 1.4,
        })
        return SimpleNamespace(orig_entities=[cluster], parameters=parameters, md_label="test_md")

    def test_prune_tunnels_truncates_offensive_tunnels(self):
        from transport_tools.libs.networks import TunnelNetwork
        fake_self = self._make_self()
        cluster = fake_self.orig_entities[0]
        offender = cluster.tunnels[3590]
        n_before = len(offender.spheres_data)
        TunnelNetwork._prune_tunnels(fake_self)
        self.assertLess(len(offender.spheres_data), n_before)
        self.assertEqual(len(offender.spheres_data), len(offender.layer_membership))


if __name__ == "__main__":
    unittest.main()
