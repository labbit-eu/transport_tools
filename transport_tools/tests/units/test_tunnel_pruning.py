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


def build_single_tunnel(datasection, parameters=None):
    """Build one Tunnel object from a CAVER 7-line block, with an option to override parameters."""
    defaults = {
        "relevant_tunnel_min_radius": 0,
        "relevant_tunnel_min_length": 5,
        "relevant_tunnel_max_curvature": 999,
        "tunnel_properties_quantile": 0.9,
        "md_label": "test_md",
        "layer_thickness": 1.5,
    }
    if parameters:
        defaults.update(parameters)
    tunnel = Tunnel(defaults, np.eye(4))
    tunnel.fill_data(datasection)
    return tunnel


def build_tunnels():
    """Build four real CAVER tunnels from the shared test data (one CAVER cluster, 4 snapshots)."""
    return [build_single_tunnel(datasection)
            for datasection in (datasection0, datasection1, datasection2, datasection3)]


def expected_length_curvature(tunnel):
    """Independent recomputation of length/curvature from the kept sphere coordinates,
    following CAVER's definition (arc length of the centerline from the first sphere,
    over the first-to-last chord). Used to verify the refresh without reusing the
    implementation under test."""
    coords = tunnel.spheres_data[:, 0:3]
    segments = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    length = float(segments.sum())
    chord = float(np.linalg.norm(coords[-1] - coords[0]))
    return length, (length / chord if chord > 0 else float("inf"))


def build_synthetic_tunnel(dists, radii, seed=0.0):
    """Build a Tunnel from raw distance/radius profiles (spheres laid on the x-axis)."""
    dists = np.asarray(dists, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(dists)
    seg = np.abs(np.diff(np.r_[0.0, dists]))
    xs = np.cumsum(seg) + seed
    ys = np.zeros(n) + seed
    zs = np.zeros(n)
    lens = np.cumsum(seg)
    data = np.column_stack([xs, ys, zs, dists, radii, lens])
    tunnel = Tunnel({
        "relevant_tunnel_min_radius": 0,
        "relevant_tunnel_min_length": 5,
        "relevant_tunnel_max_curvature": 999,
        "tunnel_properties_quantile": 0.9,
        "md_label": "test_md",
        "layer_thickness": 1.5,
    }, np.eye(4))
    tunnel.spheres_data = data
    tunnel.snapshot = "synthetic.{}.pdb".format(int(seed))
    tunnel.caver_cluster_id = 1
    tunnel.tunnel_id = 1
    tunnel.length = float(lens[-1])
    tunnel.curvature = 1.5
    tunnel.bottleneck_radius = 0.5
    tunnel.throughput = 0.5
    tunnel.cost = 0.7
    tunnel.filters_passed = True
    return tunnel


def build_flared_tunnel(max_dist: float, seed: float, passing: bool = True):
    """A long tunnel with a flared tail (radius expansion beyond ~8 A) whose
    distance wanders back toward the start at the end -- the non-monotonic,
    inflating shape the pruning feature targets."""
    rng = np.random.default_rng(42 + int(seed))
    dists = [float(rng.uniform(0.2, 1.0))]
    while dists[-1] < max_dist:
        dists.append(dists[-1] + 0.5)
    x = dists[-1]
    while x > 2.0:
        x -= 0.5
        dists.append(x)
    radii = []
    for v in dists:
        rv = 1.6 + 0.12 * min(v, 8.0)
        if v > 8.0:
            rv += (v - 8.0) * 1.1
        radii.append(rv)
    tunnel = build_synthetic_tunnel(dists, radii, seed=seed)
    tunnel.filters_passed = passing
    return tunnel


def build_stub_tunnel(seed: float, passing: bool = False):
    """A short, unflared tunnel (what fails the length filter in practice)."""
    tunnel = build_synthetic_tunnel([0.5, 1.0, 1.5, 2.0],
                                    [1.5, 1.5, 1.5, 1.5], seed=seed)
    tunnel.filters_passed = passing
    return tunnel


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

    def test_uses_per_tunnel_max_distance_not_endpoint(self):
        # The CAVER `distance` column is non-monotonic: curved/wandering tunnels can
        # descend back toward the starting point, so the last sphere is NOT the tunnel's
        # maximum distance.  Here every tunnel climbs to ~9-12 A and then descends back
        # to ~2 A at its end; using the endpoint would put the P90 valid-range bound at
        # ~2 A and yield no cut, while the true maxima (np.max) place it in the flared
        # tail region where a consensus cut legitimately exists.
        rng = np.random.default_rng(42)
        maxes = (9, 10, 11, 12)
        tunnels = []
        for i, mx in enumerate(maxes):
            dists = [float(rng.uniform(0.2, 1.0))]
            while dists[-1] < mx:
                dists.append(dists[-1] + 0.5)
            x = dists[-1]
            while x > 2.0:
                x -= 0.5
                dists.append(x)
            radii = []
            for v in dists:
                rv = 1.6 + 0.12 * min(v, 8.0)
                if v > 8.0:
                    rv += (v - 8.0) * 1.1
                radii.append(rv)
            tunnels.append(build_synthetic_tunnel(dists, radii, seed=i * 100.0))

        # document the non-monotonicity (endpoint < true max for every tunnel)
        for tunnel in tunnels:
            self.assertLess(tunnel.spheres_data[-1, tunnel_pruning._COL_DISTANCE],
                            tunnel.spheres_data[:, tunnel_pruning._COL_DISTANCE].max())

        cuts = tunnel_pruning.compute_per_cluster_cut(tunnels)
        self.assertIn(1, cuts)
        cut_distance = cuts[1][0]
        # the cut must reflect the true maxima: it lies in the flared tail, well above
        # the ~2 A endpoints (which would have precluded any cut under the old code)
        self.assertGreater(cut_distance, 8.0)


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
        # the offender's guarded position is the first sphere past the cut (idx 28),
        # which already meets the water floor -> keeps [0..28] of 31 spheres
        self.assertEqual(29, n_after)
        # the last kept sphere meets the water-floor contract
        self.assertGreaterEqual(self.offender.spheres_data[-1, tunnel_pruning._COL_RADIUS], 1.4)
        self.assertEqual(n_after, len(self.offender.layer_membership))
        # length and curvature are recomputed from the kept spheres in CAVER's own terms
        exp_len, exp_curv = expected_length_curvature(self.offender)
        self.assertAlmostEqual(self.offender.length, exp_len, places=9)
        self.assertAlmostEqual(self.offender.curvature, exp_curv, places=9)
        # ...not merely the raw CAVER length column of the last kept sphere
        self.assertNotAlmostEqual(self.offender.length, self.offender.spheres_data[-1, 5], places=9)
        self.assertEqual(self.offender.filters_passed,
                         self.offender.bottleneck_radius >= 0
                         and self.offender.length >= 5.0
                         and self.offender.curvature <= 999)

    def test_recomputation_reproduces_caver_values_on_untouched_tunnels(self):
        # The refresh must be expressed in the very terms CAVER reports, or a truncated
        # tunnel stops being comparable with the untouched ones in its own cluster.  The
        # recomputation is therefore pinned against the values parsed from CAVER: it must
        # reproduce them exactly while the geometry is still unmodified.  Note both are
        # measured from the FIRST SPHERE - the gap between the CAVER starting point and
        # that sphere (the 'distance' column at index 0) belongs to neither.
        for tunnel in self.tunnels:
            caver_length, caver_curvature = tunnel.length, tunnel.curvature
            tunnel.recompute_length_and_curvature()
            self.assertAlmostEqual(tunnel.length, caver_length, places=9,
                                   msg="length drifted for " + tunnel.snapshot)
            self.assertAlmostEqual(tunnel.curvature, caver_curvature, places=9,
                                   msg="curvature drifted for " + tunnel.snapshot)
            self.assertGreater(tunnel.spheres_data[0, tunnel_pruning._COL_DISTANCE], 0)

    def test_truncation_never_lengthens_a_tunnel(self):
        # Removing spheres must shorten the recorded length; anything else means the
        # refresh is measuring from somewhere other than CAVER's reference.
        for tunnel in self.tunnels:
            length_before = tunnel.length
            if tunnel_pruning.truncate_tunnel(tunnel, self.cut_distance):
                self.assertLess(tunnel.length, length_before,
                                msg="truncation lengthened " + tunnel.snapshot)

    def test_cost_throughput_bottleneck_keep_caver_values(self):
        # not recomputable from sphere data -> must survive truncation unchanged
        cost0, throughput0, bottleneck0 = (self.offender.cost, self.offender.throughput,
                                           self.offender.bottleneck_radius)
        self.assertTrue(tunnel_pruning.truncate_tunnel(self.offender, self.cut_distance))
        self.assertEqual(self.offender.cost, cost0)
        self.assertEqual(self.offender.throughput, throughput0)
        self.assertEqual(self.offender.bottleneck_radius, bottleneck0)

    def test_filters_passed_reflects_refreshed_length(self):
        # a minimum length above the pruned length must turn the (refreshed)
        # relevant-tunnel verdict off
        probe = build_single_tunnel(datasection2)
        caver_length = probe.length
        tunnel_pruning.truncate_tunnel(probe, self.cut_distance)
        pruned_length = probe.length
        self.assertLess(pruned_length, caver_length)  # shorter than the CAVER value

        accept = build_single_tunnel(datasection2)
        tunnel_pruning.truncate_tunnel(accept, self.cut_distance)
        self.assertTrue(accept.filters_passed)

        reject = build_single_tunnel(datasection2, {"relevant_tunnel_min_length": pruned_length + 0.5})
        tunnel_pruning.truncate_tunnel(reject, self.cut_distance)
        self.assertFalse(reject.filters_passed)

    def test_cut_index_uses_first_sphere_at_or_after_cut(self):
        # Distance column is NOT monotonic (descent between index 1 and 2): the first
        # sphere with distance >= cut is index 0, so `searchsorted` (undefined on
        # unsorted input) would return index 4 and over-truncate, whereas the guarded
        # `argmax(dists >= cut)` must start at 0.
        dists = [1.274, 0.540, 0.082, 0.033, 1.627, 1.826]
        radii = [1.5] * 6
        # sanity: the profile really descends
        self.assertTrue(np.any(np.diff(dists) < 0))
        tunnel = build_synthetic_tunnel(dists, radii, seed=3.0)
        cut_distance = 1.12

        # first sphere at/after the cut is index 0 (argmax rule)
        reach = tunnel.spheres_data[:, tunnel_pruning._COL_DISTANCE] >= cut_distance
        self.assertTrue(reach.any())
        self.assertEqual(0, int(np.argmax(reach)))
        # `searchsorted` gives a different (wrong) index on this unsorted column
        self.assertGreater(int(np.searchsorted(tunnel.spheres_data[:, tunnel_pruning._COL_DISTANCE],
                                               cut_distance, side="left")), 0)

        n_before = len(tunnel.spheres_data)
        self.assertTrue(tunnel_pruning.truncate_tunnel(tunnel, cut_distance))
        # radii are all equal so the bottleneck lands at index 0; the guard brackets it
        # at index 1 and the cut keeps through the first floor-passing sphere at/after
        # index 2 -> [0..2].  Had `searchsorted` been used, the cut would have started
        # at index 4 and kept the whole tunnel instead.
        self.assertEqual(3, len(tunnel.spheres_data))
        self.assertGreaterEqual(tunnel.spheres_data[-1, tunnel_pruning._COL_RADIUS], 1.4)
        self.assertLess(len(tunnel.spheres_data), n_before)

    def test_bottleneck_is_bracketed_by_the_guard(self):
        # `bottleneck_radius` is not refreshed after truncation, so the kept chain must
        # still contain the point it refers to.  CAVER locates the bottleneck on the
        # continuous axis while the profile only samples it, so the true minimum sits
        # BETWEEN spheres: the guard keeps one sphere past argmin(R) to bracket it.
        dists = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        radii = [1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.5, 1.5, 1.5, 1.6, 1.7, 1.8]
        tunnel = build_synthetic_tunnel(dists, radii, seed=11.0)
        # the CAVER value is below every sampled radius, as it is for 7465 of the 8151
        # md1 tunnels -- it is not one of the profile radii
        tunnel.bottleneck_radius = 0.97
        cut_distance = 2.5  # first reached at idx 4

        self.assertTrue(tunnel_pruning.truncate_tunnel(tunnel, cut_distance,
                                                       water_radius=1.4))
        # guard = max(cut idx 4, argmin(R) idx 5 bracketed to 6, +1) = 7; idx 7 meets the
        # floor, so [0..7] is kept -- the narrowest sphere (5) and its outward neighbour
        # (6), between which the true bottleneck lies, both survive
        self.assertEqual(8, len(tunnel.spheres_data))
        self.assertEqual(1.0, float(tunnel.spheres_data[5, tunnel_pruning._COL_RADIUS]))
        self.assertGreaterEqual(tunnel.spheres_data[-1, tunnel_pruning._COL_RADIUS], 1.4)

    def test_guard_ignores_a_far_sphere_matching_the_bottleneck_radius(self):
        # Regression: guarding on the profile radius numerically closest to
        # `bottleneck_radius` is not a position at all.  Here a far tail sphere (idx 10)
        # is 0.0001 wide of the CAVER value while the real bottleneck is at idx 5 -- the
        # same coincidence that, on md1, moved the guard 45 spheres out and left the
        # tunnel unprunable.  The guard must key off argmin(R), not the value match.
        dists = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        radii = [1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 1.5, 1.5, 1.5, 1.6, 1.0001, 1.8]
        tunnel = build_synthetic_tunnel(dists, radii, seed=13.0)
        tunnel.bottleneck_radius = 1.0002  # closest profile radius is the tail sphere 10
        self.assertEqual(10, int(np.argmin(np.abs(np.asarray(radii) - tunnel.bottleneck_radius))))
        self.assertEqual(5, int(np.argmin(radii)))

        self.assertTrue(tunnel_pruning.truncate_tunnel(tunnel, 2.5, water_radius=1.4))
        # guarded off argmin(R)=5 -> keeps [0..7], not the [0..11] the value match forces
        self.assertEqual(8, len(tunnel.spheres_data))

    def test_far_cut_beyond_tunnel_max_is_no_op(self):
        # with a cut beyond the tunnel's max distance the guarded position clamps to the
        # last sphere, which meets the water floor: there is no tail beyond the guard, so
        # the tunnel stays intact.  (The original good_gardener CSV writer dropped that
        # last sphere here -- the water-floor off-by-one this module now fixes.)
        far_cut = max(self.offender.spheres_data[:, 3]) + 100
        n_before = len(self.offender.spheres_data)
        truncated = tunnel_pruning.truncate_tunnel(self.offender, far_cut)
        self.assertFalse(truncated)
        self.assertEqual(n_before, len(self.offender.spheres_data))

    def test_water_floor_advances_cut_to_first_passing_sphere(self):
        # When the guarded position sits on sub-floor spheres, the cut must advance to
        # the first sphere that meets the floor and KEEP it: the old `n_keep = adj`
        # kept one sphere short, leaving the last kept sphere below the floor.
        # Distances are monotonic; the cut is first reached at idx 4, the bottleneck
        # (idx 6) pushes the guarded position to idx 7 (R=1.0, sub-floor), and idx 8
        # (R=1.6) is the first floor-passing sphere.
        dists = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        radii = [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.0, 1.0, 1.6, 2.0]
        tunnel = build_synthetic_tunnel(dists, radii, seed=7.0)
        cut_distance = 2.5

        self.assertTrue(tunnel_pruning.truncate_tunnel(tunnel, cut_distance,
                                                       water_radius=1.4))
        # keeps [0..8]: the protected bottleneck (6), the sub-floor sphere after it
        # (7) and the floor-passing guard sphere (8)
        self.assertEqual(9, len(tunnel.spheres_data))
        self.assertEqual(1.6, float(tunnel.spheres_data[-1, tunnel_pruning._COL_RADIUS]))
        self.assertGreaterEqual(tunnel.spheres_data[-1, tunnel_pruning._COL_RADIUS], 1.4)

    def test_water_floor_unmet_keeps_tunnel_intact(self):
        # No sphere at or after the guarded position meets the floor, so the tunnel
        # cannot be truncated without violating the contract -> left untouched.
        dists = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        radii = [1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.0, 1.0, 1.2, 1.3]
        tunnel = build_synthetic_tunnel(dists, radii, seed=7.0)
        n_before = len(tunnel.spheres_data)
        self.assertFalse(tunnel_pruning.truncate_tunnel(tunnel, 2.5, water_radius=1.4))
        self.assertEqual(n_before, len(tunnel.spheres_data))


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

    def test_non_passing_tunnels_do_not_influence_cut_or_pruning(self):
        # The consensus cut is computed from relevant (filters_passed) tunnels only:
        # short non-passing stubs must not shift it (they would pull the P10 of the
        # per-tunnel max distances into the core region), and the relevant tunnels'
        # truncation must be identical with or without the stubs present.
        def make_good():
            return [build_flared_tunnel(mx, seed)
                    for mx, seed in ((9, 100.0), (10, 200.0), (11, 300.0), (12, 400.0))]

        solo = make_good()
        stats_solo = tunnel_pruning.prune_tunnels(solo)
        self.assertGreaterEqual(stats_solo["n_truncated"], 1)  # not a vacuous comparison
        solo_kept = [t.spheres_data.copy() for t in solo]

        mixed = make_good() + [build_stub_tunnel(900.0 + i) for i in range(3)]
        stubs = mixed[4:]
        stats = tunnel_pruning.prune_tunnels(mixed)
        self.assertEqual(4, stats["n_relevant"])

        for tunnel, kept in zip(mixed[:4], solo_kept):
            self.assertTrue(np.array_equal(tunnel.spheres_data, kept))
        for stub in stubs:  # non-passing tunnels stay untouched
            self.assertEqual(4, len(stub.spheres_data))

    def test_non_passing_tunnel_is_never_truncated(self):
        # A non-passing tunnel with the same offensive (flared, wandering) shape as
        # the relevant tunnels of the cluster is still left entirely untouched: only
        # relevant tunnels take part in the cut and in the truncation.  It would have
        # been flagged (and truncated) had it been relevant.
        a = build_flared_tunnel(11, 500.0, passing=True)
        b = build_flared_tunnel(10, 550.0, passing=True)
        c = build_flared_tunnel(11.5, 600.0, passing=False)
        c_before = c.spheres_data.copy()
        n_a_before = len(a.spheres_data)

        # the non-passing tunnel would have been flagged if it took part in pruning
        cut = tunnel_pruning.compute_per_cluster_cut([a, b])[1][0]
        verdicts = tunnel_pruning.verify_postcut([c], cut)
        key = (c.snapshot, c.caver_cluster_id, c.tunnel_id)
        self.assertIn(key, verdicts)
        self.assertIn(verdicts[key]["category"], ("inflating", "curved", "both"))

        stats = tunnel_pruning.prune_tunnels([a, b, c])

        self.assertEqual(1, stats["n_truncated"])  # the extending relevant tunnel a
        self.assertLess(len(a.spheres_data), n_a_before)
        self.assertEqual(len(c_before), len(c.spheres_data))  # the non-passing twin
        self.assertTrue(np.array_equal(c.spheres_data, c_before))

    def test_cluster_with_fewer_than_two_relevant_tunnels_is_untouched(self):
        # A cluster with a single relevant tunnel has no consensus cut, so nothing --
        # not even the relevant tunnel -- is truncated.
        passing = build_flared_tunnel(10, 700.0, passing=True)
        stubs = [build_stub_tunnel(800.0 + i) for i in range(3)]
        before = [t.spheres_data.copy() for t in [passing] + stubs]

        stats = tunnel_pruning.prune_tunnels([passing] + stubs)

        self.assertEqual(1, stats["n_clusters"])
        self.assertEqual(1, stats["n_relevant"])
        self.assertEqual(0, stats["n_clusters_with_cut"])
        self.assertEqual(0, stats["n_truncated"])
        for tunnel, kept in zip([passing] + stubs, before):
            self.assertTrue(np.array_equal(tunnel.spheres_data, kept))

    def test_all_non_passing_cluster_is_untouched(self):
        stubs = [build_stub_tunnel(810.0 + i) for i in range(3)]
        before = [t.spheres_data.copy() for t in stubs]

        stats = tunnel_pruning.prune_tunnels(stubs)

        self.assertEqual(0, stats["n_relevant"])
        self.assertEqual(0, stats["n_clusters_with_cut"])
        self.assertEqual(0, stats["n_truncated"])
        for tunnel, kept in zip(stubs, before):
            self.assertTrue(np.array_equal(tunnel.spheres_data, kept))


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
        return SimpleNamespace(orig_entities=[cluster], parameters=parameters,
                               transform_mat=np.eye(4),
                               starting_point_coords=np.array([[0.0, 0.0, 0.0]]),
                               md_label="test_md")

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
