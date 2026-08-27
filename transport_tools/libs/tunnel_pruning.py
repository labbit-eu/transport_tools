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

"""
Pruning of CAVER tunnel profiles.

Implements the tunnel-pruning workflow previously developed as the standalone
``good_gardener`` pipeline, adapted to operate directly on the in-memory
``Tunnel`` objects built by TransportTools at stage 2.  Its goal is to identify
tunnel segments that extend beyond the natural protein surface into empty
solvent space, and truncate them while preserving geometrically meaningful
tunnel extensions.

The workflow has three stages, applied per CAVER cluster within a single MD
simulation:

1.  ``compute_per_cluster_cut`` -- determine a consensus exit distance per
    cluster (joint-transition score: radius expansion times survival drop).
2.  ``verify_postcut`` -- classify each tunnel that extends beyond the cut as
    valid or offensive (inflating into empty space and/or curved/wandering).
3.  ``truncate_tunnel`` -- cut the offensive tunnels' sphere data at the
    guarded position (bottleneck protection + water-radius floor), keeping the
    original number of spheres otherwise untouched.

Only tunnels that pass the relevant-tunnel filter (``filters_passed``) take
part in the workflow: they define the consensus cut (and the post-cut inflation
reference) and are the only tunnels that may be truncated.  Tunnels that fail
the filter are left entirely untouched and do not influence the cut, so a
cluster with fewer than two relevant tunnels has no consensus cut.

Pruning is applied in-place to the passed ``Tunnel`` objects: their
``spheres_data`` is truncated, and the derived ``length``, ``curvature``,
``layer_membership`` and ``filters_passed`` attributes are refreshed so that
all downstream stages (3+ of TransportTools) see the pruned tunnel geometry.
``cost``/``throughput``/``bottleneck_radius`` are not recomputable from the
sphere data and keep their CAVER values (logged limitation).

Sphere data columns of ``Tunnel.spheres_data``: ``[x, y, z, distance, R, length]``.
Note that the ``length`` column is not the tunnel length but CAVER's resampling
parameter along the axis (a constant ~0.5 A step offset by ``distance[0]``); the
tunnel length is the arc length of the sphere centerline, see
``Tunnel.recompute_length_and_curvature()``.
"""

from __future__ import annotations

__version__ = '0.9.8'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import numpy as np
from logging import getLogger
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from transport_tools.libs.networks import Tunnel

logger = getLogger(__name__)

# column indices in Tunnel.spheres_data
_COL_DISTANCE = 3
_COL_RADIUS = 4
_COL_LENGTH = 5


# ── small helpers ──────────────────────────────────────────────────────────


def _group_by_cluster(tunnels: List[Tunnel]) -> Dict[int, List[Tunnel]]:
    """Group Tunnel objects by their CAVER cluster id, preserving order."""
    groups: Dict[int, List[Tunnel]] = {}
    for tunnel in tunnels:
        groups.setdefault(tunnel.caver_cluster_id, []).append(tunnel)
    return groups


def _bin_spheres(
    cluster_tunnels: List[Tunnel], bin_size: float = 0.5
) -> Tuple[List[List[float]], List[float]]:
    """Bin sphere radii by distance.  Returns (binned_radii, survival_fraction)."""
    items: List[Tuple[float, float, Tuple[str, int, int]]] = []
    keys_seen: set = set()
    for tunnel in cluster_tunnels:
        key = (tunnel.snapshot, tunnel.caver_cluster_id, tunnel.tunnel_id)
        keys_seen.add(key)
        dists = tunnel.spheres_data[:, _COL_DISTANCE]
        radii = tunnel.spheres_data[:, _COL_RADIUS]
        for dist, radius in zip(dists, radii):
            items.append((dist, radius, key))

    if not items:
        return [], []

    max_dist = max(dist for dist, _, _ in items)
    n_bins = int(np.ceil(max_dist / bin_size)) or 1

    binned_radii: List[List[float]] = [[] for _ in range(n_bins)]
    bin_tunnel_keys: List[set] = [set() for _ in range(n_bins)]

    for dist, radius, key in items:
        idx = int(dist // bin_size)
        if idx >= n_bins:
            idx = n_bins - 1
        binned_radii[idx].append(radius)
        bin_tunnel_keys[idx].add(key)

    n_total = len(keys_seen)
    survival = [len(keys) / n_total for keys in bin_tunnel_keys]

    return binned_radii, survival


def _smooth(arr: np.ndarray, window: int = 3) -> np.ndarray:
    """Moving average, NaN-safe."""
    if len(arr) == 0:
        return arr
    res = np.copy(arr)
    half = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        valid = arr[lo:hi][~np.isnan(arr[lo:hi])]
        res[i] = np.mean(valid) if len(valid) > 0 else np.nan
    return res


def _slope_r_over_l(lengths: np.ndarray, radii: np.ndarray, min_points: int = 3) -> float:
    """Linear slope of radius vs length (length-ordered)."""
    if len(lengths) < min_points:
        return 0.0
    var_len = np.var(lengths)
    if var_len < 1e-12:
        return 0.0
    return float(np.cov(lengths, radii, ddof=0)[0, 1] / var_len)


# ── Step 1: cluster-level cut ──────────────────────────────────────────────


def compute_per_cluster_cut(
    tunnels: List[Tunnel],
    bin_size: float = 0.5,
    surv_perc_range: Tuple[float, float] = (0.1, 0.9),
    core_range: Tuple[float, float] = (0.3, 0.6),
    mode: str = "first",
    min_joint_threshold: float = 0.01,
) -> Dict[int, Tuple[float, float, float]]:
    """Joint transition score per cluster.

    :param tunnels: Tunnel objects of a single MD simulation.  The caller
        controls which tunnels are included -- ``prune_tunnels`` passes only the
        relevant (``filters_passed``) ones; the function itself works on the
        population it is given.
    :param bin_size: width of distance bins in Angstroms.
    :param surv_perc_range: percentile range of the per-tunnel maximum distances, bounding
        the region in which a cut may be selected.  Equivalently a survival range: the
        default ``(0.1, 0.9)`` spans the distances at which 90% and 10% of the cluster's
        tunnels still reach, excluding both the core every tunnel covers and the sparse
        tail only the longest few reach.
    :param core_range: fraction of the distance range defining the tunnel core region.
    :param mode: cut selection mode: ``"first"`` (lowest-distance bin passing the joint
        threshold) or ``"peak"`` (bin with the maximal joint score).
    :param min_joint_threshold: absolute noise floor of the joint score in ``mode="first"``.
        Applied on top of the adaptive threshold (twice the 90th percentile of the positive
        joint scores inside the core region), which is what normally selects the cut; this
        floor only suppresses clusters whose core is so quiet that the adaptive threshold
        degenerates.  Set to 0 to disable.
    :return: {cluster_id: (cut_distance, core_radius, core_q3)} where core_q3 is the
        75th percentile of per-bin median radii in the core region.
    """
    cuts: Dict[int, Tuple[float, float, float]] = {}
    for cid, cluster_tunnels in _group_by_cluster(tunnels).items():
        binned, survival = _bin_spheres(cluster_tunnels, bin_size)
        n_bins = len(binned)
        if n_bins < 3:
            continue

        centres = np.arange(n_bins) * bin_size + bin_size / 2.0
        surv = np.array(survival, dtype=np.float64)

        med_r = np.full(n_bins, np.nan)
        for bi in range(n_bins):
            if binned[bi]:
                med_r[bi] = np.median(binned[bi])

        # Valid range: [P10, P90] of per-tunnel max distances.
        # The CAVER `distance` column is NOT monotonic (curved/wandering tunnels can
        # descend back toward the start), so the endpoint is not the maximum.
        max_dists = np.array([tunnel.spheres_data[:, _COL_DISTANCE].max()
                              for tunnel in cluster_tunnels])
        if len(max_dists) < 2:
            continue
        d_min = float(np.percentile(max_dists, surv_perc_range[0] * 100))
        d_max = float(np.percentile(max_dists, surv_perc_range[1] * 100))
        in_range = (centres > d_min) & (centres <= d_max)

        # Core region
        g_min, g_max = float(np.min(centres)), float(np.max(centres))
        clo = g_min + core_range[0] * (g_max - g_min)
        chi = g_min + core_range[1] * (g_max - g_min)
        core = (centres >= clo) & (centres <= chi) & ~np.isnan(med_r)
        if np.sum(core) < 2:
            core = (
                (centres >= g_min + 0.25 * (g_max - g_min))
                & (centres <= g_min + 0.75 * (g_max - g_min))
                & ~np.isnan(med_r)
            )
        if np.sum(core) == 0:
            continue

        core_r = float(np.median(med_r[core]))
        core_q3 = float(np.percentile(med_r[core], 75))

        # Smooth survival and radius
        smooth_surv = _smooth(surv, 3)
        smooth_r = _smooth(med_r, 3)

        # Joint score: smoothed radius rise x survival drop
        rise = (
            np.maximum(0, (smooth_r - core_r) / core_r)
            if core_r > 0
            else np.zeros(n_bins)
        )
        drop = np.zeros(n_bins)
        drop[1:] = -np.diff(smooth_surv)
        drop = np.maximum(drop, 0)
        joint = rise * drop

        if mode == "first":
            core_j = joint[core]
            pos = core_j[core_j > 0]
            thresh = float(np.percentile(pos, 90)) * 2 if len(pos) > 0 else 0.01
            mask = in_range & (joint > thresh)
            if min_joint_threshold > 0:
                mask = mask & (joint > min_joint_threshold)
            if np.any(mask):
                cuts[cid] = (float(centres[np.argmax(mask)]), core_r, core_q3)

        elif mode == "peak":
            valid = np.where(in_range & ~np.isnan(joint), joint, -np.inf)
            if np.any(valid > 0):
                cuts[cid] = (float(centres[np.argmax(valid)]), core_r, core_q3)

        else:
            raise ValueError("Unknown mode {!r}".format(mode))

    return cuts


# ── Step 2: per-tunnel post-cut verification ───────────────────────────────


def verify_postcut(
    tunnels: List[Tunnel],
    cut_distance: float,
    eff_thresh: float = 0.5,
    slope_percentile: float = 90,
) -> Dict[Tuple[str, int, int], Dict[str, object]]:
    """Check each tunnel that extends beyond *cut_distance*.

    Inflation detection: the cluster-level core expansion reference is the
    ``slope_percentile``-th percentile of the positive core slopes (linear fit
    of radius vs length over spheres before the cut) of the passed tunnels --
    ``prune_tunnels`` passes only the relevant population, so the reference
    reflects it.

    Curvature detection: path efficiency (distance gain / length gain) of the
    post-cut segment below ``eff_thresh`` flags a wandering path.

    :param tunnels: Tunnel objects of a single CAVER cluster.
    :param cut_distance: consensus cluster exit distance.
    :param eff_thresh: path-efficiency threshold below which a post-cut segment is curved.
    :param slope_percentile: percentile of cluster core-expansion slopes used as the
        inflation reference.
    :return: {(snapshot, cluster_id, tunnel_id): {"inflating": bool, "curved": bool,
        "category": str}}
    """
    # Cluster-level core expansion reference
    core_slopes: List[float] = []
    for tunnel in tunnels:
        dists = tunnel.spheres_data[:, _COL_DISTANCE]
        mask = dists < cut_distance
        slope = _slope_r_over_l(tunnel.spheres_data[mask, _COL_LENGTH],
                                tunnel.spheres_data[mask, _COL_RADIUS])
        if slope > 0:
            core_slopes.append(slope)

    core_slope_ref = (
        float(np.percentile(core_slopes, slope_percentile)) if core_slopes else 0.0
    )

    verdicts: Dict[Tuple[str, int, int], Dict[str, object]] = {}
    for tunnel in tunnels:
        key = (tunnel.snapshot, tunnel.caver_cluster_id, tunnel.tunnel_id)
        distances = tunnel.spheres_data[:, _COL_DISTANCE]
        lengths = tunnel.spheres_data[:, _COL_LENGTH]
        radii = tunnel.spheres_data[:, _COL_RADIUS]

        # Extends beyond cut?
        pc = distances >= cut_distance
        if not pc.any():
            continue

        # Curvature signal
        pc_len = lengths[pc]
        pc_dst = distances[pc]
        len_gain = pc_len[-1] - pc_len[0]
        dst_gain = pc_dst[-1] - pc_dst[0]
        efficiency = dst_gain / len_gain if len_gain > 0 else 1.0
        curved = efficiency < eff_thresh

        # Inflation signal
        inflating = False
        if core_slope_ref > 0 and np.sum(pc) >= 3:
            slope_pc = _slope_r_over_l(lengths[pc], radii[pc])
            if slope_pc > core_slope_ref:
                inflating = True

        category = (
            "both"
            if inflating and curved
            else "inflating" if inflating else "curved" if curved else "none"
        )
        verdicts[key] = {"inflating": inflating, "curved": curved, "category": category}

    return verdicts


# ── Step 3: truncation ─────────────────────────────────────────────────────


def truncate_tunnel(tunnel: Tunnel, cut_distance: float, water_radius: float = 1.4) -> bool:
    """Truncate a tunnel's sphere data at the guarded cut position, in place.

    Two rules govern the actual cut point:
      1. Bottleneck protection -- the cut never reaches the tunnel's bottleneck.
         CAVER locates the bottleneck on the continuous tunnel axis while the
         profile samples that axis every ~0.5 A, so the reported
         ``bottleneck_radius`` is not one of the profile radii and the true
         narrowest point falls *between* two spheres.  Keeping one sphere past
         ``argmin(R)`` therefore brackets it on both sides, which matters
         because ``bottleneck_radius`` is deliberately not refreshed after
         truncation: the value must stay valid for the kept geometry.  The
         invariant is asserted before the cut is committed.
      2. Water-radius floor -- the cut is the first sphere at or after the
         guarded position whose radius is >= *water_radius*, and that sphere is
         kept, so the last kept sphere always meets the floor.  When no sphere
         at or after the guarded position meets the floor, the tunnel is left
         untouched.

    Note: the standalone ``good_gardener`` pipeline's CSV writer truncated one
    sphere short of this position, leaving the last kept sphere below the floor
    whenever the floor loop advanced; its own guard marker already plotted the
    floor-passing sphere, which is the contract this implementation follows.

    The tunnel's ``length`` and ``curvature`` are recomputed by
    ``Tunnel.recompute_length_and_curvature()`` -- which follows CAVER's own
    definitions and reproduces its values for an unmodified tunnel -- and
    ``layer_membership``/``filters_passed`` are refreshed to stay consistent
    with the truncated geometry.  The CAVER ``cost``, ``throughput`` and
    ``bottleneck_radius`` are *not* recomputable from the sphere data and keep
    their original values (documented, logged limitation).

    :param tunnel: Tunnel to truncate in place.
    :param cut_distance: distance cut at which the tunnel tail is removed.
    :param water_radius: minimum radius of the last kept sphere.
    :return: True when the tunnel was actually shortened, False otherwise.
    """
    dists = tunnel.spheres_data[:, _COL_DISTANCE]
    radii = tunnel.spheres_data[:, _COL_RADIUS]

    # First sphere with distance >= the cut (the CAVER `distance` column is NOT
    # monotonic — curved/wandering tunnels loop back toward the start — so
    # `searchsorted` is undefined here).  When no sphere reaches the cut, clamp to
    # the last sphere, which makes the truncation a no-op below: the tunnel does
    # not extend beyond the cut.
    reaches_cut = dists >= cut_distance
    if reaches_cut.any():
        start = int(np.argmax(reaches_cut))
    else:
        start = len(radii) - 1

    # Bottleneck protection: `argmin(R)` is the narrowest *sampled* sphere and the
    # true (continuous) bottleneck lies within one sampling step of it, so keeping
    # one sphere further out brackets the real narrowest point inside the kept
    # chain.  Deliberately positional -- searching instead for the profile radius
    # numerically closest to `bottleneck_radius` locates nothing, since the two
    # differ by ~0.006 A on average and the closest match can sit anywhere in the
    # profile.
    bottleneck_idx = int(np.argmin(radii))
    protected = min(bottleneck_idx + 1, len(radii) - 1)
    earliest = max(start, protected + 1)  # +1 because we keep it

    # First sphere at or after the guarded position whose radius meets the water
    # floor; the cut keeps through it, so the last kept sphere always has radius
    # >= water_radius.  When no such sphere exists, `adj` runs past the end and
    # the tunnel is left untouched.
    adj = earliest
    while adj < len(radii) and radii[adj] < water_radius:
        adj += 1
    n_keep = adj + 1  # keep [0, adj]; adj >= earliest >= 1 always holds

    if n_keep >= len(tunnel.spheres_data):
        return False

    # Invariant: the cut never reaches the bottleneck, so the un-refreshed CAVER
    # bottleneck_radius remains a valid property of the truncated tunnel.
    # Structurally guaranteed by `earliest` above; asserted to guard against
    # future changes to the cut logic.
    assert protected + 1 <= n_keep, (
        "Tunnel {} of cluster {}: truncation to {} sphere(s) would reach the "
        "bottleneck bracketed at index {} -- the un-refreshed CAVER "
        "bottleneck_radius must remain valid for the kept geometry"
        .format(tunnel.tunnel_id, tunnel.caver_cluster_id, n_keep, protected)
    )

    tunnel.spheres_data = tunnel.spheres_data[:n_keep]
    tunnel.recompute_length_and_curvature()
    tunnel.compute_layer_membership()
    tunnel.apply_relevance_filters()
    logger.debug("Truncated tunnel %d of cluster %d to %d sphere(s): length=%.6g, "
                 "curvature=%.6g (cost=%g, throughput=%g, bottleneck_radius=%g keep "
                 "their CAVER values).", tunnel.tunnel_id, tunnel.caver_cluster_id,
                 n_keep, tunnel.length, tunnel.curvature, tunnel.cost,
                 tunnel.throughput, tunnel.bottleneck_radius)
    return True


# ── full workflow ──────────────────────────────────────────────────────────


def prune_tunnels(
    tunnels: List[Tunnel],
    mode: str = "first",
    bin_size: float = 0.5,
    surv_perc_range: Tuple[float, float] = (0.1, 0.9),
    core_range: Tuple[float, float] = (0.3, 0.6),
    min_joint_threshold: float = 0.01,
    eff_thresh: float = 0.5,
    slope_percentile: float = 90,
    water_radius: float = 1.4,
) -> Dict[str, int]:
    """Compute cuts, verify tunnels, and truncate offensive tunnels in place.

    Applies ``compute_per_cluster_cut``, ``verify_postcut`` and
    ``truncate_tunnel`` across all clusters present in *tunnels*.  Only tunnels
    that pass the relevant-tunnel filter (``filters_passed == True``) take part:
    the consensus cut and the post-cut inflation reference are computed from
    them, and only they may be truncated -- and only if flagged as inflating
    and/or curved.  Non-passing tunnels are left entirely untouched and do not
    influence the cut; a cluster with fewer than two relevant tunnels has no
    consensus cut and is logged at debug level.

    :param tunnels: Tunnel objects of a single MD simulation.
    :param mode: cut selection mode, ``"first"`` or ``"peak"``.
    :param bin_size: width of distance bins in Angstroms.
    :param surv_perc_range: percentile range of the per-tunnel maximum distances (equivalently
        a survival range) bounding the region in which a cut may be selected.
    :param core_range: fraction of the distance range defining the tunnel core region.
    :param min_joint_threshold: absolute noise floor of the joint score in ``"first"`` mode,
        applied on top of the adaptive core-derived threshold (0 disables).
    :param eff_thresh: path-efficiency threshold below which a post-cut segment is curved.
    :param slope_percentile: percentile of cluster core-expansion slopes used as the
        inflation reference.
    :param water_radius: minimum radius of the last kept sphere.
    :return: statistics of the pruning pass: number of clusters, number of
        relevant tunnels, clusters with a cut, tunnels extending beyond a cut,
        offensive tunnels (by category) and tunnels actually truncated.
    """
    # Only relevant tunnels take part in pruning: they define the consensus cut
    # and are the only tunnels that may be truncated (design decision --
    # non-passing tunnels are excluded from layering anyway, and letting them
    # shape the cut would let e.g. short stubs pull the valid-range percentiles
    # into the core region and distort the consensus exit).
    relevant = [tunnel for tunnel in tunnels if tunnel.filters_passed]
    cuts = compute_per_cluster_cut(
        relevant,
        bin_size=bin_size,
        surv_perc_range=surv_perc_range,
        core_range=core_range,
        mode=mode,
        min_joint_threshold=min_joint_threshold,
    )

    stats: Dict[str, int] = {
        "n_clusters": len(_group_by_cluster(tunnels)),
        "n_relevant": len(relevant),
        "n_clusters_with_cut": len(cuts),
        "n_extending": 0,
        "n_offensive": 0,
        "n_inflated": 0,
        "n_curved": 0,
        "n_both": 0,
        "n_truncated": 0,
    }

    for cid, cluster_tunnels in _group_by_cluster(tunnels).items():
        cut_distance = cuts.get(cid)
        if cut_distance is None:
            logger.debug(
                "Pruning: CAVER cluster %d has no consensus cut "
                "(%d of %d tunnel(s) relevant); the cluster is left untouched.",
                cid, sum(1 for t in cluster_tunnels if t.filters_passed),
                len(cluster_tunnels),
            )
            continue
        cut_distance = cut_distance[0]

        # Only the relevant tunnels of this cluster are verified and may be
        # truncated; the inflation reference below is thus also relevant-only.
        relevant_cluster = [t for t in cluster_tunnels if t.filters_passed]

        verdicts = verify_postcut(
            relevant_cluster, cut_distance,
            eff_thresh=eff_thresh, slope_percentile=slope_percentile,
        )
        stats["n_extending"] += len(verdicts)

        for tunnel in relevant_cluster:
            flags = verdicts.get((tunnel.snapshot, tunnel.caver_cluster_id, tunnel.tunnel_id))
            if flags is None:
                continue
            category = flags["category"]
            if category == "none":
                continue
            stats["n_offensive"] += 1
            if category in ("inflating", "both"):
                stats["n_inflated"] += 1
            if category in ("curved", "both"):
                stats["n_curved"] += 1
            if category == "both":
                stats["n_both"] += 1
            if truncate_tunnel(tunnel, cut_distance, water_radius=water_radius):
                stats["n_truncated"] += 1

    return stats
