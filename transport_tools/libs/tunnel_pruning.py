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

Pruning is applied in-place to the passed ``Tunnel`` objects: their
``spheres_data`` is truncated, and the derived ``length``, ``layer_membership``
and ``filters_passed`` attributes are refreshed so that all downstream stages
(3+ of TransportTools) see the pruned tunnel geometry.

Sphere data columns of ``Tunnel.spheres_data``: ``[x, y, z, distance, R, length]``.
"""

from __future__ import annotations

__version__ = '0.9.8'
__author__ = 'Igor Marchlewski'
__mail__ = 'igomar@amu.edu.pl'

import numpy as np
from logging import getLogger
from typing import Dict, List, Tuple

logger = getLogger(__name__)

# column indices in Tunnel.spheres_data
_COL_DISTANCE = 3
_COL_RADIUS = 4
_COL_LENGTH = 5


# ── small helpers ──────────────────────────────────────────────────────────


def _group_by_cluster(tunnels: list) -> Dict[int, list]:
    """Group Tunnel objects by their CAVER cluster id, preserving order."""
    groups: Dict[int, list] = {}
    for tunnel in tunnels:
        groups.setdefault(tunnel.caver_cluster_id, []).append(tunnel)
    return groups


def _bin_spheres(
    cluster_tunnels: list, bin_size: float = 0.5
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
    tunnels: list,
    bin_size: float = 0.5,
    surv_perc_range: Tuple[float, float] = (0.1, 0.9),
    core_range: Tuple[float, float] = (0.3, 0.6),
    mode: str = "first",
    min_joint_threshold: float = 0.01,
) -> Dict[int, Tuple[float, float, float]]:
    """Joint transition score per cluster.

    :param tunnels: Tunnel objects of a single MD simulation.
    :param bin_size: width of distance bins in Angstroms.
    :param surv_perc_range: percentile range of per-tunnel maximum distances defining the
        valid region for cut selection.
    :param core_range: fraction of the distance range defining the tunnel core region.
    :param mode: cut selection mode: ``"first"`` (lowest-distance bin passing the joint
        threshold) or ``"peak"`` (bin with the maximal joint score).
    :param min_joint_threshold: minimum joint-score value for a bin to be considered in
        ``mode="first"``.  Acts as a noise floor.  Set to 0 to disable.
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

        # Valid range: [P10, P90] of per-tunnel max distances
        max_dists = np.array([tunnel.spheres_data[-1, _COL_DISTANCE]
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
    tunnels: list,
    cut_distance: float,
    eff_thresh: float = 0.5,
    slope_percentile: float = 90,
) -> Dict[Tuple[str, int, int], Dict[str, object]]:
    """Check each tunnel that extends beyond *cut_distance*.

    Inflation detection: the cluster-level core expansion reference is the
    ``slope_percentile``-th percentile of all positive core slopes (linear fit
    of radius vs length over spheres before the cut).  A tunnel is flagged as
    inflating when its post-cut expansion slope exceeds this reference.

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


def truncate_tunnel(
    tunnel, cut_distance: float, water_radius: float = 1.4
) -> bool:
    """Truncate a tunnel's sphere data at the guarded cut position, in place.

    Two rules govern the actual cut point (same as in ``good_gardener``):
      1. Bottleneck protection -- never remove the narrowest sphere.
      2. Water-radius floor -- advance until radius >= *water_radius*.

    The tunnel's ``length``, ``layer_membership`` and ``filters_passed``
    attributes are refreshed to stay consistent with the truncated geometry.
    :param tunnel: Tunnel to truncate in place.
    :param cut_distance: distance cut at which the tunnel tail is removed.
    :param water_radius: minimum radius of the last kept sphere.
    :return: True when the tunnel was actually shortened, False otherwise.
    """
    dists = tunnel.spheres_data[:, _COL_DISTANCE]
    radii = tunnel.spheres_data[:, _COL_RADIUS]

    start = int(np.searchsorted(dists, cut_distance, side="left"))
    if start >= len(radii):
        start = len(radii) - 1

    bottleneck_idx = int(np.argmin(radii))  # must keep this sphere
    earliest = max(start, bottleneck_idx + 1)  # +1 because we keep it

    adj = earliest
    while adj < len(radii) and radii[adj] < water_radius:
        adj += 1
    n_keep = adj if adj > 0 else start  # never cut before the distance cut

    if n_keep == 0 or n_keep >= len(tunnel.spheres_data):
        return False

    tunnel.spheres_data = tunnel.spheres_data[:n_keep]
    tunnel.length = float(tunnel.spheres_data[-1, _COL_LENGTH])
    _refresh_layer_membership(tunnel)
    _refresh_filters_passed(tunnel)
    return True


def _refresh_layer_membership(tunnel) -> None:
    """Recompute layer membership for the (possibly truncated) sphere data."""
    from transport_tools.libs.geometry import assign_layer_from_distances, einsum_dist

    tunnel.layer_membership = assign_layer_from_distances(
        einsum_dist(tunnel.spheres_data[:, 0:3], np.array([0.0, 0.0, 0.0])),
        tunnel.parameters["layer_thickness"],
    )[1]


def _refresh_filters_passed(tunnel) -> None:
    """Recompute the relevant-tunnel filter verdict from the (possibly truncated) length."""
    tunnel.filters_passed = (
        round(tunnel.bottleneck_radius, 6) >= tunnel.parameters["relevant_tunnel_min_radius"]
        and round(tunnel.length, 6) >= tunnel.parameters["relevant_tunnel_min_length"]
        and round(tunnel.curvature, 6) <= tunnel.parameters["relevant_tunnel_max_curvature"]
    )


# ── full workflow ──────────────────────────────────────────────────────────


def prune_tunnels(
    tunnels: list,
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
    flagged as inflating and/or curved are shortened; all other tunnels are left
    untouched.

    :param tunnels: Tunnel objects of a single MD simulation.
    :param mode: cut selection mode, ``"first"`` or ``"peak"``.
    :param bin_size: width of distance bins in Angstroms.
    :param surv_perc_range: percentile range of per-tunnel maximum distances defining the
        valid region for cut selection.
    :param core_range: fraction of the distance range defining the tunnel core region.
    :param min_joint_threshold: noise floor of the joint score in ``"first"`` mode (0 disables).
    :param eff_thresh: path-efficiency threshold below which a post-cut segment is curved.
    :param slope_percentile: percentile of cluster core-expansion slopes used as the
        inflation reference.
    :param water_radius: minimum radius of the last kept sphere.
    :return: statistics of the pruning pass: number of clusters, clusters with a cut,
        tunnels extending beyond a cut, offensive tunnels (by category) and tunnels
        actually truncated.
    """
    cuts = compute_per_cluster_cut(
        tunnels,
        bin_size=bin_size,
        surv_perc_range=surv_perc_range,
        core_range=core_range,
        mode=mode,
        min_joint_threshold=min_joint_threshold,
    )

    stats: Dict[str, int] = {
        "n_clusters": len(_group_by_cluster(tunnels)),
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
            continue
        cut_distance = cut_distance[0]

        verdicts = verify_postcut(
            cluster_tunnels, cut_distance,
            eff_thresh=eff_thresh, slope_percentile=slope_percentile,
        )
        stats["n_extending"] += len(verdicts)

        for tunnel in cluster_tunnels:
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
