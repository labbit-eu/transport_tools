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

"""
Test data fixtures for EventAssigner class tests
"""

import numpy as np

# Test parameters for EventAssigner initialization
test_parameters_minimal = {
    "event_assignment_cutoff": 0.5,
    "ambiguous_event_assignment_resolution": "penetration_depth",
    "perform_exact_matching_analysis": False,
    "perform_trace_matching_analysis": False,
    "caver_traj_offset": 1,
    "trajectory_path": "/tmp/test_traj",
    "folder_pattern4exact_matching_analysis": "md*",
    "exact_matching_details_folder": "/tmp/test_exact",
    "trace_matching_details_folder": "/tmp/test_trace",
    "trajectory_engine": "mdtraj",
    "aqauduct_ligand_effective_radius": 1.5,
    "snapshots_per_simulation": 1000
}

test_parameters_exact_matching = test_parameters_minimal.copy()
test_parameters_exact_matching["perform_exact_matching_analysis"] = True
test_parameters_exact_matching["ambiguous_event_assignment_resolution"] = "exact_matching"

test_parameters_trace_matching = test_parameters_minimal.copy()
test_parameters_trace_matching["perform_trace_matching_analysis"] = True
test_parameters_trace_matching["ambiguous_event_assignment_resolution"] = "trace_matching"

test_parameters_penetration_span = test_parameters_minimal.copy()
test_parameters_penetration_span["ambiguous_event_assignment_resolution"] = "penetration_span"

test_parameters_directionality = test_parameters_minimal.copy()
test_parameters_directionality["ambiguous_event_assignment_resolution"] = "directionality"

# Sampling-stride (frame -> CAVER snapshot) parameters for trace matching. CAVER analysed 1000
# snapshots of a 20000-frame trajectory (stride 20), so only every 20th event frame lands on an analysed
# snapshot. perform_trace_matching_analysis is off so the matching writes no detail files during the test.
test_parameters_trace_matching_strided = test_parameters_trace_matching.copy()
test_parameters_trace_matching_strided["snapshots_per_simulation"] = 1000
test_parameters_trace_matching_strided["caver_snapshot_stride"] = 20
test_parameters_trace_matching_strided["perform_trace_matching_analysis"] = False

# Stride-1 (dense) back-compatible counterpart: every event frame maps to its own snapshot (legacy f+offset).
test_parameters_trace_matching_dense = test_parameters_trace_matching.copy()
test_parameters_trace_matching_dense["snapshots_per_simulation"] = 1000
test_parameters_trace_matching_dense["caver_snapshot_stride"] = 1
test_parameters_trace_matching_dense["perform_trace_matching_analysis"] = False

# Strided counterpart with interpolation on: frames between snapshots are matched against their nearest
# analysed snapshot instead of being skipped, so every event frame resolves to some snapshot.
test_parameters_trace_matching_strided_interp = test_parameters_trace_matching_strided.copy()
test_parameters_trace_matching_strided_interp["interpolate_missing_snapshots4matching"] = True

# Per-frame ligand trace [frame, x, y, z] spanning the event window (frames 10..50) for these matching tests.
sample_event_frame_trace = np.array([[float(f), 1.0, 1.0, 1.0] for f in range(10, 51)])

# Event specification format: (md_label, event_label, (resname:resid, (start_frame, end_frame)))
sample_event_specification_1 = ("md1", "1_entry", ("WAT:123", (10, 50)))
sample_event_specification_2 = ("md1", "2_release", ("WAT:456", (20, 60)))
sample_event_specification_3 = ("md2", "3_entry", ("WAT:789", (30, 70)))

# Active filters for testing
test_active_filters = {
    "length": (0.0, 999.0),
    "radius": (0.0, 999.0),
    "curvature": (0.0, 999.0),
    "min_sims_num": 1,
    "min_snapshots_num": 1,
    "min_avg_snapshots_num": 1,
    "min_transport_events": 0,
    "min_entry_events": 0,
    "min_release_events": 0
}

# Mock LayeredPathSet nodes data
# Format: [x, y, z, layer, terminal_node_flag, ...]
# Terminal node has flag = 1 at position [4]
sample_nodes_data_entry = np.array([
    [0.0, 0.0, 0.0, 0, 0],  # Starting node
    [1.0, 1.0, 1.0, 1, 0],  # Middle node
    [2.0, 2.0, 2.0, 2, 1],  # Terminal node (entry direction: positive x,y,z)
])

sample_nodes_data_release = np.array([
    [2.0, 2.0, 2.0, 0, 0],  # Starting node (inside)
    [1.0, 1.0, 1.0, 1, 0],  # Middle node
    [0.0, 0.0, 0.0, 2, 1],  # Terminal node (release direction: negative x,y,z)
])

sample_nodes_data_no_terminal = np.array([
    [0.0, 0.0, 0.0, 0, 0],  # No terminal nodes marked
    [1.0, 1.0, 1.0, 1, 0],
    [2.0, 2.0, 2.0, 2, 0],
])

# Expected event directions
expected_direction_entry = np.array([2.0, 2.0, 2.0])
expected_direction_release = np.array([0.0, 0.0, 0.0])

# Mock buriedness and depth values
# buriedness = fraction of event path inside tunnel
# depth = maximum penetration depth into protein
sample_buriedness_high = 0.8
sample_buriedness_medium = 0.6
sample_buriedness_low = 0.3
sample_buriedness_below_cutoff = 0.4

sample_depth_high = 15.0
sample_depth_medium = 10.0
sample_depth_low = 5.0

# Minimal (shallow-end) depth among buried nodes, used by penetration_span to compute the traversed span
# (max_depth - min_buried_depth). Default value for tests where the span is irrelevant (other resolutions ignore it).
sample_min_depth = 0.0
# penetration_span fixtures: SC1 reaches deeper (larger max_depth) but over a narrow span (perpendicular-like),
# SC2 is shallower overall yet spans a wide depth range (a genuine surface->core traversal):
#   SC1 span = sample_depth_high  - sample_min_depth_narrow = 15.0 - 14.0 = 1.0
#   SC2 span = sample_depth_medium - sample_min_depth_wide   = 10.0 -  2.0 = 8.0
sample_min_depth_narrow = 14.0
sample_min_depth_wide = 2.0

# Supercluster avg_direction vectors for directionality resolution tests, compared against the entry event's
# terminal-node direction [2, 2, 2] (sample_nodes_data_entry): one aligned (0 deg), one well off-axis (~70 deg).
sample_direction_aligned = np.array([2.0, 2.0, 2.0])
sample_direction_offaxis = np.array([2.0, 2.0, -2.0])

# Expected return format from perform_assignment
# (event_specification, assigned_sc_ids, max_buriedness, max_depth)
expected_assignment_success = (
    sample_event_specification_1,
    np.array([1]),  # Assigned to SC 1
    sample_buriedness_high,
    sample_depth_high
)

expected_assignment_outlier_no_directional = (
    sample_event_specification_1,
    None,  # Not assigned (no directional alignment)
    -999,
    -999
)

expected_assignment_outlier_low_buriedness = (
    sample_event_specification_1,
    None,  # Not assigned (buriedness too low)
    sample_buriedness_below_cutoff,
    -999
)

expected_assignment_multiple_scs = (
    sample_event_specification_1,
    np.array([1, 2]),  # Assigned to multiple SCs
    sample_buriedness_high,
    sample_depth_high
)

# Exact matching buriedness data structure
sample_exact_matching_buriedness = {
    "4all_frames": {
        1: 0.75,  # SC 1: 75% of frames buried
        2: 0.50   # SC 2: 50% of frames buried
    },
    "4existing_tunnels": {
        1: 0.90,  # SC 1: 90% of frames with tunnels
        2: 0.60   # SC 2: 60% of frames with tunnels
    }
}

sample_exact_matching_no_tunnels = {
    "4all_frames": {},
    "4existing_tunnels": {}
}

# Tunnels from the candidate SCs exist in the event's frames, but the ligand is never inside any of them
# (all-zero matched buriedness) - the other degenerate matching case, which must fall back to a geometric
# resolution rather than keeping every candidate unfiltered
sample_exact_matching_all_zero = {
    "4all_frames": {
        1: 0.0,
        2: 0.0
    },
    "4existing_tunnels": {
        1: 0.0,
        2: 0.0
    }
}

# Mock trajectory coordinates for ligand
# Shape: (num_frames, num_atoms, 3)
sample_ligand_coords = np.array([
    [[1.0, 1.0, 1.0]],  # Frame 1
    [[1.5, 1.5, 1.5]],  # Frame 2
    [[2.0, 2.0, 2.0]],  # Frame 3
])

# Closest tunnel data format for exact matching
# Format: [frame, x_lig, y_lig, z_lig, dist2lig, cluster_id, x_sph, y_sph, z_sph, dist2sp, radius, length]
sample_closest_tunnels_data = np.array([
    [10, 1.0, 1.0, 1.0, 0.5, 1, 1.0, 1.2, 1.0, 5.0, 2.0, 15.0],
    [11, 1.5, 1.5, 1.5, 0.3, 1, 1.5, 1.7, 1.5, 6.0, 2.1, 15.0],
    [12, 2.0, 2.0, 2.0, 1.5, 2, 2.0, 2.5, 2.0, 7.0, 1.9, 14.0],
])

# Test cases for directional alignment
# SuperCluster properties that would make them directionally aligned
test_sc_aligned_positive = {
    "sc_id": 1,
    "terminal_direction": np.array([2.0, 2.0, 2.0]),  # Aligned with entry event
    "passes_filter": True
}

test_sc_aligned_negative = {
    "sc_id": 2,
    "terminal_direction": np.array([0.0, 0.0, 0.0]),  # Aligned with release event
    "passes_filter": True
}

test_sc_not_aligned = {
    "sc_id": 3,
    "terminal_direction": np.array([10.0, 0.0, 0.0]),  # Not aligned
    "passes_filter": True
}

test_sc_filtered_out = {
    "sc_id": 4,
    "terminal_direction": np.array([2.0, 2.0, 2.0]),  # Would be aligned
    "passes_filter": False  # But doesn't pass filters
}

# Expected list of directionally aligned SC IDs for different scenarios
expected_aligned_sc_ids_entry = [1]
expected_aligned_sc_ids_release = [2]
expected_aligned_sc_ids_multiple = [1, 2]
expected_aligned_sc_ids_none = []