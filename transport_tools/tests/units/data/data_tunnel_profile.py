# -*- coding: utf-8 -*-

# Test data for TunnelProfile4MD class tests

import numpy as np

# Sample tunnel data sections (simulating CAVER output format)
# Format: 7 lines in CSV format (X, Y, Z, distance, length, R, Upper limit)
# Format per line: snapshot, cluster_id, tunnel_id, throughput, cost, bottleneck_radius, -, -, -, curvature, length, , property_name, values...

tunnel_data_section_snapshot1 = [
    "test.1.pdb, 1, 1, 0.566, 0.568, 1.109, -, -, -, 1.199, 12.162, , X, 42.789, 42.123, 41.567, 41.012, 40.456, 39.901, 39.345, 38.789, 38.234, 37.678, 37.123\n",
    "test.1.pdb, 1, 1, 0.566, 0.568, 1.109, -, -, -, 1.199, 12.162, , Y, 42.178, 42.456, 42.789, 43.123, 43.456, 43.789, 44.123, 44.456, 44.789, 45.123, 45.456\n",
    "test.1.pdb, 1, 1, 0.566, 0.568, 1.109, -, -, -, 1.199, 12.162, , Z, 30.813, 30.512, 30.234, 29.987, 29.756, 29.534, 29.321, 29.123, 28.934, 28.756, 28.587\n",
    "test.1.pdb, 1, 1, 0.566, 0.568, 1.109, -, -, -, 1.199, 12.162, , distance, 0.000, 1.234, 2.456, 3.678, 4.890, 6.102, 7.314, 8.526, 9.738, 10.950, 12.162\n",
    "test.1.pdb, 1, 1, 0.566, 0.568, 1.109, -, -, -, 1.199, 12.162, , length, 0.000, 1.234, 2.456, 3.678, 4.890, 6.102, 7.314, 8.526, 9.738, 10.950, 12.162\n",
    "test.1.pdb, 1, 1, 0.566, 0.568, 1.109, -, -, -, 1.199, 12.162, , R, 1.661, 1.523, 1.445, 1.389, 1.334, 1.278, 1.223, 1.167, 1.112, 1.056, 1.001\n",
    "test.1.pdb, 1, 1, 0.566, 0.568, 1.109, -, -, -, 1.199, 12.162, , Upper limit of R overestimation, -, -, -, -, -, -, -, -, -, -, -\n"
]

tunnel_data_section_snapshot2 = [
    "test.2.pdb, 1, 1, 0.644, 0.439, 1.594, -, -, -, 1.156, 12.410, , X, 42.789, 42.089, 41.534, 40.978, 40.423, 39.867, 39.312, 38.756, 38.201, 37.645, 37.089\n",
    "test.2.pdb, 1, 1, 0.644, 0.439, 1.594, -, -, -, 1.156, 12.410, , Y, 42.178, 42.423, 42.756, 43.089, 43.423, 43.756, 44.089, 44.423, 44.756, 45.089, 45.423\n",
    "test.2.pdb, 1, 1, 0.644, 0.439, 1.594, -, -, -, 1.156, 12.410, , Z, 30.813, 30.534, 30.267, 30.012, 29.767, 29.534, 29.312, 29.101, 28.901, 28.712, 28.534\n",
    "test.2.pdb, 1, 1, 0.644, 0.439, 1.594, -, -, -, 1.156, 12.410, , distance, 0.000, 1.298, 2.534, 3.789, 5.012, 6.245, 7.478, 8.711, 9.944, 11.177, 12.410\n",
    "test.2.pdb, 1, 1, 0.644, 0.439, 1.594, -, -, -, 1.156, 12.410, , length, 0.000, 1.298, 2.534, 3.789, 5.012, 6.245, 7.478, 8.711, 9.944, 11.177, 12.410\n",
    "test.2.pdb, 1, 1, 0.644, 0.439, 1.594, -, -, -, 1.156, 12.410, , R, 1.784, 1.656, 1.567, 1.489, 1.412, 1.334, 1.256, 1.178, 1.101, 1.023, 0.945\n",
    "test.2.pdb, 1, 1, 0.644, 0.439, 1.594, -, -, -, 1.156, 12.410, , Upper limit of R overestimation, -, -, -, -, -, -, -, -, -, -, -\n"
]

tunnel_data_section_snapshot3 = [
    "test.3.pdb, 1, 1, 0.567, 0.570, 1.173, -, -, -, 1.124, 12.598, , X, 42.789, 42.056, 41.501, 40.945, 40.389, 39.834, 39.278, 38.723, 38.167, 37.612, 37.056\n",
    "test.3.pdb, 1, 1, 0.567, 0.570, 1.173, -, -, -, 1.124, 12.598, , Y, 42.178, 42.389, 42.723, 43.056, 43.389, 43.723, 44.056, 44.389, 44.723, 45.056, 45.389\n",
    "test.3.pdb, 1, 1, 0.567, 0.570, 1.173, -, -, -, 1.124, 12.598, , Z, 30.813, 30.556, 30.301, 30.056, 29.823, 29.601, 29.389, 29.189, 28.989, 28.801, 28.623\n",
    "test.3.pdb, 1, 1, 0.567, 0.570, 1.173, -, -, -, 1.124, 12.598, , distance, 0.000, 1.362, 2.612, 3.901, 5.134, 6.378, 7.622, 8.866, 10.110, 11.354, 12.598\n",
    "test.3.pdb, 1, 1, 0.567, 0.570, 1.173, -, -, -, 1.124, 12.598, , length, 0.000, 1.362, 2.612, 3.901, 5.134, 6.378, 7.622, 8.866, 10.110, 11.354, 12.598\n",
    "test.3.pdb, 1, 1, 0.567, 0.570, 1.173, -, -, -, 1.124, 12.598, , R, 1.892, 1.789, 1.689, 1.589, 1.489, 1.389, 1.289, 1.189, 1.089, 0.989, 0.889\n",
    "test.3.pdb, 1, 1, 0.567, 0.570, 1.173, -, -, -, 1.124, 12.598, , Upper limit of R overestimation, -, -, -, -, -, -, -, -, -, -, -\n"
]

# Parameters for creating test tunnels
test_parameters = {
    "min_tunnel_radius4clustering": 0,
    "min_tunnel_length4clustering": 5,
    "max_tunnel_curvature4clustering": 999,
    "tunnel_properties_quantile": 0.9,
    "md_label": "md1",
    "layer_thickness": 1.5,
    "snapshot_id_position": 1,
    "snapshot_delimiter": ".",
    "snapshots_per_simulation": 10000,
    "caver_traj_offset": 1
}

# Transformation matrix (identity for simplicity in tests)
test_transform_mat = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Expected CSV output format for tunnel
expected_csv_line_pattern = "md1,1,1,{length:.3f},{radius:.3f},{curvature:.3f},{throughput:.3f},11"

# Expected bottleneck residues (sample data)
sample_bottleneck_residues = {
    1: ["ALA:42", "VAL:45", "LEU:89"],
    2: ["VAL:45", "LEU:89", "PHE:102"],
    3: ["LEU:89", "PHE:102", "TRP:156"]
}

# Expected residue frequencies after processing
expected_residue_freq = {
    "ALA:42": 1,
    "VAL:45": 2,
    "LEU:89": 3,
    "PHE:102": 2,
    "TRP:156": 1
}

# Filter configurations for testing
test_filters_pass_all = {
    "length": (-1, -1),
    "radius": (-1, -1),
    "curvature": (-1, -1),
    "min_sims_num": 1,
    "min_avg_snapshots_num": 1,
    "min_avg_water_events": -1,
    "min_avg_entry_events": -1,
    "min_avg_release_events": -1
}

test_filters_strict = {
    "length": (10, 15),
    "radius": (1.0, 2.0),
    "curvature": (1, 10),
    "min_sims_num": 1,
    "min_avg_snapshots_num": 1,
    "min_avg_water_events": -1,
    "min_avg_entry_events": -1,
    "min_avg_release_events": -1
}

test_filters_length_only = {
    "length": (12, 14),
    "radius": (-1, -1),
    "curvature": (-1, -1),
    "min_sims_num": 1,
    "min_avg_snapshots_num": 1,
    "min_avg_water_events": -1,
    "min_avg_entry_events": -1,
    "min_avg_release_events": -1
}
