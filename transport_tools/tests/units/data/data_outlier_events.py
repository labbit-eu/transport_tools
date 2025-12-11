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

"""
Test data fixtures for OutlierTransportEvents class tests
"""

# Test parameters for OutlierTransportEvents initialization
test_parameters_minimal = {
    "perform_comparative_analysis": False,
    "comparative_groups_definition": None,
    "super_cluster_details_folder": "/tmp/test_details",
    "visualization_folder": "/tmp/test_viz",
    "layered_aquaduct_vis_path": "/tmp/test_layered",
    "random_seed": 42,
    "max_events_per_cluster4visualization": 10
}

test_parameters_with_comparative = {
    "perform_comparative_analysis": True,
    "comparative_groups_definition": {
        "group_A": ["md1", "md2"],
        "group_B": ["md3"]
    },
    "super_cluster_details_folder": "/tmp/test_details",
    "visualization_folder": "/tmp/test_viz",
    "layered_aquaduct_vis_path": "/tmp/test_layered",
    "random_seed": 42,
    "max_events_per_cluster4visualization": 10
}

# Mock transport event data
# Format: (path_id, (resname:residue_id, (start_frame, end_frame)))
sample_entry_event_1 = ("path_001", ("WAT:123", (10, 50)))
sample_entry_event_2 = ("path_002", ("WAT:456", (20, 60)))
sample_release_event_1 = ("path_003", ("WAT:789", (30, 70)))
sample_release_event_2 = ("path_004", ("WAT:012", (40, 80)))

# Mock traced event format
sample_traced_event_1 = ("WAT:123", (10, 50))
sample_traced_event_2 = ("WAT:456", (20, 60))
sample_traced_event_3 = ("WAT:789", (30, 70))
sample_traced_event_4 = ("WAT:012", (40, 80))

# Expected output for report_events_details
expected_details_output_header = """Unassigned transport events:
Number of entry events = 2
Number of release events = 2
"""

expected_details_entry_line = "entry: (from Simulation: AQUA-DUCT ID, (Resname:Residue), start_frame->end_frame; ... )"
expected_details_release_line = "release: (from Simulation: AQUA-DUCT ID, (Resname:Residue), start_frame->end_frame; ... )"

# Expected summary line format
expected_summary_line_format = "Total number of unassigned events:"

# Multiple MD labels for testing
test_md_labels = ["md1", "md2", "md3"]

# Sample event collections for testing
sample_events_collection = {
    "entry": {
        "md1": [
            ("path_001", ("WAT:123", (10, 50))),
            ("path_002", ("WAT:456", (20, 60)))
        ],
        "md2": [
            ("path_005", ("WAT:111", (15, 55)))
        ]
    },
    "release": {
        "md1": [
            ("path_003", ("WAT:789", (30, 70)))
        ],
        "md2": [
            ("path_006", ("WAT:222", (25, 65))),
            ("path_007", ("WAT:333", (35, 75)))
        ]
    }
}

# Expected counts for sample_events_collection
expected_counts_md1 = (3, 2, 1)  # (total, entry, release)
expected_counts_md2 = (3, 1, 2)  # (total, entry, release)
expected_counts_overall = (6, 3, 3)  # (total, entry, release)

# Widths for summary line formatting
test_column_widths = [50, 10, 10, 10]

# Expected visualization lines patterns
expected_vis_patterns = [
    "events = [",
    "for event in events:",
    "with gzip.open(event, 'rb') as in_stream:",
    "pathset = pickle.load(in_stream)",
    "for path in pathset:",
    "path[3:6] =",
    "cmd.load_cgo(path,",
    "cmd.set('cgo_line_width',"
]
