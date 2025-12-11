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

# Test configuration data for config.py unit tests

minimal_config_content = """[INPUT_PATHS]
caver_results_path = /path/to/caver
aquaduct_results_path = /path/to/aquaduct
trajectory_path = /path/to/trajectory

[CALCULATIONS_SETTINGS]
start_from_stage = 1
stop_after_stage = 5
snapshots_per_simulation = 100
min_tunnel_radius4clustering = 3.0
event_min_distance = 2.5

[OUTPUT_SETTINGS]
output_path = /path/to/output
verbose_logging = False
log_level = INFO
"""

advanced_config_content = """[INPUT_PATHS]
caver_results_path = /path/to/caver
aquaduct_results_path = /path/to/aquaduct

[CALCULATIONS_SETTINGS]
start_from_stage = 1
min_tunnel_radius4clustering = 3.0

[OUTPUT_SETTINGS]
output_path = /path/to/output

[ADVANCED_SETTINGS]
random_seed = 42
num_cpus = 4
"""

# Sample old parameters for report_updates testing
old_parameters_no_changes = {
    "start_from_stage": 1,
    "stop_after_stage": 5,
    "snapshots_per_simulation": 100,
    "min_tunnel_radius4clustering": 3.0,
    "output_path": "/path/to/output",
    "verbose_logging": False,
}

old_parameters_with_changes = {
    "start_from_stage": 2,  # Changed from 1
    "stop_after_stage": 4,  # Changed from 5
    "snapshots_per_simulation": 100,  # Same
    "min_tunnel_radius4clustering": 2.5,  # Changed from 3.0
    "output_path": "/path/to/old_output",  # Changed
    "verbose_logging": True,  # Changed from False
}

# Expected template file sections
expected_template_sections = ["INPUT_PATHS", "CALCULATIONS_SETTINGS", "OUTPUT_SETTINGS"]
expected_advanced_template_sections = ["INPUT_PATHS", "CALCULATIONS_SETTINGS", "OUTPUT_SETTINGS", "ADVANCED_SETTINGS"]

# Sample filter values for testing get_filters()
sample_filter_names = [
    "min_length", "max_length", "min_bottleneck_radius", "max_bottleneck_radius",
    "min_curvature", "max_curvature", "min_sims_num", "min_snapshots_num",
    "min_avg_snapshots_num", "min_total_events", "min_entry_events", "min_release_events"
]
