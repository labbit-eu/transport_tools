# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2021 The TransportTools Authors
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

# Test data for ui.py unit tests

# Sample command line arguments
sample_args_config = ["-c", "config.ini"]
sample_args_write_template = ["-w"]
sample_args_write_template_advanced = ["-w", "-a"]
sample_args_version = ["-v"]
sample_args_license = ["-l"]
sample_args_overwrite = ["--overwrite"]

# Expected parser argument names
expected_parser_args = ["config_filename", "write_config_file", "advanced", "print_version", "print_license", "overwrite"]

# Logging levels
valid_log_levels = ["CRITICAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "NOTSET"]
invalid_log_level = "INVALID_LEVEL"

# Messages for filtering tests
verbose_messages = [
    "Optimizing assignment of events",
    "Distance matrix calculation complete",
    "Using point 123 for alignment",
    "No points found in region",
    "Using starting point at coordinates",
    "alignment_length = 50",
    "Using CA atoms for alignment",
    "General rotation matrix calculated",
    "max_dist = 10.5 layer_thickness = 2.0",
    "Transport event assigned to cluster",
    "Optimized distance is 5.5",
]

default_filtered_messages = [
    "findfont: Font family not found",
]

# Progress bar test values
progress_test_values = [
    (0, 100),
    (25, 100),
    (50, 100),
    (75, 100),
    (100, 100),
]

# Process count test values
process_count_test_values = [
    (1, "parallel process"),
    (2, "parallel processes"),
    (10, "parallel processes"),
]
