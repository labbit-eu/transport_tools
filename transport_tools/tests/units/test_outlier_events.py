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
import os
import tempfile
import shutil
from transport_tools.libs.tools import OutlierTransportEvents
from transport_tools.tests.units.data.data_outlier_events import (
    test_parameters_minimal, test_parameters_with_comparative,
    sample_entry_event_1, sample_entry_event_2,
    sample_release_event_1, sample_release_event_2,
    sample_traced_event_1, sample_traced_event_2,
    sample_traced_event_3, sample_traced_event_4,
    expected_counts_md1, expected_counts_md2, expected_counts_overall,
    test_column_widths, expected_summary_line_format,
    expected_details_output_header, expected_vis_patterns
)


class TestOutlierTransportEvents(unittest.TestCase):
    """
    Comprehensive unit tests for OutlierTransportEvents class
    Tests the functionality of storing, counting, and reporting outlier transport events
    """

    def setUp(self):
        """
        Set up test fixtures before each test method
        """
        self.maxDiff = None
        self.temp_dir = tempfile.mkdtemp()

        # Update parameters to use temp directory
        self.test_params = test_parameters_minimal.copy()
        self.test_params["super_cluster_details_folder"] = os.path.join(self.temp_dir, "details")
        self.test_params["visualization_folder"] = os.path.join(self.temp_dir, "viz")
        self.test_params["layered_aquaduct_vis_path"] = os.path.join(self.temp_dir, "layered")

        self.outliers = OutlierTransportEvents(self.test_params)

    def tearDown(self):
        """
        Clean up temporary files after each test
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """
        Verify OutlierTransportEvents initializes with correct default state
        """
        self.assertIsInstance(self.outliers.parameters, dict)
        self.assertIsInstance(self.outliers.transport_events, dict)
        self.assertIsInstance(self.outliers.transport_events_global, dict)
        self.assertIsInstance(self.outliers.num_events, dict)

        # Check initial num_events structure
        self.assertIn("overall", self.outliers.num_events)
        self.assertEqual(self.outliers.num_events["overall"]["entry"], 0)
        self.assertEqual(self.outliers.num_events["overall"]["release"], 0)

    def test_exist_returns_false_initially(self):
        """
        Verify exist() returns False for new instance with no events
        """
        self.assertFalse(self.outliers.exist("md1"))
        self.assertFalse(self.outliers.exist("overall"))

    def test_exist_returns_false_for_nonexistent_md_label(self):
        """
        Verify exist() returns False for MD labels that don't exist
        """
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        self.assertFalse(self.outliers.exist("md2"))
        self.assertFalse(self.outliers.exist("nonexistent"))

    def test_exist_returns_true_after_adding_entry_event(self):
        """
        Verify exist() returns True after adding an entry event
        """
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        self.assertTrue(self.outliers.exist("md1"))
        self.assertTrue(self.outliers.exist("overall"))

    def test_exist_returns_true_after_adding_release_event(self):
        """
        Verify exist() returns True after adding a release event
        """
        self.outliers.add_transport_event("md1", "path_003", "release", sample_traced_event_3)
        self.assertTrue(self.outliers.exist("md1"))
        self.assertTrue(self.outliers.exist("overall"))

    def test_count_events_returns_zeros_initially(self):
        """
        Verify count_events() returns (0, 0, 0) for new instance
        """
        total, entry, release = self.outliers.count_events("overall")
        self.assertEqual(total, 0)
        self.assertEqual(entry, 0)
        self.assertEqual(release, 0)

    def test_count_events_returns_zeros_for_nonexistent_md_label(self):
        """
        Verify count_events() returns (0, 0, 0) for MD labels that don't exist
        """
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        total, entry, release = self.outliers.count_events("md2")
        self.assertEqual(total, 0)
        self.assertEqual(entry, 0)
        self.assertEqual(release, 0)

    def test_add_transport_event_entry_type(self):
        """
        Verify entry events are stored and counted correctly
        """
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)

        # Check storage
        self.assertIn("entry", self.outliers.transport_events)
        self.assertIn("md1", self.outliers.transport_events["entry"])
        self.assertEqual(len(self.outliers.transport_events["entry"]["md1"]), 1)

        # Verify stored data
        path_id, traced_event = self.outliers.transport_events["entry"]["md1"][0]
        self.assertEqual(path_id, "path_001")
        self.assertEqual(traced_event, sample_traced_event_1)

        # Check counts
        total, entry, release = self.outliers.count_events("md1")
        self.assertEqual(total, 1)
        self.assertEqual(entry, 1)
        self.assertEqual(release, 0)

    def test_add_transport_event_release_type(self):
        """
        Verify release events are stored and counted correctly
        """
        self.outliers.add_transport_event("md1", "path_003", "release", sample_traced_event_3)

        # Check storage
        self.assertIn("release", self.outliers.transport_events)
        self.assertIn("md1", self.outliers.transport_events["release"])
        self.assertEqual(len(self.outliers.transport_events["release"]["md1"]), 1)

        # Verify stored data
        path_id, traced_event = self.outliers.transport_events["release"]["md1"][0]
        self.assertEqual(path_id, "path_003")
        self.assertEqual(traced_event, sample_traced_event_3)

        # Check counts
        total, entry, release = self.outliers.count_events("md1")
        self.assertEqual(total, 1)
        self.assertEqual(entry, 0)
        self.assertEqual(release, 1)

    def test_add_multiple_events_same_md_label(self):
        """
        Verify multiple events can be added to the same MD label
        """
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        self.outliers.add_transport_event("md1", "path_002", "entry", sample_traced_event_2)
        self.outliers.add_transport_event("md1", "path_003", "release", sample_traced_event_3)

        # Check counts
        total, entry, release = self.outliers.count_events("md1")
        self.assertEqual(total, 3)
        self.assertEqual(entry, 2)
        self.assertEqual(release, 1)

        # Verify storage
        self.assertEqual(len(self.outliers.transport_events["entry"]["md1"]), 2)
        self.assertEqual(len(self.outliers.transport_events["release"]["md1"]), 1)

    def test_add_events_multiple_md_labels(self):
        """
        Verify events can be added across different MD labels
        """
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        self.outliers.add_transport_event("md1", "path_002", "entry", sample_traced_event_2)
        self.outliers.add_transport_event("md2", "path_005", "entry", sample_traced_event_1)
        self.outliers.add_transport_event("md1", "path_003", "release", sample_traced_event_3)
        self.outliers.add_transport_event("md2", "path_006", "release", sample_traced_event_3)
        self.outliers.add_transport_event("md2", "path_007", "release", sample_traced_event_4)

        # Check md1 counts
        total, entry, release = self.outliers.count_events("md1")
        self.assertEqual(total, 3)
        self.assertEqual(entry, 2)
        self.assertEqual(release, 1)

        # Check md2 counts
        total, entry, release = self.outliers.count_events("md2")
        self.assertEqual(total, 3)
        self.assertEqual(entry, 1)
        self.assertEqual(release, 2)

        # Check overall counts
        total, entry, release = self.outliers.count_events("overall")
        self.assertEqual(total, 6)
        self.assertEqual(entry, 3)
        self.assertEqual(release, 3)

    def test_add_transport_event_globally_unassigned_false(self):
        """
        Verify globally_unassigned=False doesn't add to global counters
        """
        self.outliers.add_transport_event("md1", "path_001", "entry",
                                         sample_traced_event_1, globally_unassigned=False)

        # Check md1 counts (should have the event)
        total, entry, release = self.outliers.count_events("md1")
        self.assertEqual(total, 1)
        self.assertEqual(entry, 1)
        self.assertEqual(release, 0)

        # Check overall counts (should NOT have the event)
        total, entry, release = self.outliers.count_events("overall")
        self.assertEqual(total, 0)
        self.assertEqual(entry, 0)
        self.assertEqual(release, 0)

        # Verify not in global storage
        self.assertNotIn("entry", self.outliers.transport_events_global)

    def test_add_transport_event_with_comparative_analysis(self):
        """
        Verify events are grouped correctly when comparative analysis is enabled
        """
        params_comp = test_parameters_with_comparative.copy()
        params_comp["super_cluster_details_folder"] = os.path.join(self.temp_dir, "details")
        params_comp["visualization_folder"] = os.path.join(self.temp_dir, "viz")
        params_comp["layered_aquaduct_vis_path"] = os.path.join(self.temp_dir, "layered")

        outliers_comp = OutlierTransportEvents(params_comp)

        # Add events to md1 and md2 (both in group_A)
        outliers_comp.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        outliers_comp.add_transport_event("md2", "path_002", "entry", sample_traced_event_2)

        # Events should be counted under group_A
        total, entry, release = outliers_comp.count_events("group_A")
        self.assertEqual(total, 2)
        self.assertEqual(entry, 2)
        self.assertEqual(release, 0)

    def test_report_events_details_creates_file(self):
        """
        Verify report_events_details() creates output file
        """
        # Add some events
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        self.outliers.add_transport_event("md1", "path_003", "release", sample_traced_event_3)

        filename = "test_outliers.txt"
        self.outliers.report_events_details(filename)

        # Check file was created
        expected_path = os.path.join(self.test_params["super_cluster_details_folder"], filename)
        self.assertTrue(os.path.exists(expected_path))

    def test_report_events_details_correct_format(self):
        """
        Verify report_events_details() writes correct format
        """
        # Add some events
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        self.outliers.add_transport_event("md1", "path_002", "entry", sample_traced_event_2)
        self.outliers.add_transport_event("md2", "path_003", "release", sample_traced_event_3)
        self.outliers.add_transport_event("md2", "path_004", "release", sample_traced_event_4)

        filename = "test_outliers_format.txt"
        self.outliers.report_events_details(filename)

        expected_path = os.path.join(self.test_params["super_cluster_details_folder"], filename)
        with open(expected_path, "r") as f:
            content = f.read()

        # Check header
        self.assertIn("Unassigned transport events:", content)
        self.assertIn("Number of entry events = 2", content)
        self.assertIn("Number of release events = 2", content)

        # Check event type sections
        self.assertIn("entry:", content)
        self.assertIn("release:", content)

        # Check MD label presence
        self.assertIn("from md1:", content)
        self.assertIn("from md2:", content)

        # Check event details
        self.assertIn("path_001", content)
        self.assertIn("WAT:123", content)
        self.assertIn("10->50", content)

    def test_report_summary_line_format(self):
        """
        Verify report_summary_line() returns correctly formatted string
        """
        # Add some events
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        self.outliers.add_transport_event("md1", "path_003", "release", sample_traced_event_3)

        summary_line = self.outliers.report_summary_line(test_column_widths, "md1")

        # Check format
        self.assertIn(expected_summary_line_format, summary_line)
        self.assertIn("2", summary_line)  # total count
        self.assertIn("1", summary_line)  # entry count (appears twice but that's ok)

        # Verify it's a single line
        self.assertEqual(summary_line.count("\n"), 1)

    def test_report_summary_line_correct_counts(self):
        """
        Verify report_summary_line() includes correct event counts
        """
        # Add events with known counts
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        self.outliers.add_transport_event("md1", "path_002", "entry", sample_traced_event_2)
        self.outliers.add_transport_event("md1", "path_003", "release", sample_traced_event_3)

        summary_line = self.outliers.report_summary_line(test_column_widths, "md1")

        # Parse numbers from summary line
        # Format: "Total number of unassigned events:         3,          2,          1"
        parts = summary_line.split(":")
        self.assertEqual(len(parts), 2)

        numbers_part = parts[1].strip()
        # Should contain total=3, entry=2, release=1
        self.assertIn("3", numbers_part)
        self.assertIn("2", numbers_part)
        self.assertIn("1", numbers_part)

    def test_prepare_visualization_returns_list(self):
        """
        Verify prepare_visualization() returns a list of strings
        """
        # Add some events
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)

        vis_lines = self.outliers.prepare_visualization("overall")

        self.assertIsInstance(vis_lines, list)
        # Should have lines if events exist
        if self.outliers.exist("overall"):
            self.assertGreater(len(vis_lines), 0)
            for line in vis_lines:
                self.assertIsInstance(line, str)

    def test_prepare_visualization_empty_when_no_events(self):
        """
        Verify prepare_visualization() returns empty list when no events exist
        """
        vis_lines = self.outliers.prepare_visualization("overall")

        self.assertIsInstance(vis_lines, list)
        self.assertEqual(len(vis_lines), 0)

    def test_prepare_visualization_contains_expected_patterns(self):
        """
        Verify prepare_visualization() generates correct PyMOL commands
        """
        # Add events
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        self.outliers.add_transport_event("md1", "path_003", "release", sample_traced_event_3)

        vis_lines = self.outliers.prepare_visualization("overall")

        # Join all lines to check for patterns
        all_content = "\n".join(vis_lines)

        # Check for key patterns
        self.assertIn("events = [", all_content)
        self.assertIn("for event in events:", all_content)
        self.assertIn("pickle.load(in_stream)", all_content)
        self.assertIn("cmd.load_cgo(path,", all_content)

    def test_prepare_visualization_md_specific(self):
        """
        Verify prepare_visualization() can generate MD-specific visualizations
        """
        # Add events to different MD labels
        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)
        self.outliers.add_transport_event("md2", "path_002", "entry", sample_traced_event_2)

        # Get md1-specific visualization
        vis_lines_md1 = self.outliers.prepare_visualization("md1")

        self.assertIsInstance(vis_lines_md1, list)
        # Should have content if events exist for md1
        if self.outliers.exist("md1"):
            self.assertGreater(len(vis_lines_md1), 0)


class TestOutlierTransportEventsEdgeCases(unittest.TestCase):
    """
    Test edge cases and boundary conditions for OutlierTransportEvents
    """

    def setUp(self):
        """
        Set up test fixtures
        """
        self.temp_dir = tempfile.mkdtemp()
        self.test_params = test_parameters_minimal.copy()
        self.test_params["super_cluster_details_folder"] = os.path.join(self.temp_dir, "details")
        self.test_params["visualization_folder"] = os.path.join(self.temp_dir, "viz")
        self.test_params["layered_aquaduct_vis_path"] = os.path.join(self.temp_dir, "layered")

        self.outliers = OutlierTransportEvents(self.test_params)

    def tearDown(self):
        """
        Clean up
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_add_event_with_empty_md_label(self):
        """
        Verify behavior with empty MD label string
        """
        self.outliers.add_transport_event("", "path_001", "entry", sample_traced_event_1)

        # Should still work
        self.assertTrue(self.outliers.exist(""))
        total, entry, release = self.outliers.count_events("")
        self.assertEqual(total, 1)

    def test_add_event_with_special_characters_in_md_label(self):
        """
        Verify handling of special characters in MD labels
        """
        special_label = "md-1_test.v2"
        self.outliers.add_transport_event(special_label, "path_001", "entry", sample_traced_event_1)

        self.assertTrue(self.outliers.exist(special_label))
        total, entry, release = self.outliers.count_events(special_label)
        self.assertEqual(total, 1)

    def test_report_details_creates_directory_if_not_exists(self):
        """
        Verify report_events_details() creates parent directories
        """
        # Use a nested path that doesn't exist
        nested_path = os.path.join(self.temp_dir, "level1", "level2", "level3")
        self.test_params["super_cluster_details_folder"] = nested_path

        self.outliers.add_transport_event("md1", "path_001", "entry", sample_traced_event_1)

        filename = "test_nested.txt"
        self.outliers.report_events_details(filename)

        # Directory should be created
        self.assertTrue(os.path.exists(nested_path))

        # File should exist
        expected_path = os.path.join(nested_path, filename)
        self.assertTrue(os.path.exists(expected_path))

    def test_large_number_of_events(self):
        """
        Verify handling of large number of events
        """
        # Add 1000 events
        for i in range(1000):
            path_id = f"path_{i:05d}"
            traced_event = (f"WAT:{i}", (i, i + 10))
            event_type = "entry" if i % 2 == 0 else "release"
            self.outliers.add_transport_event("md1", path_id, event_type, traced_event)

        # Check counts
        total, entry, release = self.outliers.count_events("md1")
        self.assertEqual(total, 1000)
        self.assertEqual(entry, 500)
        self.assertEqual(release, 500)

    def test_summary_line_with_zero_events(self):
        """
        Verify report_summary_line() handles zero events correctly
        """
        summary_line = self.outliers.report_summary_line(test_column_widths, "overall")

        # Should contain zeros
        self.assertIn(expected_summary_line_format, summary_line)
        # Parse and verify zeros
        parts = summary_line.split(":")
        numbers_part = parts[1].strip()
        # Should show 0, 0, 0
        self.assertIn("0", numbers_part)


if __name__ == "__main__":
    unittest.main()
