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
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from transport_tools.libs.tools import EventAssigner
from transport_tools.tests.units.data.data_event_assigner import (
    test_parameters_minimal, test_parameters_exact_matching, test_parameters_trace_matching,
    test_parameters_penetration_span, test_parameters_directionality,
    sample_event_specification_1, sample_event_specification_2,
    test_active_filters, sample_nodes_data_entry, sample_nodes_data_release,
    sample_nodes_data_no_terminal, expected_direction_entry,
    sample_buriedness_high, sample_buriedness_medium, sample_buriedness_low,
    sample_buriedness_below_cutoff, sample_depth_high, sample_depth_medium,
    sample_depth_low, sample_exact_matching_buriedness, sample_exact_matching_no_tunnels,
    sample_min_depth, sample_min_depth_narrow, sample_min_depth_wide,
    sample_direction_aligned, sample_direction_offaxis,
    test_parameters_trace_matching_strided, test_parameters_trace_matching_dense,
    test_parameters_trace_matching_strided_interp, sample_event_frame_trace
)


class TestEventAssigner(unittest.TestCase):
    """
    Unit tests for EventAssigner class
    Tests the functionality of assigning transport events to superclusters
    """

    def setUp(self):
        """
        Set up test fixtures before each test method
        """
        self.maxDiff = None
        self.temp_dir = tempfile.mkdtemp()

        # Update parameters to use temp directory
        self.test_params = test_parameters_minimal.copy()
        self.test_params["trajectory_path"] = os.path.join(self.temp_dir, "traj")
        self.test_params["exact_matching_details_folder"] = os.path.join(self.temp_dir, "exact")

        # Create mock LayeredPathSet
        self.mock_event = Mock()
        self.mock_event.entity_label = "1_entry"
        self.mock_event.nodes_data = sample_nodes_data_entry

        # Create mock SuperCluster
        self.mock_sc1 = Mock()
        self.mock_sc1.sc_id = 1
        self.mock_sc1.has_passed_filter.return_value = True
        self.mock_sc1.is_directionally_aligned.return_value = True
        self.mock_sc1.compute_distance2transport_event.return_value = (sample_buriedness_high, sample_depth_high, sample_min_depth)

        self.mock_superclusters = {1: self.mock_sc1}

    def tearDown(self):
        """
        Clean up temporary files after each test
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """
        Verify EventAssigner initializes with correct attributes
        """
        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        self.assertIsInstance(assigner.parameters, dict)
        self.assertEqual(assigner.event_specification, sample_event_specification_1)
        self.assertEqual(assigner.event, self.mock_event)
        self.assertEqual(assigner.super_clusters, self.mock_superclusters)
        self.assertEqual(assigner.active_filters, test_active_filters)

    def test_find_directionally_aligned_scs_single_match(self):
        """
        Verify _find_directionally_aligned_scs() finds single aligned SC
        """
        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        aligned_ids = assigner._find_directionally_aligned_scs()

        self.assertIsInstance(aligned_ids, list)
        self.assertEqual(len(aligned_ids), 1)
        self.assertEqual(aligned_ids[0], 1)

    def test_find_directionally_aligned_scs_multiple_matches(self):
        """
        Verify _find_directionally_aligned_scs() finds multiple aligned SCs
        """
        # Add second aligned SC
        mock_sc2 = Mock()
        mock_sc2.sc_id = 2
        mock_sc2.has_passed_filter.return_value = True
        mock_sc2.is_directionally_aligned.return_value = True

        self.mock_superclusters[2] = mock_sc2

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        aligned_ids = assigner._find_directionally_aligned_scs()

        self.assertEqual(len(aligned_ids), 2)
        self.assertIn(1, aligned_ids)
        self.assertIn(2, aligned_ids)

    def test_find_directionally_aligned_scs_no_matches(self):
        """
        Verify _find_directionally_aligned_scs() returns empty list when no SCs aligned
        """
        self.mock_sc1.is_directionally_aligned.return_value = False

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        aligned_ids = assigner._find_directionally_aligned_scs()

        self.assertEqual(len(aligned_ids), 0)

    def test_find_directionally_aligned_scs_filtered_out(self):
        """
        Verify _find_directionally_aligned_scs() excludes SCs that don't pass filters
        """
        self.mock_sc1.has_passed_filter.return_value = False

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        aligned_ids = assigner._find_directionally_aligned_scs()

        self.assertEqual(len(aligned_ids), 0)

    def test_find_directionally_aligned_scs_event_no_terminal_node(self):
        """
        Verify _find_directionally_aligned_scs() handles events with no terminal node
        """
        self.mock_event.nodes_data = sample_nodes_data_no_terminal

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        aligned_ids = assigner._find_directionally_aligned_scs()

        # Should return empty list when terminal node can't be found
        self.assertEqual(len(aligned_ids), 0)

    def test_perform_assignment_successful_single_sc(self):
        """
        Verify perform_assignment() successfully assigns to single SC
        """
        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()

        self.assertEqual(event_spec, sample_event_specification_1)
        self.assertIsNotNone(assigned_ids)
        self.assertEqual(len(assigned_ids), 1)
        self.assertEqual(assigned_ids[0], 1)
        self.assertEqual(max_buriedness, sample_buriedness_high)
        self.assertEqual(max_depth, sample_depth_high)

    def test_perform_assignment_no_directional_alignment(self):
        """
        Verify perform_assignment() returns None when no directional alignment
        """
        self.mock_sc1.is_directionally_aligned.return_value = False

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()

        self.assertEqual(event_spec, sample_event_specification_1)
        self.assertIsNone(assigned_ids)
        self.assertEqual(max_buriedness, -999)
        self.assertEqual(max_depth, -999)

    def test_perform_assignment_buriedness_below_cutoff(self):
        """
        Verify perform_assignment() returns None when buriedness below cutoff
        """
        self.mock_sc1.compute_distance2transport_event.return_value = (
            sample_buriedness_below_cutoff, sample_depth_high, sample_min_depth
        )

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()

        self.assertEqual(event_spec, sample_event_specification_1)
        self.assertIsNone(assigned_ids)
        self.assertEqual(max_buriedness, sample_buriedness_below_cutoff)
        self.assertEqual(max_depth, -999)  # Not computed when buriedness too low

    def test_perform_assignment_multiple_scs_same_buriedness(self):
        """
        Verify perform_assignment() handles multiple SCs with same buriedness
        Default behavior with penetration_depth resolution - only highest depth SC assigned
        """
        # Add second SC with same buriedness but lower depth
        mock_sc2 = Mock()
        mock_sc2.sc_id = 2
        mock_sc2.has_passed_filter.return_value = True
        mock_sc2.is_directionally_aligned.return_value = True
        mock_sc2.compute_distance2transport_event.return_value = (
            sample_buriedness_high, sample_depth_medium, sample_min_depth
        )

        self.mock_superclusters[2] = mock_sc2

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()

        self.assertEqual(event_spec, sample_event_specification_1)
        self.assertIsNotNone(assigned_ids)
        # With penetration_depth resolution (default), only SC with highest depth is assigned
        self.assertEqual(len(assigned_ids), 1)
        self.assertEqual(assigned_ids[0], 1)  # SC1 has higher depth
        self.assertEqual(max_buriedness, sample_buriedness_high)
        self.assertEqual(max_depth, sample_depth_high)  # Max of both

    def test_perform_assignment_penetration_depth_resolution(self):
        """
        Verify perform_assignment() uses penetration depth to resolve ambiguity
        """
        # Add second SC with same buriedness but lower depth
        mock_sc2 = Mock()
        mock_sc2.sc_id = 2
        mock_sc2.has_passed_filter.return_value = True
        mock_sc2.is_directionally_aligned.return_value = True
        mock_sc2.compute_distance2transport_event.return_value = (
            sample_buriedness_high, sample_depth_low, sample_min_depth
        )

        self.mock_superclusters[2] = mock_sc2

        # Use penetration_depth resolution
        params = self.test_params.copy()
        params["ambiguous_event_assignment_resolution"] = "penetration_depth"

        assigner = EventAssigner(
            params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()

        self.assertEqual(event_spec, sample_event_specification_1)
        self.assertIsNotNone(assigned_ids)
        # Only SC1 should be assigned (has higher depth)
        self.assertEqual(len(assigned_ids), 1)
        self.assertEqual(assigned_ids[0], 1)
        self.assertEqual(max_depth, sample_depth_high)

    def test_perform_assignment_different_buriedness(self):
        """
        Verify perform_assignment() selects SC with highest buriedness
        """
        # Add second SC with lower buriedness
        mock_sc2 = Mock()
        mock_sc2.sc_id = 2
        mock_sc2.has_passed_filter.return_value = True
        mock_sc2.is_directionally_aligned.return_value = True
        mock_sc2.compute_distance2transport_event.return_value = (
            sample_buriedness_medium, sample_depth_high, sample_min_depth
        )

        self.mock_superclusters[2] = mock_sc2

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()

        self.assertEqual(event_spec, sample_event_specification_1)
        self.assertIsNotNone(assigned_ids)
        # Only SC1 should be assigned (higher buriedness)
        self.assertEqual(len(assigned_ids), 1)
        self.assertEqual(assigned_ids[0], 1)
        self.assertEqual(max_buriedness, sample_buriedness_high)

    def test_perform_assignment_tolerance_threshold(self):
        """
        Verify perform_assignment() applies 0.05 tolerance for buriedness
        With penetration_depth resolution, highest depth SC is selected
        """
        # Add second SC with buriedness just slightly lower (within 0.05)
        mock_sc2 = Mock()
        mock_sc2.sc_id = 2
        mock_sc2.has_passed_filter.return_value = True
        mock_sc2.is_directionally_aligned.return_value = True
        mock_sc2.compute_distance2transport_event.return_value = (
            sample_buriedness_high - 0.03,  # Within 0.05 tolerance
            sample_depth_medium, sample_min_depth
        )

        self.mock_superclusters[2] = mock_sc2

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()

        # With penetration_depth resolution, only highest depth SC assigned
        self.assertIsNotNone(assigned_ids)
        self.assertEqual(len(assigned_ids), 1)
        self.assertEqual(assigned_ids[0], 1)  # SC1 has higher depth

    def test_perform_assignment_returns_correct_format(self):
        """
        Verify perform_assignment() returns tuple with correct format
        """
        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        result = assigner.perform_assignment()

        # Should return 4-tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

        event_spec, assigned_ids, max_buriedness, max_depth = result

        # Verify types
        self.assertIsInstance(event_spec, tuple)
        self.assertTrue(isinstance(assigned_ids, np.ndarray) or assigned_ids is None)
        self.assertIsInstance(max_buriedness, (int, float))
        self.assertIsInstance(max_depth, (int, float))


class TestEventAssignerExactMatching(unittest.TestCase):
    """
    Test exact matching functionality of EventAssigner
    Note: These tests use mocks due to trajectory dependencies
    """

    def setUp(self):
        """
        Set up test fixtures
        """
        self.temp_dir = tempfile.mkdtemp()
        self.test_params = test_parameters_exact_matching.copy()
        self.test_params["trajectory_path"] = os.path.join(self.temp_dir, "traj")
        self.test_params["exact_matching_details_folder"] = os.path.join(self.temp_dir, "exact")

        # Create mock event and superclusters
        self.mock_event = Mock()
        self.mock_event.entity_label = "1_entry"
        self.mock_event.nodes_data = sample_nodes_data_entry

        self.mock_sc1 = Mock()
        self.mock_sc1.sc_id = 1
        self.mock_sc1.has_passed_filter.return_value = True
        self.mock_sc1.is_directionally_aligned.return_value = True
        self.mock_sc1.compute_distance2transport_event.return_value = (
            sample_buriedness_high, sample_depth_high, sample_min_depth
        )

        self.mock_superclusters = {1: self.mock_sc1}

    def tearDown(self):
        """
        Clean up
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('transport_tools.libs.tools.TrajectoryFactory')
    def test_exact_event_tunnel_matching_no_trajectory(self, mock_traj_factory):
        """
        Verify _exact_event_tunnel_matching() handles missing trajectory gracefully
        """
        # Mock trajectory that doesn't exist
        mock_traj = Mock()
        mock_traj.inputs_exists.return_value = False
        mock_traj_factory.return_value = mock_traj

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            self.mock_superclusters,
            test_active_filters
        )

        # Should return empty buriedness dict
        buriedness = assigner._exact_event_tunnel_matching(np.array([1]))

        self.assertIsInstance(buriedness, dict)
        self.assertIn("4all_frames", buriedness)
        self.assertIn("4existing_tunnels", buriedness)
        self.assertEqual(len(buriedness["4all_frames"]), 0)
        self.assertEqual(len(buriedness["4existing_tunnels"]), 0)

    def test_exact_matching_resolution_with_exact_matching_disabled(self):
        """
        Verify exact_matching resolution triggers exact matching analysis
        """
        # Add second SC
        mock_sc2 = Mock()
        mock_sc2.sc_id = 2
        mock_sc2.has_passed_filter.return_value = True
        mock_sc2.is_directionally_aligned.return_value = True
        mock_sc2.compute_distance2transport_event.return_value = (
            sample_buriedness_high, sample_depth_medium, sample_min_depth
        )

        self.mock_superclusters[2] = mock_sc2

        # Use exact_matching resolution but disable exact matching analysis
        params = self.test_params.copy()
        params["ambiguous_event_assignment_resolution"] = "exact_matching"
        params["perform_exact_matching_analysis"] = False

        with patch.object(EventAssigner, '_exact_event_tunnel_matching') as mock_exact:
            mock_exact.return_value = sample_exact_matching_no_tunnels

            assigner = EventAssigner(
                params,
                sample_event_specification_1,
                self.mock_event,
                self.mock_superclusters,
                test_active_filters
            )

            # Should fail because exact matching is required but no tunnels match
            event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()

            # _exact_event_tunnel_matching should have been called
            mock_exact.assert_called_once()


class TestEventAssignerTraceMatching(unittest.TestCase):
    """
    Test trajectory-free trace matching functionality of EventAssigner
    Note: These tests mock the per-snapshot tunnel lookup; the trajectory-free trace itself is exercised
    end-to-end by the integration tests.
    """

    def setUp(self):
        """
        Set up test fixtures with two equally buried superclusters (an ambiguous assignment)
        """
        self.test_params = test_parameters_trace_matching.copy()

        self.mock_event = Mock()
        self.mock_event.entity_label = "1_entry"
        self.mock_event.nodes_data = sample_nodes_data_entry

        self.mock_sc1 = Mock()
        self.mock_sc1.sc_id = 1
        self.mock_sc1.has_passed_filter.return_value = True
        self.mock_sc1.is_directionally_aligned.return_value = True
        self.mock_sc1.compute_distance2transport_event.return_value = (sample_buriedness_high, sample_depth_high, sample_min_depth)

        self.mock_sc2 = Mock()
        self.mock_sc2.sc_id = 2
        self.mock_sc2.has_passed_filter.return_value = True
        self.mock_sc2.is_directionally_aligned.return_value = True
        self.mock_sc2.compute_distance2transport_event.return_value = (sample_buriedness_high, sample_depth_medium,
                                                                       sample_min_depth)

        self.mock_superclusters = {1: self.mock_sc1, 2: self.mock_sc2}

    def test_trace_matching_resolution_invoked_and_picks_best(self):
        """
        Verify trace_matching resolution runs _trace_event_tunnel_matching on an ambiguous event and keeps
        the supercluster with the highest matched buriedness
        """
        with patch.object(EventAssigner, '_trace_event_tunnel_matching') as mock_trace:
            mock_trace.return_value = sample_exact_matching_buriedness  # SC1 0.75 > SC2 0.50

            assigner = EventAssigner(self.test_params, sample_event_specification_1, self.mock_event,
                                     self.mock_superclusters, test_active_filters)
            _, assigned_ids, _, _ = assigner.perform_assignment()

            mock_trace.assert_called_once()
            self.assertIsNotNone(assigned_ids)
            self.assertEqual(list(assigned_ids), [1])

    def test_trace_matching_resolution_no_tunnels_unassigned(self):
        """
        Verify an ambiguous event whose trace matches no tunnels is left unassigned under trace_matching
        """
        with patch.object(EventAssigner, '_trace_event_tunnel_matching') as mock_trace:
            mock_trace.return_value = sample_exact_matching_no_tunnels

            assigner = EventAssigner(self.test_params, sample_event_specification_1, self.mock_event,
                                     self.mock_superclusters, test_active_filters)
            _, assigned_ids, _, _ = assigner.perform_assignment()

            mock_trace.assert_called_once()
            self.assertIsNone(assigned_ids)

    def test_trace_matching_analysis_runs_for_single_sc(self):
        """
        Verify perform_trace_matching_analysis triggers trace matching even for an unambiguous (single-SC)
        event, mirroring perform_exact_matching_analysis
        """
        single_sc = {1: self.mock_sc1}
        with patch.object(EventAssigner, '_trace_event_tunnel_matching') as mock_trace:
            mock_trace.return_value = sample_exact_matching_buriedness

            assigner = EventAssigner(self.test_params, sample_event_specification_1, self.mock_event,
                                     single_sc, test_active_filters)
            assigner.perform_assignment()

            # analysis pass runs the matching for the assigned event regardless of ambiguity
            mock_trace.assert_called_once()


class TestEventAssignerPenetrationSpan(unittest.TestCase):
    """
    Test penetration_span resolution of EventAssigner.
    penetration_span resolves ambiguity by the depth range an event traverses inside each supercluster
    (max_depth - shallowest buried depth) rather than by its single deepest point, so it favours genuine
    surface-to-core traversals over perpendicular crossings, without trajectories or AQUA-DUCT traces.
    """

    def setUp(self):
        """
        Set up two equally buried superclusters: SC1 reaches deeper (larger max_depth) but over a narrow span
        (perpendicular-like), SC2 is shallower yet spans a wide depth range (a genuine traversal).
        """
        self.test_params = test_parameters_penetration_span.copy()

        self.mock_event = Mock()
        self.mock_event.entity_label = "1_entry"
        self.mock_event.nodes_data = sample_nodes_data_entry

        self.mock_sc1 = Mock()  # deep but narrow span (15.0 - 14.0 = 1.0)
        self.mock_sc1.sc_id = 1
        self.mock_sc1.has_passed_filter.return_value = True
        self.mock_sc1.is_directionally_aligned.return_value = True
        self.mock_sc1.compute_distance2transport_event.return_value = (sample_buriedness_high, sample_depth_high,
                                                                       sample_min_depth_narrow)

        self.mock_sc2 = Mock()  # shallower but wide span (10.0 - 2.0 = 8.0)
        self.mock_sc2.sc_id = 2
        self.mock_sc2.has_passed_filter.return_value = True
        self.mock_sc2.is_directionally_aligned.return_value = True
        self.mock_sc2.compute_distance2transport_event.return_value = (sample_buriedness_high, sample_depth_medium,
                                                                       sample_min_depth_wide)

        self.mock_superclusters = {1: self.mock_sc1, 2: self.mock_sc2}

    def test_penetration_span_picks_wider_span_over_deeper(self):
        """
        Verify penetration_span assigns the event to the wider-span SC (SC2), not the deeper one (SC1)
        """
        assigner = EventAssigner(self.test_params, sample_event_specification_1, self.mock_event,
                                 self.mock_superclusters, test_active_filters)
        _, assigned_ids, _, _ = assigner.perform_assignment()

        self.assertIsNotNone(assigned_ids)
        self.assertEqual(list(assigned_ids), [2])

    def test_penetration_depth_picks_deeper_on_same_data(self):
        """
        Verify penetration_depth diverges from penetration_span on the same event: it keeps the deeper SC (SC1),
        confirming the two methods resolve the same ambiguity differently
        """
        params = self.test_params.copy()
        params["ambiguous_event_assignment_resolution"] = "penetration_depth"

        assigner = EventAssigner(params, sample_event_specification_1, self.mock_event,
                                 self.mock_superclusters, test_active_filters)
        _, assigned_ids, _, _ = assigner.perform_assignment()

        self.assertIsNotNone(assigned_ids)
        self.assertEqual(list(assigned_ids), [1])

    def test_penetration_span_keeps_ties(self):
        """
        Verify penetration_span keeps all SCs whose span is within the 0.05 tolerance of the best, so a genuine
        multi-SC traversal stays assigned to all of them
        """
        # SC1 span = 15.0 - 14.0 = 1.00; SC2 span = 11.0 - 10.02 = 0.98 -> within 0.05, both kept
        self.mock_sc2.compute_distance2transport_event.return_value = (sample_buriedness_high, 11.0, 10.02)

        assigner = EventAssigner(self.test_params, sample_event_specification_1, self.mock_event,
                                 self.mock_superclusters, test_active_filters)
        _, assigned_ids, _, _ = assigner.perform_assignment()

        self.assertIsNotNone(assigned_ids)
        self.assertEqual(sorted(assigned_ids), [1, 2])


class TestEventAssignerDirectionality(unittest.TestCase):
    """
    Test directionality resolution of EventAssigner.
    directionality resolves ambiguity by how closely the event's exit bearing matches each supercluster's overall
    direction, rejecting superclusters the event merely crosses perpendicularly, and breaks ties within 10 degrees
    by penetration span.
    """

    def setUp(self):
        """
        Set up two equally buried superclusters for the entry event (terminal direction [2, 2, 2]): SC1 is aligned
        (0 deg) but has a narrow span, SC2 is well off-axis (~70 deg) but has a wide span - so span alone would
        prefer SC2 while directionality must prefer the aligned SC1.
        """
        self.test_params = test_parameters_directionality.copy()

        self.mock_event = Mock()
        self.mock_event.entity_label = "1_entry"
        self.mock_event.nodes_data = sample_nodes_data_entry

        self.mock_sc1 = Mock()  # aligned (0 deg), narrow span (15.0 - 14.0 = 1.0)
        self.mock_sc1.sc_id = 1
        self.mock_sc1.has_passed_filter.return_value = True
        self.mock_sc1.is_directionally_aligned.return_value = True
        self.mock_sc1.avg_direction = sample_direction_aligned
        self.mock_sc1.compute_distance2transport_event.return_value = (sample_buriedness_high, sample_depth_high,
                                                                       sample_min_depth_narrow)

        self.mock_sc2 = Mock()  # off-axis (~70 deg), wide span (10.0 - 2.0 = 8.0)
        self.mock_sc2.sc_id = 2
        self.mock_sc2.has_passed_filter.return_value = True
        self.mock_sc2.is_directionally_aligned.return_value = True
        self.mock_sc2.avg_direction = sample_direction_offaxis
        self.mock_sc2.compute_distance2transport_event.return_value = (sample_buriedness_high, sample_depth_medium,
                                                                       sample_min_depth_wide)

        self.mock_superclusters = {1: self.mock_sc1, 2: self.mock_sc2}

    def test_directionality_picks_aligned_over_wider_span(self):
        """
        Verify directionality assigns the event to the aligned SC (SC1), rejecting the off-axis SC2 even though
        SC2 has the wider penetration span
        """
        assigner = EventAssigner(self.test_params, sample_event_specification_1, self.mock_event,
                                 self.mock_superclusters, test_active_filters)
        _, assigned_ids, _, _ = assigner.perform_assignment()

        self.assertIsNotNone(assigned_ids)
        self.assertEqual(list(assigned_ids), [1])

    def test_penetration_span_diverges_on_same_data(self):
        """
        Verify penetration_span diverges from directionality on the same event: it keeps the wider-span but
        off-axis SC2, confirming directionality's rejection of the perpendicular crossing is what differs
        """
        params = self.test_params.copy()
        params["ambiguous_event_assignment_resolution"] = "penetration_span"

        assigner = EventAssigner(params, sample_event_specification_1, self.mock_event,
                                 self.mock_superclusters, test_active_filters)
        _, assigned_ids, _, _ = assigner.perform_assignment()

        self.assertIsNotNone(assigned_ids)
        self.assertEqual(list(assigned_ids), [2])

    def test_directionality_breaks_ties_by_span(self):
        """
        Verify that among superclusters aligned within the 10 degree tolerance, directionality falls back to the
        wider penetration span: with both SCs aligned, the wider-span SC2 wins
        """
        self.mock_sc2.avg_direction = sample_direction_aligned  # now both within the 10 deg tie band

        assigner = EventAssigner(self.test_params, sample_event_specification_1, self.mock_event,
                                 self.mock_superclusters, test_active_filters)
        _, assigned_ids, _, _ = assigner.perform_assignment()

        self.assertIsNotNone(assigned_ids)
        self.assertEqual(list(assigned_ids), [2])

    def test_directionality_keeps_full_ties(self):
        """
        Verify directionality keeps all SCs that tie on both direction (within 10 deg) and span (within 0.05),
        so a genuine multi-SC traversal stays assigned to all of them
        """
        self.mock_sc2.avg_direction = sample_direction_aligned  # same direction as SC1
        # same span as SC1 (15.0 - 14.0 = 1.0) -> tie on both criteria
        self.mock_sc2.compute_distance2transport_event.return_value = (sample_buriedness_high, sample_depth_high,
                                                                       sample_min_depth_narrow)

        assigner = EventAssigner(self.test_params, sample_event_specification_1, self.mock_event,
                                 self.mock_superclusters, test_active_filters)
        _, assigned_ids, _, _ = assigner.perform_assignment()

        self.assertIsNotNone(assigned_ids)
        self.assertEqual(sorted(assigned_ids), [1, 2])


class TestEventAssignerStridedMatching(unittest.TestCase):
    """
    event frame -> CAVER snapshot tests for trajectory-free trace matching under a sampling stride.

    With caver_snapshot_stride > 1 only event frames that land on an analysed CAVER snapshot are matched;
    the frames between snapshots map to None and are skipped (exact alignment), so the ligand is never
    compared against a tunnel from a different time. At stride 1 every frame resolves to its own snapshot,
    reproducing the legacy f + caver_traj_offset behaviour.
    """

    @staticmethod
    def _make_recording_supercluster():
        """
        Mock supercluster whose single CAVER cluster records every snapshot id it is queried for and always
        reports a buried sphere (distance 0.5 <= the 1.5 effective-radius cutoff). Returns the supercluster
        and the list the queried snapshot ids accumulate into.
        """

        queried_snap_ids = []
        mock_cluster = Mock()
        mock_cluster.cluster_id = 1

        def _closest(trace_xyz, snap_id):
            queried_snap_ids.append(snap_id)
            # (distance, 6-element sphere [x, y, z, dist, radius, length]) -> closest_tunnels_data row of 12
            return 0.5, np.array([1.0, 1.0, 1.0, 0.5, 2.0, 15.0])

        mock_cluster.get_closest_tunnel_sphere_in_frame2coords.side_effect = _closest

        mock_sc = Mock()
        mock_sc.sc_id = 1
        mock_sc.get_caver_clusters.return_value = {"md1": [mock_cluster]}
        return mock_sc, queried_snap_ids

    @staticmethod
    def _make_event():
        mock_event = Mock()
        mock_event.entity_label = "1_entry"
        mock_event.nodes_data = sample_nodes_data_entry
        return mock_event

    def test_strided_matching_queries_only_grid_snapshots(self):
        """
        Verify under stride 20 only event frames on the snapshot grid (20, 40 -> snapshots 2, 3) are matched
        and the 39 in-between frames are skipped, so no tunnel is ever queried for a non-analysed frame
        """
        params = test_parameters_trace_matching_strided.copy()
        mock_sc, queried_snap_ids = self._make_recording_supercluster()

        with patch("transport_tools.libs.tools.get_event_frame_traces",
                   return_value={"1_entry": sample_event_frame_trace}):
            assigner = EventAssigner(params, sample_event_specification_1, self._make_event(),
                                     {1: mock_sc}, test_active_filters)
            buriedness = assigner._trace_event_tunnel_matching(np.array([1]))

        # event frames 20 and 40 are the only ones on the snapshot grid (stride 20) -> snapshots 2 and 3;
        # frames 10..19, 21..39, 41..50 map to None and are skipped (never queried for a tunnel)
        self.assertEqual(sorted(set(queried_snap_ids)), [2, 3])
        mock_sc.get_caver_clusters.assert_called_once_with(md_labels=["md1"], snap_ids=[2, 3])
        # both matched frames are buried (dist 0.5 <= 1.5): 2 buried of 41 reported frames; of 2 existing tunnels
        self.assertAlmostEqual(buriedness["4all_frames"][1], 2 / 41)
        self.assertAlmostEqual(buriedness["4existing_tunnels"][1], 1.0)

    def test_dense_stride1_matching_queries_every_frame(self):
        """
        Verify at the default stride 1 every event frame f (10..50) resolves to snapshot f + offset (11..51),
        so nothing is skipped - the back-compatible behaviour
        """
        params = test_parameters_trace_matching_dense.copy()
        mock_sc, queried_snap_ids = self._make_recording_supercluster()

        with patch("transport_tools.libs.tools.get_event_frame_traces",
                   return_value={"1_entry": sample_event_frame_trace}):
            assigner = EventAssigner(params, sample_event_specification_1, self._make_event(),
                                     {1: mock_sc}, test_active_filters)
            buriedness = assigner._trace_event_tunnel_matching(np.array([1]))

        # stride 1 + offset 1: every event frame f maps to snapshot f + 1, nothing dropped
        self.assertEqual(sorted(set(queried_snap_ids)), list(range(11, 52)))
        mock_sc.get_caver_clusters.assert_called_once_with(md_labels=["md1"], snap_ids=list(range(11, 52)))
        self.assertAlmostEqual(buriedness["4all_frames"][1], 41 / 41)
        self.assertAlmostEqual(buriedness["4existing_tunnels"][1], 1.0)

    def test_strided_matching_with_interpolation_matches_every_frame(self):
        """
        Verify interpolate_missing_snapshots4matching makes every event frame resolve to its nearest
        analysed snapshot (none skipped): under stride 20 frames 10..50 snap to the closest of 2, 3, 4
        """
        params = test_parameters_trace_matching_strided_interp.copy()
        mock_sc, queried_snap_ids = self._make_recording_supercluster()

        with patch("transport_tools.libs.tools.get_event_frame_traces",
                   return_value={"1_entry": sample_event_frame_trace}):
            assigner = EventAssigner(params, sample_event_specification_1, self._make_event(),
                                     {1: mock_sc}, test_active_filters)
            buriedness = assigner._trace_event_tunnel_matching(np.array([1]))

        # nothing is skipped: all 41 frames query their nearest snapshot (frames 10..29 -> 2, 30..49 -> 3,
        # 50 -> 4), the opposite of the non-interpolated run that queried only frames 20 and 40
        self.assertEqual(len(queried_snap_ids), 41)
        self.assertEqual(sorted(set(queried_snap_ids)), [2, 3, 4])
        mock_sc.get_caver_clusters.assert_called_once_with(md_labels=["md1"], snap_ids=[2, 3, 4])
        self.assertAlmostEqual(buriedness["4all_frames"][1], 41 / 41)
        self.assertAlmostEqual(buriedness["4existing_tunnels"][1], 1.0)


class TestEventAssignerEdgeCases(unittest.TestCase):
    """
    Test edge cases and boundary conditions for EventAssigner
    """

    def setUp(self):
        """
        Set up test fixtures
        """
        self.temp_dir = tempfile.mkdtemp()
        self.test_params = test_parameters_minimal.copy()
        self.test_params["trajectory_path"] = os.path.join(self.temp_dir, "traj")

        self.mock_event = Mock()
        self.mock_event.entity_label = "1_entry"
        self.mock_event.nodes_data = sample_nodes_data_entry

    def tearDown(self):
        """
        Clean up
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_empty_superclusters_dict(self):
        """
        Verify EventAssigner handles empty superclusters dictionary
        """
        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            {},  # Empty superclusters
            test_active_filters
        )

        aligned_ids = assigner._find_directionally_aligned_scs()
        self.assertEqual(len(aligned_ids), 0)

        event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()
        self.assertIsNone(assigned_ids)
        self.assertEqual(max_buriedness, -999)

    def test_event_with_zero_buriedness(self):
        """
        Verify handling of events with zero buriedness
        """
        mock_sc = Mock()
        mock_sc.sc_id = 1
        mock_sc.has_passed_filter.return_value = True
        mock_sc.is_directionally_aligned.return_value = True
        mock_sc.compute_distance2transport_event.return_value = (0.0, sample_depth_high, sample_min_depth)

        assigner = EventAssigner(
            self.test_params,
            sample_event_specification_1,
            self.mock_event,
            {1: mock_sc},
            test_active_filters
        )

        event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()

        # Should not be assigned (0.0 < cutoff of 0.5)
        self.assertIsNone(assigned_ids)
        self.assertEqual(max_buriedness, 0.0)

    def test_event_with_exact_cutoff_buriedness(self):
        """
        Verify handling of events with buriedness exactly at cutoff
        """
        cutoff = 0.5
        params = self.test_params.copy()
        params["event_assignment_cutoff"] = cutoff

        mock_sc = Mock()
        mock_sc.sc_id = 1
        mock_sc.has_passed_filter.return_value = True
        mock_sc.is_directionally_aligned.return_value = True
        mock_sc.compute_distance2transport_event.return_value = (cutoff, sample_depth_high, sample_min_depth)

        assigner = EventAssigner(
            params,
            sample_event_specification_1,
            self.mock_event,
            {1: mock_sc},
            test_active_filters
        )

        event_spec, assigned_ids, max_buriedness, max_depth = assigner.perform_assignment()

        # Should be assigned (>= cutoff)
        self.assertIsNotNone(assigned_ids)
        self.assertEqual(len(assigned_ids), 1)


if __name__ == "__main__":
    unittest.main()