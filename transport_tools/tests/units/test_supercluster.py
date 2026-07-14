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
from unittest.mock import Mock
from transport_tools.libs.networks import SuperCluster, get_md_membership4groups, subsample_events


class TestSuperCluster(unittest.TestCase):
    """
    Unit tests for SuperCluster class
    Tests critical methods for managing tunnel superclusters and transport events
    """

    def setUp(self):
        """
        Set up test fixtures
        """
        self.maxDiff = None
        self.temp_dir = tempfile.mkdtemp()

        # Basic parameters for SuperCluster
        self.parameters = {
            "perform_comparative_analysis": False,
            "comparative_groups_definition": None,
            "transformation_folder": os.path.join(self.temp_dir, "transform"),
            "caver_foldername": "caver",
            "super_cluster_path_set_folder": os.path.join(self.temp_dir, "pathsets"),
            "aqauduct_ligand_effective_radius": 1.5,
            "layer_thickness": 1.5
        }

        self.sc = SuperCluster(sc_id=1, parameters=self.parameters, total_num_md_sims=3)

    def tearDown(self):
        """
        Clean up
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """
        Verify SuperCluster initializes with correct attributes
        """
        self.assertEqual(self.sc.sc_id, 1)
        self.assertIsNone(self.sc.prioritized_sc_id)
        self.assertEqual(self.sc.total_num_md_sims, 3)
        self.assertIsInstance(self.sc.tunnel_clusters, dict)
        self.assertIsInstance(self.sc.tunnel_clusters_valid, dict)
        self.assertIsInstance(self.sc.transport_events, dict)
        self.assertIsInstance(self.sc.num_events, dict)
        self.assertIn("overall", self.sc.num_events)
        self.assertEqual(self.sc.num_events["overall"]["entry"], 0)
        self.assertEqual(self.sc.num_events["overall"]["release"], 0)

    def test_add_transport_event_entry(self):
        """
        Verify add_transport_event() correctly stores entry events
        """
        # Setup: Add properties so event can be locally assigned
        self.sc.set_properties({"md1": True})

        result = self.sc.add_transport_event(
            md_label="md1",
            path_id="path_001",
            event_type="entry",
            traced_event=("WAT:123", (10, 50))
        )

        # Should be locally assigned
        self.assertTrue(result)

        # Check storage
        self.assertIn("entry", self.sc.transport_events)
        self.assertIn("md1", self.sc.transport_events["entry"])
        self.assertEqual(len(self.sc.transport_events["entry"]["md1"]), 1)

        # Check counts
        self.assertEqual(self.sc.num_events["overall"]["entry"], 1)
        self.assertEqual(self.sc.num_events["overall"]["release"], 0)
        self.assertEqual(self.sc.num_events["md1"]["entry"], 1)

    def test_add_transport_event_release(self):
        """
        Verify add_transport_event() correctly stores release events
        """
        self.sc.set_properties({"md1": True})

        result = self.sc.add_transport_event(
            md_label="md1",
            path_id="path_002",
            event_type="release",
            traced_event=("WAT:456", (20, 60))
        )

        self.assertTrue(result)
        self.assertIn("release", self.sc.transport_events)
        self.assertEqual(self.sc.num_events["md1"]["release"], 1)

    def test_add_transport_event_multiple_md_labels(self):
        """
        Verify add_transport_event() handles events from multiple MD simulations
        """
        self.sc.set_properties({"md1": True, "md2": True})

        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        self.sc.add_transport_event("md1", "path_002", "entry", ("WAT:456", (20, 60)))
        self.sc.add_transport_event("md2", "path_003", "release", ("WAT:789", (30, 70)))

        # Check overall counts
        self.assertEqual(self.sc.num_events["overall"]["entry"], 2)
        self.assertEqual(self.sc.num_events["overall"]["release"], 1)

        # Check per-MD counts
        self.assertEqual(self.sc.num_events["md1"]["entry"], 2)
        self.assertEqual(self.sc.num_events["md2"]["release"], 1)

    def test_add_transport_event_without_local_properties(self):
        """
        Verify add_transport_event() returns False when no local properties
        """
        # Don't set properties - event can't be locally assigned
        result = self.sc.add_transport_event(
            md_label="md1",
            path_id="path_001",
            event_type="entry",
            traced_event=("WAT:123", (10, 50))
        )

        # Should not be locally assigned
        self.assertFalse(result)

        # But still counted in overall
        self.assertEqual(self.sc.num_events["overall"]["entry"], 1)

    def test_add_transport_event_with_comparative_analysis(self):
        """
        Verify add_transport_event() handles comparative analysis grouping
        """
        params_comp = self.parameters.copy()
        params_comp["perform_comparative_analysis"] = True
        params_comp["comparative_groups_definition"] = {
            "group_A": ["md1", "md2"],
            "group_B": ["md3"]
        }

        sc_comp = SuperCluster(sc_id=1, parameters=params_comp, total_num_md_sims=3)
        sc_comp.set_properties({"group_A": True})

        # Add event to md1 (in group_A)
        result = sc_comp.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))

        # Should be locally assigned to group_A
        self.assertTrue(result)
        self.assertEqual(sc_comp.num_events["group_A"]["entry"], 1)

    def test_add_caver_cluster(self):
        """
        Verify add_caver_cluster() correctly stores tunnel clusters
        """
        # Create mock LayeredPathSet
        mock_pathset = Mock()
        mock_pathset.entity_label = "Cluster_45"

        self.sc.add_caver_cluster(md_label="md1", cls_id=45, path_set=mock_pathset)

        # Check storage
        self.assertIn("md1", self.sc.tunnel_clusters)
        self.assertIn(45, self.sc.tunnel_clusters["md1"])
        self.assertEqual(self.sc.tunnel_clusters["md1"][45], mock_pathset)

        # Check validity tracking
        self.assertIn("md1", self.sc.tunnel_clusters_valid)
        self.assertIn(45, self.sc.tunnel_clusters_valid["md1"])
        self.assertTrue(self.sc.tunnel_clusters_valid["md1"][45])

    def test_add_caver_cluster_multiple_md_labels(self):
        """
        Verify add_caver_cluster() handles clusters from multiple MD simulations
        """
        mock_pathset1 = Mock()
        mock_pathset2 = Mock()
        mock_pathset3 = Mock()

        self.sc.add_caver_cluster("md1", 45, mock_pathset1)
        self.sc.add_caver_cluster("md1", 46, mock_pathset2)
        self.sc.add_caver_cluster("md2", 50, mock_pathset3)

        # Check storage
        self.assertEqual(len(self.sc.tunnel_clusters["md1"]), 2)
        self.assertEqual(len(self.sc.tunnel_clusters["md2"]), 1)
        self.assertIn(45, self.sc.tunnel_clusters["md1"])
        self.assertIn(46, self.sc.tunnel_clusters["md1"])
        self.assertIn(50, self.sc.tunnel_clusters["md2"])

    def test_set_properties(self):
        """
        Verify set_properties() correctly updates properties
        """
        test_props = {"md1": True, "md2": False, "md3": True}
        self.sc.set_properties(test_props)

        self.assertIsInstance(self.sc.properties, dict)
        self.assertEqual(self.sc.properties, test_props)
        # Verify it's a copy, not a reference
        test_props["md4"] = True
        self.assertNotIn("md4", self.sc.properties)

    def test_set_bottleneck_residue_freq(self):
        """
        Verify set_bottleneck_residue_freq() correctly updates residue frequencies
        """
        test_freq = {"ALA_10": 0.5, "VAL_20": 0.3}
        self.sc.set_bottleneck_residue_freq(test_freq)

        self.assertIsInstance(self.sc.bottleneck_residue_freq, dict)
        self.assertEqual(self.sc.bottleneck_residue_freq, test_freq)
        # Verify it's a copy
        test_freq["LEU_30"] = 0.2
        self.assertNotIn("LEU_30", self.sc.bottleneck_residue_freq)

    def test_update_caver_clusters_validity_retained(self):
        """
        Verify update_caver_clusters_validity() marks retained clusters as valid
        """
        # Add some clusters
        mock_pathset = Mock()
        self.sc.add_caver_cluster("md1", 45, mock_pathset)
        self.sc.add_caver_cluster("md1", 46, mock_pathset)
        self.sc.add_caver_cluster("md2", 50, mock_pathset)

        # Update validity - retain only cluster 45 from md1
        retained = {"md1": [45]}
        self.sc.update_caver_clusters_validity(retained)

        # Check validity
        self.assertTrue(self.sc.tunnel_clusters_valid["md1"][45])
        self.assertFalse(self.sc.tunnel_clusters_valid["md1"][46])
        self.assertFalse(self.sc.tunnel_clusters_valid["md2"][50])

    def test_update_caver_clusters_validity_all_invalidated(self):
        """
        Verify update_caver_clusters_validity() invalidates all when none retained
        """
        mock_pathset = Mock()
        self.sc.add_caver_cluster("md1", 45, mock_pathset)
        self.sc.add_caver_cluster("md2", 50, mock_pathset)

        # No clusters retained
        retained = {}
        self.sc.update_caver_clusters_validity(retained)

        # All should be invalid
        self.assertFalse(self.sc.tunnel_clusters_valid["md1"][45])
        self.assertFalse(self.sc.tunnel_clusters_valid["md2"][50])

    def test_has_passed_filter_no_properties(self):
        """
        Verify has_passed_filter() returns False when no properties set
        """
        # Don't set properties
        self.assertFalse(self.sc.has_passed_filter())

    def test_has_passed_filter_with_properties(self):
        """
        Verify has_passed_filter() returns True when properties set
        """
        self.sc.set_properties({"md1": True})
        self.assertTrue(self.sc.has_passed_filter())

    def test_has_passed_filter_empty_properties(self):
        """
        Verify has_passed_filter() returns False for empty properties dict
        """
        self.sc.set_properties({})
        self.assertFalse(self.sc.has_passed_filter())

    def test_has_passed_filter_with_transport_events_pass(self):
        """
        Verify has_passed_filter() checks transport event filters
        """
        self.sc.set_properties({"md1": True})

        # Add events
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        self.sc.add_transport_event("md1", "path_002", "release", ("WAT:456", (20, 60)))

        # Define filters requiring at least 1 event
        active_filters = {
            "min_transport_events": 1,
            "min_entry_events": 0,
            "min_release_events": 0
        }

        # Should pass
        self.assertTrue(self.sc.has_passed_filter(consider_transport_events=True, active_filters=active_filters))

    def test_has_passed_filter_with_transport_events_fail(self):
        """
        Verify has_passed_filter() fails when event requirements not met
        """
        self.sc.set_properties({"md1": True})

        # Add only 1 entry event
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))

        # Define filters requiring 2 entry events
        active_filters = {
            "min_transport_events": 0,
            "min_entry_events": 2,
            "min_release_events": 0
        }

        # Should fail
        self.assertFalse(self.sc.has_passed_filter(consider_transport_events=True, active_filters=active_filters))

    def test_has_passed_filter_requires_filters_when_considering_events(self):
        """
        Verify has_passed_filter() raises error when filters not provided
        """
        self.sc.set_properties({"md1": True})

        with self.assertRaises(RuntimeError):
            self.sc.has_passed_filter(consider_transport_events=True, active_filters=None)

    def test_count_md_labels4events_no_events(self):
        """
        Verify count_md_labels4events() returns 0 when no events
        """
        self.assertEqual(self.sc.count_md_labels4events(), 0)

    def test_count_md_labels4events_single_md_label(self):
        """
        Verify count_md_labels4events() counts simulations with events
        """
        self.sc.set_properties({"md1": True})
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        self.sc.add_transport_event("md1", "path_002", "entry", ("WAT:456", (20, 60)))

        self.assertEqual(self.sc.count_md_labels4events(), 1)

    def test_count_md_labels4events_multiple_md_labels(self):
        """
        Verify count_md_labels4events() counts unique simulations
        """
        self.sc.set_properties({"md1": True, "md2": True, "md3": True})
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        self.sc.add_transport_event("md2", "path_002", "release", ("WAT:456", (20, 60)))
        self.sc.add_transport_event("md3", "path_003", "entry", ("WAT:789", (30, 70)))
        self.sc.add_transport_event("md1", "path_004", "release", ("WAT:012", (40, 80)))

        # 3 unique MD labels contributed events
        self.assertEqual(self.sc.count_md_labels4events(), 3)

    def test_get_md_labels_no_clusters(self):
        """
        Verify get_md_labels() returns empty list when no clusters
        """
        md_labels = self.sc.get_md_labels()
        self.assertEqual(len(md_labels), 0)

    def test_get_md_labels_with_valid_clusters(self):
        """
        Verify get_md_labels() returns MD labels with valid clusters
        """
        mock_pathset = Mock()
        self.sc.add_caver_cluster("md1", 45, mock_pathset)
        self.sc.add_caver_cluster("md2", 50, mock_pathset)

        md_labels = self.sc.get_md_labels()

        self.assertEqual(len(md_labels), 2)
        self.assertIn("md1", md_labels)
        self.assertIn("md2", md_labels)

    def test_get_md_labels_excludes_invalid_clusters(self):
        """
        Verify get_md_labels() excludes MD labels with all invalid clusters
        """
        mock_pathset = Mock()
        self.sc.add_caver_cluster("md1", 45, mock_pathset)
        self.sc.add_caver_cluster("md2", 50, mock_pathset)

        # Invalidate all clusters in md2
        self.sc.update_caver_clusters_validity({"md1": [45]})

        md_labels = self.sc.get_md_labels()

        # Only md1 should be included
        self.assertEqual(len(md_labels), 1)
        self.assertIn("md1", md_labels)
        self.assertNotIn("md2", md_labels)


class TestSuperClusterPerResidueTracking(unittest.TestCase):
    """
    Unit tests for the per-residue event tracking introduced for multi-residue
    AQUA-DUCT analyses (WAT + ligand). Covers:
    - num_events_by_residue initialization and population
    - get_summary_line_data residue_names parameter (new optional argument)
    """

    def setUp(self):
        self.maxDiff = None
        self.temp_dir = tempfile.mkdtemp()
        self.parameters = {
            "perform_comparative_analysis": False,
            "comparative_groups_definition": None,
            "transformation_folder": os.path.join(self.temp_dir, "transform"),
            "caver_foldername": "caver",
            "super_cluster_path_set_folder": os.path.join(self.temp_dir, "pathsets"),
            "aqauduct_ligand_effective_radius": 1.5,
            "layer_thickness": 1.5,
        }
        self.sc = SuperCluster(sc_id=1, parameters=self.parameters, total_num_md_sims=3)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_num_events_by_residue_initialized(self):
        """Attribute exists and is initialized to {'overall': {}}"""
        self.assertIsInstance(self.sc.num_events_by_residue, dict)
        self.assertIn("overall", self.sc.num_events_by_residue)
        self.assertEqual(self.sc.num_events_by_residue["overall"], {})

    def test_add_event_populates_overall_by_residue(self):
        """Adding a WAT entry must populate num_events_by_residue['overall']['WAT']"""
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        self.assertEqual(self.sc.num_events_by_residue["overall"]["WAT"]["entry"], 1)
        self.assertEqual(self.sc.num_events_by_residue["overall"]["WAT"]["release"], 0)

    def test_add_event_two_residues_independent(self):
        """WAT and U01 must be tracked under separate keys without contamination"""
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        self.sc.add_transport_event("md1", "path_002", "release", ("U01:55", (20, 60)))
        self.assertEqual(self.sc.num_events_by_residue["overall"]["WAT"]["entry"], 1)
        self.assertEqual(self.sc.num_events_by_residue["overall"]["WAT"]["release"], 0)
        self.assertEqual(self.sc.num_events_by_residue["overall"]["U01"]["entry"], 0)
        self.assertEqual(self.sc.num_events_by_residue["overall"]["U01"]["release"], 1)

    def test_add_event_per_stat_md_label_under_comparative(self):
        """Comparative analysis: per-residue counts also tracked at group level when
        local properties are set for that group."""
        params = self.parameters.copy()
        params["perform_comparative_analysis"] = True
        params["comparative_groups_definition"] = {
            "group_A": ["md1", "md2"], "group_B": ["md3"]
        }
        sc = SuperCluster(sc_id=1, parameters=params, total_num_md_sims=3)
        sc.set_properties({"group_A": True})

        result = sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        self.assertTrue(result)
        self.assertEqual(sc.num_events_by_residue["group_A"]["WAT"]["entry"], 1)
        # overall always incremented too
        self.assertEqual(sc.num_events_by_residue["overall"]["WAT"]["entry"], 1)

    def test_get_summary_line_data_without_residue_names_unchanged_length(self):
        """Without residue_names: backwards-compatible column count.
        sc_id (1) + 13 tunnel-property dashes (no properties set for 'overall')
        + 3 event columns = 17 items."""
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        data = self.sc.get_summary_line_data(print_transport_events=True, md_label="overall")
        self.assertEqual(len(data), 17)

    def test_get_summary_line_data_with_residue_names_appends_quadruples(self):
        """Passing residue_names adds 4 columns (E_avg, E_std, R_avg, R_std) per residue."""
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        self.sc.add_transport_event("md1", "path_002", "release", ("U01:55", (20, 60)))
        data = self.sc.get_summary_line_data(print_transport_events=True,
                                              md_label="overall",
                                              residue_names=["U01", "WAT"],
                                              sims2process=["md1"])
        # 17 base columns + 8 (2 residues × 4 columns)
        self.assertEqual(len(data), 25)
        # Order matches residue_names: U01_E_avg, U01_E_std, U01_R_avg, U01_R_std, WAT_...
        self.assertEqual(data[17], "0.0")  # U01 entry avg
        self.assertEqual(data[18], "0.0")  # U01 entry std
        self.assertEqual(data[19], "1.0")  # U01 release avg
        self.assertEqual(data[20], "0.0")  # U01 release std
        self.assertEqual(data[21], "1.0")  # WAT entry avg
        self.assertEqual(data[22], "0.0")  # WAT entry std
        self.assertEqual(data[23], "0.0")  # WAT release avg
        self.assertEqual(data[24], "0.0")  # WAT release std

    def test_get_summary_line_data_absent_residue_outputs_dash(self):
        """Residue in residue_names but absent from this SC's events → '-' for all 4 cols"""
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        data = self.sc.get_summary_line_data(print_transport_events=True,
                                              md_label="overall",
                                              residue_names=["U01", "WAT"],
                                              sims2process=["md1"])
        # U01 has no events anywhere — all 4 columns are '-'
        self.assertEqual(data[17], "-")
        self.assertEqual(data[18], "-")
        self.assertEqual(data[19], "-")
        self.assertEqual(data[20], "-")
        # WAT has 1 entry, 0 release, single sim → avg=count, std=0
        self.assertEqual(data[21], "1.0")
        self.assertEqual(data[22], "0.0")
        self.assertEqual(data[23], "0.0")
        self.assertEqual(data[24], "0.0")

    def test_get_summary_line_data_residue_names_requires_sims2process(self):
        """residue_names without sims2process must raise, not silently emit nan"""
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        with self.assertRaises(ValueError):
            self.sc.get_summary_line_data(print_transport_events=True,
                                          md_label="overall",
                                          residue_names=["WAT"])

    def test_get_summary_line_data_residue_avg_std_across_multiple_sims(self):
        """Per-residue avg/std pool per-simulation counts, zero-filling silent simulations
        that belong to sims2process (the group) but contributed no event of that residue."""
        self.sc.add_transport_event("md1", "path_001", "entry", ("WAT:123", (10, 50)))
        self.sc.add_transport_event("md1", "path_002", "entry", ("WAT:124", (10, 50)))
        self.sc.add_transport_event("md2", "path_003", "entry", ("WAT:125", (10, 50)))
        # md3 contributes nothing for this SC/residue but is part of the group being summarized
        data = self.sc.get_summary_line_data(print_transport_events=True,
                                              md_label="overall",
                                              residue_names=["WAT"],
                                              sims2process=["md1", "md2", "md3"])
        # WAT entry counts per sim: [2, 1, 0] -> avg=1.0, std=sqrt(((2-1)^2+(1-1)^2+(0-1)^2)/3)=0.816
        self.assertEqual(data[17], "1.0")
        self.assertEqual(data[18], "0.8")


class TestModuleFunctions(unittest.TestCase):
    """
    Test module-level functions from networks.py
    """

    def test_get_md_membership4groups_basic(self):
        """
        Verify get_md_membership4groups() creates correct membership mapping
        """
        groups_def = {
            "group_A": ["md1", "md2"],
            "group_B": ["md3", "md4"]
        }

        membership = get_md_membership4groups(groups_def)

        self.assertEqual(membership["md1"], "group_A")
        self.assertEqual(membership["md2"], "group_A")
        self.assertEqual(membership["md3"], "group_B")
        self.assertEqual(membership["md4"], "group_B")

    def test_get_md_membership4groups_duplicate_raises_error(self):
        """
        Verify get_md_membership4groups() raises error for duplicate assignments
        """
        groups_def = {
            "group_A": ["md1", "md2"],
            "group_B": ["md2", "md3"]  # md2 is duplicate
        }

        with self.assertRaises(ValueError) as context:
            get_md_membership4groups(groups_def)

        self.assertIn("md2", str(context.exception))
        self.assertIn("multiple groups", str(context.exception))

    def test_get_md_membership4groups_empty_groups(self):
        """
        Verify get_md_membership4groups() handles empty group definitions
        """
        groups_def = {}
        membership = get_md_membership4groups(groups_def)
        self.assertEqual(len(membership), 0)

    def test_subsample_events_no_subsampling_needed(self):
        """
        Verify subsample_events() returns all events when below max
        """
        events_dict = {
            "md1": [("path_001", ("WAT:123", (10, 50))),
                   ("path_002", ("WAT:456", (20, 60)))]
        }

        result = list(subsample_events(events_dict, random_seed=42, max_events=10,
                                      md_label="overall", comparative_groups_definition=None))

        # Should return all events
        self.assertEqual(len(result), 2)

    def test_subsample_events_subsamples_when_exceeds_max(self):
        """
        Verify subsample_events() subsamples when exceeding max_events
        """
        # Create more events than max
        events_dict = {
            "md1": [(f"path_{i:03d}", ("WAT:123", (i*10, i*10+40))) for i in range(100)]
        }

        max_events = 10
        result = list(subsample_events(events_dict, random_seed=42, max_events=max_events,
                                      md_label="overall", comparative_groups_definition=None))

        # Should be subsampled to max_events
        self.assertEqual(len(result), max_events)

    def test_subsample_events_reproducible_with_same_seed(self):
        """
        Verify subsample_events() gives reproducible results with same seed
        """
        events_dict = {
            "md1": [(f"path_{i:03d}", ("WAT:123", (i*10, i*10+40))) for i in range(100)]
        }

        result1 = list(subsample_events(events_dict, random_seed=42, max_events=10,
                                        md_label="overall", comparative_groups_definition=None))
        result2 = list(subsample_events(events_dict, random_seed=42, max_events=10,
                                        md_label="overall", comparative_groups_definition=None))

        # Should be identical
        self.assertEqual(result1, result2)

    def test_subsample_events_different_with_different_seed(self):
        """
        Verify subsample_events() gives different results with different seeds
        """
        events_dict = {
            "md1": [(f"path_{i:03d}", ("WAT:123", (i*10, i*10+40))) for i in range(100)]
        }

        result1 = list(subsample_events(events_dict, random_seed=42, max_events=10,
                                        md_label="overall", comparative_groups_definition=None))
        result2 = list(subsample_events(events_dict, random_seed=123, max_events=10,
                                        md_label="overall", comparative_groups_definition=None))

        # Should be different (with very high probability)
        self.assertNotEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()