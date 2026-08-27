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

"""
One-shot generator for the per-variant golden fixtures consumed by
TestTransportProcesses.test_11assign_transport_events.

The integration test golden-compares event assignment under every non-default
ambiguous_event_assignment_resolution variant against fixtures committed under
tests/data/saved_outputs/event_assignment_variants/<variant>/. This script (re)creates those fixtures from a
real pipeline run, so it must NOT be run blindly - inspect the produced files and commit them only once they
look right.

Prerequisite: a stage-8 checkpoint must exist at
    tests/test_results/TestTransportProcesses/temp/mol_system_10.dump
Run the integration tests once through test_10create_layered_description4aquaduct_networks with the
tearDownClass rmtree() disabled so the output directory (and that checkpoint) persists, then run:

    python -m transport_tools.tests.integration.generate_event_assignment_fixtures

The variant list and the set of golden artifacts are defined here and imported by the test, so the producer
and the consumer can never drift.
"""

import os
import json
import shutil

from transport_tools.libs.utils import set_paths_from_package_root
from transport_tools.libs.tools import load_checkpoint

# Manifest file (committed, small: just file names) recording the exact set of per-event trace_matching detail
# files each simulation produced. Written by the generator and asserted by the consumer, so the full output is
# validated by identity - which (event, supercluster) pairs matched - without committing the ~68 MB of file
# contents (the curated sample files above content-compare a representative few). The count is the list length.
MIXED_EVENTS_TRACE_MANIFEST_FILE = "trace_matching_file_manifest.json"


def list_trace_detail_files(out_path):
    """Sorted file names of the per-event trace_matching detail files produced per simulation, keyed by md_label."""
    manifest = {}
    analysis_root = os.path.join(out_path, "data", "trace_matching_analysis")
    if os.path.isdir(analysis_root):
        for md_label in sorted(os.listdir(analysis_root)):
            md_dir = os.path.join(analysis_root, md_label)
            if os.path.isdir(md_dir):
                manifest[md_label] = sorted(entry.name for entry in os.scandir(md_dir) if entry.is_file())
    return manifest

# Canonical set of non-default ambiguous-assignment-resolution variants covered by the integration test.
# exact_matching is intentionally absent: it stays the default chain of test_11assign_transport_events (it
# also needs the source MD trajectory, which the other variants do not).
EVENT_ASSIGNMENT_VARIANTS = (
    ("penetration_depth", {"ambiguous_event_assignment_resolution": "penetration_depth",
                           "perform_exact_matching_analysis": False}),
    ("penetration_span", {"ambiguous_event_assignment_resolution": "penetration_span",
                          "perform_exact_matching_analysis": False}),
    ("directionality", {"ambiguous_event_assignment_resolution": "directionality",
                        "perform_exact_matching_analysis": False}),
    ("trace_matching", {"ambiguous_event_assignment_resolution": "trace_matching",
                        "perform_exact_matching_analysis": False,
                        "perform_trace_matching_analysis": True}),
    ("assign2all", {"ambiguous_event_assignment_resolution": "assign2all",
                    "perform_exact_matching_analysis": False}),
)


def variant_artifacts(out_path):
    """
    (produced_file, fixture_relative_path) pairs that define a variant's golden set - the per-event summary and
    the supercluster events details written by assign_transport_events() + generate_super_cluster_summary().
    :param out_path: pipeline output directory
    :return: list of (absolute produced path, path relative to the variant fixture folder)
    """

    return [
        (os.path.join(out_path, "statistics", "3-initial_events_summary.txt"),
         "3-initial_events_summary.txt"),
        (os.path.join(out_path, "data", "super_clusters", "details", "initial_super_cluster_events_details.txt"),
         "initial_super_cluster_events_details.txt"),
    ]


def trace_analysis_dir(out_path):
    """Per-event trace_matching analysis output folder (only produced when perform_trace_matching_analysis)."""
    return os.path.join(out_path, "data", "trace_matching_analysis", "md1")


# Mixed-events trace_matching variants exercising the CAVER sampling stride: the mixed_events fixture feeds CAVER
# a trajectory thinned to 1000 snapshots while AQUA-DUCT sees the full 20000-frame window, so the real stride is
# 20. The two variants golden-compare trajectory-free trace matching with the frame->snapshot mapping in its two
# modes - exact alignment (in-between frames contribute no tunnel) and interpolation (each frame matches its
# nearest analysed snapshot). Consumed by
# TestTransportProcessesMixedEvents.test_12assign_transport_events_trace_matching.
MIXED_EVENTS_TRACE_VARIANTS = (
    ("trace_matching_exact", {"ambiguous_event_assignment_resolution": "trace_matching",
                              "perform_exact_matching_analysis": False,
                              "perform_trace_matching_analysis": True,
                              "caver_snapshot_stride": 20,
                              "interpolate_missing_snapshots4matching": False}),
    ("trace_matching_interpolated", {"ambiguous_event_assignment_resolution": "trace_matching",
                                     "perform_exact_matching_analysis": False,
                                     "perform_trace_matching_analysis": True,
                                     "caver_snapshot_stride": 20,
                                     "interpolate_missing_snapshots4matching": True}),
)

# A curated handful of per-event trace_matching detail files to golden-compare, instead of the whole analysis
# tree (mixed_events produces thousands of events per simulation - ~68 MB - which must not be committed). Chosen
# to span both simulations, both traced residue types (WAT + the U01 ligand), entry and release events, and
# several superclusters; each shows a clear exact-vs-interpolated difference, so they exercise the frame->
# snapshot mapping in both modes. Paths are relative to a variant's fixture folder (and, under data/, to the
# pipeline output). Each file must exist in both variants.
MIXED_EVENTS_TRACE_SAMPLE_FILES = (
    os.path.join("trace_matching_analysis", "e10s0_seed43_f371m1821", "wat_100_entry_sc1.txt"),
    os.path.join("trace_matching_analysis", "e10s0_seed43_f371m1821", "wat_100_release_sc2.txt"),
    os.path.join("trace_matching_analysis", "e10s0_seed43_f371m1821", "wat_102_entry_sc6.txt"),
    os.path.join("trace_matching_analysis", "e10s0_seed43_f371m1821", "wat_103_release_sc5.txt"),
    os.path.join("trace_matching_analysis", "e10s14_e4s14f222m327_f1997m1236", "u01_1_entry_sc1.txt"),
    os.path.join("trace_matching_analysis", "e10s14_e4s14f222m327_f1997m1236", "u01_1_release_sc1.txt"),
)


def generate_mixed_events(out_path=None, saved_data=None):
    """
    Regenerate the mixed_events trace_matching golden fixtures (interpolation off vs on) from the persisted
    stage-10 checkpoint of TestTransportProcessesMixedEvents. As with generate(), inspect the produced files
    and commit them only once they look right. Only the two small assignment-outcome artifacts and the curated
    sample detail files are kept - never the full per-event analysis tree.
    :param out_path: pipeline output directory (defaults to the mixed_events integration-test results folder)
    :param saved_data: golden-data root (defaults to tests/data/saved_outputs_mixed_events)
    """

    if out_path is None:
        out_path = set_paths_from_package_root("tests", "test_results", "TestTransportProcessesMixedEvents")
    if saved_data is None:
        saved_data = os.path.join(set_paths_from_package_root("tests", "data"), "saved_outputs_mixed_events")

    checkpoint = os.path.join(out_path, "temp", "mol_system_10.dump")
    if not os.path.exists(checkpoint):
        raise SystemExit(
            "Stage-10 checkpoint not found at {}.\nRun the mixed_events integration tests once through "
            "test_10create_layered_description4aquaduct_networks with the tearDownClass rmtree() disabled "
            "so the output persists, then re-run this generator.".format(checkpoint))

    for variant, param_overrides in MIXED_EVENTS_TRACE_VARIANTS:
        # the matching code appends per-event files without clearing, so wipe the analysis folder before each
        # variant; otherwise the previous variant's files would leak into this variant's manifest
        analysis_root = os.path.join(out_path, "data", "trace_matching_analysis")
        if os.path.exists(analysis_root):
            shutil.rmtree(analysis_root)

        # fresh checkpoint per variant so assignment mutations do not leak between variants
        mol_system = load_checkpoint(checkpoint)
        for key, value in param_overrides.items():
            mol_system.parameters[key] = value
        mol_system.assign_transport_events()
        mol_system.generate_super_cluster_summary(out_filename="3-initial_events_summary.txt")

        fixtures = os.path.join(saved_data, "event_assignment_variants", variant)
        os.makedirs(fixtures, exist_ok=True)
        for produced, relpath in variant_artifacts(out_path):
            shutil.copyfile(produced, os.path.join(fixtures, relpath))

        for relpath in MIXED_EVENTS_TRACE_SAMPLE_FILES:
            produced = os.path.join(out_path, "data", relpath)
            destination = os.path.join(fixtures, relpath)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(produced, destination)

        with open(os.path.join(fixtures, MIXED_EVENTS_TRACE_MANIFEST_FILE), "w") as manifest_stream:
            json.dump(list_trace_detail_files(out_path), manifest_stream, indent=2, sort_keys=True)

        print("wrote mixed_events fixtures for variant '{}' -> {}".format(variant, fixtures))


def generate(out_path=None, saved_data=None):
    """
    Regenerate the per-variant golden fixtures from the persisted stage-8 checkpoint.
    :param out_path: pipeline output directory (defaults to the integration-test results folder)
    :param saved_data: golden-data root holding event_assignment_variants/ (defaults to tests/data/saved_outputs)
    """

    if out_path is None:
        out_path = set_paths_from_package_root("tests", "test_results", "TestTransportProcesses")
    if saved_data is None:
        saved_data = os.path.join(set_paths_from_package_root("tests", "data"), "saved_outputs")

    checkpoint = os.path.join(out_path, "temp", "mol_system_10.dump")
    if not os.path.exists(checkpoint):
        raise SystemExit(
            "Stage-8 checkpoint not found at {}.\nRun the integration tests once through "
            "test_10create_layered_description4aquaduct_networks with the tearDownClass rmtree() disabled "
            "so the output persists, then re-run this generator.".format(checkpoint))

    for variant, param_overrides in EVENT_ASSIGNMENT_VARIANTS:
        # fresh checkpoint per variant so assignment mutations do not leak between variants
        mol_system = load_checkpoint(checkpoint)
        for key, value in param_overrides.items():
            mol_system.parameters[key] = value
        mol_system.assign_transport_events()
        mol_system.generate_super_cluster_summary(out_filename="3-initial_events_summary.txt")

        fixtures = os.path.join(saved_data, "event_assignment_variants", variant)
        os.makedirs(fixtures, exist_ok=True)
        for produced, relpath in variant_artifacts(out_path):
            shutil.copyfile(produced, os.path.join(fixtures, relpath))

        if param_overrides.get("perform_trace_matching_analysis"):
            destination = os.path.join(fixtures, "trace_matching_analysis", "md1")
            if os.path.exists(destination):
                shutil.rmtree(destination)
            shutil.copytree(trace_analysis_dir(out_path), destination)

        print("wrote fixtures for variant '{}' -> {}".format(variant, fixtures))


if __name__ == "__main__":
    import sys
    # "default" (TestTransportProcesses variants), "mixed_events" (the stride/interpolation variants),
    # or "all". Each needs its own stage-10 checkpoint to be present.
    target = sys.argv[1] if len(sys.argv) > 1 else "default"
    if target not in ("default", "mixed_events", "all"):
        raise SystemExit("usage: python -m transport_tools.tests.integration."
                         "generate_event_assignment_fixtures [default|mixed_events|all]")
    if target in ("default", "all"):
        generate()
    if target in ("mixed_events", "all"):
        generate_mixed_events()
