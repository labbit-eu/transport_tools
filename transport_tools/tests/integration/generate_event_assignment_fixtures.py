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
import shutil

from transport_tools.libs.utils import set_paths_from_package_root
from transport_tools.libs.tools import load_checkpoint

# Canonical set of non-default ambiguous-assignment-resolution variants covered by the integration test.
# exact_matching is intentionally absent: it stays the default chain of test_11assign_transport_events (it
# also needs the source MD trajectory, which the other variants do not).
EVENT_ASSIGNMENT_VARIANTS = (
    ("penetration_depth", {"ambiguous_event_assignment_resolution": "penetration_depth",
                           "perform_exact_matching_analysis": False}),
    ("penetration_span", {"ambiguous_event_assignment_resolution": "penetration_span",
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
    generate()
