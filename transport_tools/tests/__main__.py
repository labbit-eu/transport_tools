#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules
# and ligand transport through them.
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
import transport_tools.tests.units.test_utils as TT_test_utils
import transport_tools.tests.units.test_geometry as TT_test_geometry
import transport_tools.tests.units.test_networks as TT_test_networks
import transport_tools.tests.units.test_tunnel_profile as TT_test_tunnel_profile
import transport_tools.tests.units.test_outlier_events as TT_test_outlier_events
import transport_tools.tests.units.test_event_assigner as TT_test_event_assigner
import transport_tools.tests.units.test_supercluster as TT_test_supercluster
import transport_tools.tests.units.test_msms as TT_test_msms
import transport_tools.tests.units.test_config as TT_test_config
import transport_tools.tests.units.test_ui as TT_test_ui
from transport_tools.libs.utils import set_paths_from_package_root

# Create unit tests suite
unit_tests_suite = unittest.defaultTestLoader.loadTestsFromModule(TT_test_utils)
unit_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_geometry))
unit_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_networks))
unit_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_tunnel_profile))
unit_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_outlier_events))
unit_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_event_assigner))
unit_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_supercluster))
unit_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_msms))
unit_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_config))
unit_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_ui))

# Run unit tests first
print("=" * 80)
print("Running Unit Tests")
print("=" * 80)
print(f"Number of unit tests loaded: {unit_tests_suite.countTestCases()}")
unit_results = unittest.TextTestRunner(verbosity=2).run(unit_tests_suite)

# Perform integration tests if data present
if os.path.exists(set_paths_from_package_root("tests", "data")):
    integration_tests_suite = unittest.TestSuite()
    import transport_tools.tests.integration.test_tools as TT_test_tools
    import transport_tools.tests.integration.test_tools_mixed_events as TT_test_tools_mixed_events
    import transport_tools.tests.integration.test_protein_files as TT_test_protein_files
    import transport_tools.tests.integration.test_config as TT_test_config
    import transport_tools.tests.integration.test_geometry as TT_test_geometry2
    import transport_tools.tests.integration.test_networks as TT_test_networks2

    integration_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_protein_files))
    integration_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_config))
    integration_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_geometry2))
    integration_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_networks2))
    integration_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_tools))
    integration_tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_tools_mixed_events))

    print("\n" + "=" * 80)
    print("Running Integration Tests")
    print("=" * 80)
    print(f"Number of integration tests loaded: {integration_tests_suite.countTestCases()}")
    integration_results = unittest.TextTestRunner(verbosity=2).run(integration_tests_suite)

    # Combine results
    all_errors = unit_results.errors + integration_results.errors
    all_failures = unit_results.failures + integration_results.failures
else:
    print("\nNo test data available, skipping integration tests.")
    all_errors = unit_results.errors
    all_failures = unit_results.failures

# Print summary
print("\n" + "=" * 80)
print("Test Summary")
print("=" * 80)

for error in all_errors:
    for e in error:
        print(e)
    print("*"+"-"*78+"*")

for failure in all_failures:
    for f in failure:
        print(f)
    print("*"+"-"*78+"*")

# Cleanup test_results folder if empty (all tests passed)
test_results_path = set_paths_from_package_root("tests", "test_results")
if os.path.exists(test_results_path):
    if not os.listdir(test_results_path):
        # Directory is empty, safe to remove
        try:
            os.rmdir(test_results_path)
            print("\nCleaned up empty test_results folder (all tests passed)")
        except OSError as e:
            print(f"\nWarning: Could not remove test_results folder: {e}")
    else:
        # Directory is not empty, keep it for investigation
        print(f"\ntest_results folder not empty - keeping for investigation: {test_results_path}")
