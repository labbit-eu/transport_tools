#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules
# and ligand transport through them.
# Copyright (C) 2021  Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej
# Surpeta, Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal  <janbre@amu.edu.pl>
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

__version__ = '0.9.0'
__author__ = 'Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta, ' \
             'Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal'
__mail__ = 'janbre@amu.edu.pl'

import unittest
import os
import transport_tools.tests.units.test_utils as TT_test_utils
import transport_tools.tests.units.test_geometry as TT_test_geometry
import transport_tools.tests.units.test_networks as TT_test_networks
from transport_tools.tests.integration.test_tools import set_paths

if os.path.exists(set_paths("tests", "data")):
    import transport_tools.tests.integration.test_tools as TT_test_tools
    import transport_tools.tests.integration.test_protein_files as TT_test_protein_files
    import transport_tools.tests.integration.test_config as TT_test_config
    import transport_tools.tests.integration.test_geometry as TT_test_geometry2
    import transport_tools.tests.integration.test_networks as TT_test_networks2
else:
    print("No test data available, running unit tests only.")

tests_suite = unittest.defaultTestLoader.loadTestsFromModule(TT_test_utils)
tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_geometry))
tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_networks))
if os.path.exists(set_paths("tests", "data")):
    tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_protein_files))
    tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_config))
    tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_geometry2))
    tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_networks2))
    tests_suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(TT_test_tools))

print("Number of tests loaded:", tests_suite.countTestCases())
results = unittest.TextTestRunner(verbosity=2).run(tests_suite)

for error in results.errors:
    for e in error:
        print(e)
    print("*"+"-"*78+"*")

for failure in results.failures:
    for f in failure:
        print(f)
    print("*"+"-"*78+"*")
