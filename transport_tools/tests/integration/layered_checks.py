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

__version__ = '0.9.8'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

"""
Strict checks of layered entities (LayeredPathSet) shared by the integration tests: exact equality of the
layering results, and the invariant that every layered network shares a single parameters dict among its
entities (pathsets layered in pool workers travel without parameters, see LayeredPathSet.detach_parameters()).
"""

import os
import pickle
from typing import Dict

import numpy as np

LAYERED_DUMP_SUFFIX = "_layered_paths.dump"


def load_layered_dump(dump_file: str) -> Dict:
    with open(dump_file, "rb") as in_stream:
        return pickle.load(in_stream)


def assert_same_layered_entities(test_case, expected: Dict, actual: Dict, label: str = ""):
    """
    Exact comparison of two {entity_id: LayeredPathSet} dicts: same ids in the same order, same attributes
    in the same order, and bit-identical data. Only the `parameters` attribute is not compared.
    """

    test_case.assertEqual(list(expected.keys()), list(actual.keys()), label)
    for entity_id, expected_pathset in expected.items():
        expected_state = vars(expected_pathset)
        actual_state = vars(actual[entity_id])
        test_case.assertEqual(list(expected_state.keys()), list(actual_state.keys()), label)
        for key, value in expected_state.items():
            if key == "parameters":
                continue
            other = actual_state[key]
            msg = "{} entity {}, attribute '{}'".format(label, entity_id, key)
            if isinstance(value, np.ndarray):
                test_case.assertEqual(value.dtype, other.dtype, msg)
                test_case.assertTrue(np.array_equal(value, other), msg)
            elif key == "node_paths":
                test_case.assertEqual(len(value), len(other), msg)
                for path, other_path in zip(value, other):
                    test_case.assertTrue(np.array_equal(path, other_path), msg)
            else:
                test_case.assertEqual(value, other, msg)


def assert_single_shared_parameters(test_case, entities: Dict, md_label: str, label: str = ""):
    """
    All entities of one layered network refer to one parameters dict, that of the network (md_label).
    """

    test_case.assertGreater(len(entities), 0, label)
    test_case.assertEqual(1, len({id(entity.parameters) for entity in entities.values()}),
                          "{}: entities hold private parameters copies".format(label))
    parameters = next(iter(entities.values())).parameters
    test_case.assertIsInstance(parameters, dict, label)
    test_case.assertEqual(md_label, parameters["md_label"], label)


def assert_layered_dumps_share_parameters(test_case, folder: str):
    """
    Every saved layered network in the folder stores a single parameters dict shared by all its entities.
    """

    dump_files = sorted(f for f in os.listdir(folder) if f.endswith(LAYERED_DUMP_SUFFIX))
    test_case.assertGreater(len(dump_files), 0, folder)
    for dump_file in dump_files:
        assert_single_shared_parameters(test_case, load_layered_dump(os.path.join(folder, dump_file)),
                                        dump_file[:-len(LAYERED_DUMP_SUFFIX)], dump_file)
