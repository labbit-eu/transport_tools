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
Layered pathsets share the parameters of the network they belong to. Pathsets that travel on their own
(pool results, SLURM shard pickles) carry none, so that unpickling them one by one does not give every
entity a private copy of the whole configuration (the stage-8 aggregation OOM).
"""

import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np


def _parameters(md_label: str = "md1") -> dict:
    # comparative_groups_definition stands in for the large expanded entries of a real configuration
    return {"sp_radius": 0.5, "layer_thickness": 1.5, "random_seed": 4, "pickle_protocol": 4,
            "comparative_groups_definition": {"g1": {"sim{:04d}".format(i) for i in range(1500)},
                                              "g2": {"sim{:04d}".format(i) for i in range(1500, 3000)}},
            "md_label": md_label}


def _make_pathset(entity_label: str, parameters: dict, offset: float = 0.):
    from transport_tools.libs.geometry import LayeredPathSet

    pathset = LayeredPathSet(entity_label, parameters["md_label"], parameters,
                             starting_point_coords=np.array([0., 0., 0.]))
    pathset.node_labels.extend(["0_1", "1_1"])
    pathset.nodes_data = np.concatenate((pathset.nodes_data,
                                         np.array([[1. + offset, 0., 0., 0, 0, 1.2, 0.1],
                                                   [2. + offset, 1., 0., 1, 1, 1.4, 0.2]])))
    pathset.node_paths.append(np.array(["SP", "0_1", "1_1"]))
    pathset.characteristics = (0.5 + offset, 3)
    return pathset


def _travel(pathset):
    """Pickle a pathset on its own and unpickle it, as a pool result or a shard pickle entry does."""
    return pickle.loads(pickle.dumps(pathset, protocol=4))


def _state_without_parameters(pathset) -> dict:
    return {key: value for key, value in vars(pathset).items() if key != "parameters"}


class TestDetachParameters(unittest.TestCase):
    def test_detach_drops_the_reference(self):
        parameters = _parameters()
        pathset = _make_pathset("Cluster_1", parameters)
        attributes = list(vars(pathset).keys())

        pathset.detach_parameters()

        self.assertIsNone(pathset.parameters)
        self.assertEqual(attributes, list(vars(pathset).keys()), "the attribute layout must be kept")
        self.assertIn("md_label", parameters, "the shared dict itself must not be modified")

    def test_detach_is_idempotent(self):
        pathset = _make_pathset("Cluster_1", _parameters())
        pathset.detach_parameters()
        pathset.detach_parameters()
        self.assertIsNone(pathset.parameters)

    def test_detached_pathset_travels_without_the_configuration(self):
        pathset = _make_pathset("Cluster_1", _parameters())
        attached_size = len(pickle.dumps(pathset, protocol=4))
        pathset.detach_parameters()
        detached = _travel(pathset)

        self.assertIsNone(detached.parameters)
        self.assertLess(len(pickle.dumps(pathset, protocol=4)) * 10, attached_size)
        self.assertTrue(detached.is_same(pathset))


class TestLayeringWorkersDetach(unittest.TestCase):
    """The pool workers of stages 3 and 8 return the result of the layering unchanged except for the
    parameters reference."""

    class _Layerable:
        def __init__(self, result):
            self.result = result

        def create_layered_cluster(self):
            return self.result

        def create_layered_event(self):
            return self.result

    def _check_worker(self, worker, entity_id):
        parameters = _parameters()
        pathset = _make_pathset("Cluster_1", parameters)
        expected_state = {key: value for key, value in _state_without_parameters(pathset).items()}

        returned_id, md_label, returned = worker(self._Layerable((entity_id, "md1", pathset)))

        self.assertEqual(entity_id, returned_id)
        self.assertEqual("md1", md_label)
        self.assertIs(pathset, returned)
        self.assertIsNone(returned.parameters)
        self.assertEqual(list(expected_state.keys()), list(_state_without_parameters(returned).keys()))
        self.assertIn("comparative_groups_definition", parameters, "the shared dict itself must not be modified")

    def test_cluster_worker(self):
        from transport_tools.libs.tools import create_layered_cluster_worker

        self._check_worker(create_layered_cluster_worker, 7)

    def test_event_worker(self):
        from transport_tools.libs.tools import create_layered_event_worker

        self._check_worker(create_layered_event_worker, "1_release")


class TestAddLayeredEntityAttachesNetworkParameters(unittest.TestCase):
    def setUp(self):
        from transport_tools.libs.networks import Network

        self.tmp_dir = tempfile.mkdtemp()
        self.network = Network(_parameters(""), "md1")
        self.network.layered_dump_file = os.path.join(self.tmp_dir, "layered", "md1_layered.dump")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_attaches_to_detached_pathset(self):
        pathset = _make_pathset("Cluster_1", _parameters())
        pathset.detach_parameters()
        self.network.add_layered_entity(1, pathset)
        self.assertIs(self.network.parameters, pathset.parameters)
        self.assertIs(pathset, self.network.layered_entities[1])

    def test_replaces_private_copy(self):
        # legacy shard results and in-process callers arrive with a parameters dict of their own
        pathset = _travel(_make_pathset("Cluster_1", _parameters()))
        self.network.add_layered_entity(1, pathset)
        self.assertIs(self.network.parameters, pathset.parameters)

    def _fill_with_travelled_pathsets(self, detach: bool) -> dict:
        originals = dict()
        for entity_id in range(1, 6):
            pathset = _make_pathset("Cluster_{}".format(entity_id), self.network.parameters, offset=entity_id)
            originals[entity_id] = pathset
            if detach:
                pathset.detach_parameters()
            self.network.add_layered_entity(entity_id, _travel(pathset))
        return originals

    def _assert_same_layering(self, originals: dict, entities: dict):
        self.assertEqual(list(originals.keys()), list(entities.keys()))
        for entity_id, original in originals.items():
            loaded = entities[entity_id]
            self.assertEqual(list(vars(original).keys()), list(vars(loaded).keys()))
            for key, value in _state_without_parameters(original).items():
                other = getattr(loaded, key)
                if isinstance(value, np.ndarray):
                    self.assertTrue(np.array_equal(value, other), key)
                elif key == "node_paths":
                    self.assertEqual(len(value), len(other))
                    for path, other_path in zip(value, other):
                        self.assertTrue(np.array_equal(path, other_path))
                else:
                    self.assertEqual(value, other, key)

    def test_saved_network_holds_a_single_configuration(self):
        for detach in (True, False):
            with self.subTest(detached_in_worker=detach):
                self.network.layered_entities = dict()
                originals = self._fill_with_travelled_pathsets(detach)
                self.network.save_layered_network()

                with open(self.network.layered_dump_file, "rb") as in_stream:
                    raw_entities = pickle.load(in_stream)
                self.assertEqual(1, len({id(entity.parameters) for entity in raw_entities.values()}))
                self.assertEqual(self.network.parameters, next(iter(raw_entities.values())).parameters)
                self._assert_same_layering(originals, raw_entities)

                from transport_tools.libs.networks import Network

                loader = Network(_parameters(""), "md1")
                loader.layered_dump_file = self.network.layered_dump_file
                loader.load_layered_network()
                self.assertEqual(1, len({id(entity.parameters) for entity in loader.layered_entities.values()}))
                self.assertEqual(loader.parameters, next(iter(loader.layered_entities.values())).parameters)
                self._assert_same_layering(originals, loader.layered_entities)

    def test_saved_network_is_smaller_than_with_private_copies(self):
        self._fill_with_travelled_pathsets(detach=True)
        shared_size = len(pickle.dumps(self.network.layered_entities))
        private = {entity_id: _travel(pathset) for entity_id, pathset in self.network.layered_entities.items()}
        self.assertEqual(len(private), len({id(entity.parameters) for entity in private.values()}))
        self.assertLess(shared_size * 3, len(pickle.dumps(private)))


if __name__ == "__main__":
    unittest.main()
