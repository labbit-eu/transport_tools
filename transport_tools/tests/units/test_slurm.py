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

"""Unit tests for the SlurmShardStage framework and the DistanceShardStage stage-4 adapter.
These exercise pure-Python logic (folder derivation, num_shards resolution, fingerprint
content, stale-result detection); they do not submit any SLURM jobs."""

__version__ = '0.9.7'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import json
import os
import tempfile
import unittest

from transport_tools.libs.slurm import (
    DistanceShardStage,
    SlurmShardStage,
    _missing_shards,
    derive_num_shards,
    shard_result_path,
    submitit_available,
)


class TestDeriveNumShards(unittest.TestCase):
    """Granularity heuristic with optional array-parallelism cap."""

    def test_returns_at_least_one_shard(self):
        self.assertEqual(1, derive_num_shards(0))
        self.assertEqual(1, derive_num_shards(1))

    def test_aims_for_target_per_shard(self):
        self.assertEqual(5, derive_num_shards(100, target_per_shard=20))

    def test_array_parallelism_caps_count(self):
        # would derive ~50 shards at default target; cap to 10
        self.assertEqual(10, derive_num_shards(50 * 20000, array_parallelism=10))

    def test_max_auto_shards_caps_count_when_no_parallelism(self):
        # very large n_jobs without array_parallelism is capped to max_auto_shards
        self.assertEqual(7, derive_num_shards(1_000_000, max_auto_shards=7))


class TestSubmititAvailable(unittest.TestCase):
    """submitit_available returns a bool whether or not the optional dep is installed."""

    def test_returns_bool(self):
        self.assertIsInstance(submitit_available(), bool)


class TestDistanceShardStageFolderLayout(unittest.TestCase):
    """DistanceShardStage publishes the stage-4 folder under slurm_root_folder."""

    def test_folder_name_uses_stage_number(self):
        self.assertEqual("stage04_distances", DistanceShardStage.folder_name)

    def test_get_slurm_folder_joins_root_and_folder_name(self):
        path = DistanceShardStage.get_slurm_folder({"slurm_root_folder": "/scratch/run/_slurm"})
        self.assertEqual("/scratch/run/_slurm/stage04_distances", path)

    def test_shard_result_path_includes_shard_id(self):
        stage = DistanceShardStage(num_clusters=10)
        path = stage.shard_result_path("/tmp/folder", 7)
        # %05d formatting must yield a zero-padded shard id so lexicographic sort matches
        # numeric order; this guarantees that re-running an interrupted job sees the same
        # filenames it wrote before
        self.assertEqual("/tmp/folder/shard_result_00007.npy", path)
        # consistent with module-level helper used by external callers
        self.assertEqual(path, shard_result_path("/tmp/folder", 7))


class TestDistanceShardStageNumShards(unittest.TestCase):
    """num_shards() picks the user-requested count when set, otherwise auto-derives."""

    def _params(self, slurm_num_shards=None, slurm_array_parallelism=None):
        return {
            "slurm_num_shards": slurm_num_shards,
            "slurm_array_parallelism": slurm_array_parallelism,
        }

    def test_zero_clusters_yields_single_shard(self):
        stage = DistanceShardStage(num_clusters=0)
        self.assertEqual(1, stage.num_shards(self._params()))

    def test_two_clusters_yields_single_shard(self):
        # n_jobs = 1 pairwise comparison; can't split below 1
        stage = DistanceShardStage(num_clusters=2)
        self.assertEqual(1, stage.num_shards(self._params()))

    def test_user_requested_count_is_capped_at_n_jobs(self):
        # 3 clusters => 3 pairwise comparisons; asking for 100 shards is silly
        stage = DistanceShardStage(num_clusters=3)
        self.assertEqual(3, stage.num_shards(self._params(slurm_num_shards=100)))

    def test_auto_derives_from_n_jobs_when_unset(self):
        # large cluster count: 200 clusters => 200*199/2 = 19900 pairs
        # derive_num_shards aims for ~20000 pairs/shard => returns 1
        stage = DistanceShardStage(num_clusters=200)
        self.assertEqual(1, stage.num_shards(self._params()))


class TestDistanceShardStageFingerprint(unittest.TestCase):
    """Fingerprint identifies the parameters this stage's shard results were computed for."""

    def _params(self, **overrides):
        params = {
            "clustering_cutoff": 2.0,
            "calculate_exact_path_distances": True,
        }
        params.update(overrides)
        return params

    def test_includes_num_clusters_and_num_shards(self):
        stage = DistanceShardStage(num_clusters=42)
        info = stage.fingerprint(self._params(), num_shards=7)
        self.assertEqual(42, info["num_clusters"])
        self.assertEqual(7, info["num_shards"])

    def test_includes_distance_affecting_parameters(self):
        stage = DistanceShardStage(num_clusters=10)
        info = stage.fingerprint(self._params(clustering_cutoff=3.5,
                                              calculate_exact_path_distances=False), num_shards=4)
        self.assertEqual(3.5, info["clustering_cutoff"])
        self.assertFalse(info["calculate_exact_path_distances"])

    def test_includes_tt_version(self):
        # bumping the TT version invalidates old shard result files automatically
        from transport_tools.libs.slurm import __version__
        stage = DistanceShardStage(num_clusters=10)
        info = stage.fingerprint(self._params(), num_shards=4)
        self.assertEqual(__version__, info["tt_version"])

    def test_fingerprint_changes_when_cutoff_changes(self):
        stage = DistanceShardStage(num_clusters=10)
        fp1 = stage.fingerprint(self._params(clustering_cutoff=2.0), num_shards=4)
        fp2 = stage.fingerprint(self._params(clustering_cutoff=2.5), num_shards=4)
        self.assertNotEqual(fp1, fp2)


class TestMissingShards(unittest.TestCase):
    """_missing_shards() reuses fresh results and discards stale ones based on fingerprint."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))
        self.params = {
            "clustering_cutoff": 2.0,
            "calculate_exact_path_distances": True,
        }
        self.stage = DistanceShardStage(num_clusters=10)

    def _touch_shard(self, shard_id):
        path = self.stage.shard_result_path(self._tmp, shard_id)
        with open(path, "wb") as out:
            out.write(b"")

    def test_no_existing_results_returns_all_shards_missing(self):
        missing = _missing_shards(self.stage, self._tmp, num_shards=3, parameters=self.params)
        self.assertEqual([0, 1, 2], missing)

    def test_existing_fresh_results_are_reused(self):
        # write fingerprint info matching what the stage would produce for these params
        with open(os.path.join(self._tmp, "shards_info.json"), "w") as out:
            json.dump(self.stage.fingerprint(self.params, num_shards=3), out)
        self._touch_shard(0)
        self._touch_shard(2)
        missing = _missing_shards(self.stage, self._tmp, num_shards=3, parameters=self.params)
        self.assertEqual([1], missing)

    def test_stale_results_are_discarded_when_fingerprint_mismatch(self):
        # persist an old fingerprint for a different cutoff
        old_fp = self.stage.fingerprint({"clustering_cutoff": 1.0,
                                         "calculate_exact_path_distances": True}, num_shards=3)
        with open(os.path.join(self._tmp, "shards_info.json"), "w") as out:
            json.dump(old_fp, out)
        self._touch_shard(0)
        self._touch_shard(2)
        missing = _missing_shards(self.stage, self._tmp, num_shards=3, parameters=self.params)
        # all three shards must be rerun; the existing files were deleted
        self.assertEqual([0, 1, 2], missing)
        self.assertFalse(os.path.exists(self.stage.shard_result_path(self._tmp, 0)))
        self.assertFalse(os.path.exists(self.stage.shard_result_path(self._tmp, 2)))


class _DummyStage(SlurmShardStage):
    """A minimal SlurmShardStage subclass that touches only the interface, used to exercise
    the abstract-class contract and the default fingerprint/get_slurm_folder behaviours."""

    folder_name = "stage99_dummy"
    stage_name = "dummy"
    job_name = "tt_dummy"
    fingerprint_keys = ("some_param",)

    def num_shards(self, parameters):
        return 3

    def shard_result_path(self, slurm_folder, shard_id):
        return os.path.join(slurm_folder, "dummy_{}.bin".format(shard_id))

    def run_shard(self, config_file, num_shards, shard_id):
        return shard_id


class TestSlurmShardStageDefaults(unittest.TestCase):
    """Default behaviours inherited from SlurmShardStage (get_slurm_folder, fingerprint)."""

    def test_get_slurm_folder_uses_classmethod(self):
        # accessible without instantiating; required so SLURM-shard workers can derive the
        # result path from just the parameters dict (no runtime state)
        path = _DummyStage.get_slurm_folder({"slurm_root_folder": "/root"})
        self.assertEqual("/root/stage99_dummy", path)

    def test_fingerprint_includes_listed_parameter_keys(self):
        stage = _DummyStage()
        info = stage.fingerprint({"some_param": "value-x"}, num_shards=4)
        self.assertEqual("value-x", info["some_param"])
        self.assertEqual(4, info["num_shards"])

    def test_cannot_instantiate_without_overriding_abstract_methods(self):
        # SlurmShardStage itself is abstract; attempting to instantiate without overriding
        # the @abstractmethod hooks must raise TypeError
        with self.assertRaises(TypeError):
            SlurmShardStage()  # type: ignore[abstract]


if __name__ == "__main__":
    unittest.main()
