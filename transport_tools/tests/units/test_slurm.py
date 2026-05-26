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
    AquaductLayeringShardStage,
    DistanceShardStage,
    EventAssignmentShardStage,
    SlurmShardStage,
    TunnelLayeringShardStage,
    _missing_shards,
    derive_num_shards,
    submitit_available,
)


class TestDeriveNumShards(unittest.TestCase):
    """Granularity heuristic with optional array-parallelism cap. target_per_shard is a
    required keyword-only arg so every caller picks the value that matches its workload -
    there is no stage-4-flavoured default lurking in the framework."""

    def test_returns_at_least_one_shard(self):
        self.assertEqual(1, derive_num_shards(0, target_per_shard=20))
        self.assertEqual(1, derive_num_shards(1, target_per_shard=20))

    def test_aims_for_target_per_shard(self):
        self.assertEqual(5, derive_num_shards(100, target_per_shard=20))

    def test_array_parallelism_caps_count(self):
        # would derive ~50 shards at target=20000; cap to 10
        self.assertEqual(10, derive_num_shards(50 * 20000, target_per_shard=20000,
                                                array_parallelism=10))

    def test_max_auto_shards_caps_count_when_no_parallelism(self):
        # very large n_jobs without array_parallelism is capped to max_auto_shards
        self.assertEqual(7, derive_num_shards(1_000_000, target_per_shard=1,
                                               max_auto_shards=7))

    def test_target_per_shard_is_keyword_only(self):
        # passing positionally raises - the parameter is keyword-only so adding new optional
        # args later doesn't shift positions or silently change behaviour for existing callers
        with self.assertRaises(TypeError):
            derive_num_shards(100, 20)  # type: ignore[misc]


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
        # %05d formatting yields a zero-padded shard id so lexicographic sort matches numeric
        # order; this guarantees that re-running an interrupted job sees the same filenames
        # it wrote before. Callable as a classmethod without instantiating the stage so
        # SLURM-shard workers can derive the output path from just (params, shard_id).
        self.assertEqual("/tmp/folder/shard_result_00007.npy",
                         DistanceShardStage.shard_result_path("/tmp/folder", 7))


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
        with open(os.path.join(self._tmp, "partial_shards_info.json"), "w") as out:
            json.dump(self.stage.fingerprint(self.params, num_shards=3), out)
        self._touch_shard(0)
        self._touch_shard(2)
        missing = _missing_shards(self.stage, self._tmp, num_shards=3, parameters=self.params)
        self.assertEqual([1], missing)

    def test_stale_results_are_discarded_when_fingerprint_mismatch(self):
        # persist an old fingerprint for a different cutoff
        old_fp = self.stage.fingerprint({"clustering_cutoff": 1.0,
                                         "calculate_exact_path_distances": True}, num_shards=3)
        with open(os.path.join(self._tmp, "partial_shards_info.json"), "w") as out:
            json.dump(old_fp, out)
        self._touch_shard(0)
        self._touch_shard(2)
        missing = _missing_shards(self.stage, self._tmp, num_shards=3, parameters=self.params)
        # all three shards must be rerun; the existing files were deleted
        self.assertEqual([0, 1, 2], missing)
        self.assertFalse(os.path.exists(self.stage.shard_result_path(self._tmp, 0)))
        self.assertFalse(os.path.exists(self.stage.shard_result_path(self._tmp, 2)))


class TestShardSideEffectsArchive(unittest.TestCase):
    """write_side_effects_archive + restore_side_effects_from_archive: round-trip and the
    failure modes that drive `_missing_shards()` resume decisions."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))
        # write side-effects into a separate sub-tree so we can wipe it without touching
        # the slurm folder (mirrors the layout in the integration tests where the visualisation
        # folder lives elsewhere)
        self._target = os.path.join(self._tmp, "target")
        os.makedirs(self._target, exist_ok=True)
        self._slurm = os.path.join(self._tmp, "slurm")
        os.makedirs(self._slurm, exist_ok=True)

    def _make_file(self, relpath: str, content: bytes) -> str:
        path = os.path.join(self._target, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as out:
            out.write(content)
        return path

    def test_round_trip_restores_byte_identical_content(self):
        # write a couple of side-effect files, archive them, wipe, restore - bytes must match
        p1 = self._make_file("Cluster_1.py", b"# pymol script\ncmd.show_as('wire', 'all')\n")
        p2 = self._make_file("nodes/cls001_layer0_pts.pdb", b"ATOM   1  C  ...\nEND\n")
        TunnelLayeringShardStage.write_side_effects_archive(self._slurm, 7, [p1, p2])
        # archive must exist
        self.assertTrue(os.path.exists(
            TunnelLayeringShardStage.shard_side_effects_archive_path(self._slurm, 7)))
        # wipe and restore
        __import__("shutil").rmtree(self._target)
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(self._slurm, 7, [p1, p2])
        self.assertTrue(ok)
        with open(p1, "rb") as f:
            self.assertEqual(b"# pymol script\ncmd.show_as('wire', 'all')\n", f.read())
        with open(p2, "rb") as f:
            self.assertEqual(b"ATOM   1  C  ...\nEND\n", f.read())

    def test_restore_returns_false_when_archive_missing(self):
        # never write the archive; restore must report failure rather than raise
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(
            self._slurm, 0, ["/nope"])
        self.assertFalse(ok)

    def test_restore_returns_false_when_archive_is_corrupt(self):
        # write garbage where the archive should be; restore must report failure
        archive_path = TunnelLayeringShardStage.shard_side_effects_archive_path(self._slurm, 0)
        with open(archive_path, "wb") as out:
            out.write(b"not a tarball")
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(
            self._slurm, 0, ["/nope"])
        self.assertFalse(ok)

    def test_restore_returns_false_when_expected_file_not_in_archive(self):
        # archive contains file A but caller asks to restore B - restore reports failure
        # without writing anything to B's path (so the caller can fall back to resubmit
        # without confusing partial state)
        p_in = self._make_file("Cluster_1.py", b"in-archive")
        TunnelLayeringShardStage.write_side_effects_archive(self._slurm, 0, [p_in])
        p_out = os.path.join(self._target, "Cluster_9_not_in_archive.py")
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(
            self._slurm, 0, [p_out])
        self.assertFalse(ok)
        self.assertFalse(os.path.exists(p_out))

    def test_empty_archive_round_trips(self):
        # stages with no side-effects (e.g. visualize_layered_clusters=False) still write
        # an archive so the manifest semantics stay uniform; restore with empty list must
        # succeed trivially
        TunnelLayeringShardStage.write_side_effects_archive(self._slurm, 0, [])
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(self._slurm, 0, [])
        self.assertTrue(ok)


class TestMissingShardsSideEffectRecovery(unittest.TestCase):
    """`_missing_shards()` resume path for stages with `has_side_effects=True`: restore
    from the per-shard tar.gz backup instead of resubmitting when output files were wiped
    between runs; fall back to resubmit only when the backup is also unrecoverable."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))
        self._target = os.path.join(self._tmp, "target")
        os.makedirs(self._target, exist_ok=True)
        self._slurm = os.path.join(self._tmp, "slurm")
        os.makedirs(self._slurm, exist_ok=True)
        # tunnel layering with one md_label and two clusters - small enough to inspect each
        # shard's outcome individually
        self.params = {
            "layer_thickness": 1.5,
            "min_tunnel_radius4clustering": 0.75,
            "min_tunnel_length4clustering": 5.0,
            "max_tunnel_curvature4clustering": 2.0,
            "random_seed": 4,
            "clustering_max_num_rep_frag": 3,
            "use_cluster_spread": True,
            "visualize_layered_clusters": True,
            "layered_caver_vis_path": self._target,
            "slurm_root_folder": self._tmp,
        }
        self.stage = TunnelLayeringShardStage([("md1", 1), ("md1", 2)])

    def _bootstrap_shard(self, shard_id: int, side_effect_paths) -> None:
        # mimic what compute_tunnel_layering_shard() writes on a successful run:
        # the result pickle, the manifest, the side-effects archive, all atomic
        with open(self.stage.shard_result_path(self._slurm, shard_id), "wb") as f:
            f.write(b"")
        for path in side_effect_paths:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as out:
                out.write(b"side-effect-content-" + os.path.basename(path).encode() + b"\n")
        TunnelLayeringShardStage.write_side_effects_archive(
            self._slurm, shard_id, side_effect_paths)
        TunnelLayeringShardStage.write_shard_manifest(
            self._slurm, shard_id, side_effect_paths)
        # match the fingerprint so the resume check sees a fresh cache
        with open(os.path.join(self._slurm, "partial_shards_info.json"), "w") as out:
            json.dump(self.stage.fingerprint(self.params, num_shards=1), out)

    def _side_effect_for(self, md_label, cls_id):
        return os.path.join(self._target, md_label, "Cluster_{}.py".format(cls_id))

    def test_side_effects_present_skips_resubmission(self):
        # both side-effect files exist on disk - the shard is fully usable, missing list empty
        files = [self._side_effect_for("md1", 1), self._side_effect_for("md1", 2)]
        self._bootstrap_shard(0, files)
        missing = _missing_shards(self.stage, self._slurm, num_shards=1, parameters=self.params)
        self.assertEqual([], missing)

    def test_side_effects_wiped_restored_from_archive_skips_resubmission(self):
        # simulate the user wiping the visualisation folder between runs; the archive lives
        # in the slurm folder so the helper restores from it instead of resubmitting
        files = [self._side_effect_for("md1", 1), self._side_effect_for("md1", 2)]
        self._bootstrap_shard(0, files)
        __import__("shutil").rmtree(self._target)
        missing = _missing_shards(self.stage, self._slurm, num_shards=1, parameters=self.params)
        self.assertEqual([], missing)
        for path in files:
            self.assertTrue(os.path.exists(path),
                            "expected restored side-effect at {}".format(path))

    def test_side_effects_wiped_with_no_archive_resubmits(self):
        # the archive was also wiped (or never written by an older shard) - the helper has
        # no choice but to invalidate the cached pickle and ask the launcher to resubmit
        files = [self._side_effect_for("md1", 1), self._side_effect_for("md1", 2)]
        self._bootstrap_shard(0, files)
        os.remove(TunnelLayeringShardStage.shard_side_effects_archive_path(self._slurm, 0))
        __import__("shutil").rmtree(self._target)
        missing = _missing_shards(self.stage, self._slurm, num_shards=1, parameters=self.params)
        self.assertEqual([0], missing)
        # the cached pickle + manifest should also have been removed so a fresh resubmission
        # produces consistent state
        self.assertFalse(os.path.exists(self.stage.shard_result_path(self._slurm, 0)))
        self.assertFalse(os.path.exists(
            TunnelLayeringShardStage.shard_manifest_path(self._slurm, 0)))

    def test_manifest_missing_invalidates(self):
        # has_side_effects=True but the manifest is missing - cannot validate, so the cache
        # is treated as broken and the shard is resubmitted
        files = [self._side_effect_for("md1", 1)]
        self._bootstrap_shard(0, files)
        os.remove(TunnelLayeringShardStage.shard_manifest_path(self._slurm, 0))
        missing = _missing_shards(self.stage, self._slurm, num_shards=1, parameters=self.params)
        self.assertEqual([0], missing)

    def test_distance_stage_fast_path_unchanged(self):
        # DistanceShardStage keeps has_side_effects=False so the resume check still only
        # validates the result pickle - no archive lookup, no manifest required
        from transport_tools.libs.slurm import DistanceShardStage
        stage = DistanceShardStage(num_clusters=5)
        with open(stage.shard_result_path(self._slurm, 0), "wb") as f:
            f.write(b"")
        with open(os.path.join(self._slurm, "partial_shards_info.json"), "w") as out:
            json.dump(stage.fingerprint({"clustering_cutoff": 2.0,
                                          "calculate_exact_path_distances": True},
                                         num_shards=1), out)
        missing = _missing_shards(stage, self._slurm, num_shards=1,
                                  parameters={"clustering_cutoff": 2.0,
                                              "calculate_exact_path_distances": True})
        self.assertEqual([], missing)


class _DummyStage(SlurmShardStage):
    """A minimal SlurmShardStage subclass that touches only the interface, used to exercise
    the abstract-class contract and the default fingerprint/get_slurm_folder behaviours."""

    folder_name = "stage99_dummy"
    stage_name = "dummy"
    job_name = "tt_dummy"
    fingerprint_keys = ("some_param",)

    def num_shards(self, parameters):
        return 3

    @classmethod
    def shard_result_path(cls, slurm_folder, shard_id):
        return os.path.join(slurm_folder, "dummy_{}.bin".format(shard_id))

    @classmethod
    def run_shard(cls, config_file, num_shards, shard_id):
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


class TestTunnelLayeringShardStageFolderLayout(unittest.TestCase):
    """TunnelLayeringShardStage publishes its folder under slurm_root_folder."""

    def test_folder_name_uses_stage_number(self):
        self.assertEqual("stage03_tunnel_layering", TunnelLayeringShardStage.folder_name)

    def test_get_slurm_folder_joins_root_and_folder_name(self):
        path = TunnelLayeringShardStage.get_slurm_folder({"slurm_root_folder": "/scratch/run/_slurm"})
        self.assertEqual("/scratch/run/_slurm/stage03_tunnel_layering", path)

    def test_shard_result_path_is_pickle_extension(self):
        # tunnel-layering shard results contain LayeredPathSet instances - pickle, not npy
        path = TunnelLayeringShardStage.shard_result_path("/tmp/folder", 3)
        self.assertEqual("/tmp/folder/shard_result_00003.pkl", path)

    def test_shard_result_path_is_classmethod(self):
        # callable without instantiating (so SLURM shard workers can derive their output path
        # from just the parameters dict + shard_id without needing the runtime items list)
        path = TunnelLayeringShardStage.shard_result_path("/x", 0)
        self.assertTrue(path.endswith("shard_result_00000.pkl"))


class TestTunnelLayeringShardStageNumShards(unittest.TestCase):
    """num_shards() honors user override, caps at item count, auto-derives at 200/shard."""

    def _params(self, slurm_num_shards=None, slurm_array_parallelism=None):
        return {
            "slurm_num_shards": slurm_num_shards,
            "slurm_array_parallelism": slurm_array_parallelism,
        }

    def test_empty_items_yields_single_shard(self):
        stage = TunnelLayeringShardStage([])
        self.assertEqual(1, stage.num_shards(self._params()))

    def test_user_count_capped_at_item_count(self):
        # 5 items, asking for 100 shards is silly - cap to 5
        stage = TunnelLayeringShardStage([("md{}".format(i), 1) for i in range(5)])
        self.assertEqual(5, stage.num_shards(self._params(slurm_num_shards=100)))

    def test_auto_derives_at_200_per_shard(self):
        # 600 items / 200 per shard => 3 shards
        stage = TunnelLayeringShardStage([("md{}".format(i), j) for i in range(60) for j in range(10)])
        self.assertEqual(3, stage.num_shards(self._params()))

    def test_array_parallelism_caps_auto_derived(self):
        # 1000 items / 200 per shard would be 5 shards; cap to 2
        stage = TunnelLayeringShardStage([("md{}".format(i), j) for i in range(100) for j in range(10)])
        self.assertEqual(2, stage.num_shards(self._params(slurm_array_parallelism=2)))


class TestTunnelLayeringShardStageFingerprint(unittest.TestCase):
    """Fingerprint identifies the parameters AND the work-item enumeration."""

    def _params(self, **overrides):
        params = {
            "layer_thickness": 1.5,
            "min_tunnel_radius4clustering": 0.75,
            "min_tunnel_length4clustering": 5.0,
            "max_tunnel_curvature4clustering": 2.0,
            "random_seed": 4,
            "clustering_max_num_rep_frag": 3,
            "use_cluster_spread": True,
            # visualize_layered_clusters is part of the fingerprint (toggles whether the
            # worker writes per-cluster .py + nodes/ side-effects); supplied here so the
            # fingerprint() call has every key it needs
            "visualize_layered_clusters": True,
        }
        params.update(overrides)
        return params

    def test_fingerprint_includes_items_hash_and_count(self):
        stage = TunnelLayeringShardStage([("a", 1), ("b", 2)])
        info = stage.fingerprint(self._params(), num_shards=2)
        self.assertEqual(2, info["num_items"])
        self.assertEqual(64, len(info["items_hash"]))  # sha256 hex digest

    def test_fingerprint_changes_when_items_differ(self):
        stage1 = TunnelLayeringShardStage([("a", 1), ("b", 2)])
        stage2 = TunnelLayeringShardStage([("a", 1), ("b", 3)])
        self.assertNotEqual(stage1.fingerprint(self._params(), num_shards=2),
                            stage2.fingerprint(self._params(), num_shards=2))

    def test_fingerprint_changes_when_clustering_param_differs(self):
        stage = TunnelLayeringShardStage([("a", 1)])
        fp1 = stage.fingerprint(self._params(layer_thickness=1.5), num_shards=1)
        fp2 = stage.fingerprint(self._params(layer_thickness=2.0), num_shards=1)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_is_stable_across_tuple_vs_list_input(self):
        # the constructor canonicalises tuples to lists so JSON round-trips preserve the hash
        stage_from_tuples = TunnelLayeringShardStage([("a", 1), ("b", 2)])
        stage_from_lists = TunnelLayeringShardStage([["a", 1], ["b", 2]])
        self.assertEqual(stage_from_tuples.fingerprint(self._params(), num_shards=2),
                         stage_from_lists.fingerprint(self._params(), num_shards=2))


class TestTunnelLayeringShardStateRoundTrip(unittest.TestCase):
    """prepare_state writes items.json; load_items reads it back unchanged."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))

    def test_prepare_state_writes_items_json(self):
        items = [("md_alpha", 7), ("md_beta", 0), ("md_alpha", 13)]
        stage = TunnelLayeringShardStage(items)
        stage.prepare_state({}, self._tmp)
        items_file = os.path.join(self._tmp, "items.json")
        self.assertTrue(os.path.exists(items_file))

    def test_load_items_returns_canonicalised_pairs(self):
        items = [("md_alpha", 7), ("md_beta", 0)]
        stage = TunnelLayeringShardStage(items)
        stage.prepare_state({}, self._tmp)
        loaded = TunnelLayeringShardStage.load_items(self._tmp)
        # JSON normalises tuples to lists; consumers iterate as `for md, cls in loaded`
        self.assertEqual([["md_alpha", 7], ["md_beta", 0]], loaded)

    def test_load_items_missing_raises_with_diagnostic(self):
        with self.assertRaises(FileNotFoundError) as cm:
            TunnelLayeringShardStage.load_items(self._tmp)
        # message must point at the missing file and explain who writes it
        self.assertIn("items.json", str(cm.exception))
        self.assertIn("prepare_state", str(cm.exception))

    def test_prepare_state_overwrites_stale_items_atomically(self):
        # writing must use a tmp+rename so a partially-written items.json is not visible
        TunnelLayeringShardStage([("a", 1)]).prepare_state({}, self._tmp)
        TunnelLayeringShardStage([("b", 2), ("c", 3)]).prepare_state({}, self._tmp)
        self.assertEqual([["b", 2], ["c", 3]],
                         TunnelLayeringShardStage.load_items(self._tmp))


class TestAquaductLayeringShardStageFolderLayout(unittest.TestCase):
    """AquaductLayeringShardStage publishes its folder under slurm_root_folder."""

    def test_folder_name_uses_stage_number(self):
        # stage 8 in the tt_engine.py workflow (create_layered_description4aquaduct_networks)
        self.assertEqual("stage08_aquaduct_layering", AquaductLayeringShardStage.folder_name)

    def test_get_slurm_folder_joins_root_and_folder_name(self):
        path = AquaductLayeringShardStage.get_slurm_folder({"slurm_root_folder": "/scratch/run/_slurm"})
        self.assertEqual("/scratch/run/_slurm/stage08_aquaduct_layering", path)

    def test_shard_result_path_is_pickle_extension(self):
        # aquaduct-layering shard results contain LayeredPathSet instances - pickle, not npy
        path = AquaductLayeringShardStage.shard_result_path("/tmp/folder", 3)
        self.assertEqual("/tmp/folder/shard_result_00003.pkl", path)

    def test_shard_result_path_is_classmethod(self):
        # callable without instantiating (so SLURM shard workers can derive their output path
        # from just the parameters dict + shard_id without needing the runtime items list)
        path = AquaductLayeringShardStage.shard_result_path("/x", 0)
        self.assertTrue(path.endswith("shard_result_00000.pkl"))

    def test_does_not_collide_with_tunnel_layering_folder(self):
        # both layering stages live under the same slurm_root_folder; their per-stage subfolders
        # must be distinct so per-shard result files cannot clobber each other
        root = {"slurm_root_folder": "/scratch/run/_slurm"}
        self.assertNotEqual(
            AquaductLayeringShardStage.get_slurm_folder(root),
            TunnelLayeringShardStage.get_slurm_folder(root))


class TestAquaductLayeringShardStageNumShards(unittest.TestCase):
    """num_shards() honors user override, caps at item count, auto-derives at 200/shard."""

    def _params(self, slurm_num_shards=None, slurm_array_parallelism=None):
        return {
            "slurm_num_shards": slurm_num_shards,
            "slurm_array_parallelism": slurm_array_parallelism,
        }

    def test_empty_items_yields_single_shard(self):
        stage = AquaductLayeringShardStage([])
        self.assertEqual(1, stage.num_shards(self._params()))

    def test_user_count_capped_at_item_count(self):
        # 5 items, asking for 100 shards is silly - cap to 5
        stage = AquaductLayeringShardStage(
            [("md{}".format(i), "wat_{}_release".format(i)) for i in range(5)])
        self.assertEqual(5, stage.num_shards(self._params(slurm_num_shards=100)))

    def test_auto_derives_at_200_per_shard(self):
        # 600 items / 200 per shard => 3 shards
        stage = AquaductLayeringShardStage(
            [("md{}".format(i), "wat_{}_release".format(j)) for i in range(60) for j in range(10)])
        self.assertEqual(3, stage.num_shards(self._params()))

    def test_array_parallelism_caps_auto_derived(self):
        # 1000 items / 200 per shard would be 5 shards; cap to 2
        stage = AquaductLayeringShardStage(
            [("md{}".format(i), "wat_{}_release".format(j)) for i in range(100) for j in range(10)])
        self.assertEqual(2, stage.num_shards(self._params(slurm_array_parallelism=2)))


class TestAquaductLayeringShardStageFingerprint(unittest.TestCase):
    """Fingerprint identifies the parameters AND the work-item enumeration."""

    def _params(self, **overrides):
        params = {
            "layer_thickness": 1.5,
            "random_seed": 4,
            "clustering_max_num_rep_frag": 3,
            "use_cluster_spread": True,
            # visualize_layered_events is part of the fingerprint (toggles whether the
            # worker writes per-event .py + nodes/ side-effects); supplied here so the
            # fingerprint() call has every key it needs
            "visualize_layered_events": True,
        }
        params.update(overrides)
        return params

    def test_fingerprint_includes_items_hash_and_count(self):
        stage = AquaductLayeringShardStage([("a", "wat_1_entry"), ("b", "wat_2_release")])
        info = stage.fingerprint(self._params(), num_shards=2)
        self.assertEqual(2, info["num_items"])
        self.assertEqual(64, len(info["items_hash"]))  # sha256 hex digest

    def test_fingerprint_changes_when_items_differ(self):
        stage1 = AquaductLayeringShardStage([("a", "wat_1_entry"), ("b", "wat_2_release")])
        stage2 = AquaductLayeringShardStage([("a", "wat_1_entry"), ("b", "wat_3_release")])
        self.assertNotEqual(stage1.fingerprint(self._params(), num_shards=2),
                            stage2.fingerprint(self._params(), num_shards=2))

    def test_fingerprint_changes_when_layer_thickness_differs(self):
        stage = AquaductLayeringShardStage([("a", "wat_1_release")])
        fp1 = stage.fingerprint(self._params(layer_thickness=1.5), num_shards=1)
        fp2 = stage.fingerprint(self._params(layer_thickness=2.0), num_shards=1)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_is_stable_across_tuple_vs_list_input(self):
        # the constructor canonicalises tuples to lists so JSON round-trips preserve the hash
        stage_from_tuples = AquaductLayeringShardStage(
            [("a", "wat_1_release"), ("b", "wat_2_entry")])
        stage_from_lists = AquaductLayeringShardStage(
            [["a", "wat_1_release"], ["b", "wat_2_entry"]])
        self.assertEqual(stage_from_tuples.fingerprint(self._params(), num_shards=2),
                         stage_from_lists.fingerprint(self._params(), num_shards=2))

    def test_fingerprint_omits_upstream_event_min_distance(self):
        # event_min_distance is an upstream filter applied during stage 7; its effect is baked
        # into the stage-7 orig_networks and reflected in items_hash. Including it explicitly
        # would create false-positive invalidations when the user only reruns stage 8.
        stage = AquaductLayeringShardStage([("a", "wat_1_release")])
        info = stage.fingerprint(self._params(), num_shards=1)
        self.assertNotIn("event_min_distance", info)


class TestAquaductLayeringShardStateRoundTrip(unittest.TestCase):
    """prepare_state writes items.json; load_items reads it back unchanged."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))

    def test_prepare_state_writes_items_json(self):
        items = [("md_alpha", "wat_7_release"), ("md_beta", "wat_0_entry"),
                 ("md_alpha", "wat_13_release")]
        stage = AquaductLayeringShardStage(items)
        stage.prepare_state({}, self._tmp)
        items_file = os.path.join(self._tmp, "items.json")
        self.assertTrue(os.path.exists(items_file))

    def test_load_items_returns_canonicalised_pairs(self):
        items = [("md_alpha", "wat_7_release"), ("md_beta", "wat_0_entry")]
        stage = AquaductLayeringShardStage(items)
        stage.prepare_state({}, self._tmp)
        loaded = AquaductLayeringShardStage.load_items(self._tmp)
        # JSON normalises tuples to lists; consumers iterate as `for md, event in loaded`
        self.assertEqual([["md_alpha", "wat_7_release"], ["md_beta", "wat_0_entry"]], loaded)

    def test_load_items_missing_raises_with_diagnostic(self):
        with self.assertRaises(FileNotFoundError) as cm:
            AquaductLayeringShardStage.load_items(self._tmp)
        # message must point at the missing file and explain who writes it
        self.assertIn("items.json", str(cm.exception))
        self.assertIn("prepare_state", str(cm.exception))

    def test_prepare_state_overwrites_stale_items_atomically(self):
        # writing must use a tmp+rename so a partially-written items.json is not visible
        AquaductLayeringShardStage([("a", "wat_1_release")]).prepare_state({}, self._tmp)
        AquaductLayeringShardStage(
            [("b", "wat_2_entry"), ("c", "wat_3_release")]).prepare_state({}, self._tmp)
        self.assertEqual([["b", "wat_2_entry"], ["c", "wat_3_release"]],
                         AquaductLayeringShardStage.load_items(self._tmp))


# Helper for the event-assignment stage: a tiny picklable stand-in for the supercluster /
# active-filter state. The real classes are heavy and depend on a full TransportProcesses
# instance; everything `EventAssignmentShardStage` does with them is pickle + hash + read
# back, so any picklable Python object exercises the contract just fine.
def _fake_state(super_clusters=None, active_filters=None):
    return ({1: "sc-one", 2: "sc-two"} if super_clusters is None else super_clusters,
            {"min_length": 5} if active_filters is None else active_filters)


class TestEventAssignmentShardStageFolderLayout(unittest.TestCase):
    """EventAssignmentShardStage publishes its folder under slurm_root_folder."""

    def test_folder_name_uses_stage_number(self):
        # stage 9 in the tt_engine.py workflow (assign_transport_events)
        self.assertEqual("stage09_event_assignment", EventAssignmentShardStage.folder_name)

    def test_get_slurm_folder_joins_root_and_folder_name(self):
        path = EventAssignmentShardStage.get_slurm_folder({"slurm_root_folder": "/scratch/run/_slurm"})
        self.assertEqual("/scratch/run/_slurm/stage09_event_assignment", path)

    def test_shard_result_path_is_pickle_extension(self):
        # event-assignment shard results contain tuples with numpy arrays - pickle, not npy
        path = EventAssignmentShardStage.shard_result_path("/tmp/folder", 3)
        self.assertEqual("/tmp/folder/shard_result_00003.pkl", path)

    def test_does_not_collide_with_other_stage_folders(self):
        # stage 9 sits alongside the other SLURM stages under the same slurm_root_folder
        root = {"slurm_root_folder": "/scratch/run/_slurm"}
        folders = {
            EventAssignmentShardStage.get_slurm_folder(root),
            AquaductLayeringShardStage.get_slurm_folder(root),
            TunnelLayeringShardStage.get_slurm_folder(root),
            DistanceShardStage.get_slurm_folder(root),
        }
        self.assertEqual(4, len(folders))


class TestEventAssignmentShardStageNumShards(unittest.TestCase):
    """num_shards() honors user override, caps at item count, auto-derives at 200/shard."""

    def _params(self, slurm_num_shards=None, slurm_array_parallelism=None):
        return {
            "slurm_num_shards": slurm_num_shards,
            "slurm_array_parallelism": slurm_array_parallelism,
        }

    def _stage(self, items):
        super_clusters, active_filters = _fake_state()
        return EventAssignmentShardStage(items, super_clusters, active_filters, pickle_protocol=4)

    def test_empty_items_yields_single_shard(self):
        stage = self._stage([])
        self.assertEqual(1, stage.num_shards(self._params()))

    def test_user_count_capped_at_item_count(self):
        # 5 events, asking for 100 shards is silly - cap to 5
        stage = self._stage([("md{}".format(i), "wat_{}_release".format(i)) for i in range(5)])
        self.assertEqual(5, stage.num_shards(self._params(slurm_num_shards=100)))

    def test_auto_derives_at_200_per_shard(self):
        # 600 events / 200 per shard => 3 shards
        stage = self._stage(
            [("md{}".format(i), "wat_{}_release".format(j)) for i in range(60) for j in range(10)])
        self.assertEqual(3, stage.num_shards(self._params()))

    def test_array_parallelism_caps_auto_derived(self):
        # 1000 events / 200 per shard would be 5 shards; cap to 2
        stage = self._stage(
            [("md{}".format(i), "wat_{}_release".format(j)) for i in range(100) for j in range(10)])
        self.assertEqual(2, stage.num_shards(self._params(slurm_array_parallelism=2)))


class TestEventAssignmentShardStateRoundTrip(unittest.TestCase):
    """prepare_state writes items.json + state.pkl; load_items / load_state round-trip them."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))

    def _stage(self, items, super_clusters=None, active_filters=None):
        sc, af = _fake_state(super_clusters, active_filters)
        return EventAssignmentShardStage(items, sc, af, pickle_protocol=4)

    def test_prepare_state_writes_items_and_state_files(self):
        stage = self._stage([("md_alpha", "wat_7_release"), ("md_beta", "wat_0_entry")])
        stage.prepare_state({}, self._tmp)
        self.assertTrue(os.path.exists(os.path.join(self._tmp, "items.json")))
        self.assertTrue(os.path.exists(os.path.join(self._tmp, "state.pkl")))

    def test_load_items_returns_canonicalised_pairs(self):
        stage = self._stage([("md_alpha", "wat_7_release"), ("md_beta", "wat_0_entry")])
        stage.prepare_state({}, self._tmp)
        loaded = EventAssignmentShardStage.load_items(self._tmp)
        self.assertEqual([["md_alpha", "wat_7_release"], ["md_beta", "wat_0_entry"]], loaded)

    def test_load_state_round_trips_super_clusters_and_filters(self):
        super_clusters = {1: "sc-one", 2: "sc-two"}
        active_filters = {"min_length": 5, "min_sims_num": 2}
        stage = self._stage([("md_alpha", "wat_7_release")],
                            super_clusters=super_clusters, active_filters=active_filters)
        stage.prepare_state({}, self._tmp)
        loaded_sc, loaded_af = EventAssignmentShardStage.load_state(self._tmp)
        self.assertEqual(super_clusters, loaded_sc)
        self.assertEqual(active_filters, loaded_af)

    def test_load_items_missing_raises_with_diagnostic(self):
        with self.assertRaises(FileNotFoundError) as cm:
            EventAssignmentShardStage.load_items(self._tmp)
        self.assertIn("items.json", str(cm.exception))
        self.assertIn("prepare_state", str(cm.exception))

    def test_load_state_missing_raises_with_diagnostic(self):
        with self.assertRaises(FileNotFoundError) as cm:
            EventAssignmentShardStage.load_state(self._tmp)
        self.assertIn("state.pkl", str(cm.exception))
        self.assertIn("prepare_state", str(cm.exception))

    def test_prepare_state_overwrites_stale_artifacts_atomically(self):
        # writing must use a tmp+rename so a partially-written file is never visible
        self._stage([("a", "wat_1_release")]).prepare_state({}, self._tmp)
        self._stage([("b", "wat_2_entry"), ("c", "wat_3_release")],
                    super_clusters={42: "different-sc"}).prepare_state({}, self._tmp)
        self.assertEqual([["b", "wat_2_entry"], ["c", "wat_3_release"]],
                         EventAssignmentShardStage.load_items(self._tmp))
        loaded_sc, _ = EventAssignmentShardStage.load_state(self._tmp)
        self.assertEqual({42: "different-sc"}, loaded_sc)


class TestEventAssignmentShardStageFingerprint(unittest.TestCase):
    """Fingerprint identifies parameters, the event set, AND the supercluster + filter state."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))

    def _params(self, **overrides):
        params = {
            "event_assignment_cutoff": 0.85,
            "ambiguous_event_assignment_resolution": "penetration_depth",
            "perform_exact_matching_analysis": False,
        }
        params.update(overrides)
        return params

    def _stage(self, items, super_clusters=None, active_filters=None):
        sc, af = _fake_state(super_clusters, active_filters)
        return EventAssignmentShardStage(items, sc, af, pickle_protocol=4)

    def test_fingerprint_includes_items_and_state_hashes(self):
        stage = self._stage([("a", "wat_1_entry"), ("b", "wat_2_release")])
        stage.prepare_state(self._params(), self._tmp)
        info = stage.fingerprint(self._params(), num_shards=2)
        self.assertEqual(2, info["num_items"])
        self.assertEqual(64, len(info["items_hash"]))  # sha256 hex digest
        self.assertEqual(64, len(info["state_hash"]))

    def test_fingerprint_changes_when_items_differ(self):
        stage1 = self._stage([("a", "wat_1_entry"), ("b", "wat_2_release")])
        stage2 = self._stage([("a", "wat_1_entry"), ("b", "wat_3_release")])
        stage1.prepare_state(self._params(), self._tmp)
        # second stage writes into the same folder but we only need its in-memory hash
        stage2.prepare_state(self._params(), self._tmp)
        self.assertNotEqual(stage1.fingerprint(self._params(), num_shards=2)["items_hash"],
                            stage2.fingerprint(self._params(), num_shards=2)["items_hash"])

    def test_fingerprint_changes_when_super_clusters_differ(self):
        stage1 = self._stage([("a", "wat_1_release")], super_clusters={1: "alpha"})
        stage2 = self._stage([("a", "wat_1_release")], super_clusters={1: "beta"})
        stage1.prepare_state(self._params(), self._tmp)
        stage2.prepare_state(self._params(), self._tmp)
        self.assertNotEqual(stage1.fingerprint(self._params(), num_shards=1)["state_hash"],
                            stage2.fingerprint(self._params(), num_shards=1)["state_hash"])

    def test_fingerprint_changes_when_active_filters_differ(self):
        stage1 = self._stage([("a", "wat_1_release")], active_filters={"min_length": 5})
        stage2 = self._stage([("a", "wat_1_release")], active_filters={"min_length": 8})
        stage1.prepare_state(self._params(), self._tmp)
        stage2.prepare_state(self._params(), self._tmp)
        self.assertNotEqual(stage1.fingerprint(self._params(), num_shards=1)["state_hash"],
                            stage2.fingerprint(self._params(), num_shards=1)["state_hash"])

    def test_fingerprint_changes_when_assignment_cutoff_differs(self):
        stage = self._stage([("a", "wat_1_release")])
        stage.prepare_state(self._params(), self._tmp)
        fp1 = stage.fingerprint(self._params(event_assignment_cutoff=0.85), num_shards=1)
        fp2 = stage.fingerprint(self._params(event_assignment_cutoff=0.95), num_shards=1)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_before_prepare_state_raises(self):
        # state_hash must be computed by prepare_state(); calling fingerprint() first must
        # fail loudly rather than silently produce a fingerprint without the state component
        stage = self._stage([("a", "wat_1_release")])
        with self.assertRaises(RuntimeError) as cm:
            stage.fingerprint(self._params(), num_shards=1)
        self.assertIn("prepare_state", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
