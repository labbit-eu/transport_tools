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
import tarfile
import tempfile
import unittest
from typing import Optional

from transport_tools.libs.slurm import (
    AquaductLayeringShardStage,
    AquaductNetworksShardStage,
    DistanceShardStage,
    EventAssignmentShardStage,
    SlurmShardStage,
    TunnelLayeringShardStage,
    TunnelNetworksShardStage,
    _categorize_failure,
    _detect_parent_slurm_allocation,
    _emit_failure_categorisation,
    _emit_runtime_statistics,
    _ensure_cpu_bind_none,
    _missing_shards,
    _parse_slurm_time_limit_min,
    _percentile,
    _report_run_stats,
    _warn_on_driver_misallocation,
    cleanup_stage_folder,
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
        TunnelLayeringShardStage.write_side_effects_archive(
            self._slurm, 7, [p1, p2], self._target)
        # archive must exist
        self.assertTrue(os.path.exists(
            TunnelLayeringShardStage.shard_side_effects_archive_path(self._slurm, 7)))
        # arcnames are stored relative to base_path - inspecting the archive shows short
        # portable paths (no `home/.../target/...` prefix)
        with tarfile.open(
                TunnelLayeringShardStage.shard_side_effects_archive_path(self._slurm, 7),
                "r:gz") as tar:
            self.assertEqual({"Cluster_1.py", "nodes/cls001_layer0_pts.pdb"},
                             {m.name for m in tar.getmembers() if m.isfile()})
        # wipe and restore
        __import__("shutil").rmtree(self._target)
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(
            self._slurm, 7, [p1, p2], self._target)
        self.assertTrue(ok)
        with open(p1, "rb") as f:
            self.assertEqual(b"# pymol script\ncmd.show_as('wire', 'all')\n", f.read())
        with open(p2, "rb") as f:
            self.assertEqual(b"ATOM   1  C  ...\nEND\n", f.read())

    def test_restore_returns_false_when_archive_missing(self):
        # never write the archive; restore must report failure rather than raise
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(
            self._slurm, 0, [os.path.join(self._target, "nope")], self._target)
        self.assertFalse(ok)

    def test_restore_returns_false_when_archive_is_corrupt(self):
        # write garbage where the archive should be; restore must report failure
        archive_path = TunnelLayeringShardStage.shard_side_effects_archive_path(self._slurm, 0)
        with open(archive_path, "wb") as out:
            out.write(b"not a tarball")
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(
            self._slurm, 0, [os.path.join(self._target, "nope")], self._target)
        self.assertFalse(ok)

    def test_restore_returns_false_when_expected_file_not_in_archive(self):
        # archive contains file A but caller asks to restore B - restore reports failure
        # without writing anything to B's path (so the caller can fall back to resubmit
        # without confusing partial state)
        p_in = self._make_file("Cluster_1.py", b"in-archive")
        TunnelLayeringShardStage.write_side_effects_archive(
            self._slurm, 0, [p_in], self._target)
        p_out = os.path.join(self._target, "Cluster_9_not_in_archive.py")
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(
            self._slurm, 0, [p_out], self._target)
        self.assertFalse(ok)
        self.assertFalse(os.path.exists(p_out))

    def test_empty_archive_round_trips(self):
        # stages with no side-effects (e.g. visualize_layered_clusters=False) still write
        # an archive so the manifest semantics stay uniform; restore with empty list must
        # succeed trivially
        TunnelLayeringShardStage.write_side_effects_archive(
            self._slurm, 0, [], self._target)
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(
            self._slurm, 0, [], self._target)
        self.assertTrue(ok)

    def test_archive_restores_after_output_folder_moves(self):
        # the relative-arcname design must allow the user to relocate the output folder
        # (and the slurm folder underneath it): writing the archive at the original location,
        # then physically moving target + slurm to a new path and calling restore with the
        # new base_path must reconstruct the side-effect files at the new location -
        # nothing inside the archive encodes the old absolute path.
        p1 = self._make_file("md1/Cluster_1.py", b"# moved\n")
        p2 = self._make_file("md1/nodes/cls001-0-0.pdb", b"ATOM\nEND\n")
        TunnelLayeringShardStage.write_side_effects_archive(
            self._slurm, 3, [p1, p2], self._target)
        # relocate the whole tree: new target + new slurm folder under a different parent
        new_root = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(new_root, ignore_errors=True))
        new_target = os.path.join(new_root, "target")
        new_slurm = os.path.join(new_root, "slurm")
        __import__("shutil").move(self._target, new_target)
        __import__("shutil").move(self._slurm, new_slurm)
        # now wipe the side-effect files at the new location to simulate a resume against
        # the moved tree, then restore using the new base_path
        for rel in ("md1/Cluster_1.py", "md1/nodes/cls001-0-0.pdb"):
            os.remove(os.path.join(new_target, rel))
        new_p1 = os.path.join(new_target, "md1", "Cluster_1.py")
        new_p2 = os.path.join(new_target, "md1", "nodes", "cls001-0-0.pdb")
        ok = TunnelLayeringShardStage.restore_side_effects_from_archive(
            new_slurm, 3, [new_p1, new_p2], new_target)
        self.assertTrue(ok)
        with open(new_p1, "rb") as f:
            self.assertEqual(b"# moved\n", f.read())
        with open(new_p2, "rb") as f:
            self.assertEqual(b"ATOM\nEND\n", f.read())


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
            "output_path": self._target,
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
            self._slurm, shard_id, side_effect_paths, self.params["output_path"])
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


class TestTunnelNetworksShardStageFolderLayout(unittest.TestCase):
    """TunnelNetworksShardStage publishes its folder under slurm_root_folder."""

    def test_folder_name_uses_stage_number(self):
        # stage 2 in the tt_engine.py workflow (process_tunnel_networks)
        self.assertEqual("stage02_tunnel_networks", TunnelNetworksShardStage.folder_name)

    def test_get_slurm_folder_joins_root_and_folder_name(self):
        path = TunnelNetworksShardStage.get_slurm_folder({"slurm_root_folder": "/scratch/run/_slurm"})
        self.assertEqual("/scratch/run/_slurm/stage02_tunnel_networks", path)

    def test_shard_result_path_is_pickle_extension(self):
        # tunnel-networks shard result is a tiny pickled list of processed md_labels
        path = TunnelNetworksShardStage.shard_result_path("/tmp/folder", 3)
        self.assertEqual("/tmp/folder/shard_result_00003.pkl", path)

    def test_does_not_collide_with_other_stage_folders(self):
        # stage 2 sits alongside the other SLURM stages under the same slurm_root_folder;
        # per-stage subfolders must be distinct so per-shard artefacts cannot clobber each
        # other across stages
        root = {"slurm_root_folder": "/scratch/run/_slurm"}
        folders = {
            TunnelNetworksShardStage.get_slurm_folder(root),
            TunnelLayeringShardStage.get_slurm_folder(root),
            DistanceShardStage.get_slurm_folder(root),
            AquaductNetworksShardStage.get_slurm_folder(root),
            AquaductLayeringShardStage.get_slurm_folder(root),
            EventAssignmentShardStage.get_slurm_folder(root),
        }
        self.assertEqual(6, len(folders))


class TestTunnelNetworksShardStageNumShards(unittest.TestCase):
    """num_shards() honors user override, caps at item count, auto-derives at 50/shard."""

    def _params(self, slurm_num_shards=None, slurm_array_parallelism=None):
        return {
            "slurm_num_shards": slurm_num_shards,
            "slurm_array_parallelism": slurm_array_parallelism,
        }

    def test_empty_items_yields_single_shard(self):
        stage = TunnelNetworksShardStage([])
        self.assertEqual(1, stage.num_shards(self._params()))

    def test_user_count_capped_at_item_count(self):
        # 5 md_labels, asking for 100 shards is silly - cap to 5
        stage = TunnelNetworksShardStage(["md{}".format(i) for i in range(5)])
        self.assertEqual(5, stage.num_shards(self._params(slurm_num_shards=100)))

    def test_auto_derives_at_50_per_shard(self):
        # 150 md_labels / 50 per shard => 3 shards (tunnel-networks runs coarser than the
        # layering stages since per-MD work is moderate rather than highly variable)
        stage = TunnelNetworksShardStage(["md{}".format(i) for i in range(150)])
        self.assertEqual(3, stage.num_shards(self._params()))

    def test_array_parallelism_caps_auto_derived(self):
        # 1000 items / 50 per shard would be 20 shards; cap to 2
        stage = TunnelNetworksShardStage(["md{}".format(i) for i in range(1000)])
        self.assertEqual(2, stage.num_shards(self._params(slurm_array_parallelism=2)))


class TestTunnelNetworksShardStageFingerprint(unittest.TestCase):
    """Fingerprint identifies the parameters AND the md_label enumeration."""

    def _params(self, **overrides):
        params = {
            "caver_results_folder_pattern": "*",
            "snapshots_per_simulation": 10,
            "caver_traj_offset": 1,
            "process_bottleneck_residues": False,
            "visualize_transformed_tunnels": False,
        }
        params.update(overrides)
        return params

    def test_fingerprint_includes_items_hash_and_count(self):
        stage = TunnelNetworksShardStage(["md1", "md2", "md3"])
        info = stage.fingerprint(self._params(), num_shards=2)
        self.assertEqual(3, info["num_items"])
        self.assertEqual(64, len(info["items_hash"]))  # sha256 hex digest

    def test_fingerprint_changes_when_items_differ(self):
        stage1 = TunnelNetworksShardStage(["md1", "md2"])
        stage2 = TunnelNetworksShardStage(["md1", "md3"])
        self.assertNotEqual(stage1.fingerprint(self._params(), num_shards=2),
                            stage2.fingerprint(self._params(), num_shards=2))

    def test_fingerprint_changes_when_pattern_differs(self):
        stage = TunnelNetworksShardStage(["md1"])
        fp1 = stage.fingerprint(self._params(caver_results_folder_pattern="*"), num_shards=1)
        fp2 = stage.fingerprint(self._params(caver_results_folder_pattern="md*"), num_shards=1)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_changes_when_snapshots_per_simulation_differs(self):
        stage = TunnelNetworksShardStage(["md1"])
        fp1 = stage.fingerprint(self._params(snapshots_per_simulation=10), num_shards=1)
        fp2 = stage.fingerprint(self._params(snapshots_per_simulation=20), num_shards=1)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_changes_when_visualize_transformed_tunnels_flips(self):
        # the fingerprint must invalidate the cache on a False->True transition so the resume
        # path does not silently re-use a cache that has no visualisation side-effects backed
        # up (the empty manifest would otherwise pass the resume check trivially)
        stage = TunnelNetworksShardStage(["md1"])
        fp1 = stage.fingerprint(self._params(visualize_transformed_tunnels=False), num_shards=1)
        fp2 = stage.fingerprint(self._params(visualize_transformed_tunnels=True), num_shards=1)
        self.assertNotEqual(fp1, fp2)


class TestTunnelNetworksShardStateRoundTrip(unittest.TestCase):
    """prepare_state writes items.json; load_items reads it back unchanged."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))

    def test_prepare_state_writes_items_json(self):
        items = ["md_alpha", "md_beta", "md_gamma"]
        stage = TunnelNetworksShardStage(items)
        stage.prepare_state({}, self._tmp)
        items_file = os.path.join(self._tmp, "items.json")
        self.assertTrue(os.path.exists(items_file))

    def test_load_items_returns_canonicalised_list(self):
        items = ["md_alpha", "md_beta"]
        stage = TunnelNetworksShardStage(items)
        stage.prepare_state({}, self._tmp)
        loaded = TunnelNetworksShardStage.load_items(self._tmp)
        # JSON round-trip keeps the order; consumers iterate as `for md_label in loaded`
        self.assertEqual(["md_alpha", "md_beta"], loaded)

    def test_load_items_missing_raises_with_diagnostic(self):
        with self.assertRaises(FileNotFoundError) as cm:
            TunnelNetworksShardStage.load_items(self._tmp)
        # message must point at the missing file and explain who writes it
        self.assertIn("items.json", str(cm.exception))
        self.assertIn("prepare_state", str(cm.exception))

    def test_prepare_state_overwrites_stale_items_atomically(self):
        # writing must use a tmp+rename so a partially-written items.json is not visible
        TunnelNetworksShardStage(["md_a"]).prepare_state({}, self._tmp)
        TunnelNetworksShardStage(["md_b", "md_c"]).prepare_state({}, self._tmp)
        self.assertEqual(["md_b", "md_c"],
                         TunnelNetworksShardStage.load_items(self._tmp))


class TestTunnelNetworksShardSideEffectPaths(unittest.TestCase):
    """derive_side_effect_paths reports the per-md_label outputs the worker is expected to
    write. Without visualisation, only the `<md>_caver.dump` files; with visualisation, also
    the per-md visualisation folder contents."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))

    def _params(self, visualize):
        return {
            "orig_caver_network_data_path": os.path.join(self._tmp, "internal"),
            "orig_caver_vis_path": os.path.join(self._tmp, "vis"),
            "visualize_transformed_tunnels": visualize,
        }

    def test_dump_only_when_visualization_disabled(self):
        params = self._params(visualize=False)
        paths = TunnelNetworksShardStage.derive_side_effect_paths(params, ["md1", "md2"])
        # exactly one dump per md_label, no viz files
        self.assertEqual([
            os.path.join(params["orig_caver_network_data_path"], "md1_caver.dump"),
            os.path.join(params["orig_caver_network_data_path"], "md2_caver.dump"),
        ], paths)

    def test_dump_plus_viz_files_when_visualization_enabled(self):
        params = self._params(visualize=True)
        # create the viz folder + one CGO file per md so the glob finds them
        for md in ("md1", "md2"):
            md_vis = os.path.join(params["orig_caver_vis_path"], md)
            os.makedirs(md_vis)
            for fname in ("{}_C_trans_rot.pdb".format(md), "view_network.py",
                          "cls_001_cgo.dump.gz", "cls_002_cgo.dump.gz"):
                open(os.path.join(md_vis, fname), "w").close()
        paths = TunnelNetworksShardStage.derive_side_effect_paths(params, ["md1", "md2"])
        # dump per md + transformed PDB + view_network.py + every *_cgo.dump.gz file
        self.assertIn(os.path.join(params["orig_caver_network_data_path"], "md1_caver.dump"),
                      paths)
        self.assertIn(os.path.join(params["orig_caver_vis_path"], "md1",
                                    "md1_C_trans_rot.pdb"), paths)
        self.assertIn(os.path.join(params["orig_caver_vis_path"], "md1", "view_network.py"),
                      paths)
        self.assertIn(os.path.join(params["orig_caver_vis_path"], "md1",
                                    "cls_001_cgo.dump.gz"), paths)
        self.assertIn(os.path.join(params["orig_caver_vis_path"], "md2",
                                    "cls_002_cgo.dump.gz"), paths)

    def test_empty_md_list_yields_empty_paths(self):
        paths = TunnelNetworksShardStage.derive_side_effect_paths(
            self._params(visualize=True), [])
        self.assertEqual([], paths)


class TestAquaductNetworksShardStageFolderLayout(unittest.TestCase):
    """AquaductNetworksShardStage publishes its folder under slurm_root_folder."""

    def test_folder_name_uses_stage_number(self):
        # stage 7 in the tt_engine.py workflow (process_aquaduct_networks)
        self.assertEqual("stage07_aquaduct_networks",
                         AquaductNetworksShardStage.folder_name)

    def test_get_slurm_folder_joins_root_and_folder_name(self):
        path = AquaductNetworksShardStage.get_slurm_folder(
            {"slurm_root_folder": "/scratch/run/_slurm"})
        self.assertEqual("/scratch/run/_slurm/stage07_aquaduct_networks", path)

    def test_shard_result_path_is_pickle_extension(self):
        # aquaduct-networks shard result is a tiny pickled list of processed md_labels
        path = AquaductNetworksShardStage.shard_result_path("/tmp/folder", 3)
        self.assertEqual("/tmp/folder/shard_result_00003.pkl", path)

    def test_does_not_collide_with_tunnel_networks_folder(self):
        # both per-MD network-preparation stages live under the same slurm_root_folder;
        # their per-stage subfolders must be distinct so per-shard artefacts cannot clobber
        # each other
        root = {"slurm_root_folder": "/scratch/run/_slurm"}
        self.assertNotEqual(
            AquaductNetworksShardStage.get_slurm_folder(root),
            TunnelNetworksShardStage.get_slurm_folder(root))


class TestAquaductNetworksShardStageNumShards(unittest.TestCase):
    """num_shards() honors user override, caps at item count, auto-derives at 50/shard."""

    def _params(self, slurm_num_shards=None, slurm_array_parallelism=None):
        return {
            "slurm_num_shards": slurm_num_shards,
            "slurm_array_parallelism": slurm_array_parallelism,
        }

    def test_empty_items_yields_single_shard(self):
        stage = AquaductNetworksShardStage([])
        self.assertEqual(1, stage.num_shards(self._params()))

    def test_user_count_capped_at_item_count(self):
        # 5 md_labels, asking for 100 shards is silly - cap to 5
        stage = AquaductNetworksShardStage(["md{}".format(i) for i in range(5)])
        self.assertEqual(5, stage.num_shards(self._params(slurm_num_shards=100)))

    def test_auto_derives_at_50_per_shard(self):
        # 150 md_labels / 50 per shard => 3 shards (matches tunnel-networks granularity:
        # per-MD work is moderate, not highly variable)
        stage = AquaductNetworksShardStage(["md{}".format(i) for i in range(150)])
        self.assertEqual(3, stage.num_shards(self._params()))

    def test_array_parallelism_caps_auto_derived(self):
        # 1000 items / 50 per shard would be 20 shards; cap to 2
        stage = AquaductNetworksShardStage(["md{}".format(i) for i in range(1000)])
        self.assertEqual(2, stage.num_shards(self._params(slurm_array_parallelism=2)))


class TestAquaductNetworksShardStageFingerprint(unittest.TestCase):
    """Fingerprint identifies the parameters AND the md_label enumeration."""

    def _params(self, **overrides):
        params = {
            "aquaduct_results_folder_pattern": "*",
            "aquaduct_results_pdb_filename": "ref.pdb",
            "aquaduct_results_relative_tarfile": "6_visualize_results.tar.gz",
            "aquaduct_traced_residues_filter": None,
            "event_min_distance": 1.0,
            "aquaduct_allow_empty_folders": False,
            "visualize_transformed_transport_events": False,
        }
        params.update(overrides)
        return params

    def test_fingerprint_includes_items_hash_and_count(self):
        stage = AquaductNetworksShardStage(["md1", "md2", "md3"])
        info = stage.fingerprint(self._params(), num_shards=2)
        self.assertEqual(3, info["num_items"])
        self.assertEqual(64, len(info["items_hash"]))  # sha256 hex digest

    def test_fingerprint_changes_when_items_differ(self):
        stage1 = AquaductNetworksShardStage(["md1", "md2"])
        stage2 = AquaductNetworksShardStage(["md1", "md3"])
        self.assertNotEqual(stage1.fingerprint(self._params(), num_shards=2),
                            stage2.fingerprint(self._params(), num_shards=2))

    def test_fingerprint_changes_when_pattern_differs(self):
        stage = AquaductNetworksShardStage(["md1"])
        fp1 = stage.fingerprint(self._params(aquaduct_results_folder_pattern="*"),
                                 num_shards=1)
        fp2 = stage.fingerprint(self._params(aquaduct_results_folder_pattern="md*"),
                                 num_shards=1)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_changes_when_event_min_distance_differs(self):
        stage = AquaductNetworksShardStage(["md1"])
        fp1 = stage.fingerprint(self._params(event_min_distance=1.0), num_shards=1)
        fp2 = stage.fingerprint(self._params(event_min_distance=2.0), num_shards=1)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_changes_when_visualize_flag_flips(self):
        # flipping visualize_transformed_transport_events must invalidate the cache so a
        # False->True transition does not silently re-use a cache that has no visualisation
        # side-effects backed up
        stage = AquaductNetworksShardStage(["md1"])
        fp1 = stage.fingerprint(
            self._params(visualize_transformed_transport_events=False), num_shards=1)
        fp2 = stage.fingerprint(
            self._params(visualize_transformed_transport_events=True), num_shards=1)
        self.assertNotEqual(fp1, fp2)


class TestAquaductNetworksShardStateRoundTrip(unittest.TestCase):
    """prepare_state writes items.json; load_items reads it back unchanged."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))

    def test_prepare_state_writes_items_json(self):
        items = ["md_alpha", "md_beta", "md_gamma"]
        stage = AquaductNetworksShardStage(items)
        stage.prepare_state({}, self._tmp)
        items_file = os.path.join(self._tmp, "items.json")
        self.assertTrue(os.path.exists(items_file))

    def test_load_items_returns_canonicalised_list(self):
        items = ["md_alpha", "md_beta"]
        stage = AquaductNetworksShardStage(items)
        stage.prepare_state({}, self._tmp)
        loaded = AquaductNetworksShardStage.load_items(self._tmp)
        # JSON round-trip keeps the order; consumers iterate as `for md_label in loaded`
        self.assertEqual(["md_alpha", "md_beta"], loaded)

    def test_load_items_missing_raises_with_diagnostic(self):
        with self.assertRaises(FileNotFoundError) as cm:
            AquaductNetworksShardStage.load_items(self._tmp)
        # message must point at the missing file and explain who writes it
        self.assertIn("items.json", str(cm.exception))
        self.assertIn("prepare_state", str(cm.exception))

    def test_prepare_state_overwrites_stale_items_atomically(self):
        # writing must use a tmp+rename so a partially-written items.json is not visible
        AquaductNetworksShardStage(["md_a"]).prepare_state({}, self._tmp)
        AquaductNetworksShardStage(["md_b", "md_c"]).prepare_state({}, self._tmp)
        self.assertEqual(["md_b", "md_c"],
                         AquaductNetworksShardStage.load_items(self._tmp))


class TestAquaductNetworksShardSideEffectPaths(unittest.TestCase):
    """derive_side_effect_paths reports the per-md_label outputs the worker is expected to
    write. Without visualisation, only the `<md>_aqua.dump` files; with visualisation, also
    the per-md visualisation folder contents (transformed PDB + view_network.py + every
    `*_cgo.dump.gz`)."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))

    def _params(self, visualize):
        return {
            "orig_aquaduct_network_data_path": os.path.join(self._tmp, "internal"),
            "orig_aquaduct_vis_path": os.path.join(self._tmp, "vis"),
            "visualize_transformed_transport_events": visualize,
        }

    def test_dump_only_when_visualization_disabled(self):
        params = self._params(visualize=False)
        paths = AquaductNetworksShardStage.derive_side_effect_paths(params, ["md1", "md2"])
        # exactly one dump per md_label, no viz files
        self.assertEqual([
            os.path.join(params["orig_aquaduct_network_data_path"], "md1_aqua.dump"),
            os.path.join(params["orig_aquaduct_network_data_path"], "md2_aqua.dump"),
        ], paths)

    def test_dump_plus_viz_files_when_visualization_enabled(self):
        params = self._params(visualize=True)
        # create the viz folder + one CGO file per md so the glob finds them
        for md in ("md1", "md2"):
            md_vis = os.path.join(params["orig_aquaduct_vis_path"], md)
            os.makedirs(md_vis)
            for fname in ("{}_A_trans_rot.pdb".format(md), "view_network.py",
                          "thr_85_release_cgo.dump.gz", "thr_85_entry_cgo.dump.gz"):
                open(os.path.join(md_vis, fname), "w").close()
        paths = AquaductNetworksShardStage.derive_side_effect_paths(params, ["md1", "md2"])
        # dump per md + transformed PDB + view_network.py + every *_cgo.dump.gz file
        self.assertIn(os.path.join(params["orig_aquaduct_network_data_path"], "md1_aqua.dump"),
                      paths)
        self.assertIn(os.path.join(params["orig_aquaduct_vis_path"], "md1",
                                    "md1_A_trans_rot.pdb"), paths)
        self.assertIn(os.path.join(params["orig_aquaduct_vis_path"], "md1", "view_network.py"),
                      paths)
        self.assertIn(os.path.join(params["orig_aquaduct_vis_path"], "md1",
                                    "thr_85_release_cgo.dump.gz"), paths)
        self.assertIn(os.path.join(params["orig_aquaduct_vis_path"], "md2",
                                    "thr_85_entry_cgo.dump.gz"), paths)

    def test_empty_md_list_yields_empty_paths(self):
        paths = AquaductNetworksShardStage.derive_side_effect_paths(
            self._params(visualize=True), [])
        self.assertEqual([], paths)


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
            AquaductNetworksShardStage.get_slurm_folder(root),
            TunnelLayeringShardStage.get_slurm_folder(root),
            TunnelNetworksShardStage.get_slurm_folder(root),
            DistanceShardStage.get_slurm_folder(root),
        }
        self.assertEqual(6, len(folders))


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


class TestCleanupStageFolder(unittest.TestCase):
    """`cleanup_stage_folder()` discards the per-stage SLURM tree after a successful
    assembly, gated by the `slurm_keep_shard_results` parameter (default False = clean)."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))
        # use a real stage class so get_slurm_folder + folder_name behave the same as in
        # production; the distance stage is the simplest (no items list, fingerprint_keys
        # only needs num_clusters)
        self._stage = DistanceShardStage(num_clusters=2)
        self._slurm_folder = self._stage.get_slurm_folder(
            {"slurm_root_folder": self._tmp})
        os.makedirs(self._slurm_folder)
        # populate the folder with a representative mix of artefacts the launcher creates:
        # per-shard result, manifest companion, side-effects archive, items.json,
        # partial_shards_info.json, and a fake submitit log file
        self._artefacts = [
            os.path.join(self._slurm_folder, "shard_result_00000.npy"),
            os.path.join(self._slurm_folder, "shard_result_00000.manifest.json"),
            os.path.join(self._slurm_folder, "shard_result_00000.side_effects.tar.gz"),
            os.path.join(self._slurm_folder, "items.json"),
            os.path.join(self._slurm_folder, "partial_shards_info.json"),
            os.path.join(self._slurm_folder, "12345_0_log.out"),
        ]
        for path in self._artefacts:
            open(path, "wb").close()

    def test_removes_stage_folder_when_keep_is_false(self):
        # default behaviour: the whole stage folder is wiped on success
        cleanup_stage_folder(self._stage,
                              {"slurm_root_folder": self._tmp,
                               "slurm_keep_shard_results": False})
        self.assertFalse(os.path.exists(self._slurm_folder))

    def test_keeps_folder_when_keep_is_true(self):
        cleanup_stage_folder(self._stage,
                              {"slurm_root_folder": self._tmp,
                               "slurm_keep_shard_results": True})
        self.assertTrue(os.path.isdir(self._slurm_folder))
        # every artefact must still be present - the helper is a no-op in this mode
        for path in self._artefacts:
            self.assertTrue(os.path.exists(path), path)

    def test_default_is_clean(self):
        # the helper falls back to keep=False when the key is missing, matching the config
        # default (we don't want a typo in the parameters dict to silently keep gigabytes of
        # cache around)
        cleanup_stage_folder(self._stage, {"slurm_root_folder": self._tmp})
        self.assertFalse(os.path.exists(self._slurm_folder))

    def test_is_idempotent_when_folder_missing(self):
        # second cleanup call after the folder is already gone must not raise
        cleanup_stage_folder(self._stage,
                              {"slurm_root_folder": self._tmp,
                               "slurm_keep_shard_results": False})
        cleanup_stage_folder(self._stage,
                              {"slurm_root_folder": self._tmp,
                               "slurm_keep_shard_results": False})  # would raise if not idempotent

    def test_does_not_touch_sibling_stage_folders(self):
        # cleanup is scoped to this stage's get_slurm_folder() only; another stage's folder
        # under the same slurm_root_folder must survive
        sibling = TunnelNetworksShardStage(["md1"]).get_slurm_folder(
            {"slurm_root_folder": self._tmp})
        os.makedirs(sibling)
        sibling_file = os.path.join(sibling, "shard_result_00000.pkl")
        open(sibling_file, "wb").close()

        cleanup_stage_folder(self._stage,
                              {"slurm_root_folder": self._tmp,
                               "slurm_keep_shard_results": False})

        self.assertFalse(os.path.exists(self._slurm_folder))
        self.assertTrue(os.path.exists(sibling_file))


class TestEnsureCpuBindNone(unittest.TestCase):
    """`_ensure_cpu_bind_none()` guarantees the --cpu-bind=none workaround stays in
    slurm_srun_args even when the user overrides the knob, but honours an explicit
    --cpu-bind=... they supply themselves."""

    def test_empty_list_adds_workaround(self):
        # the default path (user did not override slurm_srun_args; config.py left it []):
        # the workaround must still be present so a recursive SLURM run does not break
        self.assertEqual(["--cpu-bind=none"], _ensure_cpu_bind_none([]))

    def test_prepends_to_user_args(self):
        # user added --mpi=pmix; workaround is prepended, user's args preserved in order
        self.assertEqual(["--cpu-bind=none", "--mpi=pmix"],
                         _ensure_cpu_bind_none(["--mpi=pmix"]))

    def test_does_not_duplicate_when_already_present(self):
        # user explicitly asked for --cpu-bind=none; no double-insertion
        self.assertEqual(["--cpu-bind=none"], _ensure_cpu_bind_none(["--cpu-bind=none"]))

    def test_honours_explicit_user_override(self):
        # user explicitly asked for --cpu-bind=cores; we do NOT overwrite their choice
        # with --cpu-bind=none. The recursive-SLURM workaround is a default, not a hard
        # requirement when the user knows what they're doing.
        self.assertEqual(["--cpu-bind=cores", "--mpi=pmix"],
                         _ensure_cpu_bind_none(["--cpu-bind=cores", "--mpi=pmix"]))

    def test_returns_new_list(self):
        # helper must not mutate the caller's input - the parameters dict is shared
        # between stage calls and an in-place append would silently corrupt later calls
        original = ["--mpi=pmix"]
        result = _ensure_cpu_bind_none(original)
        self.assertEqual(["--mpi=pmix"], original)
        self.assertIsNot(original, result)


class TestParseSlurmTimeLimitMin(unittest.TestCase):
    """SLURM TIMELIMIT-format parsing: minutes / mm:ss / hh:mm:ss / days-hh:mm:ss."""

    def test_plain_minutes(self):
        self.assertEqual(120, _parse_slurm_time_limit_min("120"))

    def test_hours_minutes_seconds(self):
        self.assertEqual(270, _parse_slurm_time_limit_min("4:30:00"))

    def test_days_hours_minutes_seconds(self):
        # 1 day + 12 hours = 24*60 + 12*60 = 2160 min
        self.assertEqual(2160, _parse_slurm_time_limit_min("1-12:00:00"))

    def test_minutes_seconds(self):
        # 90 minutes 0 seconds -> 90 min
        self.assertEqual(90, _parse_slurm_time_limit_min("90:00"))

    def test_unparseable_returns_none(self):
        # any garbage -> None so callers can degrade gracefully
        self.assertIsNone(_parse_slurm_time_limit_min("not-a-time"))
        self.assertIsNone(_parse_slurm_time_limit_min(""))
        self.assertIsNone(_parse_slurm_time_limit_min("1:2:3:4"))


class TestDetectParentSlurmAllocation(unittest.TestCase):
    """Env-var-based detection of a parent SLURM allocation (recursive-SLURM case)."""

    def setUp(self):
        # snapshot the env so the tests don't leak SLURM_* settings between cases
        self._saved = {k: os.environ.pop(k, None)
                       for k in ("SLURM_JOB_ID", "SLURM_JOBID",
                                 "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE",
                                 "SLURM_TIMELIMIT", "SBATCH_TIMELIMIT")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]

    def test_returns_none_when_not_inside_slurm(self):
        self.assertIsNone(_detect_parent_slurm_allocation())

    def test_detects_via_slurm_job_id(self):
        os.environ["SLURM_JOB_ID"] = "12345"
        info = _detect_parent_slurm_allocation()
        self.assertIsNotNone(info)
        self.assertEqual("12345", info["job_id"])

    def test_detects_via_legacy_slurm_jobid(self):
        # older SLURM versions export SLURM_JOBID (no underscore); we accept both
        os.environ["SLURM_JOBID"] = "98765"
        info = _detect_parent_slurm_allocation()
        self.assertEqual("98765", info["job_id"])

    def test_parses_cpu_count_from_slurm_cpus_on_node(self):
        os.environ["SLURM_JOB_ID"] = "12345"
        os.environ["SLURM_CPUS_ON_NODE"] = "16"
        self.assertEqual(16, _detect_parent_slurm_allocation()["cpus"])

    def test_parses_compressed_cpu_count(self):
        # SLURM_JOB_CPUS_PER_NODE on multi-node allocations uses compressed form
        os.environ["SLURM_JOB_ID"] = "12345"
        os.environ["SLURM_JOB_CPUS_PER_NODE"] = "16(x2)"
        self.assertEqual(16, _detect_parent_slurm_allocation()["cpus"])

    def test_unparseable_cpu_count_yields_none(self):
        os.environ["SLURM_JOB_ID"] = "12345"
        os.environ["SLURM_CPUS_ON_NODE"] = "garbage"
        self.assertIsNone(_detect_parent_slurm_allocation()["cpus"])

    def test_parses_time_limit(self):
        os.environ["SLURM_JOB_ID"] = "12345"
        os.environ["SLURM_TIMELIMIT"] = "4:30:00"
        self.assertEqual(270, _detect_parent_slurm_allocation()["time_min"])


class TestWarnOnDriverMisallocation(unittest.TestCase):
    """The two heuristic warnings emitted when the recursive driver looks under-allocated
    for the array it's about to submit. Both are warn-only (the launcher proceeds either
    way) - users can know what they're doing."""

    def _capture_warnings(self, parent_info, parameters, num_shards):
        import logging
        from transport_tools.libs.slurm import logger as slurm_logger

        captured: list = []
        class _Capture(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    captured.append(record.getMessage())

        handler = _Capture()
        slurm_logger.addHandler(handler)
        try:
            _warn_on_driver_misallocation(parent_info, parameters, num_shards, "test-stage")
        finally:
            slurm_logger.removeHandler(handler)
        return captured

    def test_warns_when_driver_walltime_shorter_than_array(self):
        warnings = self._capture_warnings(
            {"job_id": "12345", "cpus": 4, "time_min": 30},
            {"slurm_timeout_min": 240}, num_shards=4)
        self.assertTrue(any("wall-time limit" in w for w in warnings))
        self.assertTrue(any("12345" in w for w in warnings))

    def test_no_walltime_warning_when_driver_outlasts_array(self):
        warnings = self._capture_warnings(
            {"job_id": "12345", "cpus": 4, "time_min": 480},
            {"slurm_timeout_min": 240}, num_shards=4)
        self.assertFalse(any("wall-time limit" in w for w in warnings))

    def test_no_walltime_warning_when_parent_time_unknown(self):
        # parent time_min is None (not in env / unparseable) -> don't false-positive on
        # every plain salloc
        warnings = self._capture_warnings(
            {"job_id": "12345", "cpus": 4, "time_min": None},
            {"slurm_timeout_min": 240}, num_shards=4)
        self.assertFalse(any("wall-time limit" in w for w in warnings))

    def test_warns_on_single_cpu_with_large_array(self):
        warnings = self._capture_warnings(
            {"job_id": "12345", "cpus": 1, "time_min": 480},
            {"slurm_timeout_min": 240}, num_shards=64)
        self.assertTrue(any("1 CPU" in w for w in warnings))

    def test_no_cpu_warning_for_small_array(self):
        # single-core driver is fine for a handful of shards - the threshold is 16
        warnings = self._capture_warnings(
            {"job_id": "12345", "cpus": 1, "time_min": 480},
            {"slurm_timeout_min": 240}, num_shards=4)
        self.assertFalse(any("1 CPU" in w for w in warnings))

    def test_no_cpu_warning_when_multi_cpu(self):
        # >1 CPU driver: polling overhead is fine even for large arrays
        warnings = self._capture_warnings(
            {"job_id": "12345", "cpus": 4, "time_min": 480},
            {"slurm_timeout_min": 240}, num_shards=1000)
        self.assertFalse(any("1 CPU" in w for w in warnings))


class _FakeJob:
    """Minimal stand-in for `submitit.Job` exposing only the surface `_categorize_failure`,
    `_safe_job_state`, and `_job_stderr_path` read: `state`, `job_id`, `paths.stderr`. The
    real Job object pulls state via sacct and is otherwise untestable in a unit context;
    this stub lets us drive every failure category from a deterministic input."""

    def __init__(self, state=None, job_id="999", stderr_path=None,
                 state_raises=False):
        self._state = state
        self.job_id = job_id
        self._state_raises = state_raises
        if stderr_path is not None:
            self.paths = type("P", (), {"stderr": stderr_path})()
        else:
            self.paths = type("P", (), {"stderr": None})()

    @property
    def state(self):
        if self._state_raises:
            raise RuntimeError("sacct hiccup")
        return self._state


class TestCategorizeFailure(unittest.TestCase):
    """`_categorize_failure()` maps SLURM state -> (category, remediation_hint) for the
    failure-summary reporter. Falls back to a stderr scan for OOM markers when SLURM state
    is just 'FAILED' (cgroup OOM not always reflected in state); falls back to 'error' with
    a 'inspect stderr' hint otherwise."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self._tmp, ignore_errors=True))

    def test_timeout_state(self):
        category, hint = _categorize_failure(_FakeJob(state="TIMEOUT"),
                                              RuntimeError("ignored"))
        self.assertEqual("timeout", category)
        self.assertIn("slurm_timeout_min", hint)

    def test_oom_state(self):
        category, hint = _categorize_failure(_FakeJob(state="OUT_OF_MEMORY"),
                                              RuntimeError("ignored"))
        self.assertEqual("oom", category)
        self.assertIn("slurm_mem_gb", hint)

    def test_node_failure_state(self):
        category, _ = _categorize_failure(_FakeJob(state="NODE_FAIL"),
                                           RuntimeError("ignored"))
        self.assertEqual("node_failure", category)

    def test_cancelled_state(self):
        category, _ = _categorize_failure(_FakeJob(state="CANCELLED"),
                                           RuntimeError("ignored"))
        self.assertEqual("cancelled", category)

    def test_preempted_state(self):
        category, _ = _categorize_failure(_FakeJob(state="PREEMPTED"),
                                           RuntimeError("ignored"))
        self.assertEqual("preempted", category)

    def test_stderr_oom_marker_when_state_is_just_failed(self):
        # SLURM reported FAILED with no further info; stderr has the kernel OOM marker -
        # the helper must still classify this as oom rather than generic error
        stderr_path = os.path.join(self._tmp, "stderr")
        with open(stderr_path, "w") as out:
            out.write("worker started\nallocating ...\nKilled\n")
        category, hint = _categorize_failure(
            _FakeJob(state="FAILED", stderr_path=stderr_path),
            RuntimeError("subprocess died"))
        self.assertEqual("oom", category)
        self.assertIn("Killed", hint)

    def test_generic_error_fallback(self):
        # SLURM state FAILED, stderr clean -> generic 'error' with 'inspect stderr' hint
        stderr_path = os.path.join(self._tmp, "stderr")
        with open(stderr_path, "w") as out:
            out.write("Traceback (most recent call last):\nValueError: bad input\n")
        category, hint = _categorize_failure(
            _FakeJob(state="FAILED", stderr_path=stderr_path),
            ValueError("bad input"))
        self.assertEqual("error", category)
        self.assertIn("stderr", hint)

    def test_safe_when_state_unreadable(self):
        # transient sacct hiccup -> _safe_job_state returns None -> we fall through to the
        # stderr scan / generic-error path without raising
        category, _ = _categorize_failure(_FakeJob(state_raises=True),
                                           RuntimeError("dunno"))
        self.assertEqual("error", category)


class TestPercentile(unittest.TestCase):
    """Pure-Python percentile helper, used by the reporter. Matches numpy's linear-interp
    default so users see numbers consistent with what they'd compute outside the framework."""

    def test_empty_yields_zero(self):
        # empty list -> 0.0 (the caller uses this in a log line; raising would mask the
        # primary stats output)
        self.assertEqual(0.0, _percentile([], 95))

    def test_single_value(self):
        self.assertEqual(42.0, _percentile([42.0], 50))
        self.assertEqual(42.0, _percentile([42.0], 95))

    def test_median_of_odd_count(self):
        self.assertEqual(3.0, _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50))

    def test_p95_linear_interp(self):
        # 5 values, 95th percentile interpolates between index 3.8 in [1, 2, 3, 4, 5]
        # = 4 * 0.2 + 5 * 0.8 = 4.8. We don't assert exact float equality, just close.
        self.assertAlmostEqual(4.8, _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95), places=2)


class _LogCapture:
    """unittest helper: capture every log record emitted on the slurm logger as a list of
    formatted message strings. Used by the reporter tests to assert on the lines without
    relying on assertLogs (which is awkward to compose when multiple log levels are emitted
    from a single helper call). Resets the logger level to INFO for the duration of the
    capture so default-WARNING setups still see the reporter's INFO output."""

    def __init__(self):
        import logging
        from transport_tools.libs.slurm import logger as slurm_logger
        self._logger = slurm_logger
        self._messages: list = []
        self._saved_level: Optional[int] = None

        class _Handler(logging.Handler):
            def emit(_inner_self, record):
                self._messages.append((record.levelno, record.getMessage()))

        self._handler = _Handler()
        self._handler.setLevel(logging.DEBUG)

    def __enter__(self):
        import logging
        self._saved_level = self._logger.level
        self._logger.setLevel(logging.DEBUG)
        self._logger.addHandler(self._handler)
        return self._messages

    def __exit__(self, *_exc):
        self._logger.removeHandler(self._handler)
        if self._saved_level is not None:
            self._logger.setLevel(self._saved_level)


def _make_outcome(shard_id, **overrides):
    """Build a default-success per-shard outcome dict matching the shape that `_poll_jobs`
    produces, then apply overrides. Used by reporter tests."""

    outcome = {
        "shard_id": shard_id, "job_id": str(shard_id),
        "state": "COMPLETED", "start_mono": 100.0, "end_mono": 200.0,
        "category": "success", "stderr_path": None, "exception": None,
    }
    outcome.update(overrides)
    return outcome


class TestReportRunStatsCategorisation(unittest.TestCase):
    """`_report_run_stats()` emits one warning line per non-empty failure category. All
    successful shards produce no failure log; categories are listed in sorted order with
    affected shard IDs."""

    def _params(self):
        return {"slurm_poll_wait_seconds": 30, "slurm_array_parallelism": None}

    def test_all_success_no_failure_log(self):
        outcomes = {sid: _make_outcome(sid) for sid in range(3)}
        with _LogCapture() as logs:
            _report_run_stats(outcomes, num_shards=3, parameters=self._params(),
                              stage=DistanceShardStage(num_clusters=4))
        warnings = [m for (lvl, m) in logs if lvl >= 30]
        self.assertFalse(any("ended as" in w for w in warnings))

    def test_one_oom_logs_remediation_hint(self):
        outcomes = {
            0: _make_outcome(0),
            1: _make_outcome(1, category="oom", state="OUT_OF_MEMORY"),
        }
        with _LogCapture() as logs:
            _report_run_stats(outcomes, num_shards=2, parameters=self._params(),
                              stage=DistanceShardStage(num_clusters=4))
        warnings = [m for (lvl, m) in logs if lvl >= 30]
        self.assertTrue(any("oom" in w and "slurm_mem_gb" in w for w in warnings))

    def test_groups_failures_per_category(self):
        outcomes = {
            0: _make_outcome(0, category="timeout", state="TIMEOUT"),
            1: _make_outcome(1, category="timeout", state="TIMEOUT"),
            2: _make_outcome(2, category="oom", state="OUT_OF_MEMORY"),
        }
        with _LogCapture() as logs:
            _emit_failure_categorisation(outcomes, "test-stage")
        warnings = [m for (lvl, m) in logs if lvl >= 30]
        # one line per category, sorted: oom < timeout
        self.assertEqual(2, len(warnings))
        self.assertIn("oom", warnings[0])
        self.assertIn("[2]", warnings[0])  # shard id
        self.assertIn("timeout", warnings[1])
        self.assertIn("[0, 1]", warnings[1])


class TestReportRunStatsTimings(unittest.TestCase):
    """`_emit_runtime_statistics()` emits per-shard median / p95 / max / skew, total array
    wall, and parallel efficiency. Skew > 3 + efficiency < 50% trigger tuning hints."""

    def _params(self, slurm_array_parallelism=None):
        return {"slurm_poll_wait_seconds": 30,
                "slurm_array_parallelism": slurm_array_parallelism}

    def test_balanced_workload_emits_stats_without_tuning_hints(self):
        # 4 shards, ~equal wall time, run concurrently -> efficiency ~100%, skew ~1
        outcomes = {sid: _make_outcome(sid, start_mono=100.0, end_mono=200.0)
                    for sid in range(4)}
        with _LogCapture() as logs:
            _emit_runtime_statistics(outcomes, num_shards=4, parameters=self._params(),
                                      stage=DistanceShardStage(num_clusters=4))
        info = [m for (lvl, m) in logs if lvl == 20]
        self.assertTrue(any("median" in m and "skew" in m for m in info))
        # neither tuning hint should fire on a balanced workload
        self.assertFalse(any("strided split is unbalanced" in m for m in info))
        self.assertFalse(any("queue / scheduling overhead" in m for m in info))

    def test_skewed_workload_triggers_target_items_hint(self):
        # 3 fast shards + 1 slow shard -> skew > 3, hint must fire
        outcomes = {
            0: _make_outcome(0, start_mono=100.0, end_mono=110.0),
            1: _make_outcome(1, start_mono=100.0, end_mono=110.0),
            2: _make_outcome(2, start_mono=100.0, end_mono=110.0),
            3: _make_outcome(3, start_mono=100.0, end_mono=1100.0),
        }
        with _LogCapture() as logs:
            _emit_runtime_statistics(outcomes, num_shards=4, parameters=self._params(),
                                      stage=DistanceShardStage(num_clusters=4))
        info = [m for (lvl, m) in logs if lvl == 20]
        self.assertTrue(any("strided split is unbalanced" in m for m in info))
        self.assertTrue(any("target_items_per_shard" in m for m in info))

    def test_low_efficiency_triggers_array_parallelism_hint(self):
        # 4 shards that run sequentially (start-end staggered) -> total wall = 4 × per-shard,
        # efficiency ~ 25%, hint must fire and mention the current slurm_array_parallelism
        outcomes = {
            0: _make_outcome(0, start_mono=100.0, end_mono=200.0),
            1: _make_outcome(1, start_mono=200.0, end_mono=300.0),
            2: _make_outcome(2, start_mono=300.0, end_mono=400.0),
            3: _make_outcome(3, start_mono=400.0, end_mono=500.0),
        }
        with _LogCapture() as logs:
            _emit_runtime_statistics(outcomes, num_shards=4,
                                      parameters=self._params(slurm_array_parallelism=10),
                                      stage=DistanceShardStage(num_clusters=4))
        info = [m for (lvl, m) in logs if lvl == 20]
        self.assertTrue(any("queue / scheduling overhead" in m for m in info))
        self.assertTrue(any("slurm_array_parallelism (currently 10)" in m for m in info))

    def test_no_timing_data_logs_fallback_message(self):
        # All shards completed between poll cycles (no observed RUNNING -> done) so we have
        # no timing data; the helper must still emit a coherent log line rather than crash
        outcomes = {sid: _make_outcome(sid, start_mono=None, end_mono=200.0)
                    for sid in range(2)}
        with _LogCapture() as logs:
            _emit_runtime_statistics(outcomes, num_shards=2, parameters=self._params(),
                                      stage=DistanceShardStage(num_clusters=4))
        info = [m for (lvl, m) in logs if lvl == 20]
        self.assertTrue(any("no shard ran long enough" in m for m in info))

    def test_internal_error_does_not_raise(self):
        # The reporter must never let a formatting bug fail a successful stage. Drive it
        # with a malformed outcome that triggers an exception inside the emitter and assert
        # only a warning is logged.
        outcomes = {0: {"shard_id": 0, "job_id": "x",
                         "state": "COMPLETED",
                         "start_mono": "not-a-number",  # type mismatch -> arithmetic raises
                         "end_mono": "still-not-a-number",
                         "category": "success", "stderr_path": None, "exception": None}}
        with _LogCapture() as logs:
            _report_run_stats(outcomes, num_shards=1, parameters=self._params(),
                              stage=DistanceShardStage(num_clusters=4))
        warnings = [m for (lvl, m) in logs if lvl >= 30]
        self.assertTrue(any("post-run reporter failed" in w for w in warnings))


class TestExecutorParamsCpuBindGuard(unittest.TestCase):
    """`SlurmShardStage.executor_params()` wires _ensure_cpu_bind_none + the optional
    slurm_additional_parameters passthrough into the dict it returns to submitit."""

    def _params(self, **overrides):
        params = {
            "slurm_timeout_min": 240, "slurm_cpus_per_task": 4, "slurm_mem_gb": 8.0,
            "slurm_partition": None, "slurm_account": None,
            "slurm_array_parallelism": None, "slurm_setup": None,
            "slurm_srun_args": None, "slurm_additional_parameters": None,
        }
        params.update(overrides)
        return params

    def test_default_srun_args_has_cpu_bind_none(self):
        # user did not set slurm_srun_args -> default workaround is applied
        stage = DistanceShardStage(num_clusters=2)
        ep = stage.executor_params(self._params())
        self.assertEqual(["--cpu-bind=none"], ep["slurm_srun_args"])

    def test_user_srun_args_get_cpu_bind_prepended(self):
        # user added other args; workaround must still be present
        stage = DistanceShardStage(num_clusters=2)
        ep = stage.executor_params(self._params(slurm_srun_args=["--mpi=pmix"]))
        self.assertEqual(["--cpu-bind=none", "--mpi=pmix"], ep["slurm_srun_args"])

    def test_user_explicit_cpu_bind_is_honoured(self):
        # explicit user override wins
        stage = DistanceShardStage(num_clusters=2)
        ep = stage.executor_params(
            self._params(slurm_srun_args=["--cpu-bind=cores"]))
        self.assertEqual(["--cpu-bind=cores"], ep["slurm_srun_args"])

    def test_additional_parameters_passthrough(self):
        # dict passed through verbatim to submitit's slurm_additional_parameters arg
        stage = DistanceShardStage(num_clusters=2)
        ep = stage.executor_params(self._params(
            slurm_additional_parameters={"qos": "high", "constraint": "icelake"}))
        self.assertEqual({"qos": "high", "constraint": "icelake"},
                         ep["slurm_additional_parameters"])

    def test_empty_additional_parameters_omitted(self):
        # an empty dict (or None) must not produce an empty slurm_additional_parameters
        # key in the kwargs - submitit would still echo it as an empty #SBATCH block
        stage = DistanceShardStage(num_clusters=2)
        ep = stage.executor_params(self._params(slurm_additional_parameters={}))
        self.assertNotIn("slurm_additional_parameters", ep)


if __name__ == "__main__":
    unittest.main()
