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

# Reusable framework for distributing TransportTools stages over SLURM array jobs. Each stage is
# implemented as a SlurmShardStage subclass that owns its per-shard work; the generic launcher
# run_stage_on_slurm() handles fingerprint-based resumability, missing-shard discovery, submitit
# submission, completion-order polling with per-shard runtime timeouts, and per-shard
# cancellation. Concrete adapters in this module:
#   * TunnelNetworksShardStage     - stage 2 (process_tunnel_networks)
#   * TunnelLayeringShardStage     - stage 3 (create_layered_description4tunnel_networks)
#   * DistanceShardStage           - stage 4 (compute_tunnel_clusters_distances)
#   * AquaductNetworksShardStage   - stage 7 (process_aquaduct_networks)
#   * AquaductLayeringShardStage   - stage 8 (create_layered_description4aquaduct_networks)
#   * EventAssignmentShardStage    - stage 9 (assign_transport_events)
#
# Submission is handled by the optional 'submitit' package (an extra dependency); it is imported
# lazily so that the rest of TransportTools installs and runs without it.

__version__ = '0.9.8'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import os
import json
import pickle
import shutil
import subprocess
import tarfile
import time
import hashlib
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
from logging import getLogger

logger = getLogger(__name__)

# default upper bound on the auto-derived number of shards, to keep the SLURM array reasonably
# sized; only applied when the user did not set slurm_array_parallelism (which then governs
# the shard count instead, regardless of whether it is larger or smaller than this default)
_DEFAULT_MAX_AUTO_SHARDS = 200

def submitit_available() -> bool:
    """
    Return True if the optional 'submitit' package is importable.
    """

    try:
        import submitit  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# srun-args + parent-allocation helpers (recursive SLURM workaround)
# ---------------------------------------------------------------------------
#
# When `tt_engine.py` is itself launched from inside a SLURM allocation (the most common
# production driver pattern) and then submits its own SLURM array, the parent job's
# `SLURM_CPU_BIND*` environment variables leak into the array job's sbatch script and make
# the nested `srun` fail with "CPU binding outside of job step allocation". The fix is to
# pass `--cpu-bind=none` to that nested srun. The default behaviour of `executor_params()`
# substitutes `['--cpu-bind=none']` when the user did not set `slurm_srun_args`, but when
# the user overrides the knob to add other args (`--mpi=pmix`, `--exact`, ...), we must
# still guarantee the workaround is present so a recursive run does not break.
# `_ensure_cpu_bind_none()` is what enforces that invariant.

_CPU_BIND_NONE = "--cpu-bind=none"


def _ensure_cpu_bind_none(srun_args: List[str]) -> List[str]:
    """
    Guarantee that `--cpu-bind=none` is present in the srun-args list, preserving the user's
    other args. If the user already supplied any form of `--cpu-bind=...` we leave their
    choice alone (so an explicit override is honoured even when it differs from `none`);
    otherwise we prepend the workaround. Always returns a new list so the caller's input is
    not mutated.

    :param srun_args: list of srun args (already parsed by config.py into a list); may be
                      empty but must not be None
    :return: srun args list with the `--cpu-bind=none` workaround guaranteed when no
             explicit `--cpu-bind=...` is already present
    """

    has_cpu_bind = any(str(arg).startswith("--cpu-bind") for arg in srun_args)
    if has_cpu_bind:
        return list(srun_args)
    # prepend rather than append so any later `--exact` / `--mpi=...` style args still come
    # in the canonical order users expect when reading sbatch logs
    return [_CPU_BIND_NONE, *srun_args]


def _detect_parent_slurm_allocation() -> Optional[dict]:
    """
    Detect whether the launcher itself is running inside a SLURM allocation (the recursive
    SLURM case: `tt_engine.py` was started by `sbatch driver.sh` or by `salloc --interactive`
    and is now about to submit its own array job). Returns a dict with the parent job's
    identifying details, or `None` when not running inside a SLURM allocation.

    The detection is purely env-var-based (no `scontrol` shell-out) so it is cheap and safe
    to call from every launcher invocation. Keys in the returned dict:
      * `job_id`   - parent `SLURM_JOB_ID` (always present when this returns non-None)
      * `cpus`     - parent CPU count parsed from `SLURM_CPUS_ON_NODE` /
                     `SLURM_JOB_CPUS_PER_NODE`, or `None` if unparseable
      * `time_min` - parent wall-time limit in minutes parsed from `SLURM_TIMELIMIT` /
                     `SBATCH_TIMELIMIT` formats `[days-]hours:minutes[:seconds]` or plain
                     minutes, or `None` if unparseable / absent
    The keys are minimal on purpose: the misallocation warning below only needs CPU count
    and time limit to fire; we don't try to enumerate every SLURM_* var.
    """

    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not job_id:
        return None

    info: dict = {"job_id": job_id, "cpus": None, "time_min": None}

    cpus_raw = (os.environ.get("SLURM_CPUS_ON_NODE")
                or os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
    if cpus_raw:
        # SLURM_JOB_CPUS_PER_NODE can be a compressed form like "16(x2)" on multi-node
        # allocations; take the leading integer.
        try:
            info["cpus"] = int(cpus_raw.split("(")[0].split(",")[0])
        except (ValueError, IndexError):
            pass

    time_raw = os.environ.get("SLURM_TIMELIMIT") or os.environ.get("SBATCH_TIMELIMIT")
    if time_raw:
        info["time_min"] = _parse_slurm_time_limit_min(time_raw)

    return info


def _parse_slurm_time_limit_min(value: str) -> Optional[int]:
    """
    Parse SLURM time-limit strings into minutes. Accepted forms (the same submitit / sbatch
    accept):
      * 'minutes'                       e.g. '120'         -> 120
      * 'minutes:seconds'               e.g. '90:00'       -> 90
      * 'hours:minutes:seconds'         e.g. '4:30:00'     -> 270
      * 'days-hours:minutes:seconds'    e.g. '1-12:00:00'  -> 2160
    Returns `None` for any unparseable value (so callers can degrade gracefully without
    raising).
    """

    value = value.strip()
    if not value:
        return None
    days_str = "0"
    rest = value
    if "-" in value:
        days_str, _, rest = value.partition("-")
    parts = rest.split(":")
    try:
        days = int(days_str)
        if len(parts) == 1:           # plain minutes
            return days * 24 * 60 + int(parts[0])
        if len(parts) == 2:           # mm:ss
            mm, ss = int(parts[0]), int(parts[1])
            return days * 24 * 60 + mm + (1 if ss else 0)
        if len(parts) == 3:           # hh:mm:ss
            hh, mm, ss = int(parts[0]), int(parts[1]), int(parts[2])
            return days * 24 * 60 + hh * 60 + mm + (1 if ss else 0)
    except ValueError:
        return None
    return None


def _warn_on_driver_misallocation(parent_info: dict, parameters: dict,
                                  num_shards: int, stage_name: str) -> None:
    """
    Emit a warning when the recursive driver's allocation looks too small to outlive the
    array it is about to submit. Two heuristic signals, both warn-only (the launcher does
    not refuse to submit - the user may know what they are doing):

      * **Tiny driver wall-time vs. array wall-time.** If the parent's `SLURM_TIMELIMIT` is
        smaller than the array's `slurm_timeout_min`, the driver will die before its array
        finishes and the user will need to manually `sacct` to retrieve results. No warning
        when the parent's wall-time is unparseable (we'd false-positive on every plain
        salloc).
      * **Single-CPU driver with a large array.** A driver running on 1 CPU has only enough
        memory / Python event-loop bandwidth to babysit a small number of pending jobs;
        polling a thousand-shard array from a single core can run the driver out of memory
        (submitit holds an in-memory Job object per shard). We warn when parent CPUs == 1
        AND `num_shards` >= 16 (a soft threshold; below this the polling overhead is
        negligible).
    """

    if parent_info.get("time_min") is not None \
            and parent_info["time_min"] < parameters["slurm_timeout_min"]:
        logger.warning(
            "Driver job %s has a wall-time limit of %d min, shorter than the SLURM %s "
            "array's per-shard limit of %d min. The driver will likely exit before the "
            "array finishes; results will still be on disk but the launcher won't be there "
            "to assemble them. Increase the driver's --time or reduce slurm_timeout_min.",
            parent_info["job_id"], parent_info["time_min"], stage_name,
            parameters["slurm_timeout_min"])

    if parent_info.get("cpus") == 1 and num_shards >= 16:
        logger.warning(
            "Driver job %s is allocated 1 CPU but is about to poll %d SLURM %s shard(s). "
            "Polling a large array from a single-core driver can run the driver out of "
            "memory (submitit holds an in-memory Job object per shard). Consider "
            "allocating the driver more CPUs / memory, or splitting the array into fewer "
            "larger shards via slurm_num_shards.",
            parent_info["job_id"], num_shards, stage_name)


def derive_num_shards(n_jobs: int, *, target_per_shard: int,
                      array_parallelism: Optional[int] = None,
                      max_auto_shards: int = _DEFAULT_MAX_AUTO_SHARDS) -> int:
    """
    Derive a sensible number of SLURM array tasks for a job, aiming for roughly
    `target_per_shard` work items per shard while keeping the array size bounded. Each stage's
    natural work-item unit is different (pairwise comparisons for distances, (md_label, cls_id)
    pairs for tunnel layering, ...), so `target_per_shard` is a required keyword-only argument -
    every caller picks the value that matches its workload.

    When `array_parallelism` is provided, it governs the cap directly (the user already expressed
    their preferred array size, and there is no benefit to deriving more shards than tasks that
    can run concurrently); otherwise the count falls back to `max_auto_shards`.
    :param n_jobs: total number of work items to perform
    :param target_per_shard: average number of work items per shard the heuristic aims for
    :param array_parallelism: user-configured maximum number of concurrent array tasks; when set,
                              it replaces `max_auto_shards` as the cap on the derived shard count
    :param max_auto_shards: fallback upper bound used when `array_parallelism` is None
    :return: number of shards to use
    """

    num_shards = -(-max(n_jobs, 1) // max(1, target_per_shard))  # ceiling division
    cap = array_parallelism if array_parallelism is not None else max_auto_shards
    return max(1, min(num_shards, cap))


# ---------------------------------------------------------------------------
# Framework: SlurmShardStage + run_stage_on_slurm
# ---------------------------------------------------------------------------


class SlurmShardStage(ABC):
    """
    Base class for a SLURM-distributed stage. A concrete stage owns:
      * its sharding scheme (`num_shards`),
      * the per-shard runner that runs on a SLURM compute node (`run_shard`),
      * the per-shard result file layout (`shard_result_path`),
      * the fingerprint that invalidates stale shard results when parameters change
        (`fingerprint`).
    The generic launcher `run_stage_on_slurm()` reuses the resumability, polling, and
    cancellation logic across every stage.

    Subclasses set:
      * `folder_name` - subdirectory under `parameters["slurm_root_folder"]` where this
        stage's per-shard result files and submitit artifacts live (e.g. "stage04_distances")
      * `stage_name` - short label used in log messages
      * `job_name` - passed to submitit as the SLURM job name
    For stages where the work units depend on runtime-derived state (e.g. the distance
    stage needs num_clusters), pass that state into the subclass constructor;
    `SlurmShardStage` holds no mutable state of its own beyond the fixed class attributes.

    Determinism contract. The default expectation for a new stage is that its output is
    **bit-identical to the local backend** across any `slurm_num_shards` choice: workers
    compute their assigned items independently, and the parent assembles them in a
    deterministic order (either a fixed scatter into a pre-allocated array, or an
    in-submission-order commit driven by the original work-item enumeration). The
    integration parity tests rely on this contract. Subclass docstrings restate the
    guarantee with the per-stage rationale; a subclass whose parent step does a cross-shard
    reduction (averaging, summing, ...) where the floating-point result depends on
    summation order MUST flag the divergence prominently in its docstring so users know
    output's last few float bits can shift on re-shard. None of the currently implemented
    stages fall in that second bucket.
    """

    #: subdirectory name under slurm_root_folder where this stage's shards live
    folder_name: str = "stage"
    #: short identifier used in log messages
    stage_name: str = "stage"
    #: SLURM job name (also the prefix of submitit's per-task log files)
    job_name: str = "tt_stage"
    #: pipeline stage number this shard executes (e.g. 4 for the distance stage). Passed to the
    #: rebuilt config as active_stages=(stage_number, stage_number) so the shard only validates and
    #: sets up the prerequisites of its own stage (see AnalysisConfig._runs_stage). Subclasses MUST
    #: override; None means "unset" and would scope the rebuild to no stage.
    stage_number: int | None = None

    items_filename: str = "items.json"

    #: subclasses with worker-side side-effect files (e.g. layering stages writing per-cluster
    #: .py + nodes/*.pdb) set this to True. Toggles the resume-path side-effect validation in
    #: `_missing_shards()`: when True, the launcher requires a per-shard manifest + tarball
    #: backup and will try to restore side-effects from the backup before resubmitting any
    #: shard whose visualisation files have been wiped between runs. Default False matches the
    #: distance-shard backend whose workers produce nothing besides the result pickle.
    has_side_effects: bool = False

    @classmethod
    def shard_manifest_path(cls, slurm_folder: str, shard_id: int) -> str:
        """
        Companion to `shard_result_path`: the per-shard "side-effects manifest" - a tiny JSON
        file listing every on-disk side-effect (e.g. visualisation .py / .pdb files written by
        the inner-Pool workers) the shard produced beyond its main result pickle. The default
        layout is `<slurm_folder>/shard_result_<id>.manifest.json`; subclasses that don't want
        the default companion-suffix can override.

        Why it exists: the resume optimisation in `_missing_shards()` skips submitting a shard
        whose result file is already on disk with a matching fingerprint. But for stages whose
        workers also write side-effect files (e.g. layering writes per-cluster Pymol scripts +
        per-layer node PDBs), those side-effects can be wiped by the user (clearing the output
        folder) or by a test between runs while the result pickle stays in the SLURM cache.
        Without a manifest, the resume optimisation would silently reuse a stale shard and the
        side-effect files would never be regenerated. With a manifest, `_missing_shards()`
        treats any shard with a missing side-effect file as missing and resubmits it.

        Stages with no worker-side side-effects (e.g. the distance stage, whose workers only
        return numbers) don't write a manifest and don't need to override anything; the
        default `expected_side_effect_files()` returns an empty list and the resume check
        validates only the result pickle, preserving the existing fast-path.
        """

        return os.path.join(slurm_folder, "shard_result_{:05d}.manifest.json".format(shard_id))

    def expected_side_effect_files(self, parameters: dict, slurm_folder: str,
                                   shard_id: int) -> Optional[List[str]]:
        """
        Hook reporting the per-shard side-effect files this shard is expected to have produced.
        Default: `None` - no manifest is expected, resume check validates only the result
        pickle (preserves the existing behaviour for stages without worker-side side-effects,
        e.g. the distance stage).

        A stage that has worker-side side-effects (e.g. visualisation files) overrides this
        to return either an empty list (manifest is expected but stage's parameters mean no
        side-effects produced for this run) or the list of files read from a per-shard
        manifest written by `run_shard()`. The launcher's `_missing_shards()` uses this to
        decide whether to invalidate a shard whose result pickle is present but whose
        side-effect files have been wiped between runs.

        :param parameters: job configuration parameters
        :param slurm_folder: shared folder for this stage's SLURM execution
        :param shard_id: 0-based ID of the shard whose side-effects are being checked
        :return: list of absolute side-effect file paths the shard is expected to have
                 written; `None` if this stage has no concept of side-effect files (don't
                 perform the side-effect check at all)
        """

        return None

    @classmethod
    def write_shard_manifest(cls, slurm_folder: str, shard_id: int,
                             side_effect_files: List[str]) -> None:
        """
        Atomically persist a shard's side-effect manifest to
        `shard_manifest_path(slurm_folder, shard_id)` (write to `.tmp`, then os.replace). Called
        by a stage's `run_shard()` after all its workers have completed but before the result
        pickle is written, so that on resume the manifest is always at least as up-to-date as
        the result pickle - never the other way round (which would risk a stale manifest
        invalidating a fresh result).
        """

        manifest_path = cls.shard_manifest_path(slurm_folder, shard_id)
        tmp_path = manifest_path + ".tmp"
        with open(tmp_path, "w") as out_stream:
            json.dump({"side_effect_files": list(side_effect_files)}, out_stream)
        os.replace(tmp_path, manifest_path)

    @classmethod
    def read_shard_manifest(cls, slurm_folder: str, shard_id: int) -> Optional[List[str]]:
        """
        Read back the side-effect manifest written by `write_shard_manifest`. Returns the list
        of side-effect file paths if the manifest is present and well-formed; returns `None` if
        it's absent or unreadable (signal to the caller that the side-effect check cannot be
        performed - typically meaning the shard was produced by a TT version that did not
        write manifests, OR that the manifest was wiped along with the side-effects).
        """

        manifest_path = cls.shard_manifest_path(slurm_folder, shard_id)
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path) as in_stream:
                data = json.load(in_stream)
            return list(data["side_effect_files"])
        except (json.JSONDecodeError, OSError, KeyError, TypeError):
            return None

    @classmethod
    def shard_side_effects_archive_path(cls, slurm_folder: str, shard_id: int) -> str:
        """
        Path of the per-shard side-effects archive: a gzipped tarball that bundles all on-disk
        side-effect files the shard produced (per-cluster .py scripts, per-layer node PDBs,
        per-event exact-matching dumps, etc.). The archive serves as a local backup that
        survives accidental wipes of the user-facing output folder. On resume, when the manifest
        check finds any side-effect missing on disk, the launcher extracts the archive back to
        original paths instead of resubmitting the shard - so the heavy per-cluster sklearn /
        per-event trajectory work is not repeated.

        Default layout: `<slurm_folder>/shard_result_<id>.side_effects.tar.gz`. Compressed
        (text-heavy side-effects compress ~3-5x), atomically written with tmp + os.replace.
        """

        return os.path.join(slurm_folder,
                            "shard_result_{:05d}.side_effects.tar.gz".format(shard_id))

    @classmethod
    def write_side_effects_archive(cls, slurm_folder: str, shard_id: int,
                                   file_paths: List[str], base_path: str) -> None:
        """
        Bundle the listed side-effect files into a per-shard tarball at
        `shard_side_effects_archive_path()`. Each file is stored with its path relative to
        `base_path` as the tar member name, so the archive is self-describing and can be
        inspected (`tar -tzf ...`) or extracted (`tar -xzf ...`) anywhere without recreating
        the full absolute path tree. `restore_side_effects_from_archive()` reapplies the same
        `base_path` to place bytes back at the original location. Empty file_paths produces
        an empty archive (so the missing-shard check always finds an artefact to inspect).

        Atomic via tmp + os.replace: a partially-written archive is never visible to the
        resume check.

        :param slurm_folder: shared folder for this stage's SLURM execution
        :param shard_id: 0-based shard ID
        :param file_paths: list of absolute paths to side-effect files this shard produced;
                           caller must ensure every path exists on disk at call time
        :param base_path: directory the member arcnames are stored relative to. In production
                          this is `parameters["output_path"]` - every side-effect file lives
                          under it, so arcnames become short, portable, relative paths like
                          `vis/sources/layered/caver/md1/Cluster_1.py`.
        """

        archive_path = cls.shard_side_effects_archive_path(slurm_folder, shard_id)
        tmp_path = archive_path + ".tmp"
        with tarfile.open(tmp_path, "w:gz") as tar:
            for path in file_paths:
                tar.add(path, arcname=os.path.relpath(path, base_path))
        os.replace(tmp_path, archive_path)

    @classmethod
    def restore_side_effects_from_archive(cls, slurm_folder: str, shard_id: int,
                                          expected_files: List[str],
                                          base_path: str) -> bool:
        """
        Extract the per-shard side-effects archive back to its original absolute paths so a
        wiped output folder can be reconstructed WITHOUT re-running the heavy shard work
        (per-cluster sklearn for layering; per-event trajectory scans for assignment).

        :param slurm_folder: shared folder for this stage's SLURM execution
        :param shard_id: 0-based shard ID
        :param expected_files: list of absolute paths that the caller expects to be restored;
                               returns False if any one of them is missing from the archive
                               or fails to extract (caller should treat the shard as missing
                               and resubmit)
        :param base_path: same `base_path` used by `write_side_effects_archive()` (in
                          production: `parameters["output_path"]`); each expected file is
                          located inside the archive via its path relative to this base.
        :return: True if every entry in `expected_files` was successfully written back to its
                 original path; False if anything went wrong (archive missing, corrupt, or
                 missing an expected member). The caller can then invalidate the shard and
                 fall back to resubmission.
        """

        archive_path = cls.shard_side_effects_archive_path(slurm_folder, shard_id)
        if not os.path.exists(archive_path):
            return False
        expected_arcnames = {os.path.relpath(p, base_path) for p in expected_files}
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}
                if not expected_arcnames.issubset(members.keys()):
                    return False
                for path in expected_files:
                    arcname = os.path.relpath(path, base_path)
                    member = members[arcname]
                    src = tar.extractfile(member)
                    if src is None:
                        return False
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tmp_path = path + ".tmp.restore"
                    with open(tmp_path, "wb") as out:
                        out.write(src.read())
                    os.replace(tmp_path, path)
        except (tarfile.TarError, OSError):
            return False
        return True

    @classmethod
    def get_slurm_folder(cls, parameters: dict) -> str:
        """
        Per-stage SLURM folder: `<slurm_root_folder>/<folder_name>/`. Computed as a
        classmethod so that SLURM-shard workers (which only have a config_file, not a stage
        instance with runtime state) can derive the result path the same way the launcher does.
        """

        return os.path.join(parameters["slurm_root_folder"], cls.folder_name)

    def prepare_state(self, parameters: dict, slurm_folder: str) -> None:
        """
        Optional hook called once in the launcher (after the SLURM folder is created and before
        the fingerprint check), giving the stage a chance to write any per-stage state artifact
        the shards will load on startup. Default: no-op. Stages that need the launcher to share
        a work-item enumeration (e.g. the (md_label, cluster_id) list for tunnel layering)
        override this to write a single artifact next to the per-shard result files; shards
        then read that artifact in `run_shard()` instead of re-deriving the enumeration from
        scratch in every shard process.
        """

        return None

    @abstractmethod
    def num_shards(self, parameters: dict) -> int:
        """
        Total number of SLURM array tasks to submit for this stage. Resolved from the user's
        `slurm_num_shards` knob if set, otherwise auto-derived via `derive_num_shards()`.
        :param parameters: job configuration parameters
        """

    @classmethod
    @abstractmethod
    def shard_result_path(cls, slurm_folder: str, shard_id: int) -> str:
        """
        Path of the per-shard result file that this shard's worker writes and the launcher
        reads back during assembly. Must be deterministic given (slurm_folder, shard_id) and
        must include `shard_id` to keep concurrent shards from clobbering each other.
        Declared as a `@classmethod` so SLURM-shard workers (which only have a config_file +
        shard_id, no stage instance with runtime state) can derive the output path directly
        from the class, the same way `get_slurm_folder()` works.
        """

    @classmethod
    @abstractmethod
    def run_shard(cls, config_file: str, num_shards: int, shard_id: int) -> Any:
        """
        Execute one shard. Runs inside a SLURM compute-node Python process. Declared as a
        `@classmethod` so submitit pickles only a class-qualified function reference (plus
        the args), not the launcher's runtime state on the stage instance - on a large run
        with many shards, pickling instance state like the tunnel-layering items list into
        every task's submitit payload would be wasted bytes (the shards rebuild that state
        from disk anyway). The implementation is responsible for writing
        `shard_result_path()` atomically (write to a `.tmp` neighbour, then os.replace).
        :param config_file: path to the INI configuration file of the analysis
        :param num_shards: total number of shards the calculation is split into
        :param shard_id: 0-based ID of the shard to compute
        :return: an arbitrary value (typically a per-shard count) for the launcher to log
        """

    def fingerprint(self, parameters: dict, num_shards: int) -> dict:
        """
        Fingerprint dict identifying the parameters this stage's shard results were
        computed for. The launcher persists this to `shards_info.json` in the SLURM
        folder and discards stale partial results whenever the current fingerprint
        does not match the persisted one (e.g. after the user changed a parameter
        that affects results, or after a TransportTools version bump that changes the
        on-disk format).

        The default implementation includes `num_shards`, the TransportTools version,
        and the values of every parameter listed in `fingerprint_keys`; subclasses
        override or extend it to include runtime-derived values (e.g. the number of
        clusters for the distance stage).
        """

        info: dict = {
            "num_shards": num_shards,
            "tt_version": __version__,
        }
        for key in getattr(self, "fingerprint_keys", ()):
            info[key] = parameters[key]
        return info

    def executor_params(self, parameters: dict) -> dict:
        """
        SLURM submission parameters for this stage. The defaults below honor the
        existing global `slurm_*` knobs; subclasses can override to apply per-stage
        resource overrides (e.g. layering shards need more memory than profile-filter
        shards) without affecting other stages. The returned dict is passed directly
        to `submitit.AutoExecutor.update_parameters(**...)`.
        """

        params: dict = {
            "name": self.job_name,
            "timeout_min": parameters["slurm_timeout_min"],
            "cpus_per_task": parameters["slurm_cpus_per_task"],
            "mem_gb": parameters["slurm_mem_gb"],
            "nodes": 1,
            "tasks_per_node": 1,
        }
        if parameters["slurm_partition"] is not None:
            params["slurm_partition"] = parameters["slurm_partition"]
        if parameters["slurm_account"] is not None:
            params["slurm_account"] = parameters["slurm_account"]
        if parameters["slurm_array_parallelism"] is not None:
            params["slurm_array_parallelism"] = parameters["slurm_array_parallelism"]
        if parameters["slurm_setup"]:  # non-empty list of setup commands; submitit expects a list of str
            params["slurm_setup"] = parameters["slurm_setup"]
        # Guarantee --cpu-bind=none is in slurm_srun_args even when the user overrides the
        # knob. When a SLURM stage is itself launched from inside a SLURM allocation, the
        # parent job's SLURM_CPU_BIND* variables leak into this array job's sbatch script
        # and make the nested srun fail with "CPU binding outside of job step allocation".
        # _ensure_cpu_bind_none() prepends the workaround unless the user already supplied
        # an explicit --cpu-bind=... (in which case their choice is honoured). The job
        # stays confined to its allocated cores via cgroups regardless.
        params["slurm_srun_args"] = _ensure_cpu_bind_none(parameters["slurm_srun_args"] or [])
        # Cluster-specific sbatch headers (--qos, --constraint, --gres, --reservation, ...)
        # parsed by config.py from the comma-separated 'key=value' INI form into a dict.
        # submitit translates each entry into a #SBATCH --key=value header.
        if parameters["slurm_additional_parameters"]:
            params["slurm_additional_parameters"] = parameters["slurm_additional_parameters"]
        return params


def _missing_shards(stage: SlurmShardStage, slurm_folder: str, num_shards: int,
                    parameters: dict) -> List[int]:
    """
    Validate any partial shard results already present in the SLURM folder and report which
    shards still need to be computed. Partial results are reused only if they were produced for
    the same fingerprint (same num_shards, same fingerprint_keys values, same TT version) AND
    every side-effect file the shard was expected to produce (per its on-disk manifest) is
    still present; otherwise the stale result files are discarded so the calculation starts
    fresh. The side-effect check handles the case where the user (or a test) wipes an output
    folder between runs without also clearing this SLURM cache - without it, the launcher
    would silently reuse a stale shard and the wiped side-effects would never be regenerated.
    :param stage: the SLURM stage providing the fingerprint and result-path layout
    :param slurm_folder: shared folder used for this stage's SLURM execution
    :param num_shards: total number of shards
    :param parameters: job configuration parameters
    :return: sorted list of shard IDs whose results are still missing
    """

    info = stage.fingerprint(parameters, num_shards)
    # filename recording the parameters a set of partial shard results was computed for
    info_file = os.path.join(slurm_folder, "partial_shards_info.json")

    previous_info = None
    if os.path.exists(info_file):
        try:
            with open(info_file) as in_stream:
                previous_info = json.load(in_stream)
        except (json.JSONDecodeError, OSError):
            previous_info = None

    if previous_info != info:
        # absent or stale partial results - discard them and start a fresh calculation. Only
        # files that look like shard results are removed; user files dropped into the folder
        # (e.g. submitit's own *.pkl / *_log files) are left alone. Manifest + side-effects
        # archive companions are cleaned up alongside their result pickles so a stale
        # manifest can never outlive its result.
        existing_paths: set = set()
        for sid in range(max(num_shards,
                             previous_info.get("num_shards", 0) if previous_info else 0)):
            existing_paths.add(stage.shard_result_path(slurm_folder, sid))
            existing_paths.add(stage.shard_manifest_path(slurm_folder, sid))
            existing_paths.add(stage.shard_side_effects_archive_path(slurm_folder, sid))
        for path in existing_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Failed to remove stale shard artefact '%s': %s", path, exc)
        with open(info_file, "w") as out_stream:
            json.dump(info, out_stream, indent=2)

    missing: List[int] = []
    for sid in range(num_shards):
        result_path = stage.shard_result_path(slurm_folder, sid)
        if not os.path.exists(result_path):
            missing.append(sid)
            continue
        # fast path for stages with no worker-side side-effects (distance stage, future
        # numerical-only stages): result pickle existence is sufficient.
        if not stage.has_side_effects:
            continue
        # side-effect check for layering / event-matching stages: validate manifest + on-disk
        # side-effect files. When any side-effect is missing (typically: user wiped an output
        # folder between runs), restore from the per-shard tar.gz backup so the heavy worker
        # computation is NOT repeated. Only fall back to resubmission when both the backup
        # and the manifest are unrecoverable.
        expected = stage.expected_side_effect_files(parameters, slurm_folder, sid)
        manifest_missing = expected is None
        if manifest_missing:
            logger.info("SLURM %s shard %d: manifest missing on a stage that tracks "
                        "side-effects; cannot validate. Invalidating the cached pickle and "
                        "resubmitting.", stage.stage_name, sid)
            _invalidate_cached_shard(stage, slurm_folder, sid)
            missing.append(sid)
            continue
        if any(not os.path.exists(p) for p in expected):
            logger.info("SLURM %s shard %d: side-effect file(s) wiped from disk since the "
                        "result was cached; restoring them from the per-shard backup archive.",
                        stage.stage_name, sid)
            restored = stage.restore_side_effects_from_archive(
                slurm_folder, sid, expected, parameters["output_path"])
            if restored:
                logger.info("SLURM %s shard %d: side-effects restored from backup; skipping "
                            "resubmission.", stage.stage_name, sid)
                continue
            logger.info("SLURM %s shard %d: side-effects backup missing or unusable; "
                        "invalidating the cached pickle and resubmitting so the worker "
                        "regenerates them.", stage.stage_name, sid)
            _invalidate_cached_shard(stage, slurm_folder, sid)
            missing.append(sid)
    return missing


def _invalidate_cached_shard(stage: SlurmShardStage, slurm_folder: str, shard_id: int) -> None:
    """Remove a shard's cached artefacts (result pickle + manifest + side-effects archive) so
    the next resume check sees the shard as missing and the launcher resubmits it. Centralised
    so the cleanup paths in `_missing_shards()` (broken manifest, unrecoverable archive) stay
    in sync as new artefact types are added in the future. Removal failures are logged but
    not raised - the worst case is the next run re-removes a stale file."""

    for stale in (stage.shard_result_path(slurm_folder, shard_id),
                  stage.shard_manifest_path(slurm_folder, shard_id),
                  stage.shard_side_effects_archive_path(slurm_folder, shard_id)):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError as exc:
                logger.warning("Failed to remove invalidated shard artefact '%s': %s",
                               stale, exc)


#: Seconds of grace between a shard's first absence from squeue and the launcher giving up on
#: it. Bounds the NFS/race window between slurmctld dropping a job from its active view and the
#: shard's result pickle becoming visible to the launcher's filesystem - if the pickle was
#: about to land, it lands within this window and the next `job.done()` check at the top of the
#: poll loop catches it on the success path before the squeue branch abandons the shard.
_SQUEUE_MISS_GRACE_S = 5.0

#: Subprocess timeout for the squeue probe. Bounded so a sick slurmctld cannot hang the
#: launcher: on timeout, `_squeue_live_jobs()` returns None and the caller falls back to the
#: submission-time backstop rather than false-abandoning anything.
_SQUEUE_PROBE_TIMEOUT_S = 30.0

#: Hard cap on the number of element ids a single bracketed squeue token may expand to. A
#: well-formed `jobid_[a-b]` range is bounded by the cluster's MaxArraySize (typically <= a few
#: thousand), but a corrupt squeue line must not make `_expand_squeue_array_ids` allocate an
#: unbounded set; past the cap we keep the raw token only (still correct - it just won't match
#: per-element ids, which degrades to the pre-expansion behaviour for that one pathological row).
_SQUEUE_ARRAY_EXPANSION_CAP = 100000


def _expand_squeue_array_ids(token: str) -> set:
    """
    Expand one `squeue -o %i` id token into the set of concrete job-id strings it represents.

    SLURM collapses still-PENDING elements of an array job into a single bracketed row, e.g.
    `347_[2-199]` (and on a throttled array, `347_[2-199%10]`, or a sparse `347_[1,3,5-9]`),
    while elements that have already started show individually as `347_2`, `347_3`, ...
    submitit's per-task `Job.job_id` is always the concrete element form (`347_2`), so a raw
    string match against the collapsed row would miss every pending element and make `_poll_jobs`
    false-abandon perfectly-queued shards. Expanding the bracket here lets both forms resolve to
    the same `<parent>_<index>` ids the poll loop compares against.

    Returns a set that always contains the raw token (so non-array ids like `12345` and any
    form we don't recognise still match by identity) plus, for a bracketed array token, one
    `<parent>_<index>` string per element in the range/list. The `%throttle` suffix inside the
    bracket is stripped before parsing; malformed bracket bodies degrade to the raw token only.
    """

    ids = {token}
    open_bracket = token.find("[")
    if open_bracket == -1 or not token.endswith("]"):
        return ids
    prefix = token[:open_bracket]              # e.g. '347_'
    body = token[open_bracket + 1:-1]          # e.g. '2-199%10' or '1,3,5-9'
    body = body.split("%", 1)[0]               # drop the array-throttle suffix if present
    indices: List[int] = []
    for part in body.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, _, hi_str = part.partition("-")
            try:
                lo, hi = int(lo_str), int(hi_str)
            except ValueError:
                return ids                     # malformed range -> keep raw token only
            if hi < lo or (hi - lo + 1) + len(indices) > _SQUEUE_ARRAY_EXPANSION_CAP:
                return ids                     # nonsensical or oversized -> raw token only
            indices.extend(range(lo, hi + 1))
        else:
            try:
                indices.append(int(part))
            except ValueError:
                return ids
            if len(indices) > _SQUEUE_ARRAY_EXPANSION_CAP:
                return ids
    for index in indices:
        ids.add("{}{}".format(prefix, index))
    return ids


def _squeue_live_jobs(job_ids: List[str],
                      timeout_s: float = _SQUEUE_PROBE_TIMEOUT_S) -> Optional[set]:
    """Batched 'is this job still listed by slurmctld?' check, used by `_poll_jobs` as a
    filesystem-independent dead-shard signal that works even when sacct is unavailable
    (slurmdbd missing - submitit's job.done() then never flips to True for scancel'd /
    node-failed / preempted shards because both the result pickle and the SLURM state path
    fail). One `squeue -h -j <comma-list>` call covers every still-pending shard per poll
    cycle regardless of shard count.

    Returns:
        - the set of job_ids that squeue currently lists, when the controller is healthy;
        - an empty set when slurmctld is healthy but knows nothing about any of the supplied
          ids (every one of them has left its active view -> caller should abandon them after
          the usual grace);
        - None when squeue itself failed (binary missing, timed out, or an unexpected error
          message). The caller MUST treat None as 'unknown' and not abandon any shard based
          on it - the submission-time backstop is the only remaining safety net in that case.

    job_ids must be submitit `Job.job_id` strings, including the '<parent>_<arr>' array-task
    suffix where applicable (squeue accepts that format directly).
    """

    if not job_ids:
        return set()
    try:
        proc = subprocess.run(
            ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%i"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout_s, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("squeue probe failed (%s); deferring to backstop timeouts.", exc)
        return None

    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace").strip()

    # squeue exits non-zero when at least one of the requested ids is unknown to the
    # controller. Distinguish the benign "controller dropped these ids" case (we can trust
    # an empty result -> caller abandons after grace) from "squeue itself is broken" (return
    # None -> caller falls back to the backstop and does not abandon). Heuristic: if every
    # non-blank stderr line is an 'invalid job id' message, it's the benign case.
    only_unknown_ids = all(
        "invalid job id" in line.lower()
        for line in stderr.splitlines() if line.strip()
    )
    if proc.returncode != 0 and not (only_unknown_ids and stderr):
        logger.debug("squeue probe returned exit=%d stderr=%r; deferring to backstop "
                     "timeouts.", proc.returncode, stderr)
        return None

    live: set = set()
    for line in stdout.splitlines():
        token = line.strip()
        if token:
            # Expand any bracketed array range (`347_[2-199]`, pending elements squeue
            # collapses into one row) into its concrete `347_2`, `347_3`, ... element ids so
            # the caller's per-element membership test matches still-pending shards. Non-array
            # and individual ids pass through unchanged (the raw token is always retained).
            live.update(_expand_squeue_array_ids(token))
    return live


def _poll_jobs(missing_shards: List[int], jobs: List[Any], num_shards: int,
               parameters: dict, stage_name: str) -> dict:
    """
    Poll SLURM jobs in completion order with a per-shard runtime timeout. The timeout is derived
    from the SLURM wall-time limit: 2 * slurm_timeout_min + 5 min covers the actual run time
    plus the SLURM-reporting latency that can delay a "done" notification after the job has
    exited. The clock only starts ticking once we observe the shard transition into RUNNING, so
    shards that legitimately spend a long time in the SLURM queue (e.g. limited array
    parallelism on a busy cluster) are never abandoned for queueing alone. A shard that runs
    past its deadline is logged, cancelled, and skipped; the per-shard result files plus the
    fingerprint check in `_missing_shards()` let the next run pick up the abandoned shards
    without recomputing the finished ones. Failed shards (state FAILED, OOM, etc.) are reported
    via job.result() raising and do not abort collection.

    A complementary fast-path catches the case where submitit's `job.done()` never flips True
    because the cluster has no slurmdbd (sacct returns non-zero -> watcher.state is "UNKNOWN")
    and the worker was scancel'd / node-failed / preempted before writing its result pickle.
    Once per poll cycle, the still-pending shards' job ids are passed in one batched
    `_squeue_live_jobs()` call; any shard the controller has dropped for at least one full
    `_SQUEUE_MISS_GRACE_S` window is abandoned with category 'cancelled' (the grace covers
    NFS lag between squeue dropping the job and the result pickle becoming visible). When
    squeue itself fails, the helper returns None and this branch is skipped.

    A final submission-time backstop (~6 x slurm_timeout_min) abandons any still-pending shard
    even if both job.done() and squeue have been silent the whole time - the only remaining
    case is a completely sick controller, where running indefinitely is worse than failing
    loudly so the user can re-run.

    Returns a per-shard `outcomes` dict (keyed by shard_id) that the post-run reporter
    (`_report_run_stats`) consumes for failure categorization and runtime statistics. Each
    entry holds the shard's SLURM `state`, the monotonic timestamps of the observed
    RUNNING -> done transition (`start_mono`, `end_mono`), the categorized outcome
    (`success` / `oom` / `timeout` / `cancelled` / `node_failure` / `preempted` / `error`),
    a path to the SLURM stderr file (handy for the failure-categorisation log line), and the
    exception text when applicable. start_mono is `None` for shards that completed between
    poll cycles before we could observe the RUNNING state (timing for those shards is
    treated as missing in stats); category is `None` only on the impossible "still pending
    when the function returned" branch.
    """

    shard_wait_timeout_s = parameters["slurm_timeout_min"] * 2 * 60 + 5 * 60
    # Absolute backstop measured from submission. 6x walltime + 10 min so that any legitimate
    # queue-wait-plus-run never trips it, but a permanently-stuck poll (sacct AND squeue both
    # silent) still fails the stage in finite time instead of hanging forever.
    submission_backstop_s = parameters["slurm_timeout_min"] * 6 * 60 + 10 * 60
    submitted_mono = time.monotonic()
    pending: List[Tuple[int, Any]] = list(zip(missing_shards, jobs))
    running_since: dict = {}  # job_id -> monotonic time of first observed RUNNING state
    first_squeue_miss: dict = {}  # shard_id -> monotonic time of first squeue absence
    outcomes: dict = {sid: {
        "shard_id": sid,
        "job_id": job.job_id,
        "state": None,
        "start_mono": None,
        "end_mono": None,
        "category": None,
        "stderr_path": _job_stderr_path(job),
        "exception": None,
    } for sid, job in pending}
    logger.info("SLURM array job submitted as %d task(s); waiting for completion "
                "(per-shard runtime timeout: %ds, measured from RUNNING state; "
                "submission-time backstop: %ds).",
                len(jobs), shard_wait_timeout_s, submission_backstop_s)

    while pending:
        now = time.monotonic()
        still_pending: List[Tuple[int, Any]] = []
        for shard_id, job in pending:
            if job.done():
                outcomes[shard_id]["end_mono"] = now
                outcomes[shard_id]["state"] = _safe_job_state(job)
                try:
                    result = job.result()
                    outcomes[shard_id]["category"] = "success"
                    logger.info("SLURM %s shard %d/%d finished (%r).",
                                stage_name, shard_id + 1, num_shards, result)
                except Exception as error:
                    category, hint = _categorize_failure(job, error)
                    outcomes[shard_id]["category"] = category
                    outcomes[shard_id]["exception"] = "{}: {}".format(
                        type(error).__name__, error)
                    stderr_msg = " (stderr: {})".format(outcomes[shard_id]["stderr_path"]) \
                        if outcomes[shard_id]["stderr_path"] else ""
                    logger.error("SLURM %s shard %d (job %s, state=%s) failed [%s]: %s%s. %s",
                                 stage_name, shard_id, job.job_id,
                                 outcomes[shard_id]["state"] or "?", category, error,
                                 stderr_msg, hint)
                first_squeue_miss.pop(shard_id, None)
                continue

            # Record the first time we observe this shard as RUNNING; once recorded, the
            # per-shard runtime clock starts. Shards that are still PENDING in the queue are
            # left alone (no deadline yet) - that matches the user's expectation that the
            # timeout caps actual compute time, not queue waiting.
            if job.job_id not in running_since:
                state = _safe_job_state(job)
                if state == "RUNNING":
                    running_since[job.job_id] = now
                    outcomes[shard_id]["start_mono"] = now

            if (job.job_id in running_since
                    and now - running_since[job.job_id] > shard_wait_timeout_s):
                logger.error("SLURM %s shard %d (job %s) ran for more than %ds without finishing; "
                             "abandoning it and continuing with the shards that already finished. "
                             "Re-run the stage to resume.",
                             stage_name, shard_id, job.job_id, shard_wait_timeout_s)
                outcomes[shard_id]["state"] = "TIMEOUT"
                outcomes[shard_id]["end_mono"] = now
                outcomes[shard_id]["category"] = "timeout"
                try:
                    job.cancel()
                except Exception as cancel_error:
                    logger.warning("Failed to cancel SLURM job %s: %s", job.job_id, cancel_error)
                first_squeue_miss.pop(shard_id, None)
                continue

            still_pending.append((shard_id, job))

        # squeue-based dead-shard detection: one batched probe per poll cycle catches the
        # case where job.done() will never flip True (no slurmdbd -> watcher state is forever
        # "UNKNOWN" + worker was killed before writing the result pickle). A shard missing
        # from squeue for at least _SQUEUE_MISS_GRACE_S is treated as dropped by the
        # controller; the grace covers NFS lag between squeue dropping the job and the
        # pickle becoming visible (if the pickle was about to land, it lands within the
        # grace and the next iteration's job.done() catches the success path first).
        if still_pending:
            live = _squeue_live_jobs([job.job_id for _, job in still_pending])
            if live is not None:
                kept: List[Tuple[int, Any]] = []
                for shard_id, job in still_pending:
                    if job.job_id in live:
                        first_squeue_miss.pop(shard_id, None)
                        kept.append((shard_id, job))
                        continue
                    if shard_id not in first_squeue_miss:
                        first_squeue_miss[shard_id] = now
                        kept.append((shard_id, job))
                        continue
                    if now - first_squeue_miss[shard_id] < _SQUEUE_MISS_GRACE_S:
                        kept.append((shard_id, job))
                        continue
                    logger.error("SLURM %s shard %d (job %s) is no longer listed by squeue "
                                 "and produced no result pickle within %.0fs; treating as "
                                 "cancelled. Re-run the stage to resume.",
                                 stage_name, shard_id, job.job_id, _SQUEUE_MISS_GRACE_S)
                    outcomes[shard_id]["state"] = "CANCELLED"
                    outcomes[shard_id]["end_mono"] = now
                    outcomes[shard_id]["category"] = "cancelled"
                    try:
                        job.cancel()
                    except Exception as cancel_error:
                        logger.warning("Failed to cancel SLURM job %s: %s",
                                       job.job_id, cancel_error)
                    first_squeue_miss.pop(shard_id, None)
                still_pending = kept

        # Submission-time backstop: only reached when both job.done() and the squeue probe
        # have been silent for the entire backstop window. That's the "controller is sick"
        # case - failing loudly so the user can re-run is strictly better than hanging.
        if still_pending and now - submitted_mono > submission_backstop_s:
            for shard_id, job in still_pending:
                logger.error("SLURM %s shard %d (job %s) has been pending for more than "
                             "%ds since submission with no resolution from either "
                             "submitit's job state or squeue; abandoning. Re-run the "
                             "stage to resume.",
                             stage_name, shard_id, job.job_id, submission_backstop_s)
                outcomes[shard_id]["state"] = outcomes[shard_id]["state"] or "TIMEOUT"
                outcomes[shard_id]["end_mono"] = now
                outcomes[shard_id]["category"] = "timeout"
                try:
                    job.cancel()
                except Exception as cancel_error:
                    logger.warning("Failed to cancel SLURM job %s: %s",
                                   job.job_id, cancel_error)
                first_squeue_miss.pop(shard_id, None)
            still_pending = []

        pending = still_pending
        if pending:
            time.sleep(parameters["slurm_poll_wait_seconds"])

    return outcomes


def _safe_job_state(job: Any) -> Optional[str]:
    """Read job.state, swallowing any transient backend error (sacct hiccup, NFS lag) and
    returning None instead. Used both for the polling loop's RUNNING detection and for the
    post-run reporter's failure categorisation, where missing state must not crash the
    whole reporter."""

    try:
        return job.state
    except Exception:
        return None


def _job_stderr_path(job: Any) -> Optional[str]:
    """Best-effort recovery of the SLURM stderr file path for a job. Returns the absolute
    path as a string when submitit exposes it (the usual case), or `None` when the path
    cannot be derived for some reason. Used both in the per-shard failure log line and in
    the post-run reporter so users get a clickable file path to inspect without having to
    construct it from the submitit folder + job id themselves (5.15.16)."""

    try:
        return str(job.paths.stderr)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Post-run reporter: failure categorisation + runtime statistics
# ---------------------------------------------------------------------------
#
# After the launcher's poll loop completes, the SLURM stage's shards have ended in some mix
# of success / OOM / timeout / cancelled / node_failure / preempted / code-error. Failure
# categorisation with remediation hints and per-shard wall time stats + parallel efficiency
# share most of their inputs (job state, observed RUNNING -> done timestamps), so they are
# implemented as a single reporter pass `_report_run_stats()` driven from the per-shard
# outcomes recorded in `_poll_jobs()`.

#: SLURM job-state strings -> (category, remediation_hint). Categories drive the failure
#: summary in `_report_run_stats`; remediation_hint is logged inline next to the per-shard
#: failure line in `_poll_jobs`. Categories the reporter recognises beyond this map: "error"
#: (fell-through Python exception when state is FAILED but no specific SLURM-side cause was
#: detected) and "success" (set in the success path of `_poll_jobs`).
_SLURM_STATE_CATEGORIES: dict = {
    "TIMEOUT":         ("timeout",      "Bump slurm_timeout_min, or reduce slurm_num_shards "
                                        "so each shard handles fewer items."),
    "DEADLINE":        ("timeout",      "Bump slurm_timeout_min, or reduce slurm_num_shards."),
    "OUT_OF_MEMORY":   ("oom",          "Bump slurm_mem_gb."),
    "OOM":             ("oom",          "Bump slurm_mem_gb."),
    "NODE_FAIL":       ("node_failure", "Transient node failure; re-run the stage to resume "
                                        "from the surviving shards."),
    "BOOT_FAIL":       ("node_failure", "Transient node failure; re-run the stage."),
    "CANCELLED":       ("cancelled",    "Shard was cancelled (manual scancel or framework "
                                        "timeout); re-run the stage to resume."),
    "PREEMPTED":       ("preempted",    "Job was preempted by the scheduler; re-run the stage."),
}

#: stderr-content markers used as a fallback when SLURM state is "FAILED" but the cause is
#: actually an external kill that didn't reflect in the SLURM state. Kernel OOM-killer puts
#: "Killed" into stderr; Python MemoryError surfaces as such. Order matters: first match
#: wins.
_STDERR_OOM_MARKERS: tuple = ("Killed", "MemoryError", "out of memory")


def _categorize_failure(job: Any, error: Exception) -> Tuple[str, str]:
    """
    Classify a failed SLURM job into one of `oom / timeout / cancelled / node_failure /
    preempted / error` and pair it with a one-line remediation hint to nudge the user toward
    the right knob without forcing them to read shard logs. Resolution order:
      1. The SLURM job state itself - the cluster has already labelled the cause for OOM,
         TIMEOUT, NODE_FAIL, CANCELLED, PREEMPTED, etc., and that is the authoritative
         answer when available (see `_SLURM_STATE_CATEGORIES`).
      2. Fallback stderr-marker scan for the case where SLURM state is just "FAILED" but the
         worker process was actually OOM-killed by the kernel (the cgroup didn't catch it
         and SLURM just reports the non-zero exit). Looks for `_STDERR_OOM_MARKERS` in the
         job's stderr file; bounded to a single read so a multi-GB stderr does not eat
         memory in the launcher.
      3. "error" - genuine Python-side exception in the shard's `run_shard()`. The
         remediation is to inspect the stderr file (we already log the path next to the
         failure line in `_poll_jobs`).
    Returns (category, hint).
    """

    state = _safe_job_state(job)
    if state and state in _SLURM_STATE_CATEGORIES:
        return _SLURM_STATE_CATEGORIES[state]

    # Fallback: SLURM didn't categorize; scan stderr for OOM markers. Worth catching because
    # cgroup OOM kills sometimes only surface in stderr as a single "Killed" line.
    stderr_path = _job_stderr_path(job)
    if stderr_path and os.path.exists(stderr_path):
        try:
            # cap read at ~64 KB - the marker we're after is usually one of the very last
            # lines, and a multi-GB stderr from a chatty worker would otherwise stall the
            # launcher
            with open(stderr_path, "rb") as in_stream:
                in_stream.seek(0, os.SEEK_END)
                size = in_stream.tell()
                in_stream.seek(max(0, size - 65536))
                tail = in_stream.read().decode("utf-8", errors="replace")
        except OSError:
            tail = ""
        for marker in _STDERR_OOM_MARKERS:
            if marker in tail:
                return ("oom", "Bump slurm_mem_gb (OOM marker '{}' seen in stderr).".format(
                    marker))

    return ("error", "Inspect the shard's stderr for the exception.")


def _percentile(values: List[float], percentile: float) -> float:
    """
    Compute the requested percentile via linear interpolation, matching numpy's default
    behaviour. Reimplemented in pure Python so the reporter has no numpy dependency at the
    point it runs (the framework module itself is otherwise numpy-free for everything
    outside the distance-shard adapter). Returns 0.0 on empty input so the caller can keep
    its log message uniform.
    """

    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    rank = (len(sorted_vals) - 1) * (percentile / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _report_run_stats(outcomes: dict, num_shards: int, parameters: dict,
                      stage: SlurmShardStage) -> None:
    """
    Emit the post-run reporter pass after `_poll_jobs()` completes: a failure-categorisation
    summary (5.15.3) followed by per-shard runtime statistics with parallel-efficiency and
    skew hints (5.15.4). The reporter never raises - it is purely observational and a bug
    in its formatting must not turn a successful stage into a failed run.

    Failure categorisation: groups shards by category (`oom`, `timeout`, ...) and emits one
    log line per non-empty category with the matching remediation hint, plus the list of
    affected shard IDs so users can correlate with the per-shard stderr paths logged inline
    in `_poll_jobs`.

    Runtime statistics: derived from the monotonic timestamps recorded in `_poll_jobs`
    (start = first time we observed RUNNING; end = first time job.done() returned True).
    Shards that completed between poll cycles (no observed RUNNING -> done transition) are
    counted in the "fast" bucket and excluded from median / p95 / max so they do not skew
    the timing picture. Skew = max / median; parallel_efficiency =
    sum(per-shard wall time) / (total array wall time × num_shards). Skew > 3 hints at an
    unbalanced strided split (suggest fewer / bigger shards via stage's
    `target_items_per_shard`); efficiency < 50 % hints at queue / scheduling overhead
    dominating (suggest reducing `slurm_array_parallelism`).

    :param outcomes: per-shard outcomes dict returned by `_poll_jobs`
    :param num_shards: total number of shards this run targeted (incl. shards that resumed
                       from cache and were not submitted - the reporter only stats the ones
                       it actually observed via the polling loop)
    :param parameters: job configuration parameters (read for `slurm_array_parallelism`
                       remediation hint and total wall-time computation)
    :param stage: the SLURM stage instance (read for `target_items_per_shard` skew hint)
    """

    if not outcomes:
        return

    try:
        _emit_failure_categorisation(outcomes, stage.stage_name)
        _emit_runtime_statistics(outcomes, num_shards, parameters, stage)
    except Exception as exc:  # purely observational - never let formatting bugs fail a run
        logger.warning("SLURM %s post-run reporter failed: %s. Stage results are unaffected.",
                       stage.stage_name, exc)


def _emit_failure_categorisation(outcomes: dict, stage_name: str) -> None:
    """One log line per non-empty failure category, with remediation hint and shard IDs."""

    by_category: dict = {}
    for outcome in outcomes.values():
        if outcome["category"] and outcome["category"] != "success":
            by_category.setdefault(outcome["category"], []).append(outcome)
    if not by_category:
        return

    # remediation hints - same map drives _categorize_failure, but here we look up by
    # category, so build a category -> hint table from _SLURM_STATE_CATEGORIES values
    hints: dict = {category: hint for (category, hint) in _SLURM_STATE_CATEGORIES.values()}
    hints.setdefault("error", "Inspect the shard's stderr for the exception.")

    for category in sorted(by_category):
        shard_ids = sorted(o["shard_id"] for o in by_category[category])
        hint = hints.get(category, "")
        logger.warning("SLURM %s post-run: %d shard(s) ended as '%s' (shards %s). %s",
                       stage_name, len(shard_ids), category, shard_ids, hint)


def _emit_runtime_statistics(outcomes: dict, num_shards: int, parameters: dict,
                             stage: SlurmShardStage) -> None:
    """Per-shard runtime stats + parallel-efficiency log line. Only counts shards we
    observed transitioning RUNNING -> done; shards completed between polls are reported
    separately as 'fast (< poll_wait)' so users don't think we lost them."""

    timed: List[float] = []
    untimed = 0
    earliest_start: Optional[float] = None
    latest_end: Optional[float] = None
    for outcome in outcomes.values():
        start, end = outcome["start_mono"], outcome["end_mono"]
        if start is not None and end is not None and end >= start:
            timed.append(end - start)
            if earliest_start is None or start < earliest_start:
                earliest_start = start
            if latest_end is None or end > latest_end:
                latest_end = end
        elif end is not None:
            untimed += 1
            if latest_end is None or end > latest_end:
                latest_end = end

    if not timed:
        # nothing to report on (every shard completed too fast to observe, or everything
        # failed before reaching RUNNING) - just note the headcount
        logger.info("SLURM %s post-run: %d/%d shard(s) completed; no shard ran long enough "
                    "to be timed by the %ds polling loop.", stage.stage_name,
                    len(outcomes), num_shards, parameters["slurm_poll_wait_seconds"])
        return

    timed_sorted = sorted(timed)
    median = timed_sorted[len(timed_sorted) // 2]
    p95 = _percentile(timed_sorted, 95.0)
    max_wall = timed_sorted[-1]
    skew = max_wall / median if median > 0 else float("inf")
    total_wall = (latest_end - earliest_start) if (latest_end is not None
                                                    and earliest_start is not None) else 0.0
    sum_wall = sum(timed)
    # parallel_efficiency = sum / (total_wall × num_shards). Use len(outcomes) instead of
    # num_shards when the stage was a resume (only a subset of shards was actually
    # submitted), so the denominator reflects what we observed not the theoretical max.
    parallelism_denom = total_wall * max(len(outcomes), 1)
    efficiency = (sum_wall / parallelism_denom * 100.0) if parallelism_denom > 0 else 0.0
    fast_note = (", {} shard(s) too fast to time (< {}s poll interval)".format(
        untimed, parameters["slurm_poll_wait_seconds"]) if untimed else "")

    logger.info(
        "SLURM %s post-run timings (%d/%d timed%s): "
        "median %.1fs, p95 %.1fs, max %.1fs, skew %.2f; total array wall %.1fs, "
        "parallel efficiency %.0f%%.",
        stage.stage_name, len(timed), num_shards, fast_note,
        median, p95, max_wall, skew, total_wall, efficiency)

    # Tuning hints - both fire only on the relevant signal so they aren't noise on a
    # well-balanced run
    if skew > 3 and len(timed) > 1:
        logger.info(
            "SLURM %s: shard runtime skew is %.1fx (median %.1fs vs. max %.1fs). The "
            "strided split is unbalanced for this workload; consider increasing the stage's "
            "target_items_per_shard (currently %s) so each shard handles more items and the "
            "long-tail clusters amortise across the bulk work.",
            stage.stage_name, skew, median, max_wall,
            getattr(stage, "target_items_per_shard", "?"))
    if efficiency < 50.0 and total_wall > 0:
        array_parallelism = parameters.get("slurm_array_parallelism")
        hint = ("Reduce slurm_array_parallelism (currently {}) so fewer shards queue "
                "concurrently and the polling / scheduling overhead drops.".format(
                    array_parallelism)) if array_parallelism is not None else \
               ("Set slurm_array_parallelism to a smaller number so fewer shards queue "
                "concurrently and the polling / scheduling overhead drops.")
        logger.info("SLURM %s: parallel efficiency %.0f%% suggests queue / scheduling "
                    "overhead dominates. %s", stage.stage_name, efficiency, hint)


def run_stage_on_slurm(stage: SlurmShardStage, config_file: str,
                       parameters: dict) -> Tuple[int, List[int]]:
    """
    Generic launcher: distribute one SlurmShardStage across SLURM array tasks. Resumable -
    only shards whose result files are missing (or whose fingerprint no longer matches) are
    (re)submitted. Returns the total number of shards and the sorted list of shard IDs that
    are still missing after the run completed (typically empty; a non-empty list means some
    shards failed or timed out and the caller should either raise or note it for the user).
    :param stage: the SLURM stage to run
    :param config_file: path to the INI configuration file of the analysis (the shard
                        workers rebuild the analysis state from this file)
    :param parameters: job configuration parameters
    :return: (num_shards, missing_shards) - num_shards is the total number of shards this run
             targeted; missing_shards is the still-missing list after collection
    """

    try:
        import submitit  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("The 'slurm' backend requires the optional 'submitit' package, which is not "
                           "installed.\nInstall it with 'pip install submitit' or "
                           "'conda install -c conda-forge submitit', or set the corresponding stage "
                           "backend to 'local'.")

    slurm_folder = stage.get_slurm_folder(parameters)
    os.makedirs(slurm_folder, exist_ok=True)
    num_shards = stage.num_shards(parameters)
    stage.prepare_state(parameters, slurm_folder)

    missing = _missing_shards(stage, slurm_folder, num_shards, parameters)
    num_done = num_shards - len(missing)
    if num_done:
        logger.info("Reusing %d/%d shard result(s) already present in '%s'; resuming.",
                    num_done, num_shards, slurm_folder)

    if missing:
        # Recursive-SLURM detection: log the parent allocation explicitly so the user sees
        # in the launcher log that an outer SLURM job is hosting this driver, then warn if
        # the driver's allocation looks too small to outlive the array it's about to
        # submit. This catches the common "submitted a 1-core 30-min driver, then asked it
        # to babysit a thousand-core 4-hour array" misallocation that would otherwise be
        # discovered only when the driver vanishes mid-poll.
        parent_info = _detect_parent_slurm_allocation()
        if parent_info is not None:
            logger.debug("Detected recursive SLURM execution: driver runs inside parent "
                        "job %s (cpus=%s, time_min=%s). The launcher will pass "
                        "--cpu-bind=none to the array's srun to avoid the nested "
                        "'CPU binding outside of job step allocation' error.",
                        parent_info["job_id"], parent_info.get("cpus"),
                        parent_info.get("time_min"))
            _warn_on_driver_misallocation(parent_info, parameters, num_shards,
                                          stage.stage_name)

        # cluster="slurm" makes submission fail loudly if SLURM is unavailable, rather than
        # silently falling back to local execution - the user explicitly requested this backend
        executor = submitit.AutoExecutor(folder=slurm_folder, cluster="slurm")
        executor.update_parameters(**stage.executor_params(parameters))

        logger.info("Submitting %d SLURM %s shard(s) (of %d total); submitit folder: %s.",
                    len(missing), stage.stage_name, num_shards, slurm_folder)
        jobs = executor.map_array(stage.run_shard,
                                  [config_file] * len(missing),
                                  [num_shards] * len(missing),
                                  missing)
        outcomes = _poll_jobs(missing, jobs, num_shards, parameters, stage.stage_name)
        # Post-run reporter: per-failure category remediation hints + per-shard timing
        # statistics (median / p95 / max / skew) + parallel-efficiency hint. Observational
        # only - never raises (any internal error is logged as a warning).
        _report_run_stats(outcomes, num_shards, parameters, stage)

    # report which shards are still missing after the polling loop completed; the caller
    # decides what to do (typically: raise with a "re-run to resume" message)
    still_missing = [sid for sid in range(num_shards)
                     if not os.path.exists(stage.shard_result_path(slurm_folder, sid))]
    return num_shards, still_missing


def cleanup_stage_folder(stage: SlurmShardStage, parameters: dict) -> None:
    """
    Discard the per-stage SLURM folder after a successful stage assembly. Without this, the
    `_slurm/` tree accumulates per-shard pickles, manifests, side-effects tarballs,
    `items.json` / `state.pkl` artifacts, and submitit per-job logs across every stage * run *
    resubmission - on a large analysis the folder can grow into the GB range with no
    automatic mitigation. Cleanup only fires after the launcher's assembly step has succeeded
    end-to-end (the real outputs are already in their final destinations under the analysis
    folder), so resumability of an interrupted run is unaffected.

    The behaviour is gated by `parameters["slurm_keep_shard_results"]`: when False (default)
    the entire `<slurm_root_folder>/<stage.folder_name>/` tree is removed; when True the
    folder is left untouched. Callers invoke this at the very end of each `_slurm` assembler
    in `tools.py`, after every check that could raise has already passed - a partial assembly
    must not delete the cache.

    Idempotent and tolerant of a missing folder; removal failures are logged but never
    raised, so a transient I/O hiccup during teardown does not mask the (already successful)
    stage result.

    :param stage: the SLURM stage whose folder is being cleaned up
    :param parameters: job configuration parameters
    """

    if parameters.get("slurm_keep_shard_results", False):
        return
    slurm_folder = stage.get_slurm_folder(parameters)
    if not os.path.isdir(slurm_folder):
        return
    try:
        shutil.rmtree(slurm_folder)
    except OSError as exc:
        # Don't raise: the stage already succeeded and the user has their outputs. A failure
        # here is purely a disk-hygiene issue, and surfacing it as an error would convert a
        # successful run into a spurious failure.
        logger.warning("Failed to clean up SLURM %s folder '%s' after successful assembly: "
                       "%s. The folder can be removed manually; set slurm_keep_shard_results"
                       "=True to disable automatic cleanup.",
                       stage.stage_name, slurm_folder, exc)
        return
    logger.info("Removed SLURM %s shard cache at '%s' after successful assembly "
                "(set slurm_keep_shard_results=True to keep it).",
                stage.stage_name, slurm_folder)


# ---------------------------------------------------------------------------
# Shared shard-process bootstrap
# ---------------------------------------------------------------------------


def _rebuild_mol_system(config_file: str, stage_number: int | None = None):
    """
    Boilerplate shared by every SLURM shard's `run_shard()` entry point: pin the multiprocessing
    start method to spawn (so the inner Pool the shard's compute_* method will construct does not
    fork from a multi-threaded BLAS parent), load the analysis configuration from the INI file,
    and rebuild the TransportProcesses state. The shard process is a fresh Python interpreter
    launched by srun and does not inherit the launcher's start-method choice, so the spawn pin
    must happen here even though the launcher already ran it.
    :param config_file: path to the INI configuration file of the analysis
    :param stage_number: pipeline stage this shard executes (cls.stage_number). The config is rebuilt
        with active_stages=(stage_number, stage_number) so only that stage's prerequisites are set up
        and validated - the shard does not re-walk input trees or re-detect setup values for stages it
        will not run (see AnalysisConfig._runs_stage). None scopes to no stage (used only by callers
        with no stage-specific prerequisites).
    :return: (AnalysisConfig, TransportProcesses) - the config is returned alongside the
             TransportProcesses so callers can also read out the per-stage SLURM folder, etc.
    """

    from transport_tools.libs.utils import configure_multiprocessing_start_method
    configure_multiprocessing_start_method()

    from transport_tools.libs.config import AnalysisConfig, ShardContext
    from transport_tools.libs.tools import TransportProcesses

    # is_shard=True: this rebuild runs on a SLURM compute node whose CPU count is unrelated to the
    # launcher's, and num_cpus is inert in a shard (the inner pool is sized from slurm_cpus_per_task).
    # The shard context skips the launcher-only num_cpus validation block that would otherwise abort
    # the shard whenever num_cpus exceeds the compute node's core count. active_stages pins the rebuild
    # to this shard's single stage so its setup/validation is scoped to that stage alone.
    active_stages = None if stage_number is None else (stage_number, stage_number)
    config = AnalysisConfig(config_file, logging=False, context=ShardContext(is_shard=True),
                            active_stages=active_stages)
    return config, TransportProcesses(config)


# ---------------------------------------------------------------------------
# Stage 2 adapter: TunnelNetworksShardStage
# ---------------------------------------------------------------------------


class TunnelNetworksShardStage(SlurmShardStage):
    """
    Stage-2 adapter: distributes `process_tunnel_networks` over a SLURM array. Each shard takes
    a strided subset of the flat list of MD labels (CAVER input folders) and runs
    `_pre_process_single_tunnel_network` for every md_label it owns - reading the CAVER results
    from the shared filesystem, applying the per-MD transformation matrix produced by stage 1,
    and writing the per-md_label `<md_label>_caver.dump` into the shared internal folder. The
    per-shard result pickle stored under `<slurm_folder>/` is a tiny JSON-shaped marker holding
    the list of md_labels the shard processed (it lets the launcher assert completeness without
    re-walking the user-facing folders); the real outputs live under
    `orig_caver_network_data_path` (the `.dump` files) and, when `visualize_transformed_tunnels`
    is True, under `orig_caver_vis_path/<md_label>/` (the transformed PDB plus per-cluster CGO
    artefacts) - both tracked as side-effects backed up to a per-shard tar.gz so a wiped output
    folder can be restored on resume without re-reading CAVER.

    The md_label enumeration is built once in the launcher (`self.caver_input_folders` in the
    submission-order locked by `AnalysisConfig.get_input_folders()`) and persisted to
    `<slurm_folder>/items.json` so every shard reads its strided assignment from a single
    artifact instead of re-deriving the enumeration.

    Determinism. Bit-identical to the local backend across any `slurm_num_shards` choice:
    workers process their assigned md_labels independently (no cross-shard state), and each
    md_label's `<md>_caver.dump` plus per-MD visualisation folder are written by exactly one
    worker. No assembly reduction step in the parent - just a completeness check over the
    set of processed md_labels.
    """

    folder_name = "stage02_tunnel_networks"
    stage_name = "tunnel-networks"
    stage_number = 2
    job_name = "tt_tunnel_networks"
    # workers write `<md_label>_caver.dump` to `orig_caver_network_data_path` AND, when
    # visualize_transformed_tunnels=True, per-md_label visualisation files (transformed PDB +
    # entity CGO pickles + view_network.py) to `orig_caver_vis_path/<md_label>/`. The SLURM
    # helper backs these up to a sidecar tar.gz so they can be restored on resume without
    # re-running the CAVER parsing.
    has_side_effects = True
    # heuristic granularity when auto-deriving the shard count:
    # Per-MD work is moderate (CAVER
    # parsing + numpy transformation + optional CGO serialisation), so 50/shard hits a
    # reasonable load balance for studies with hundreds of MD sims without inflating SLURM
    # array overhead.
    target_items_per_shard = 50
    # parameters that affect tunnel-networks output: changing any of them invalidates the
    # already-computed shards. The items-list hash is added in `fingerprint()` so changes to
    # the set of MD labels (different `caver_results_folder_pattern` matches) also invalidate
    # stale shards. `visualize_transformed_tunnels` is included for the same reason
    # `visualize_layered_clusters` is on stage 3: flipping it toggles whether the worker writes
    # the per-md visualisation side-effects, so a False→True transition would otherwise reuse
    # a cache with no side-effect backup and the user would be left with no visualisations.
    fingerprint_keys: Tuple[str, ...] = (
        "caver_results_folder_pattern",
        "snapshots_per_simulation",
        "caver_traj_offset",
        "process_bottleneck_residues",
        "visualize_transformed_tunnels",
    )

    def __init__(self, items: List[str]):
        # canonicalise to a list of plain strings (detaches from caller's mutable state) so
        # JSON round-trip yields the same hash
        self.items: List[str] = [str(md) for md in items]

    def num_shards(self, parameters: dict) -> int:
        num_shards = parameters["slurm_num_shards"]
        n_items = len(self.items)
        if num_shards is None:
            num_shards = derive_num_shards(
                n_items, array_parallelism=parameters["slurm_array_parallelism"],
                target_per_shard=self.target_items_per_shard)
        return max(1, min(num_shards, max(n_items, 1)))

    @classmethod
    def shard_result_path(cls, slurm_folder: str, shard_id: int) -> str:
        return os.path.join(slurm_folder, "shard_result_{:05d}.pkl".format(shard_id))

    @classmethod
    def run_shard(cls, config_file: str, num_shards: int, shard_id: int) -> int:
        _config, mol_system = _rebuild_mol_system(config_file, cls.stage_number)
        return mol_system.compute_tunnel_networks_shard(num_shards, shard_id)

    def fingerprint(self, parameters: dict, num_shards: int) -> dict:
        info = super().fingerprint(parameters, num_shards)
        info["num_items"] = len(self.items)
        info["items_hash"] = hashlib.sha256(
            json.dumps(self.items, sort_keys=False).encode("utf-8")).hexdigest()
        return info

    def prepare_state(self, parameters: dict, slurm_folder: str) -> None:
        """
        Write the canonical md_label enumeration into the SLURM folder so every shard reads
        its strided assignment from a single source. The launcher always overwrites this file
        so it reflects the current run; stale enumerations from earlier runs are detected by
        the `items_hash` fingerprint key, not by trusting the file's age.
        """

        items_file = os.path.join(slurm_folder, self.items_filename)
        tmp_file = items_file + ".tmp"
        with open(tmp_file, "w") as out_stream:
            json.dump(self.items, out_stream)
        os.replace(tmp_file, items_file)

    @classmethod
    def load_items(cls, slurm_folder: str) -> List[str]:
        """
        Read back the md_label enumeration that the launcher wrote via `prepare_state()`.
        Called by each shard process at startup to recover its assignment. Returns a list of
        md_label strings in the original submission order.
        """

        items_file = os.path.join(slurm_folder, cls.items_filename)
        if not os.path.exists(items_file):
            raise FileNotFoundError(
                "Tunnel-networks items file not found at '{}'. The SLURM launcher writes this "
                "via TunnelNetworksShardStage.prepare_state() before submitting shards.".format(
                    items_file))
        with open(items_file) as in_stream:
            return json.load(in_stream)

    def expected_side_effect_files(self, parameters: dict, slurm_folder: str,
                                   shard_id: int) -> Optional[List[str]]:
        """
        Tunnel-networks worker writes one `<md_label>_caver.dump` per md_label into
        `orig_caver_network_data_path` and, when `visualize_transformed_tunnels` is True, a
        per-md_label visualisation folder under `orig_caver_vis_path/<md_label>/` (transformed
        PDB + view_network.py + entity CGO pickles). Both are NOT in the shard result pickle
        (which is just a list of processed md_labels) - they live in the user-facing internal
        / visualisation folders and can be wiped between runs without invalidating the SLURM
        cache. We track every per-md side-effect path in the manifest written by `run_shard()`;
        if any one of them is missing, the framework either restores from the per-shard tar.gz
        backup or invalidates the cache and resubmits.

        Unlike the layering stages, the dump file is part of the side-effects too (not just
        the visualisation) - it lives outside the SLURM folder and is what the downstream
        stages consume, so wiping it must be detected.
        """

        # When the manifest is missing (older shard, or wiped alongside the side-effects),
        # return None to force resubmission. The dump files are always produced by the worker
        # regardless of the visualize_* knob, so there is no "manifest expected to be empty"
        # path here - every successful shard writes at least one dump file per md_label.
        return self.read_shard_manifest(slurm_folder, shard_id)

    @classmethod
    def derive_side_effect_paths(cls, parameters: dict,
                                 processed_md_labels: List[str]) -> List[str]:
        """
        Derive the list of per-md side-effect file paths from the shard's processed md_labels.
        For every md_label the worker writes:
          * `<orig_caver_network_data_path>/<md_label>_caver.dump` - the orig_network dump,
            mandatory (consumed by downstream stages)
          * `<orig_caver_vis_path>/<md_label>/<md_label>_C_trans_rot.pdb` - transformed PDB,
            only when `visualize_transformed_tunnels` is True
          * `<orig_caver_vis_path>/<md_label>/view_network.py` - PyMOL view script, only when
            `visualize_transformed_tunnels` is True
          * `<orig_caver_vis_path>/<md_label>/*.cgo.dump.gz` - per-entity CGO pickles, only
            when `visualize_transformed_tunnels` is True (count is data-dependent, so we glob
            the on-disk visualisation folder after every worker has flushed its output - this
            method is called from the shard process on the SLURM compute node after the inner
            Pool has finished, so the listing is authoritative)
        Centralised here so the worker side and a future direct validation path in tests stay
        consistent on the path convention.
        """

        import glob
        paths: List[str] = []
        visualize = bool(parameters.get("visualize_transformed_tunnels", False))
        for md_label in processed_md_labels:
            dump_path = os.path.join(parameters["orig_caver_network_data_path"],
                                     "{}_caver.dump".format(md_label))
            paths.append(dump_path)
            # size sidecar written by save_orig_network() next to the dump (suffix
            # TunnelNetwork._orig_summary_suffix = ".cluster_sizes"); the stage-3 layering
            # launcher reads it to skip the full orig_network load, so it must be backed up /
            # restored with the dump or a resume that wipes network_data would lose it.
            paths.append(dump_path + ".cluster_sizes")
            if visualize:
                md_vis_folder = os.path.join(parameters["orig_caver_vis_path"], md_label)
                paths.append(os.path.join(md_vis_folder,
                                          "{}_C_trans_rot.pdb".format(md_label)))
                paths.append(os.path.join(md_vis_folder, "view_network.py"))
                # per-entity CGO pickles: each TunnelCluster.save_cgo() writes
                # `<entity_pymol_label>_cgo.dump.gz` into the md_vis_folder (see
                # networks.py:648 - `vis_file = entity_pymol_label + "_cgo.dump.gz"`). Glob
                # the folder rather than re-deriving entity_pymol_labels so the manifest
                # matches whatever the worker actually wrote, regardless of how many tunnel
                # clusters CAVER produced for this md_label.
                paths.extend(sorted(glob.glob(os.path.join(md_vis_folder, "*_cgo.dump.gz"))))
        return paths


# ---------------------------------------------------------------------------
# Stage 3 adapter: TunnelLayeringShardStage
# ---------------------------------------------------------------------------

class TunnelLayeringShardStage(SlurmShardStage):
    """
    Stage-3 adapter: distributes `create_layered_description4tunnel_networks` over a SLURM
    array. Each shard takes a strided subset of the flat (md_label, cluster_id) enumeration
    (so sklearn-heavy clusters balance across shards regardless of their position in the
    enumeration) and writes a per-shard `.pkl` file containing `[(cls_id, md_label,
    LayeredPathSet), ...]`. The launcher streams shard pickles back, buffers per md_label,
    and commits each md_label's layered entities in original submission order so the saved
    `layered_network.dump` is byte-identical to what the local backend produces.

    The (md_label, cluster_id) enumeration is built once in the launcher (by walking the
    orig_network.dump files from stage 2) and persisted to `<slurm_folder>/items.json`. Each
    shard reads that single artifact instead of re-deriving the enumeration in every shard
    process, so on a 200-shard run the orig_network files are not parsed 200 times just to
    discover work-item ids.

    Determinism. Bit-identical to the local backend across any `slurm_num_shards` choice.
    Each `(md_label, cluster_id)` is layered by exactly one worker (no cross-shard state),
    and the assembler buffers per-md_label then commits in original submission order, so
    the saved `layered_network.dump` pickle layout - which encodes dict-insertion order -
    matches the local backend byte-for-byte regardless of completion order.
    """

    folder_name = "stage03_tunnel_layering"
    stage_name = "tunnel-layering"
    stage_number = 3
    job_name = "tt_tunnel_layering"
    # workers write per-cluster .py + per-layer node PDBs into the visualisation folder when
    # visualize_layered_clusters=True; the SLURM helper backs these up to a sidecar tar.gz so
    # they can be restored on resume without re-running the per-cluster sklearn stack
    has_side_effects = True
    # heuristic granularity when auto-deriving the shard count:
    # Layering tasks have very variable per-task cost (sklearn DBSCAN/KMeans/IsolationForest/
    # hdbscan), so striding inside the shard plus 200/shard hits a reasonable load balance
    # without inflating SLURM overhead.
    target_items_per_shard = 200
    # parameters that affect tunnel-layering output: changing any of them invalidates the
    # already-computed shards. The items-list hash is added in `fingerprint()` so changes to
    # the upstream stage-2 orig_networks (different MD labels, different cluster counts) also
    # invalidate stale shards. `visualize_layered_clusters` is included because flipping it
    # toggles whether the worker writes per-cluster .py + nodes/ side-effects: a cache built
    # with visualize=False has an empty manifest and no tar.gz backup, so the side-effect
    # resume path would trivially pass even though no side-effects exist - the fingerprint
    # entry forces a full resubmission on the False→True transition.
    fingerprint_keys: Tuple[str, ...] = (
        "layer_thickness",
        "relevant_tunnel_min_radius",
        "relevant_tunnel_min_length",
        "relevant_tunnel_max_curvature",
        "random_seed",
        "clustering_max_num_rep_frag",
        "use_cluster_spread",
        "visualize_layered_clusters",
    )

    def __init__(self, items: List[Tuple[str, int]]):
        # canonicalise to lists-of-lists so JSON round-trip yields the same hash (tuples
        # serialise to lists, and json.load returns lists); also detaches the cached list
        # from the caller's mutable state
        self.items: List[List[Any]] = [[md, int(cls)] for md, cls in items]

    def num_shards(self, parameters: dict) -> int:
        num_shards = parameters["slurm_num_shards"]
        n_items = len(self.items)
        if num_shards is None:
            num_shards = derive_num_shards(
                n_items, array_parallelism=parameters["slurm_array_parallelism"],
                target_per_shard=self.target_items_per_shard)
        return max(1, min(num_shards, max(n_items, 1)))

    @classmethod
    def shard_result_path(cls, slurm_folder: str, shard_id: int) -> str:
        return os.path.join(slurm_folder, "shard_result_{:05d}.pkl".format(shard_id))

    @classmethod
    def run_shard(cls, config_file: str, num_shards: int, shard_id: int) -> int:
        _config, mol_system = _rebuild_mol_system(config_file, cls.stage_number)
        return mol_system.compute_tunnel_layering_shard(num_shards, shard_id)

    def fingerprint(self, parameters: dict, num_shards: int) -> dict:
        info = super().fingerprint(parameters, num_shards)
        info["num_items"] = len(self.items)
        info["items_hash"] = hashlib.sha256(
            json.dumps(self.items, sort_keys=False).encode("utf-8")).hexdigest()
        return info

    def prepare_state(self, parameters: dict, slurm_folder: str) -> None:
        """
        Write the canonical (md_label, cluster_id) enumeration into the SLURM folder so every
        shard can read its strided assignment from a single source. The launcher always
        overwrites this file so it reflects the current run; stale enumerations from earlier
        runs are detected by the `items_hash` fingerprint key, not by trusting the file's age.
        """

        items_file = os.path.join(slurm_folder, self.items_filename)
        tmp_file = items_file + ".tmp"
        with open(tmp_file, "w") as out_stream:
            json.dump(self.items, out_stream)
        os.replace(tmp_file, items_file)

    @classmethod
    def load_items(cls, slurm_folder: str) -> List[List[Any]]:
        """
        Read back the `(md_label, cluster_id)` enumeration that the launcher wrote via
        `prepare_state()`. Called by each shard process at startup to recover its assignment.
        Returns a list of `[md_label, cls_id]` pairs (JSON tuples are deserialised as lists).
        """

        items_file = os.path.join(slurm_folder, cls.items_filename)
        if not os.path.exists(items_file):
            raise FileNotFoundError(
                "Tunnel-layering items file not found at '{}'. The SLURM launcher writes this "
                "via TunnelLayeringShardStage.prepare_state() before submitting shards.".format(
                    items_file))
        with open(items_file) as in_stream:
            return json.load(in_stream)

    def expected_side_effect_files(self, parameters: dict, slurm_folder: str,
                                   shard_id: int) -> Optional[List[str]]:
        """
        Tunnel-layering worker writes a per-cluster `Cluster_<id>.py` PyMOL script (and a
        `nodes/cls<NNN>_layer<X>_pts.pdb` family) into the visualisation folder when
        `visualize_layered_clusters` is True. These are NOT in the result pickle - their data
        is consumed inside the worker - so if a prior interrupted run left the result pickle
        but the visualisation folder was wiped between runs, the resume optimisation would
        silently skip regeneration. We track the canonical per-cluster `.py` file (one per
        cluster_id this shard owns) in the manifest: if any one of them is missing, the
        framework invalidates the cached pickle and resubmits the shard.

        When `visualize_layered_clusters` is False, the shard has no worker side-effects and
        we return an empty manifest (the read manifest will also be empty - both branches
        validate trivially as "present").
        """

        if not parameters.get("visualize_layered_clusters", False):
            return []
        # Read the manifest written by `run_shard()`. If absent (older shard, or wiped
        # together with the side-effects), return None to force resubmission.
        return self.read_shard_manifest(slurm_folder, shard_id)

    @classmethod
    def derive_side_effect_paths(cls, parameters: dict,
                                 cluster_results: List[Tuple[int, str, Any]]) -> List[str]:
        """
        Derive the list of per-cluster side-effect file paths from the shard's result tuples.
        For every layered cluster `prep_visualization` writes two artefacts into
        `<layered_caver_vis_path>/<md_label>/`:
          * one PyMOL script at `Cluster_<id>.py` (deterministic name)
          * a family of PDB files at `nodes/cls<NNN>-<layer_id>-<sub_cls_id>.pdb` (one per
            sub-cluster of each layer; the count depends on the layering result)
        We resolve the PDB family by globbing the `nodes/` folder with the cluster's
        `visualization_prefix` ("cls<NNN>"); this runs on the SLURM compute node after every
        worker has flushed its output, so the on-disk listing is authoritative. Empty
        clusters (worker returned an empty LayeredPathSet) skip `prep_visualization` and
        contribute nothing. Centralised here so the worker side and a future direct
        validation path in tests stay consistent on the path convention.
        """

        import glob
        paths: List[str] = []
        for cls_id, md_label, layered_path_set in cluster_results:
            if layered_path_set.is_empty():
                continue  # empty clusters skip prep_visualization, so no side-effect file
            md_folder = os.path.join(parameters["layered_caver_vis_path"], md_label)
            paths.append(os.path.join(md_folder, "Cluster_{}.py".format(int(cls_id))))
            # enumerate the per-layer node PDBs the worker wrote into nodes/ for this cluster
            paths.extend(sorted(glob.glob(os.path.join(
                md_folder, "nodes", "cls{:03d}-*.pdb".format(int(cls_id))))))
        return paths


# ---------------------------------------------------------------------------
# Stage 4 adapter: DistanceShardStage
# ---------------------------------------------------------------------------


class DistanceShardStage(SlurmShardStage):
    """
    Stage-4 adapter: per-shard inter-cluster distance computation. The shard writes a compact
    (K, 3) float64 array of (cluster1_id, cluster2_id, distance) rows; the launcher
    concatenates them to assemble the condensed distance matrix.

    Determinism. Bit-identical to the local backend across any `slurm_num_shards` choice.
    Each upper-triangle pair is computed by exactly one worker and the launcher scatters
    pairs straight into a pre-allocated condensed vector via `condensed_pair_index(rows,
    cols, num_clusters)` - the destination slot is a deterministic function of
    (cluster1_id, cluster2_id), not of completion order. The per-shard rounding to
    `precision` happens inside the worker (same code path the local backend uses), so the
    matrix is identical down to the last decimal.
    """

    folder_name = "stage04_distances"
    stage_name = "distance"
    stage_number = 4
    job_name = "tt_distances"
    # heuristic granularity when auto-deriving the shard count: aim for ~20000 cluster pairs
    # per shard (cluster-cluster comparisons are cheap individually but plentiful for large
    # supercluster counts; smaller shards would inflate SLURM array overhead)
    target_items_per_shard = 20000
    # parameters that affect distance results: changing any of them invalidates the
    # already-computed shards. num_clusters is added in `fingerprint()` from runtime state.
    fingerprint_keys: Tuple[str, ...] = ("clustering_cutoff", "calculate_exact_path_distances")

    def __init__(self, num_clusters: int):
        self.num_clusters = num_clusters
        self.n_jobs = (num_clusters ** 2 - num_clusters) // 2

    def num_shards(self, parameters: dict) -> int:
        num_shards = parameters["slurm_num_shards"]
        if num_shards is None:
            num_shards = derive_num_shards(
                self.n_jobs, array_parallelism=parameters["slurm_array_parallelism"],
                target_per_shard=self.target_items_per_shard)
        return max(1, min(num_shards, self.n_jobs))

    @classmethod
    def shard_result_path(cls, slurm_folder: str, shard_id: int) -> str:
        return os.path.join(slurm_folder, "shard_result_{:05d}.npy".format(shard_id))

    @classmethod
    def run_shard(cls, config_file: str, num_shards: int, shard_id: int) -> int:
        config, mol_system = _rebuild_mol_system(config_file, cls.stage_number)
        results = mol_system.compute_distance_shard(num_shards, shard_id)

        # write the per-shard (K, 3) float64 array atomically (tmp + os.replace)
        array = np.array(results, dtype=np.float64) if results \
            else np.empty((0, 3), dtype=np.float64)
        out_file = cls.shard_result_path(cls.get_slurm_folder(config.parameters), shard_id)
        tmp_file = out_file + ".tmp.npy"
        np.save(tmp_file, array)
        os.replace(tmp_file, out_file)
        return len(results)

    def fingerprint(self, parameters: dict, num_shards: int) -> dict:
        info = super().fingerprint(parameters, num_shards)
        info["num_clusters"] = self.num_clusters
        return info


# ---------------------------------------------------------------------------
# Stage 7 adapter: AquaductNetworksShardStage
# ---------------------------------------------------------------------------


class AquaductNetworksShardStage(SlurmShardStage):
    """
    Stage-7 adapter: distributes `process_aquaduct_networks` over a SLURM array. Mirror of
    `TunnelNetworksShardStage` but for AQUA-DUCT inputs. Each shard takes a strided subset of the
    flat list of MD labels (AQUA-DUCT input folders) and runs
    `_pre_process_single_aquaduct_network` for every md_label it owns - reading the tar.gz of
    raw paths from the shared filesystem, applying the per-MD transformation matrix, and
    writing the per-md_label `<md_label>_aqua.dump` into the shared internal folder. The
    per-shard result pickle stored under `<slurm_folder>/` is a tiny JSON-shaped marker holding
    the list of md_labels the shard processed (it lets the launcher assert completeness without
    re-walking the user-facing folders); the real outputs live under
    `orig_aquaduct_network_data_path` (the `.dump` files) and, when
    `visualize_transformed_transport_events` is True, under
    `orig_aquaduct_vis_path/<md_label>/` (the transformed PDB plus per-path CGO artefacts) -
    both tracked as side-effects backed up to a per-shard tar.gz so a wiped output folder can
    be restored on resume without re-extracting the AQUA-DUCT tarballs.

    Nested-pool note: the local backend has TWO branches - when `_aquaduct_single_event_inputs`
    is True (typical big-MD studies), it runs the per-MD work in an outer Pool with
    `parallel_processing=False` to avoid nested pools; otherwise it runs sequentially with
    `parallel_processing=True` so each MD's raw-path reading itself parallelises. The SLURM
    backend always takes the outer-pool path - each shard runs its own per-MD pool with
    `parallel_processing=False` - so the inner raw-paths pool is never spawned from a worker
    process; this side-steps the nested-pool guard at the top of
    `_pre_process_single_aquaduct_network` and lets the heavy I/O on the tar.gz file (which
    dominates per-MD wall time) overlap across md_labels within each shard.

    The md_label enumeration is built once in the launcher (`self.aquaduct_input_folders` in
    the submission-order locked by `AnalysisConfig.get_input_folders()`) and persisted to
    `<slurm_folder>/items.json` so every shard reads its strided assignment from a single
    artifact instead of re-deriving the enumeration.

    Determinism. Bit-identical to the local backend across any `slurm_num_shards` choice:
    workers process their assigned md_labels independently (no cross-shard state), and each
    md_label's `<md>_aqua.dump` plus per-MD visualisation folder are written by exactly one
    worker. No assembly reduction step in the parent - just a completeness check over the
    set of processed md_labels. The local-backend's two branches (outer-pool vs.
    inner-pool) are equivalent at the per-MD output level - they only differ in where the
    raw-path Pool runs - so the SLURM-always-outer-pool choice does not affect the dump
    contents.
    """

    folder_name = "stage07_aquaduct_networks"
    stage_name = "aquaduct-networks"
    stage_number = 7
    job_name = "tt_aquaduct_networks"
    # workers write `<md_label>_aqua.dump` to `orig_aquaduct_network_data_path` AND, when
    # visualize_transformed_transport_events=True, per-md_label visualisation files (transformed
    # PDB + per-path CGO pickles + view_network.py) to `orig_aquaduct_vis_path/<md_label>/`. The
    # SLURM helper backs these up to a sidecar tar.gz so they can be restored on resume without
    # re-extracting the AQUA-DUCT tarballs.
    has_side_effects = True
    # heuristic granularity when auto-deriving the shard count:
    # Per-MD work is moderate-to-heavy (tar.gz extraction + numpy transformation of every
    # traced raw path + optional CGO serialisation). 50/shard hits a reasonable load balance
    # for studies with hundreds of MD sims without inflating SLURM array overhead, matching
    # the tunnel-networks granularity.
    target_items_per_shard = 50
    # parameters that affect aquaduct-networks output: changing any of them invalidates the
    # already-computed shards. The items-list hash is added in `fingerprint()` so changes to
    # the set of AQUA-DUCT input folders (different `aquaduct_results_folder_pattern` matches,
    # or different `aquaduct_results_path` roots) also invalidate stale shards.
    # `visualize_transformed_transport_events` is included for the same reason
    # `visualize_transformed_tunnels` is on stage 2: flipping it toggles whether the worker
    # writes the per-md visualisation side-effects, so a False->True transition would otherwise
    # reuse a cache with no side-effect backup and the user would be left with no
    # visualisations.
    fingerprint_keys: Tuple[str, ...] = (
        "aquaduct_results_folder_pattern",
        "aquaduct_results_pdb_filename",
        "aquaduct_results_relative_tarfile",
        "aquaduct_traced_residues_filter",
        "event_min_distance",
        "aquaduct_allow_empty_folders",
        "visualize_transformed_transport_events",
    )

    def __init__(self, items: List[str]):
        # canonicalise to a list of plain strings (detaches from caller's mutable state) so
        # JSON round-trip yields the same hash
        self.items: List[str] = [str(md) for md in items]

    def num_shards(self, parameters: dict) -> int:
        num_shards = parameters["slurm_num_shards"]
        n_items = len(self.items)
        if num_shards is None:
            num_shards = derive_num_shards(
                n_items, array_parallelism=parameters["slurm_array_parallelism"],
                target_per_shard=self.target_items_per_shard)
        return max(1, min(num_shards, max(n_items, 1)))

    @classmethod
    def shard_result_path(cls, slurm_folder: str, shard_id: int) -> str:
        return os.path.join(slurm_folder, "shard_result_{:05d}.pkl".format(shard_id))

    @classmethod
    def run_shard(cls, config_file: str, num_shards: int, shard_id: int) -> int:
        _config, mol_system = _rebuild_mol_system(config_file, cls.stage_number)
        return mol_system.compute_aquaduct_networks_shard(num_shards, shard_id)

    def fingerprint(self, parameters: dict, num_shards: int) -> dict:
        info = super().fingerprint(parameters, num_shards)
        info["num_items"] = len(self.items)
        info["items_hash"] = hashlib.sha256(
            json.dumps(self.items, sort_keys=False).encode("utf-8")).hexdigest()
        return info

    def prepare_state(self, parameters: dict, slurm_folder: str) -> None:
        """
        Write the canonical md_label enumeration into the SLURM folder so every shard reads
        its strided assignment from a single source. The launcher always overwrites this file
        so it reflects the current run; stale enumerations from earlier runs are detected by
        the `items_hash` fingerprint key, not by trusting the file's age.
        """

        items_file = os.path.join(slurm_folder, self.items_filename)
        tmp_file = items_file + ".tmp"
        with open(tmp_file, "w") as out_stream:
            json.dump(self.items, out_stream)
        os.replace(tmp_file, items_file)

    @classmethod
    def load_items(cls, slurm_folder: str) -> List[str]:
        """
        Read back the md_label enumeration that the launcher wrote via `prepare_state()`.
        Called by each shard process at startup to recover its assignment. Returns a list of
        md_label strings in the original submission order.
        """

        items_file = os.path.join(slurm_folder, cls.items_filename)
        if not os.path.exists(items_file):
            raise FileNotFoundError(
                "Aquaduct-networks items file not found at '{}'. The SLURM launcher writes "
                "this via AquaductNetworksShardStage.prepare_state() before submitting "
                "shards.".format(items_file))
        with open(items_file) as in_stream:
            return json.load(in_stream)

    def expected_side_effect_files(self, parameters: dict, slurm_folder: str,
                                   shard_id: int) -> Optional[List[str]]:
        """
        Aquaduct-networks worker writes one `<md_label>_aqua.dump` per md_label into
        `orig_aquaduct_network_data_path` and, when `visualize_transformed_transport_events`
        is True, a per-md_label visualisation folder under `orig_aquaduct_vis_path/<md_label>/`
        (transformed PDB + view_network.py + per-path CGO pickles). Both are NOT in the shard
        result pickle (which is just a list of processed md_labels) - they live in the
        user-facing internal / visualisation folders and can be wiped between runs without
        invalidating the SLURM cache. We track every per-md side-effect path in the manifest
        written by `run_shard()`; if any one of them is missing, the framework either restores
        from the per-shard tar.gz backup or invalidates the cache and resubmits.

        Like the tunnel-networks stage, the dump file is part of the side-effects too (not
        just the visualisation) - it lives outside the SLURM folder and is what the downstream
        layering stage consumes, so wiping it must be detected.
        """

        # When the manifest is missing (older shard, or wiped alongside the side-effects),
        # return None to force resubmission. The dump files are always produced by the worker
        # regardless of the visualize_* knob, so there is no "manifest expected to be empty"
        # path here - every successful shard writes at least one dump file per md_label.
        return self.read_shard_manifest(slurm_folder, shard_id)

    @classmethod
    def derive_side_effect_paths(cls, parameters: dict,
                                 processed_md_labels: List[str]) -> List[str]:
        """
        Derive the list of per-md side-effect file paths from the shard's processed md_labels.
        For every md_label the worker writes:
          * `<orig_aquaduct_network_data_path>/<md_label>_aqua.dump` - the orig_network dump,
            mandatory (consumed by the downstream stage-8 layering)
          * `<orig_aquaduct_vis_path>/<md_label>/<md_label>_A_trans_rot.pdb` - transformed
            PDB, only when `visualize_transformed_transport_events` is True
          * `<orig_aquaduct_vis_path>/<md_label>/view_network.py` - PyMOL view script, only
            when `visualize_transformed_transport_events` is True
          * `<orig_aquaduct_vis_path>/<md_label>/*_cgo.dump.gz` - per-path CGO pickles, only
            when `visualize_transformed_transport_events` is True (count is data-dependent,
            so we glob the on-disk visualisation folder after every worker has flushed its
            output - this method is called from the shard process on the SLURM compute node
            after the inner Pool has finished, so the listing is authoritative)
        Centralised here so the worker side and a future direct validation path in tests stay
        consistent on the path convention. Mirrors `TunnelNetworksShardStage` with paths
        rebased on `orig_aquaduct_network_data_path` / `orig_aquaduct_vis_path` and the
        AQUA-DUCT PDB filename convention.
        """

        import glob
        paths: List[str] = []
        visualize = bool(parameters.get("visualize_transformed_transport_events", False))
        for md_label in processed_md_labels:
            dump_path = os.path.join(parameters["orig_aquaduct_network_data_path"],
                                     "{}_aqua.dump".format(md_label))
            paths.append(dump_path)
            # event-id sidecar written by save_orig_network() next to the dump (suffix
            # AquaductNetwork._orig_summary_suffix = ".event_ids"); the stage-8 layering
            # launcher reads it to skip the full orig_network load, so it must be backed up /
            # restored with the dump or a resume that wipes network_data would force the slow
            # full-load fallback.
            paths.append(dump_path + ".event_ids")
            if visualize:
                md_vis_folder = os.path.join(parameters["orig_aquaduct_vis_path"], md_label)
                paths.append(os.path.join(md_vis_folder,
                                          "{}_A_trans_rot.pdb".format(md_label)))
                paths.append(os.path.join(md_vis_folder, "view_network.py"))
                # per-path CGO pickles: each AquaductPath.save_cgo() writes
                # `<path_label>_cgo.dump.gz` into the md_vis_folder. Glob the folder rather
                # than re-deriving path_labels so the manifest matches whatever the worker
                # actually wrote, regardless of how many transport paths AQUA-DUCT produced
                # for this md_label.
                paths.extend(sorted(glob.glob(os.path.join(md_vis_folder, "*_cgo.dump.gz"))))
        return paths


# ---------------------------------------------------------------------------
# Stage 8 adapter: AquaductLayeringShardStage
# ---------------------------------------------------------------------------

class AquaductLayeringShardStage(SlurmShardStage):
    """
    Stage-8 adapter: distributes `create_layered_description4aquaduct_networks` over a SLURM
    array. Mirror of `TunnelLayeringShardStage` but with AQUA-DUCT transport events instead of
    tunnel clusters as the work-item unit. Each shard takes a strided subset of the flat
    (md_label, event_label) enumeration (so sklearn-heavy events balance across shards
    regardless of their position in the enumeration) and writes a per-shard `.pkl` file
    containing `[(event_label, md_label, LayeredPathSet), ...]`. The launcher streams shard
    pickles back, buffers per md_label, and commits each md_label's layered entities in
    original submission order so the saved `aqua_layered_network.dump` is byte-identical to
    what the local backend produces.

    The (md_label, event_label) enumeration is built once in the launcher (by walking the
    aqua_orig_network.dump files from stage 7) and persisted to `<slurm_folder>/items.json`.
    Each shard reads that single artifact instead of re-deriving the enumeration in every
    shard process, so on a 200-shard run the aqua_orig_network files are not parsed 200 times
    just to discover work-item ids.

    Determinism. Bit-identical to the local backend across any `slurm_num_shards` choice.
    Each `(md_label, event_label)` is layered by exactly one worker (no cross-shard state),
    and the assembler buffers per-md_label then commits in original submission order, so
    the saved `aqua_layered_network.dump` pickle layout - which encodes dict-insertion
    order - matches the local backend byte-for-byte regardless of completion order.
    """

    folder_name = "stage08_aquaduct_layering"
    stage_name = "aquaduct-layering"
    stage_number = 8
    job_name = "tt_aquaduct_layering"
    # workers write per-event .py + per-layer node PDBs into the visualisation folder
    # unconditionally; the SLURM helper backs these up to a sidecar tar.gz so they can be
    # restored on resume without re-running the per-event sklearn stack
    has_side_effects = True
    # heuristic granularity when auto-deriving the shard count:
    # Event-layering tasks have variable per-task cost (same sklearn stack as tunnel-layering:
    # DBSCAN/KMeans/IsolationForest/hdbscan), so striding inside the shard plus 200/shard hits
    # a reasonable load balance without inflating SLURM overhead.
    target_items_per_shard = 200
    # parameters that affect aquaduct-layering output for a fixed event set: changing any of
    # them invalidates the already-computed shards. Upstream filters (event_min_distance,
    # aquaduct_traced_residues_filter) are intentionally omitted - their effect is baked into
    # the stage-7 orig_networks and reflected in `items_hash`, so changing them only
    # invalidates stage-8 shards once the user reruns stage 7. `visualize_layered_events` is
    # included for the same reason `visualize_layered_clusters` is on the tunnel-layering
    # fingerprint: flipping it toggles whether the worker writes per-event .py / node-PDB
    # side-effects, so a False→True transition would otherwise reuse a cache with no
    # side-effect backup and the user would be left with an empty visualisation folder.
    fingerprint_keys: Tuple[str, ...] = (
        "layer_thickness",
        "random_seed",
        "clustering_max_num_rep_frag",
        "use_cluster_spread",
        "visualize_layered_events",
    )

    def __init__(self, items: List[Tuple[str, str]]):
        # canonicalise to lists-of-lists so JSON round-trip yields the same hash (tuples
        # serialise to lists, and json.load returns lists); also detaches the cached list
        # from the caller's mutable state. event_label is a str like "thr_85_release", not
        # an int - cast explicitly to keep the JSON shape stable even if callers pass in
        # exotic string-like types.
        self.items: List[List[Any]] = [[md, str(event)] for md, event in items]

    def num_shards(self, parameters: dict) -> int:
        num_shards = parameters["slurm_num_shards"]
        n_items = len(self.items)
        if num_shards is None:
            num_shards = derive_num_shards(
                n_items, array_parallelism=parameters["slurm_array_parallelism"],
                target_per_shard=self.target_items_per_shard)
        return max(1, min(num_shards, max(n_items, 1)))

    @classmethod
    def shard_result_path(cls, slurm_folder: str, shard_id: int) -> str:
        return os.path.join(slurm_folder, "shard_result_{:05d}.pkl".format(shard_id))

    @classmethod
    def run_shard(cls, config_file: str, num_shards: int, shard_id: int) -> int:
        _config, mol_system = _rebuild_mol_system(config_file, cls.stage_number)
        return mol_system.compute_aquaduct_layering_shard(num_shards, shard_id)

    def fingerprint(self, parameters: dict, num_shards: int) -> dict:
        info = super().fingerprint(parameters, num_shards)
        info["num_items"] = len(self.items)
        info["items_hash"] = hashlib.sha256(
            json.dumps(self.items, sort_keys=False).encode("utf-8")).hexdigest()
        return info

    def prepare_state(self, parameters: dict, slurm_folder: str) -> None:
        """
        Write the canonical (md_label, event_label) enumeration into the SLURM folder so every
        shard can read its strided assignment from a single source. The launcher always
        overwrites this file so it reflects the current run; stale enumerations from earlier
        runs are detected by the `items_hash` fingerprint key, not by trusting the file's age.
        """

        items_file = os.path.join(slurm_folder, self.items_filename)
        tmp_file = items_file + ".tmp"
        with open(tmp_file, "w") as out_stream:
            json.dump(self.items, out_stream)
        os.replace(tmp_file, items_file)

    @classmethod
    def load_items(cls, slurm_folder: str) -> List[List[Any]]:
        """
        Read back the `(md_label, event_label)` enumeration that the launcher wrote via
        `prepare_state()`. Called by each shard process at startup to recover its assignment.
        Returns a list of `[md_label, event_label]` pairs (JSON tuples are deserialised as
        lists).
        """

        items_file = os.path.join(slurm_folder, cls.items_filename)
        if not os.path.exists(items_file):
            raise FileNotFoundError(
                "Aquaduct-layering items file not found at '{}'. The SLURM launcher writes "
                "this via AquaductLayeringShardStage.prepare_state() before submitting "
                "shards.".format(items_file))
        with open(items_file) as in_stream:
            return json.load(in_stream)

    def expected_side_effect_files(self, parameters: dict, slurm_folder: str,
                                   shard_id: int) -> Optional[List[str]]:
        """
        Aquaduct-layering worker writes per-event `<entity_label>.py` PyMOL scripts (and the
        `nodes/<prefix>_layer<X>_pts.pdb` family) into the visualisation folder when
        `visualize_layered_events` is True - that knob is passed through to
        `LayeredRepresentationOfEvents.find_representative_paths()` as the `visualize` arg
        and gates whether `prep_visualization()` is invoked. When the knob is False, no
        worker side-effects are produced and we expect an empty manifest.

        For the True case, the per-shard manifest written by `run_shard()` enumerates the
        side-effect paths and we return that list. If the manifest is absent (older shard or
        wiped together with the side-effects), we return None to force resubmission. See
        `TunnelLayeringShardStage.expected_side_effect_files` for the broader rationale.
        """

        if not parameters.get("visualize_layered_events", False):
            return []
        return self.read_shard_manifest(slurm_folder, shard_id)

    @classmethod
    def derive_side_effect_paths(cls, parameters: dict,
                                 event_results: List[Tuple[str, str, Any]]) -> List[str]:
        """
        Derive the list of per-event side-effect file paths from the shard's result tuples.
        Mirrors `TunnelLayeringShardStage.derive_side_effect_paths`: every layered event
        produces a PyMOL script at `<layered_aquaduct_vis_path>/<md_label>/<entity_label>.py`
        plus a family of node PDBs at `nodes/<entity_label>-<layer_id>-<sub_cls_id>.pdb`
        (count is data-dependent, so we glob the on-disk nodes folder using the per-event
        prefix == `entity_label`). Empty events skip `prep_visualization` and contribute
        nothing.
        """

        import glob
        paths: List[str] = []
        for event_label, md_label, layered_path_set in event_results:
            if layered_path_set.is_empty():
                continue  # empty events skip prep_visualization, so no side-effect file
            md_folder = os.path.join(parameters["layered_aquaduct_vis_path"], md_label)
            paths.append(os.path.join(md_folder, "{}.py".format(event_label)))
            paths.extend(sorted(glob.glob(os.path.join(
                md_folder, "nodes", "{}-*.pdb".format(event_label)))))
        return paths


# ---------------------------------------------------------------------------
# Stage 9 adapter: EventAssignmentShardStage
# ---------------------------------------------------------------------------


class _HashingWriter:
    """
    Pickle-compatible binary stream wrapper that forwards every write to an underlying file
    while updating a hashlib hasher. Used in EventAssignmentShardStage.prepare_state() to
    compute the SHA-256 of state.pkl in one pass without holding the entire pickle in memory
    or re-reading the bytes from (possibly NFS) disk afterwards. Implements only the
    `write()` method; that's all `pickle.dump()` needs.
    """

    def __init__(self, stream, hasher):
        self._stream = stream
        self._hasher = hasher

    def write(self, data):
        self._hasher.update(data)
        return self._stream.write(data)


class EventAssignmentShardStage(SlurmShardStage):
    """
    Stage-9 adapter: distributes `assign_transport_events` over a SLURM array. Each shard takes
    a strided subset of the flat (md_label, event_label) enumeration; the shard process loads
    a launcher-prepared `state.pkl` (containing the supercluster dict + active filter set)
    plus its strided assignment from `items.json`, runs the existing
    `init_event_assigner_worker` / `assign_event_worker` over an inner spawn-Pool sized to
    slurm_cpus_per_task, and writes its per-shard results as a single pickle. The launcher
    streams shard pickles back and applies the assignment to `_super_clusters` /
    `OutlierTransportEvents` in the original submission order so the produced state is
    byte-identical to what the local backend yields.

    Two artifacts are written by `prepare_state()` and consumed by every shard:
      * `<slurm_folder>/items.json`  - the (md_label, event_label) enumeration
      * `<slurm_folder>/state.pkl`   - pickle of (super_clusters, active_filters)
    The state pickle is hashed (`state_hash` in the fingerprint) at write time using a
    streaming `_HashingWriter` so a single pass produces both the on-disk artifact and the
    fingerprint key - no double-memory copy and no extra read pass. A change to either the
    superclusters or the filter set invalidates all stage-9 shards automatically; the
    `items_hash` covers changes to the event set itself.
    
    Note: stage 9 keeps `has_side_effects=False` for now - manifest support for
    `_exact_event_tunnel_matching` side-effect files is deferred until it is needed.
    So no "resume from backup" path applies here; only the restart-by-missing-pickle path.

    Determinism. Bit-identical to the local backend across any `slurm_num_shards` choice.
    Every event is assigned by exactly one worker over a launcher-prepared state snapshot
    (the supercluster dict + filter set captured in `state.pkl`), and the assembler walks
    the original event-specification keys in submission order via `_apply_event_assignments`
    to mutate `_super_clusters` / `OutlierTransportEvents`. The per-event assignment is a
    pure function of `(event, super_clusters, active_filters)` - no cross-event reduction
    in the parent - so the produced state matches the local backend bit-for-bit regardless
    of worker scheduling or shard count.
    """

    folder_name = "stage09_event_assignment"
    stage_name = "event-assignment"
    stage_number = 9
    job_name = "tt_event_assignment"
    # filename of the per-shard-state artifact that holds (super_clusters, active_filters);
    # written by the launcher in `prepare_state()`, read by every shard in `load_state()`
    state_filename: str = "state.pkl"
    # heuristic granularity when auto-deriving the shard count:
    # Per-event work is moderate (numpy + optional sklearn for exact matching); ~200 events
    # per shard hits a reasonable load balance without inflating SLURM overhead.
    target_items_per_shard = 200
    # parameters that affect assignment output for a fixed (event set, supercluster set, filter
    # set): changing any of them invalidates the already-computed shards. The state hash
    # captures the superclusters + active filters (computed at write time in prepare_state);
    # the items hash captures the event set.
    fingerprint_keys: Tuple[str, ...] = (
        "event_assignment_cutoff",
        "ambiguous_event_assignment_resolution",
        "perform_exact_matching_analysis",
        "perform_trace_matching_analysis",
    )

    def __init__(self, items: List[Tuple[str, str]], super_clusters: dict,
                 active_filters: dict, pickle_protocol: int):
        # canonicalise items to lists-of-lists so JSON round-trip yields the same hash;
        # cast event_label to str explicitly so the items_hash is stable across exotic
        # string-like callers
        self.items: List[List[Any]] = [[md, str(event)] for md, event in items]
        self._super_clusters = super_clusters
        self._active_filters = active_filters
        self._pickle_protocol = pickle_protocol
        # populated by `prepare_state()` during the launcher pass before the fingerprint
        # check; `fingerprint()` reads it
        self._state_hash: Optional[str] = None

    def num_shards(self, parameters: dict) -> int:
        num_shards = parameters["slurm_num_shards"]
        n_items = len(self.items)
        if num_shards is None:
            num_shards = derive_num_shards(
                n_items, array_parallelism=parameters["slurm_array_parallelism"],
                target_per_shard=self.target_items_per_shard)
        return max(1, min(num_shards, max(n_items, 1)))

    @classmethod
    def shard_result_path(cls, slurm_folder: str, shard_id: int) -> str:
        return os.path.join(slurm_folder, "shard_result_{:05d}.pkl".format(shard_id))

    @classmethod
    def run_shard(cls, config_file: str, num_shards: int, shard_id: int) -> int:
        _config, mol_system = _rebuild_mol_system(config_file, cls.stage_number)
        return mol_system.compute_event_assignment_shard(num_shards, shard_id)

    def fingerprint(self, parameters: dict, num_shards: int) -> dict:
        if self._state_hash is None:
            raise RuntimeError("EventAssignmentShardStage.fingerprint() was called before "
                               "prepare_state(); the launcher must invoke prepare_state() so "
                               "that state.pkl is written and its hash is available for the "
                               "fingerprint check.")
        info = super().fingerprint(parameters, num_shards)
        info["num_items"] = len(self.items)
        info["items_hash"] = hashlib.sha256(
            json.dumps(self.items, sort_keys=False).encode("utf-8")).hexdigest()
        info["state_hash"] = self._state_hash
        return info

    def prepare_state(self, parameters: dict, slurm_folder: str) -> None:
        """
        Write the canonical (md_label, event_label) enumeration AND the state.pkl artifact
        (containing the supercluster dict and the active filter set) into the SLURM folder,
        so every shard reads from a single source instead of re-deriving them. The state
        pickle is hashed in a single pass via `_HashingWriter` so `fingerprint()` can detect
        upstream-state changes (re-run stages 5/6 with different inputs or different filter
        set) without re-reading the file. Both writes use the tmp+rename pattern so a
        partially-written artifact is never visible to the shards.
        """

        # 1) items.json (cheap; no streaming hash needed - items_hash is computed from the
        #    same in-memory list whenever `fingerprint()` is called)
        items_file = os.path.join(slurm_folder, self.items_filename)
        tmp_items = items_file + ".tmp"
        with open(tmp_items, "w") as out_stream:
            json.dump(self.items, out_stream)
        os.replace(tmp_items, items_file)

        # 2) state.pkl (potentially hundreds of MB); stream-write + hash in one pass
        state_file = os.path.join(slurm_folder, self.state_filename)
        tmp_state = state_file + ".tmp"
        hasher = hashlib.sha256()
        with open(tmp_state, "wb") as out_stream:
            pickle.dump((self._super_clusters, self._active_filters),
                        _HashingWriter(out_stream, hasher),
                        protocol=self._pickle_protocol)
        os.replace(tmp_state, state_file)
        self._state_hash = hasher.hexdigest()

    @classmethod
    def load_items(cls, slurm_folder: str) -> List[List[Any]]:
        """
        Read back the (md_label, event_label) enumeration that the launcher wrote via
        `prepare_state()`. Called by each shard process at startup to recover its assignment.
        Returns a list of `[md_label, event_label]` pairs (JSON tuples are deserialised as
        lists).
        """

        items_file = os.path.join(slurm_folder, cls.items_filename)
        if not os.path.exists(items_file):
            raise FileNotFoundError(
                "Event-assignment items file not found at '{}'. The SLURM launcher writes "
                "this via EventAssignmentShardStage.prepare_state() before submitting "
                "shards.".format(items_file))
        with open(items_file) as in_stream:
            return json.load(in_stream)

    @classmethod
    def load_state(cls, slurm_folder: str) -> Tuple[dict, dict]:
        """
        Read back the (super_clusters, active_filters) state pickle that the launcher wrote
        via `prepare_state()`. Called once per shard process at startup; the returned
        super_clusters dict + active_filters are injected into the shard's
        `TransportProcesses` instance so `assign_event_worker` runs unchanged.
        """

        state_file = os.path.join(slurm_folder, cls.state_filename)
        if not os.path.exists(state_file):
            raise FileNotFoundError(
                "Event-assignment state file not found at '{}'. The SLURM launcher writes "
                "this via EventAssignmentShardStage.prepare_state() before submitting "
                "shards.".format(state_file))
        with open(state_file, "rb") as in_stream:
            return pickle.load(in_stream)
