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
# submission, completion-order polling with per-shard runtime timeouts, and stderr triage on
# failure. The legacy entry point run_distance_shards_on_slurm() is preserved as a thin wrapper
# around DistanceShardStage for backwards compatibility with existing call sites.
#
# Submission is handled by the optional 'submitit' package (an extra dependency); it is imported
# lazily so that the rest of TransportTools installs and runs without it.

__version__ = '0.9.7'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import os
import json
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
from logging import getLogger

logger = getLogger(__name__)

# average number of pairwise comparisons targeted per shard when the shard count is auto-derived
_TARGET_PAIRS_PER_SHARD = 20000
# default upper bound on the auto-derived number of shards, to keep the SLURM array reasonably
# sized; only applied when the user did not set slurm_array_parallelism (which then governs
# the shard count instead, regardless of whether it is larger or smaller than this default)
_MAX_AUTO_SHARDS = 200
# filename recording the parameters a set of partial shard results was computed for
_SHARDS_INFO_FILE = "shards_info.json"


def submitit_available() -> bool:
    """
    Return True if the optional 'submitit' package is importable.
    """

    try:
        import submitit  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return False
    return True


def derive_num_shards(n_jobs: int, array_parallelism: Optional[int] = None,
                      target_per_shard: int = _TARGET_PAIRS_PER_SHARD,
                      max_auto_shards: int = _MAX_AUTO_SHARDS) -> int:
    """
    Derive a sensible number of SLURM array tasks for a job, aiming for roughly
    `target_per_shard` work items per shard while keeping the array size bounded.
    When `array_parallelism` is provided, it governs the cap directly (the user already expressed
    their preferred array size, and there is no benefit to deriving more shards than tasks that
    can run concurrently); otherwise the count falls back to `max_auto_shards`.
    :param n_jobs: total number of work items to perform
    :param array_parallelism: user-configured maximum number of concurrent array tasks; when set,
                              it replaces `max_auto_shards` as the cap on the derived shard count
    :param target_per_shard: average number of work items per shard the heuristic aims for
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
    """

    #: subdirectory name under slurm_root_folder where this stage's shards live
    folder_name: str = "stage"
    #: short identifier used in log messages
    stage_name: str = "stage"
    #: SLURM job name (also the prefix of submitit's per-task log files)
    job_name: str = "tt_stage"

    @classmethod
    def get_slurm_folder(cls, parameters: dict) -> str:
        """
        Per-stage SLURM folder: `<slurm_root_folder>/<folder_name>/`. Computed as a
        classmethod so that SLURM-shard workers (which only have a config_file, not a stage
        instance with runtime state) can derive the result path the same way the launcher does.
        """

        return os.path.join(parameters["slurm_root_folder"], cls.folder_name)

    @abstractmethod
    def num_shards(self, parameters: dict) -> int:
        """
        Total number of SLURM array tasks to submit for this stage. Resolved from the user's
        `slurm_num_shards` knob if set, otherwise auto-derived via `derive_num_shards()`.
        :param parameters: job configuration parameters
        """

    @abstractmethod
    def shard_result_path(self, slurm_folder: str, shard_id: int) -> str:
        """
        Path of the per-shard result file that this shard's worker writes and the launcher
        reads back during assembly. Must be deterministic given (slurm_folder, shard_id) and
        must include `shard_id` to keep concurrent shards from clobbering each other.
        """

    @abstractmethod
    def run_shard(self, config_file: str, num_shards: int, shard_id: int) -> Any:
        """
        Execute one shard. Runs inside a SLURM compute-node Python process; must be a
        top-level callable so that submitit can pickle it. The implementation is responsible
        for writing `shard_result_path()` atomically (write to `<path>.tmp`, then os.replace).
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
        # disable srun CPU pinning by default: when a SLURM stage is itself launched from inside
        # a SLURM allocation, the parent job's SLURM_CPU_BIND* variables leak into this array
        # job's sbatch script and make the nested srun fail with "CPU binding outside of job step
        # allocation". The job stays confined to its allocated cores via cgroups regardless.
        params["slurm_srun_args"] = parameters["slurm_srun_args"] or ["--cpu-bind=none"]
        return params


def _missing_shards(stage: SlurmShardStage, slurm_folder: str, num_shards: int,
                    parameters: dict) -> List[int]:
    """
    Validate any partial shard results already present in the SLURM folder and report which
    shards still need to be computed. Partial results are reused only if they were produced for
    the same fingerprint (same num_shards, same fingerprint_keys values, same TT version);
    otherwise the stale result files are discarded so the calculation starts fresh.
    :param stage: the SLURM stage providing the fingerprint and result-path layout
    :param slurm_folder: shared folder used for this stage's SLURM execution
    :param num_shards: total number of shards
    :param parameters: job configuration parameters
    :return: sorted list of shard IDs whose results are still missing
    """

    info = stage.fingerprint(parameters, num_shards)
    info_file = os.path.join(slurm_folder, _SHARDS_INFO_FILE)

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
        # (e.g. submitit's own *.pkl / *_log files) are left alone.
        existing_paths = {stage.shard_result_path(slurm_folder, sid) for sid in range(max(
            num_shards, previous_info.get("num_shards", 0) if previous_info else 0))}
        for path in existing_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as exc:
                    logger.warning("Failed to remove stale shard result '%s': %s", path, exc)
        with open(info_file, "w") as out_stream:
            json.dump(info, out_stream, indent=2)

    return [sid for sid in range(num_shards)
            if not os.path.exists(stage.shard_result_path(slurm_folder, sid))]


def _poll_jobs(missing_shards: List[int], jobs: List[Any], num_shards: int,
               parameters: dict, stage_name: str) -> None:
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
    """

    shard_wait_timeout_s = parameters["slurm_timeout_min"] * 2 * 60 + 5 * 60
    pending: List[Tuple[int, Any]] = list(zip(missing_shards, jobs))
    running_since: dict = {}  # job_id -> monotonic time of first observed RUNNING state
    logger.info("SLURM array job submitted as %d task(s); waiting for completion "
                "(per-shard runtime timeout: %ds, measured from RUNNING state).",
                len(jobs), shard_wait_timeout_s)

    while pending:
        now = time.monotonic()
        still_pending: List[Tuple[int, Any]] = []
        for shard_id, job in pending:
            if job.done():
                try:
                    result = job.result()
                    logger.info("SLURM %s shard %d/%d finished (%r).",
                                stage_name, shard_id + 1, num_shards, result)
                except Exception as error:
                    logger.error("SLURM %s shard %d (job %s) failed: %s",
                                 stage_name, shard_id, job.job_id, error)
                continue

            # Record the first time we observe this shard as RUNNING; once recorded, the
            # per-shard runtime clock starts. Shards that are still PENDING in the queue are
            # left alone (no deadline yet) - that matches the user's expectation that the
            # timeout caps actual compute time, not queue waiting.
            if job.job_id not in running_since:
                try:
                    state = job.state
                except Exception:
                    state = None
                if state == "RUNNING":
                    running_since[job.job_id] = now

            if (job.job_id in running_since
                    and now - running_since[job.job_id] > shard_wait_timeout_s):
                logger.error("SLURM %s shard %d (job %s) ran for more than %ds without finishing; "
                             "abandoning it and continuing with the shards that already finished. "
                             "Re-run the stage to resume.",
                             stage_name, shard_id, job.job_id, shard_wait_timeout_s)
                try:
                    job.cancel()
                except Exception as cancel_error:
                    logger.warning("Failed to cancel SLURM job %s: %s", job.job_id, cancel_error)
                continue

            still_pending.append((shard_id, job))
        pending = still_pending
        if pending:
            time.sleep(parameters["slurm_poll_wait_seconds"])


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

    missing = _missing_shards(stage, slurm_folder, num_shards, parameters)
    num_done = num_shards - len(missing)
    if num_done:
        logger.info("Reusing %d/%d shard result(s) already present in '%s'; resuming.",
                    num_done, num_shards, slurm_folder)

    if missing:
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
        _poll_jobs(missing, jobs, num_shards, parameters, stage.stage_name)

    # report which shards are still missing after the polling loop completed; the caller
    # decides what to do (typically: raise with a "re-run to resume" message)
    still_missing = [sid for sid in range(num_shards)
                     if not os.path.exists(stage.shard_result_path(slurm_folder, sid))]
    return num_shards, still_missing


# ---------------------------------------------------------------------------
# Stage 4 adapter: DistanceShardStage + back-compat wrapper
# ---------------------------------------------------------------------------


def shard_result_path(slurm_folder: str, shard_id: int) -> str:
    """
    Path of the per-shard result file written by the distance shard runner. Kept as a
    module-level helper for backwards compatibility with callers that import it directly.
    """

    return os.path.join(slurm_folder, "shard_result_{:05d}.npy".format(shard_id))


def _run_distance_shard(config_file: str, num_shards: int, shard_id: int) -> int:
    """
    Compute one shard of the inter-cluster distance matrix. This is executed on a SLURM compute
    node and must remain a top-level function so that submitit can pickle it.

    The analysis state is rebuilt from the configuration file - the layered networks the shard
    needs are already present on the shared filesystem - so the stage-3 checkpoint is not required.
    The result is written atomically to a per-shard file in the SLURM folder (keyed by shard_id),
    so that a run interrupted partway can be resumed without recomputing finished shards.
    :param config_file: path to the INI configuration file of the analysis
    :param num_shards: total number of shards the calculation is split into
    :param shard_id: 0-based ID of the shard to compute
    :return: number of cluster pairs this shard computed
    """

    # The shard is a fresh Python interpreter launched by srun and does not inherit the
    # launcher's multiprocessing start-method choice; pin it to spawn before the inner pool
    # in compute_distance_shard() is constructed (see utils.configure_multiprocessing_start_method).
    from transport_tools.libs.utils import configure_multiprocessing_start_method
    configure_multiprocessing_start_method()

    from transport_tools.libs.config import AnalysisConfig
    from transport_tools.libs.tools import TransportProcesses

    config = AnalysisConfig(config_file, logging=False)
    mol_system = TransportProcesses(config)
    results = mol_system.compute_distance_shard(num_shards, shard_id)

    array = np.array(results, dtype=np.float64) if results else np.empty((0, 3), dtype=np.float64)
    slurm_folder = DistanceShardStage.get_slurm_folder(config.parameters)
    out_file = shard_result_path(slurm_folder, shard_id)
    tmp_file = out_file + ".tmp.npy"  # write to a temporary file, then atomically rename it
    np.save(tmp_file, array)
    os.replace(tmp_file, out_file)

    return len(results)


class DistanceShardStage(SlurmShardStage):
    """
    Stage-4 adapter: per-shard inter-cluster distance computation. The shard writes a compact
    (K, 3) float64 array of (cluster1_id, cluster2_id, distance) rows; the launcher
    concatenates them to assemble the condensed distance matrix.
    """

    folder_name = "stage04_distances"
    stage_name = "distance shards"
    job_name = "tt_distances"
    # parameters that affect distance results: changing any of them invalidates the
    # already-computed shards. num_clusters is added in `fingerprint()` from runtime state.
    fingerprint_keys: Tuple[str, ...] = ("clustering_cutoff", "calculate_exact_path_distances")

    def __init__(self, num_clusters: int):
        self.num_clusters = num_clusters
        self.n_jobs = (num_clusters ** 2 - num_clusters) // 2

    def num_shards(self, parameters: dict) -> int:
        num_shards = parameters["slurm_num_shards"]
        if num_shards is None:
            num_shards = derive_num_shards(self.n_jobs, parameters["slurm_array_parallelism"])
        return max(1, min(num_shards, self.n_jobs))

    def shard_result_path(self, slurm_folder: str, shard_id: int) -> str:
        return shard_result_path(slurm_folder, shard_id)

    def run_shard(self, config_file: str, num_shards: int, shard_id: int) -> int:
        return _run_distance_shard(config_file, num_shards, shard_id)

    def fingerprint(self, parameters: dict, num_shards: int) -> dict:
        info = super().fingerprint(parameters, num_shards)
        info["num_clusters"] = self.num_clusters
        return info


def run_distance_shards_on_slurm(config_file: str, num_clusters: int,
                                 parameters: dict) -> Tuple[np.ndarray, int]:
    """
    Compute the inter-cluster distances as a SLURM array job (one task per shard) via submitit.
    Only the shards whose results are not already present on disk are (re)submitted, so a run that
    was interrupted - or in which some shards failed - can be resumed by simply running stage 4
    again, without recomputing the shards that already succeeded.
    :param config_file: path to the INI configuration file of the analysis
    :param num_clusters: total number of tunnel clusters
    :param parameters: job configuration parameters
    :return: (P, 3) float64 array of (cluster1_id, cluster2_id, distance) rows concatenated from
             all shards, and the number of shards that were used
    """

    stage = DistanceShardStage(num_clusters)
    slurm_folder = stage.get_slurm_folder(parameters)

    if num_clusters <= 1:
        # nothing to distribute; preserve the original empty-return contract
        return np.empty((0, 3), dtype=np.float64), 1

    num_shards, still_missing = run_stage_on_slurm(stage, config_file, parameters)

    # assemble the matrix entries from the per-shard result files written by the shards
    # themselves; the shards already store their pairs as compact (K, 3) float64 arrays, so they
    # are concatenated directly instead of being expanded into a Python list of tuples (which
    # would cost ~150 B/pair, i.e. tens of GB for large studies)
    shard_arrays: List[np.ndarray] = []
    for shard_id in range(num_shards):
        if shard_id in still_missing:
            continue
        shard_arrays.append(np.load(stage.shard_result_path(slurm_folder, shard_id)))

    if still_missing:
        raise RuntimeError("SLURM shard(s) {} did not produce results; inspect their logs in '{}'. "
                           "{}/{} shard(s) completed - re-run stage 4 to resume from "
                           "them.".format(still_missing, slurm_folder,
                                          num_shards - len(still_missing), num_shards))

    if shard_arrays:
        results = np.concatenate(shard_arrays)
    else:
        results = np.empty((0, 3), dtype=np.float64)

    return results, num_shards
