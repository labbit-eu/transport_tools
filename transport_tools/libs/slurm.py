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
#   * DistanceShardStage         - stage 4 (compute_tunnel_clusters_distances)
#   * TunnelLayeringShardStage   - stage 3 (create_layered_description4tunnel_networks)
#
# Submission is handled by the optional 'submitit' package (an extra dependency); it is imported
# lazily so that the rest of TransportTools installs and runs without it.

__version__ = '0.9.7'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import os
import json
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
    stage.prepare_state(parameters, slurm_folder)

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
# Shared shard-process bootstrap
# ---------------------------------------------------------------------------


def _rebuild_mol_system(config_file: str):
    """
    Boilerplate shared by every SLURM shard's `run_shard()` entry point: pin the multiprocessing
    start method to spawn (so the inner Pool the shard's compute_* method will construct does not
    fork from a multi-threaded BLAS parent), load the analysis configuration from the INI file,
    and rebuild the TransportProcesses state. The shard process is a fresh Python interpreter
    launched by srun and does not inherit the launcher's start-method choice, so the spawn pin
    must happen here even though the launcher already ran it.
    :param config_file: path to the INI configuration file of the analysis
    :return: (AnalysisConfig, TransportProcesses) - the config is returned alongside the
             TransportProcesses so callers can also read out the per-stage SLURM folder, etc.
    """

    from transport_tools.libs.utils import configure_multiprocessing_start_method
    configure_multiprocessing_start_method()

    from transport_tools.libs.config import AnalysisConfig
    from transport_tools.libs.tools import TransportProcesses

    config = AnalysisConfig(config_file, logging=False)
    return config, TransportProcesses(config)


# ---------------------------------------------------------------------------
# Stage 4 adapter: DistanceShardStage
# ---------------------------------------------------------------------------


class DistanceShardStage(SlurmShardStage):
    """
    Stage-4 adapter: per-shard inter-cluster distance computation. The shard writes a compact
    (K, 3) float64 array of (cluster1_id, cluster2_id, distance) rows; the launcher
    concatenates them to assemble the condensed distance matrix.
    """

    folder_name = "stage04_distances"
    stage_name = "distance"
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
        config, mol_system = _rebuild_mol_system(config_file)
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
# Stage 3 adapter: TunnelLayeringShardStage
# ---------------------------------------------------------------------------

_TUNNEL_LAYERING_ITEMS_FILE = "items.json"


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
    """

    folder_name = "stage03_tunnel_layering"
    stage_name = "tunnel-layering"
    job_name = "tt_tunnel_layering"
    # heuristic granularity when auto-deriving the shard count (per audit plan §5.15.15).
    # Layering tasks have very variable per-task cost (sklearn DBSCAN/KMeans/IsolationForest/
    # hdbscan), so striding inside the shard plus 200/shard hits a reasonable load balance
    # without inflating SLURM overhead.
    target_items_per_shard = 200
    # parameters that affect tunnel-layering output: changing any of them invalidates the
    # already-computed shards. The items-list hash is added in `fingerprint()` so changes to
    # the upstream stage-2 orig_networks (different MD labels, different cluster counts) also
    # invalidate stale shards.
    fingerprint_keys: Tuple[str, ...] = (
        "layer_thickness",
        "min_tunnel_radius4clustering",
        "min_tunnel_length4clustering",
        "max_tunnel_curvature4clustering",
        "random_seed",
        "clustering_max_num_rep_frag",
        "use_cluster_spread",
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
        _config, mol_system = _rebuild_mol_system(config_file)
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

        items_file = os.path.join(slurm_folder, _TUNNEL_LAYERING_ITEMS_FILE)
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

        items_file = os.path.join(slurm_folder, _TUNNEL_LAYERING_ITEMS_FILE)
        if not os.path.exists(items_file):
            raise FileNotFoundError(
                "Tunnel-layering items file not found at '{}'. The SLURM launcher writes this "
                "via TunnelLayeringShardStage.prepare_state() before submitting shards.".format(
                    items_file))
        with open(items_file) as in_stream:
            return json.load(in_stream)
