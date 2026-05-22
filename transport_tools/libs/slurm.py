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

# Distribution of the stage-4 cluster-cluster distance calculation over SLURM array jobs.
# Submission is handled by the optional 'submitit' package (an extra dependency); it is imported
# lazily so that the rest of TransportTools installs and runs without it.

__version__ = '0.9.7'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import os
import json
import time
import numpy as np
from typing import List, Optional, Tuple
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


def shard_result_path(slurm_folder: str, shard_id: int) -> str:
    """
    Returns the path of the result file a given shard writes (and the launcher reads back).
    :param slurm_folder: shared folder used for the SLURM distance calculation
    :param shard_id: 0-based ID of the shard
    """

    return os.path.join(slurm_folder, "shard_result_{:05d}.npy".format(shard_id))


def submitit_available() -> bool:
    """
    Return True if the optional 'submitit' package is importable.
    """

    try:
        import submitit  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return False
    return True


def derive_num_shards(n_jobs: int, array_parallelism: Optional[int] = None) -> int:
    """
    Derive a sensible number of SLURM array tasks for a distance job, aiming for roughly
    _TARGET_PAIRS_PER_SHARD pairwise comparisons per shard while keeping the array size bounded.
    When `array_parallelism` is provided, it governs the cap directly (the user already expressed
    their preferred array size, and there is no benefit to deriving more shards than tasks that
    can run concurrently); otherwise the count falls back to the _MAX_AUTO_SHARDS default.
    :param n_jobs: total number of pairwise distance comparisons to perform
    :param array_parallelism: user-configured maximum number of concurrent array tasks; when set,
                              it replaces _MAX_AUTO_SHARDS as the cap on the derived shard count
    :return: number of shards to use
    """

    num_shards = -(-max(n_jobs, 1) // _TARGET_PAIRS_PER_SHARD)  # ceiling division
    cap = array_parallelism if array_parallelism is not None else _MAX_AUTO_SHARDS
    return max(1, min(num_shards, cap))


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
    out_file = shard_result_path(config.parameters["slurm_folder"], shard_id)
    tmp_file = out_file + ".tmp.npy"  # write to a temporary file, then atomically rename it
    np.save(tmp_file, array)
    os.replace(tmp_file, out_file)

    return len(results)


def _resolve_num_shards(n_jobs: int, parameters: dict) -> int:
    """
    Determine the number of shards to use - either the user-requested value or an auto-derived
    one - capped so that there is never more shards than pairwise comparisons.
    :param n_jobs: total number of pairwise distance comparisons
    :param parameters: job configuration parameters
    """

    num_shards = parameters["slurm_num_shards"]
    if num_shards is None:
        num_shards = derive_num_shards(n_jobs, parameters["slurm_array_parallelism"])
    return max(1, min(num_shards, n_jobs))


def _prepare_shard_results(slurm_folder: str, num_shards: int, num_clusters: int,
                           parameters: dict) -> List[int]:
    """
    Validate any partial shard results already present in the SLURM folder and report which shards
    still need to be computed. Partial results are reused only if they were produced for the same
    problem (cluster count, shard count) and the same distance-affecting parameters; otherwise the
    stale result files are discarded so the calculation starts fresh.
    :param slurm_folder: shared folder used for the SLURM distance calculation
    :param num_shards: total number of shards
    :param num_clusters: total number of tunnel clusters
    :param parameters: job configuration parameters
    :return: sorted list of shard IDs whose results are still missing
    """

    info = {
        "num_shards": num_shards,
        "num_clusters": num_clusters,
        "clustering_cutoff": parameters["clustering_cutoff"],
        "calculate_exact_path_distances": parameters["calculate_exact_path_distances"],
    }
    info_file = os.path.join(slurm_folder, _SHARDS_INFO_FILE)

    previous_info = None
    if os.path.exists(info_file):
        try:
            with open(info_file) as in_stream:
                previous_info = json.load(in_stream)
        except (json.JSONDecodeError, OSError):
            previous_info = None

    if previous_info != info:
        # absent or stale partial results - discard them and start a fresh calculation
        for filename in os.listdir(slurm_folder):
            if filename.startswith("shard_result_") and filename.endswith(".npy"):
                os.remove(os.path.join(slurm_folder, filename))
        with open(info_file, "w") as out_stream:
            json.dump(info, out_stream, indent=2)

    return [sid for sid in range(num_shards)
            if not os.path.exists(shard_result_path(slurm_folder, sid))]


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

    try:
        import submitit  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("The 'slurm' distance backend requires the optional 'submitit' package, which is not "
                            "installed.\nInstall it with 'pip install submitit' or "
                            "'conda install -c conda-forge submitit', or set 'distance_backend = local'.")

    n_jobs = (num_clusters ** 2 - num_clusters) // 2
    num_shards = _resolve_num_shards(n_jobs, parameters)
    slurm_folder = parameters["slurm_folder"]
    os.makedirs(slurm_folder, exist_ok=True)

    missing_shards = _prepare_shard_results(slurm_folder, num_shards, num_clusters, parameters)
    num_done = num_shards - len(missing_shards)
    if num_done:
        logger.info("Reusing %d/%d shard result(s) already present in '%s'; resuming.",
                    num_done, num_shards, slurm_folder)

    if missing_shards:
        # cluster="slurm" makes submission fail loudly if SLURM is unavailable, rather than
        # silently falling back to local execution - the user explicitly requested this backend
        executor = submitit.AutoExecutor(folder=slurm_folder, cluster="slurm")
        update_params = {
            "name": "tt_distances",
            "timeout_min": parameters["slurm_timeout_min"],
            "cpus_per_task": parameters["slurm_cpus_per_task"],
            "mem_gb": parameters["slurm_mem_gb"],
            "nodes": 1,
            "tasks_per_node": 1,
        }
        if parameters["slurm_partition"] is not None:
            update_params["slurm_partition"] = parameters["slurm_partition"]
        if parameters["slurm_account"] is not None:
            update_params["slurm_account"] = parameters["slurm_account"]
        if parameters["slurm_array_parallelism"] is not None:
            update_params["slurm_array_parallelism"] = parameters["slurm_array_parallelism"]
        if parameters["slurm_setup"]:  # non-empty list of setup commands; submitit expects a list of str
            update_params["slurm_setup"] = parameters["slurm_setup"]
        # disable srun CPU pinning by default: when stage 4 is itself launched from inside a SLURM
        # allocation, the parent job's SLURM_CPU_BIND* variables leak into this array job's sbatch
        # script and make the nested srun fail with "CPU binding outside of job step allocation".
        # The job stays confined to its allocated cores via cgroups regardless.
        update_params["slurm_srun_args"] = parameters["slurm_srun_args"] or ["--cpu-bind=none"]
        executor.update_parameters(**update_params)

        logger.info("Submitting %d SLURM shard(s) (of %d total) to compute %d cluster-cluster "
                    "distances (submitit folder: %s).", len(missing_shards), num_shards, n_jobs,
                    slurm_folder)
        jobs = executor.map_array(_run_distance_shard,
                                  [config_file] * len(missing_shards),
                                  [num_shards] * len(missing_shards),
                                  missing_shards)
        # Poll shards in completion order with a per-shard runtime timeout. The timeout is derived
        # from the SLURM wall-time limit: 2 * slurm_timeout_min + 5 min covers the actual run time
        # plus the SLURM-reporting latency that can delay a "done" notification after the job has
        # exited. The clock only starts ticking once we observe the shard transition into RUNNING,
        # so shards that legitimately spend a long time in the SLURM queue (e.g. limited array
        # parallelism on a busy cluster) are never abandoned for queueing alone. A shard that runs
        # past its deadline is logged, cancelled, and skipped; the atomic per-shard result files
        # plus the resumability path in _prepare_shard_results() let the next stage-4 invocation
        # pick up the abandoned shards without recomputing the finished ones. Failed shards (state
        # FAILED, OOM, etc.) are reported via job.result() raising and do not abort collection.
        shard_wait_timeout_s = parameters["slurm_timeout_min"] * 2 * 60 + 5 * 60
        pending: List[Tuple[int, "submitit.Job"]] = list(zip(missing_shards, jobs))
        running_since: dict = {}  # job_id -> monotonic time of first observed RUNNING state
        logger.info("SLURM array job submitted as %d task(s); waiting for completion "
                    "(per-shard runtime timeout: %ds, measured from RUNNING state).",
                    len(jobs), shard_wait_timeout_s)

        while pending:
            now = time.monotonic()
            still_pending: List[Tuple[int, "submitit.Job"]] = []
            for shard_id, job in pending:
                if job.done():
                    try:
                        num_pairs = job.result()
                        logger.info("SLURM shard %d/%d finished (%d distances).",
                                    shard_id + 1, num_shards, num_pairs)
                    except Exception as error:
                        logger.error("SLURM shard %d (job %s) failed: %s", shard_id, job.job_id, error)
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
                    logger.error("SLURM shard %d (job %s) ran for more than %ds without finishing; "
                                 "abandoning it and continuing with the shards that already finished. "
                                 "Re-run stage 4 to resume.",
                                 shard_id, job.job_id, shard_wait_timeout_s)
                    try:
                        job.cancel()
                    except Exception as cancel_error:
                        logger.warning("Failed to cancel SLURM job %s: %s", job.job_id, cancel_error)
                    continue

                still_pending.append((shard_id, job))
            pending = still_pending
            if pending:
                time.sleep(parameters["slurm_poll_wait_seconds"])

    # assemble the matrix entries from the per-shard result files written by the shards
    # themselves; the shards already store their pairs as compact (K, 3) float64 arrays, so they
    # are concatenated directly instead of being expanded into a Python list of tuples (which
    # would cost ~150 B/pair, i.e. tens of GB for large studies)
    shard_arrays: List[np.ndarray] = []
    still_missing: List[int] = []
    for shard_id in range(num_shards):
        result_file = shard_result_path(slurm_folder, shard_id)
        if not os.path.exists(result_file):
            still_missing.append(shard_id)
            continue
        shard_arrays.append(np.load(result_file))

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
