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

import numpy as np
import os
from pathlib import Path
from re import search
from typing import List, Tuple
from logging import getLogger

logger = getLogger(__name__)


def configure_multiprocessing_start_method() -> None:
    """
    Set the multiprocessing start method to 'spawn' if it has not been chosen yet in this process.
    Must be called at the top of every process entry point (the launcher script, every SLURM
    shard runner) before any Pool/Process/Queue is constructed. Avoids the fork-with-threads
    deadlock that intermittently hangs child workers when they are forked from a parent that
    has multi-threaded numpy/BLAS state.
    """

    import multiprocessing as mp
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn", force=True)


def iter_pool_results(labeled_tasks, *, timeout, pool):
    """
    Iterate (label, AsyncResult) pairs in submission order with a per-task timeout and
    structured error reporting. Used by every pool.apply_async() + p.get() call site in
    place of an unwrapped for-loop, so a hang or a worker exception fails loudly with the
    offending task's identity instead of stalling indefinitely or aborting anonymously.

    On multiprocessing.TimeoutError, logs the failing task's label plus the timeout that was
    exceeded, calls pool.terminate() so the program does not hang waiting for an unresponsive
    worker, and re-raises. On any other worker exception, logs the failing task's label and
    re-raises so the caller can pinpoint which item caused the failure.

    :param labeled_tasks: iterable of (label, AsyncResult) pairs; the label is a short string
        identifying the task (e.g. md_label, sc_id, event_specification) used in error logs
    :param timeout: per-task timeout in seconds, or None to disable
    :param pool: the multiprocessing.Pool that produced the AsyncResults; terminated on timeout
    :return: generator yielding the worker's return value for each task, in submission order
    """

    import multiprocessing as mp
    for label, async_result in labeled_tasks:
        try:
            yield async_result.get(timeout=timeout)
        except mp.TimeoutError:
            logger.error("Pool worker exceeded %ss timeout on task '%s'; terminating pool.",
                         timeout, label)
            pool.terminate()
            raise
        except Exception as exc:
            logger.error("Pool worker failed on task '%s': %s", label, exc)
            raise


def iter_imap_results(imap_iterator, *, timeout, pool, label: str = "imap_unordered",
                      extract_task_label=None, last_n_labels: int = 3):
    """
    Iterate a pool.imap_unordered() result iterator with an inter-completion timeout and
    structured error reporting. Functionally analogous to iter_pool_results() but for the
    streaming imap_unordered pattern; results arrive out of order, so the failing task's
    identity is not directly available - the caller-supplied iteration label, a count of
    successfully completed tasks, and (optionally) the most recent task identities are
    used to narrow down where in the workload a stall happened.

    Note on timeout semantics: `imap_iterator.next(timeout)` blocks up to `timeout` seconds
    for the next result from any worker (the timer resets on each yield). A timeout therefore
    fires on "the pool went `timeout` seconds without any task completing", not on "a specific
    task took `timeout` seconds".

    :param imap_iterator: iterator returned by pool.imap_unordered(...)
    :param timeout: inter-completion timeout in seconds, or None to disable
    :param pool: the multiprocessing.Pool that produced the iterator; terminated on timeout
    :param label: short string identifying this iteration in error logs
    :param extract_task_label: optional callable result -> str that pulls a short identity
        out of a worker's return value (e.g. lambda r: "sc_id={}".format(r[0])); when
        provided, the last `last_n_labels` completed identities are appended to the
        error log on timeout/failure to give operators a region to investigate
    :param last_n_labels: how many recent task identities to retain when extract_task_label
        is supplied (default 3); ignored when extract_task_label is None
    :return: generator yielding successive worker results until the iterator is exhausted
    """

    import multiprocessing as mp
    from collections import deque
    completed = 0
    recent_labels: deque = deque(maxlen=max(1, last_n_labels))
    while True:
        try:
            result = imap_iterator.next(timeout=timeout)
        except StopIteration:
            return
        except mp.TimeoutError:
            suffix = " {} task(s) completed before the stall.".format(completed)
            if recent_labels:
                suffix += " Last completed: {}.".format(", ".join(reversed(recent_labels)))
            logger.error("Pool worker exceeded %ss inter-completion timeout during '%s';"
                         " terminating pool.%s", timeout, label, suffix)
            pool.terminate()
            raise
        except Exception as exc:
            suffix = " {} task(s) completed before the failure.".format(completed)
            if recent_labels:
                suffix += " Last completed: {}.".format(", ".join(reversed(recent_labels)))
            logger.error("Pool worker failed during '%s': %s%s", label, exc, suffix)
            raise
        completed += 1
        if extract_task_label is not None:
            try:
                recent_labels.append(str(extract_task_label(result)))
            except Exception:
                pass  # label extraction must never break the iteration
        yield result


def cap_blas_threads_for_worker(allocated_cpus: int = 1, pool_size: int = 1) -> None:
    """
    Cap the BLAS thread budget of the calling worker process so that
    pool_size * threads_per_worker <= allocated_cpus. Must be called inside the Pool
    initializer, before the worker issues any BLAS-backed numpy operation.

    Caller-supplied budget:
      - allocated_cpus: the CPU budget the parent actually has, e.g.
            - num_cpus config parameter on the local backend
            - slurm_cpus_per_task config parameter inside a SLURM shard
      - pool_size: number of worker processes in this pool (equal to allocated_cpus for the
        existing pools; the formula generalises to future stages with a different split).
    For the current pool_size == allocated_cpus design, this collapses to 1 thread per worker.

    Sets OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS / NUMEXPR_NUM_THREADS /
    BLIS_NUM_THREADS via setdefault, so any value the user (or SLURM, or the cluster admin's
    job prolog) already exported is respected.
    """

    threads = str(max(1, allocated_cpus // max(1, pool_size)))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ.setdefault(var, threads)


def get_caver_color(color_id: int | None) -> List[float]:
    """
    Converts Pymol color IDs to RGB format, keeping within the set of 'reasonable' colors from CAVER rgb.py
    :param color_id: Pymol color ID
    :return: RBG colors in Pymol format Pymol
    """

    colors = [
        [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
        [0.71, 0.71, 0.97], [0.5, 0.99, 0.42], [0.99, 0.5, 0.42], [0.21, 0.5, 0.85], [0.5, 0.06, 0.87],
        [0.0, 0.87, 0.5], [0.86, 0.01, 0.5], [0.58, 0.82, 0.0], [0.89, 0.89, 0.6], [1.0, 0.41, 0.84], [0.41, 1.0, 0.85],
        [0.87, 0.49, 0.0], [0.62, 0.42, 0.69], [0.66, 0.67, 0.35], [0.4, 0.71, 0.62], [0.32, 1.0, 0.13],
        [0.72, 0.3, 1.0], [1.0, 0.82, 0.28], [0.96, 0.21, 0.25], [0.24, 0.22, 1.0], [0.0, 0.74, 0.79], [0.4, 0.71, 1.0],
        [0.75, 1.0, 0.2], [0.79, 0.07, 0.79], [0.87, 0.29, 0.6], [0.14, 1.0, 0.73], [0.45, 0.42, 1.0],
        [0.86, 0.59, 0.66], [0.61, 0.89, 0.66], [0.0, 0.36, 0.99], [0.86, 0.77, 0.03], [0.23, 0.99, 0.39],
        [1.0, 0.26, 0.0], [0.0, 0.98, 0.27], [0.15, 0.79, 1.0], [0.53, 0.62, 0.81], [0.88, 0.61, 0.22],
        [0.75, 0.47, 0.47], [0.73, 0.89, 0.42], [0.34, 0.94, 0.6], [0.77, 0.49, 0.87], [0.49, 0.85, 0.22],
        [0.69, 0.0, 1.0], [1.0, 0.1, 0.68], [0.24, 0.78, 0.78], [0.94, 0.23, 0.99], [0.87, 0.7, 0.46], [0.01, 0.6, 1.0],
        [0.33, 0.0, 1.0], [1.0, 0.42, 0.17], [0.61, 0.89, 0.89], [0.99, 0.0, 0.32], [0.22, 0.99, 0.96],
        [0.49, 0.28, 0.83], [0.48, 0.77, 0.43], [1.0, 0.17, 0.47], [0.91, 1.0, 0.36], [0.76, 0.28, 0.78],
        [0.7, 0.79, 0.18], [0.57, 1.0, 0.1], [0.79, 0.97, 0.0], [0.4, 0.47, 0.79], [0.19, 0.81, 0.56],
        [0.88, 0.33, 0.4], [0.62, 0.53, 1.0], [1.0, 0.45, 0.62], [0.43, 0.8, 0.81], [1.0, 0.73, 0.65],
        [0.91, 0.64, 0.84], [1.0, 0.63, 0.06], [0.44, 0.19, 1.0], [0.94, 0.97, 0.18], [0.23, 0.61, 1.0],
        [0.89, 0.41, 1.0], [0.05, 0.18, 0.99], [0.33, 0.64, 0.84], [0.94, 0.24, 0.8], [0.26, 0.4, 1.0],
        [0.01, 0.93, 0.83], [0.13, 0.99, 0.13], [0.42, 0.91, 0.0], [0.66, 0.16, 0.88], [0.78, 0.13, 0.62],
        [0.37, 0.9, 1.0], [0.13, 0.99, 0.54], [0.74, 1.0, 0.59], [0.32, 0.84, 0.45], [1.0, 0.86, 0.46],
        [0.96, 0.0, 0.82], [0.04, 0.57, 0.82], [0.62, 0.4, 0.88], [0.86, 0.42, 0.73], [0.85, 0.08, 0.96],
        [0.82, 0.79, 0.31], [0.05, 0.86, 0.67], [0.37, 0.96, 0.29], [0.61, 0.95, 0.29], [0.48, 0.99, 0.69],
        [0.82, 0.48, 0.31], [1.0, 0.08, 0.15], [0.63, 0.0, 0.78], [0.78, 0.62, 0.07], [1.0, 0.65, 0.34],
        [0.55, 0.77, 1.0], [1.0, 0.82, 0.11], [0.16, 0.05, 0.99], [0.48, 0.88, 0.55], [0.63, 0.8, 0.52],
        [0.98, 0.61, 0.54], [0.34, 0.3, 0.9], [0.14, 0.68, 0.88], [0.27, 0.94, 0.81], [0.11, 0.48, 0.99],
        [0.0, 0.84, 0.97], [0.88, 0.0, 0.66], [0.71, 0.31, 0.63], [1.0, 0.54, 0.75], [0.56, 0.3, 1.0],
        [0.45, 0.57, 0.94], [0.13, 0.69, 0.72], [0.22, 0.98, 0.0], [1.0, 0.33, 0.51], [0.57, 0.11, 1.0],
        [0.78, 0.8, 0.54], [0.67, 0.16, 0.73], [0.05, 1.0, 0.42], [0.75, 0.46, 0.63], [0.6, 0.8, 0.33],
        [0.14, 0.32, 0.94], [0.59, 1.0, 0.55], [0.85, 0.85, 0.17], [0.28, 0.79, 0.92], [0.78, 0.58, 1.0],
        [0.14, 0.88, 0.88], [1.0, 0.36, 0.32], [0.89, 0.44, 0.51], [0.85, 0.18, 0.5], [0.73, 0.65, 0.21],
        [1.0, 0.3, 0.67], [0.61, 0.99, 0.77], [0.7, 0.72, 0.0], [0.0, 1.0, 0.63], [0.23, 0.94, 0.24],
        [0.81, 0.22, 0.91], [0.28, 0.66, 0.7], [0.68, 0.91, 0.08], [0.42, 0.59, 0.72], [1.0, 0.41, 0.02],
        [0.8, 0.62, 0.35], [0.5, 0.8, 0.68], [0.35, 0.83, 0.69], [0.36, 0.99, 0.46], [0.98, 0.13, 0.89],
        [1.0, 0.01, 0.56], [0.67, 0.6, 0.88], [0.99, 0.7, 0.2], [0.86, 0.35, 0.87], [0.14, 0.89, 0.44],
        [0.99, 0.53, 0.27], [0.87, 0.48, 0.15], [0.58, 0.75, 0.86], [0.84, 0.58, 0.52], [0.85, 0.99, 0.49],
        [0.62, 0.29, 0.76], [0.74, 0.44, 1.0], [0.49, 1.0, 0.22], [0.32, 0.52, 0.94], [0.92, 0.51, 0.9],
        [0.93, 0.12, 0.36], [0.47, 0.0, 0.99], [0.95, 0.29, 0.14], [0.91, 0.9, 0.03], [0.21, 0.89, 0.68],
        [0.86, 0.85, 0.43], [0.73, 0.14, 1.0], [0.54, 0.49, 0.8], [0.87, 0.15, 0.71], [0.74, 0.68, 0.47],
        [0.33, 0.13, 0.94], [0.77, 1.0, 0.34], [0.73, 0.01, 0.68], [0.47, 0.89, 0.91], [0.56, 0.65, 0.95],
        [0.01, 0.46, 0.89], [0.99, 0.74, 0.51], [0.34, 1.0, 0.72], [0.64, 1.0, 0.42], [0.02, 0.71, 0.93],
        [0.72, 0.88, 0.28], [0.72, 0.41, 0.78], [0.46, 0.88, 0.36], [0.97, 0.2, 0.59], [0.79, 0.35, 0.51],
        [0.99, 0.13, 0.03], [0.88, 0.7, 0.59], [0.52, 0.89, 0.77], [0.83, 0.72, 0.16], [0.58, 0.82, 0.14],
        [0.12, 0.94, 0.32], [0.86, 0.92, 0.28], [0.89, 0.39, 0.24], [0.1, 1.0, 0.91], [0.0, 0.99, 0.14],
        [0.25, 0.93, 0.51], [0.99, 0.75, 0.0], [0.57, 0.88, 0.43], [0.42, 0.91, 0.13], [0.32, 0.79, 0.57],
        [0.62, 0.04, 0.9], [0.3, 0.42, 0.86], [0.53, 0.19, 0.9], [0.11, 0.81, 0.77], [0.86, 0.54, 0.78],
        [0.76, 0.84, 0.0], [0.48, 0.4, 0.87], [0.89, 0.62, 0.0], [0.71, 0.96, 0.71], [0.46, 0.71, 0.89],
        [0.99, 0.04, 0.44], [0.44, 1.0, 0.55], [0.64, 0.99, 0.0], [0.84, 1.0, 0.11], [0.2, 0.61, 0.79],
        [0.11, 0.91, 1.0], [0.9, 0.09, 0.58], [0.96, 0.76, 0.39], [0.7, 0.77, 0.39], [0.99, 0.25, 0.37],
        [0.76, 0.0, 0.9], [0.98, 0.51, 0.09], [1.0, 0.29, 0.89], [0.12, 0.66, 0.99], [0.62, 0.41, 1.0],
        [0.36, 0.88, 0.87], [0.94, 0.65, 0.73], [0.38, 0.71, 0.74], [0.62, 0.28, 0.9], [0.74, 0.35, 0.89],
        [0.91, 0.6, 0.41], [0.84, 0.3, 1.0], [0.91, 0.71, 0.28], [0.54, 0.49, 0.93], [0.86, 0.29, 0.72],
        [0.01, 0.74, 0.67], [0.24, 0.88, 1.0], [0.7, 0.89, 0.56], [0.71, 0.57, 0.42], [0.23, 1.0, 0.62],
        [0.98, 0.94, 0.29], [0.59, 0.72, 0.43], [0.48, 0.76, 0.55], [0.88, 0.14, 0.84], [0.44, 0.31, 0.96],
        [0.12, 0.57, 0.91], [0.35, 1.0, 0.0], [0.63, 0.21, 0.99], [0.66, 0.51, 0.8], [0.53, 0.37, 0.77],
        [0.51, 0.99, 0.0], [0.88, 0.48, 0.63], [0.21, 0.75, 0.67], [0.8, 0.61, 0.89], [0.33, 0.99, 0.93],
        [0.98, 0.41, 0.73], [0.59, 0.11, 0.8], [0.4, 0.91, 0.77], [0.31, 0.53, 0.79], [0.28, 0.9, 0.35],
        [0.78, 0.23, 0.68], [0.1, 0.88, 0.56], [0.52, 0.89, 0.06], [0.9, 0.8, 0.53], [0.44, 0.89, 0.65],
        [0.53, 0.97, 0.85], [0.65, 0.91, 0.19], [0.64, 0.73, 0.25], [0.97, 0.11, 1.0], [0.85, 0.5, 0.42],
        [0.3, 0.68, 0.94], [0.75, 0.17, 0.81], [0.41, 0.21, 0.89], [0.99, 0.1, 0.26], [0.94, 0.72, 0.1],
        [0.97, 0.55, 0.64], [0.92, 0.39, 0.09], [0.03, 0.99, 0.74], [0.99, 0.19, 0.13], [0.76, 0.5, 0.73],
        [0.24, 0.6, 0.89], [0.92, 0.44, 0.36], [0.65, 0.8, 0.94], [0.0, 0.99, 0.52], [0.9, 0.6, 0.11],
        [0.14, 0.42, 0.9], [0.33, 0.75, 0.83], [0.51, 0.73, 0.77], [0.05, 0.82, 0.87], [0.45, 0.81, 0.99],
        [0.71, 0.69, 0.12], [0.11, 1.0, 0.0], [0.99, 0.58, 0.18], [0.61, 1.0, 0.65], [0.81, 0.37, 0.65],
        [0.9, 0.28, 0.49], [0.36, 0.61, 1.0], [0.44, 0.57, 0.82], [0.97, 0.0, 0.71], [1.0, 0.14, 0.78],
        [0.22, 0.51, 0.99], [0.92, 0.01, 0.92], [0.74, 0.99, 0.48], [0.1, 0.78, 0.62], [1.0, 0.93, 0.09],
        [0.67, 0.49, 0.91], [0.18, 0.78, 0.88], [0.92, 0.78, 0.21], [0.76, 0.11, 0.9], [0.91, 0.3, 0.3],
        [0.79, 0.56, 0.26], [0.18, 1.0, 0.84], [0.42, 0.9, 0.46], [0.67, 0.8, 0.08], [0.79, 0.91, 0.63],
        [0.92, 0.37, 0.65], [0.4, 0.04, 0.92], [0.36, 0.37, 0.99], [0.35, 0.26, 1.0], [0.4, 0.78, 0.49],
        [0.17, 0.91, 0.78], [0.78, 0.9, 0.09], [0.1, 0.96, 0.64], [0.99, 0.5, 0.52], [0.01, 0.48, 1.0],
        [0.93, 0.87, 0.35], [0.75, 0.7, 0.31], [0.97, 0.21, 0.7], [0.9, 0.55, 0.31], [0.41, 0.48, 0.9],
        [0.24, 0.34, 0.91], [0.24, 0.69, 0.84], [0.77, 0.78, 0.1], [0.66, 1.0, 0.14], [0.92, 0.8, 0.63],
        [0.58, 0.0, 1.0], [0.0, 0.81, 0.59], [1.0, 0.39, 0.42], [0.43, 1.0, 0.07], [0.54, 0.92, 0.17],
        [0.68, 0.62, 1.0], [0.9, 1.0, 0.0], [0.87, 0.19, 0.62], [0.57, 0.21, 0.8], [0.69, 0.08, 0.83], [0.8, 0.9, 0.5],
        [0.33, 0.92, 0.2], [1.0, 0.88, 0.2], [0.0, 0.84, 0.76], [0.89, 0.05, 0.76], [0.59, 0.89, 0.54],
        [0.87, 0.45, 0.83], [0.1, 0.99, 0.23], [0.15, 0.16, 0.99], [0.47, 0.1, 0.96], [0.85, 0.18, 1.0],
        [0.85, 0.0, 0.84], [1.0, 0.86, 0.01], [0.92, 0.93, 0.45], [0.89, 0.22, 0.4], [1.0, 0.11, 0.55],
        [0.25, 0.11, 1.0], [0.38, 0.79, 0.91], [0.8, 0.77, 0.43], [0.78, 0.24, 0.57], [0.92, 0.11, 0.48],
        [0.99, 0.65, 0.45], [0.94, 0.34, 0.8], [0.95, 0.33, 0.97], [0.07, 0.66, 0.8], [0.47, 0.97, 0.32],
        [0.99, 0.55, 0.0], [0.99, 0.3, 0.23], [0.81, 0.41, 0.93], [0.22, 0.72, 0.99], [0.53, 0.93, 0.62],
        [0.42, 0.81, 0.6], [0.81, 0.4, 0.43], [0.91, 0.25, 0.9], [0.53, 0.56, 1.0], [0.75, 0.88, 0.19],
        [0.72, 0.24, 0.87], [0.64, 0.9, 0.36], [0.84, 0.23, 0.81], [0.02, 0.93, 0.93], [0.24, 0.98, 0.72],
        [0.15, 0.99, 0.45], [0.59, 0.83, 0.23], [0.85, 1.0, 0.21], [0.26, 0.89, 0.9], [0.81, 0.57, 0.16],
        [0.55, 0.38, 0.94], [0.07, 0.08, 1.0], [0.8, 0.88, 0.35], [0.81, 0.02, 0.59], [0.85, 0.5, 0.98],
        [0.31, 0.85, 0.79], [0.4, 0.51, 1.0], [0.05, 0.28, 0.97], [0.27, 0.87, 0.59], [0.64, 0.69, 0.91],
        [0.98, 0.0, 0.22], [0.79, 0.01, 1.0], [1.0, 0.51, 0.84], [0.79, 0.78, 0.22], [0.39, 1.0, 0.19],
        [0.47, 0.51, 0.73], [0.54, 0.88, 0.3], [0.57, 0.57, 0.89], [0.68, 0.4, 0.62], [0.81, 0.69, 0.0],
        [0.0, 0.55, 0.91], [0.75, 0.1, 0.71], [0.31, 0.74, 0.65], [0.66, 0.93, 0.49], [0.69, 0.35, 0.71],
        [0.69, 1.0, 0.28], [0.81, 0.48, 0.55], [0.92, 0.41, 0.9], [0.34, 0.79, 1.0], [0.01, 0.91, 0.59],
        [0.91, 0.49, 0.24], [0.16, 0.39, 1.0], [0.84, 0.71, 0.36], [0.55, 0.81, 0.6], [0.51, 0.8, 0.91],
        [0.23, 0.98, 0.15], [0.71, 0.99, 0.06], [0.39, 0.37, 0.85], [0.53, 0.2, 1.0], [0.62, 0.71, 1.0],
        [0.37, 0.87, 0.54], [0.09, 0.97, 0.82], [0.58, 1.0, 0.21], [0.41, 0.87, 0.27], [1.0, 0.0, 0.1],
        [0.49, 0.7, 1.0], [0.95, 0.65, 0.63], [0.81, 0.36, 0.79], [0.72, 0.78, 0.27], [0.69, 0.25, 0.7],
        [0.9, 0.16, 0.93], [0.92, 0.5, 0.71], [0.39, 0.08, 1.0], [0.28, 0.99, 0.31], [0.92, 0.36, 0.55],
        [0.96, 0.63, 0.26], [0.84, 0.94, 0.41], [0.39, 0.65, 0.91], [0.16, 0.73, 0.79], [0.93, 1.0, 0.09],
        [0.1, 0.77, 0.92], [0.0, 0.93, 0.36], [0.61, 0.9, 0.77], [0.98, 0.81, 0.57], [0.18, 0.91, 0.59],
        [0.91, 0.82, 0.09], [0.26, 0.31, 1.0], [0.54, 0.81, 0.49], [0.23, 0.87, 0.44], [0.14, 0.84, 0.68],
        [0.67, 0.33, 0.83], [0.6, 0.91, 0.02], [0.66, 0.09, 0.97], [0.1, 0.49, 0.86], [1.0, 0.33, 0.07],
        [0.71, 0.82, 0.48], [0.08, 0.9, 0.75], [0.37, 0.92, 0.38], [0.81, 0.67, 0.52], [0.3, 0.91, 0.7],
        [0.82, 0.68, 0.24], [0.44, 0.66, 0.8], [0.5, 0.8, 0.34], [0.07, 0.37, 0.93], [0.23, 0.44, 0.92],
        [0.91, 0.54, 0.48], [0.4, 1.0, 0.64], [0.05, 0.64, 0.89], [0.07, 0.94, 0.49], [0.73, 0.0, 0.79],
        [0.53, 0.29, 0.91], [0.62, 0.43, 0.79], [0.62, 0.81, 0.42], [0.57, 0.96, 0.37], [0.8, 0.62, 0.44],
        [0.55, 0.81, 0.79], [0.33, 0.61, 0.75], [0.58, 0.33, 0.83], [0.24, 0.0, 0.97], [0.51, 0.96, 0.5],
        [0.9, 0.55, 0.57], [0.9, 0.64, 0.52], [0.9, 0.91, 0.12], [0.56, 0.0, 0.84], [0.74, 0.42, 0.55],
        [0.13, 0.57, 0.82], [0.77, 0.53, 0.38], [0.99, 0.17, 0.33], [0.66, 0.35, 0.95], [0.82, 0.0, 0.73],
        [0.47, 0.99, 0.78], [0.46, 0.72, 0.69], [0.04, 0.76, 1.0], [0.36, 0.94, 0.06], [0.99, 0.57, 0.35],
        [1.0, 0.35, 0.6], [0.66, 0.22, 0.8], [0.3, 0.22, 0.93], [0.93, 0.03, 0.37], [1.0, 0.46, 0.33],
        [0.71, 0.53, 0.99], [0.73, 0.62, 0.3], [0.0, 0.67, 0.74], [0.9, 0.8, 0.3], [0.83, 0.08, 0.68], [0.7, 0.44, 0.7],
        [0.56, 0.85, 0.95], [0.84, 0.4, 0.57], [0.92, 0.37, 0.46], [0.91, 0.87, 0.22], [0.57, 0.94, 0.71],
        [0.14, 0.25, 1.0], [0.84, 0.54, 0.07], [0.91, 0.0, 0.58], [0.4, 0.83, 0.4], [0.26, 1.0, 0.88],
        [0.86, 0.67, 0.08], [0.16, 0.57, 0.99], [0.33, 0.45, 0.99], [0.65, 0.96, 0.59], [0.08, 0.74, 0.84],
        [0.94, 0.88, 0.52], [0.43, 0.83, 0.72], [0.26, 0.83, 0.51], [0.71, 0.92, 0.0], [0.78, 0.23, 0.99],
        [0.93, 0.68, 0.39], [0.78, 0.36, 0.99], [0.32, 0.77, 0.75], [0.33, 1.0, 0.82], [0.99, 0.73, 0.29],
        [0.91, 0.99, 0.28], [1.0, 0.04, 0.89], [0.53, 0.0, 0.92], [0.1, 0.0, 1.0], [0.06, 1.0, 0.07],
        [0.22, 0.85, 0.84], [0.98, 0.61, 0.8], [0.48, 1.0, 0.13], [0.91, 0.7, 0.02], [0.41, 0.94, 0.93],
        [0.21, 0.93, 0.33], [0.26, 0.82, 0.7], [0.37, 0.56, 0.88], [0.3, 1.0, 0.53], [0.91, 0.67, 0.17],
        [0.77, 0.94, 0.27], [0.83, 0.97, 0.58], [0.99, 0.21, 0.86], [0.73, 0.64, 0.93], [0.71, 0.42, 0.87],
        [0.99, 0.25, 0.52], [0.46, 0.92, 0.83], [0.98, 0.42, 0.5], [0.93, 0.81, 0.01], [0.82, 0.93, 0.17],
        [0.76, 0.31, 0.7], [0.3, 0.93, 0.43], [0.59, 0.12, 0.92], [0.17, 1.0, 0.28], [0.94, 0.47, 0.79],
        [0.92, 0.08, 0.68], [0.6, 0.49, 0.87], [0.84, 0.85, 0.04], [0.46, 0.14, 0.88], [0.92, 0.07, 0.86],
        [0.19, 0.84, 0.94], [0.37, 0.65, 0.67], [0.97, 0.5, 0.18], [0.41, 1.0, 0.39], [0.54, 0.86, 0.85],
        [0.94, 0.35, 0.0], [0.67, 0.07, 0.74], [0.41, 0.39, 0.93], [0.8, 0.85, 0.26], [0.66, 0.84, 0.0],
        [0.8, 0.29, 0.86], [0.6, 0.92, 0.11], [0.96, 0.3, 0.44], [1.0, 0.29, 0.76], [0.08, 0.86, 0.93],
        [0.48, 0.51, 0.86], [0.56, 0.85, 0.72], [0.81, 0.51, 0.66], [0.0, 1.0, 0.88], [0.88, 0.59, 0.91],
        [0.76, 0.71, 0.39], [0.88, 0.79, 0.38], [0.88, 0.01, 1.0], [0.68, 0.74, 0.46], [0.49, 0.86, 0.43],
        [0.5, 0.63, 0.89], [0.54, 0.72, 0.93], [0.97, 0.94, 0.38], [0.78, 0.76, 0.0], [0.44, 0.92, 0.21],
        [0.97, 0.42, 0.25], [0.92, 0.27, 0.66], [0.88, 0.64, 0.32], [0.22, 0.69, 0.75], [0.21, 0.27, 0.94],
        [0.6, 0.72, 0.35], [0.05, 1.0, 0.33], [0.69, 0.01, 0.86], [0.31, 1.0, 0.38], [0.26, 0.97, 0.07],
        [0.84, 0.08, 0.88], [0.98, 0.06, 0.76], [0.08, 0.76, 0.71], [0.22, 0.67, 0.92], [0.77, 0.41, 0.72],
        [0.43, 0.3, 0.88], [0.8, 0.16, 0.74], [0.92, 0.16, 0.54], [0.29, 1.0, 0.21], [1.0, 0.12, 0.4],
        [0.07, 0.55, 0.98], [0.73, 0.55, 0.91], [0.92, 0.46, 0.44], [1.0, 0.57, 0.46], [0.98, 0.04, 0.63],
        [0.71, 0.21, 0.95], [0.29, 0.94, 0.99], [0.0, 0.92, 0.71], [0.17, 0.93, 0.94], [0.8, 0.46, 0.79],
        [0.31, 0.69, 0.78], [0.44, 0.64, 0.99], [0.53, 0.44, 0.72], [0.35, 0.86, 0.62], [0.7, 0.96, 0.37],
        [0.76, 0.56, 0.49], [0.73, 0.95, 0.14], [0.75, 0.72, 0.19], [0.91, 0.47, 0.08], [0.78, 0.65, 0.15],
        [0.93, 0.37, 0.17], [0.82, 0.27, 0.5], [0.49, 0.86, 0.14], [0.85, 0.82, 0.59], [0.62, 0.77, 0.18],
        [0.85, 0.09, 0.52], [0.19, 0.88, 0.51], [0.16, 0.5, 0.92], [0.58, 0.99, 0.47], [0.92, 0.56, 0.83],
        [0.0, 0.94, 0.44], [0.77, 0.54, 0.8], [0.93, 0.81, 0.45], [0.32, 0.18, 1.0], [0.25, 0.8, 1.0],
        [1.0, 0.61, 0.69], [0.09, 0.92, 0.39], [0.87, 0.37, 0.33], [0.41, 0.95, 0.7], [0.63, 0.59, 0.95],
        [0.74, 0.06, 0.96], [0.03, 0.68, 1.0], [0.94, 0.04, 0.51], [0.82, 0.58, 0.0], [0.54, 0.44, 0.99],
        [0.95, 0.28, 0.59], [0.08, 0.89, 0.83], [0.07, 0.41, 1.0], [0.58, 0.67, 0.86], [0.77, 0.29, 0.94],
        [0.14, 0.92, 0.7], [0.73, 0.85, 0.12], [0.84, 0.74, 0.52], [0.48, 0.23, 0.94], [0.99, 0.43, 0.09],
        [0.99, 0.15, 0.2], [0.85, 0.63, 0.58], [0.3, 0.6, 0.95], [0.95, 0.74, 0.58], [1.0, 0.17, 0.96],
        [0.39, 0.76, 0.68], [0.85, 0.53, 0.87], [0.87, 0.92, 0.52], [0.91, 0.42, 0.01], [0.38, 0.55, 0.77],
        [0.47, 0.42, 0.78], [0.91, 0.52, 0.38], [0.94, 0.15, 0.65], [0.7, 0.91, 0.64], [0.24, 0.88, 0.76],
        [0.5, 0.36, 1.0], [0.33, 0.86, 0.95], [1.0, 0.83, 0.36], [0.19, 1.0, 0.07], [0.92, 0.55, 0.04],
        [0.68, 0.83, 0.34], [0.26, 1.0, 0.46], [0.99, 0.37, 0.92], [0.43, 0.98, 0.47], [0.25, 0.8, 0.62],
        [0.43, 0.92, 0.58], [0.48, 0.5, 0.98], [0.08, 0.99, 0.99], [0.41, 0.73, 0.83], [0.64, 0.29, 1.0],
        [0.79, 0.32, 0.59], [0.9, 0.58, 0.73], [0.34, 0.48, 0.85], [0.92, 0.21, 0.32], [0.84, 0.94, 0.06],
        [0.0, 0.68, 0.84], [0.06, 0.96, 0.56], [0.37, 0.96, 0.53], [0.56, 0.42, 0.83], [0.99, 0.25, 0.08],
        [0.93, 0.33, 0.71], [0.99, 0.24, 0.19], [0.5, 0.57, 0.77], [0.87, 0.35, 0.95], [0.99, 0.49, 0.69],
        [0.84, 0.01, 0.92], [0.79, 0.7, 0.08], [0.64, 0.87, 0.28], [0.92, 0.13, 0.77], [0.53, 0.94, 0.26],
        [0.73, 0.63, 0.38], [1.0, 0.78, 0.19], [0.21, 1.0, 0.54], [0.93, 0.09, 0.93], [0.3, 0.39, 0.93],
        [0.89, 0.22, 0.74], [0.92, 0.2, 0.47], [0.68, 0.42, 0.94], [0.36, 0.72, 0.9], [0.93, 0.43, 0.59],
        [0.96, 0.48, 0.02], [0.15, 0.71, 0.95], [0.3, 0.92, 0.28], [0.31, 0.99, 0.65], [0.91, 0.56, 0.17],
        [0.58, 0.55, 0.81], [0.86, 0.82, 0.24], [0.83, 0.99, 0.3], [0.73, 0.17, 0.67], [1.0, 0.68, 0.57],
        [0.87, 0.3, 0.8], [0.55, 0.81, 0.41], [0.6, 0.46, 0.94], [0.28, 0.52, 0.87], [0.26, 0.78, 0.85],
        [0.84, 0.51, 0.22], [0.5, 1.0, 0.59], [0.16, 0.78, 0.73], [0.0, 0.12, 0.99], [0.83, 0.55, 0.59],
        [0.74, 0.22, 0.75], [0.53, 0.13, 0.84], [0.87, 0.23, 0.55], [0.46, 0.82, 0.5], [0.97, 0.66, 0.13],
        [0.3, 0.73, 1.0], [0.94, 0.85, 0.15], [0.17, 0.93, 0.39], [0.69, 1.0, 0.53], [0.66, 0.98, 0.22],
        [0.91, 0.0, 0.44], [0.77, 0.49, 0.95], [0.5, 0.77, 0.84], [0.53, 0.81, 0.27], [0.51, 0.92, 0.4],
        [0.99, 0.08, 0.34], [0.94, 0.34, 0.87], [0.77, 0.81, 0.36], [0.85, 0.34, 0.47], [0.3, 0.06, 0.96],
        [0.65, 0.85, 0.14], [0.17, 1.0, 0.2], [0.95, 0.33, 0.36], [0.45, 0.89, 0.98], [0.59, 0.94, 0.83],
        [0.09, 0.34, 1.0], [0.66, 0.14, 0.8], [0.68, 0.66, 0.42], [0.85, 0.78, 0.12], [0.66, 0.87, 0.44],
        [0.51, 0.9, 0.69], [0.78, 0.41, 0.85], [0.46, 0.0, 0.89], [0.12, 0.99, 0.38], [0.76, 0.94, 0.55],
        [0.87, 0.24, 0.96], [0.41, 0.15, 0.94], [0.85, 0.56, 0.37], [0.68, 0.74, 0.33], [0.4, 0.98, 0.77],
        [0.83, 0.14, 0.91], [0.48, 0.21, 0.85], [0.27, 0.59, 0.82], [0.82, 0.15, 0.56], [0.6, 0.22, 0.87],
        [0.75, 0.75, 0.49], [0.92, 0.72, 0.52], [0.82, 0.42, 0.5], [0.47, 0.35, 0.81], [0.83, 0.44, 0.66],
        [0.62, 0.85, 0.07], [0.95, 0.72, 0.45], [0.33, 0.92, 0.77], [0.2, 0.92, 0.86], [0.13, 0.63, 0.77],
        [0.08, 0.83, 1.0], [0.69, 0.99, 0.64], [0.57, 0.24, 0.95], [0.16, 0.63, 0.94], [0.86, 0.65, 0.41],
        [0.41, 0.74, 0.55], [0.68, 0.09, 0.9], [0.97, 0.93, 0.0], [0.46, 0.93, 0.07], [0.82, 0.29, 0.66],
        [0.31, 0.94, 0.87], [1.0, 0.26, 0.3], [0.89, 0.93, 0.35], [0.03, 0.9, 1.0], [0.99, 0.67, 0.0],
        [0.67, 0.83, 0.24], [0.71, 0.37, 1.0], [0.63, 0.98, 0.07], [0.79, 0.85, 0.44], [0.0, 0.76, 0.87],
        [0.86, 0.85, 0.32], [0.43, 0.85, 0.86], [0.86, 0.74, 0.23], [0.16, 0.99, 0.65], [0.93, 0.56, 0.24],
        [0.84, 0.83, 0.49], [0.79, 1.0, 0.43], [0.8, 0.12, 1.0], [0.14, 0.84, 0.61], [0.3, 0.83, 0.86],
        [0.37, 0.82, 0.76], [0.46, 0.97, 0.89], [0.86, 0.46, 0.9], [0.61, 0.82, 0.88], [0.19, 0.32, 1.0],
        [0.73, 0.48, 0.81], [0.61, 0.36, 0.74], [0.12, 0.81, 0.85], [0.64, 1.0, 0.34], [0.17, 0.61, 0.87],
        [0.64, 0.94, 0.71], [0.44, 0.76, 0.75], [0.93, 0.44, 0.16], [0.46, 0.75, 0.62], [0.73, 0.38, 0.66],
        [0.88, 0.38, 0.79], [0.54, 0.99, 0.31], [1.0, 0.35, 0.14], [0.18, 0.83, 0.77], [0.32, 0.92, 0.51],
        [1.0, 0.07, 0.08], [0.54, 0.92, 0.9], [0.52, 0.05, 1.0], [0.4, 0.25, 0.95], [0.47, 0.91, 0.28],
        [0.84, 0.49, 0.73], [0.78, 0.87, 0.57], [0.0, 1.0, 0.81], [0.07, 0.49, 0.93], [0.28, 0.55, 1.0],
        [0.34, 0.45, 0.92], [0.48, 0.84, 0.61], [0.45, 0.76, 0.94], [0.82, 0.35, 0.72], [0.08, 0.69, 0.94],
        [0.82, 0.56, 0.93], [0.92, 0.33, 0.23], [0.43, 1.0, 0.27], [0.94, 0.44, 0.67], [0.8, 0.64, 0.95],
        [0.84, 0.43, 0.37], [0.95, 0.16, 0.83], [0.27, 0.95, 0.57], [1.0, 0.71, 0.06], [0.87, 0.78, 0.46],
        [1.0, 0.7, 0.4], [0.15, 0.93, 0.5], [0.7, 0.54, 0.85], [0.69, 0.67, 0.28], [0.8, 0.01, 0.66],
        [0.24, 0.74, 0.91], [0.93, 0.64, 0.06], [0.96, 0.18, 0.39], [0.81, 0.25, 0.74], [0.66, 0.47, 1.0],
        [0.37, 0.94, 0.83], [0.64, 0.87, 0.59], [0.85, 0.11, 0.78], [0.94, 0.49, 0.3], [0.37, 0.9, 0.67],
        [0.53, 0.89, 0.49], [0.2, 0.56, 0.94], [0.85, 0.59, 0.83], [0.01, 0.88, 0.89], [0.73, 0.9, 0.49],
        [0.38, 0.86, 0.34], [0.73, 0.18, 0.89], [0.26, 0.93, 0.65], [0.82, 0.16, 0.67], [0.19, 0.97, 0.78],
        [0.99, 0.09, 0.83], [0.49, 0.92, 0.0], [1.0, 0.33, 0.83], [0.7, 0.29, 0.93], [0.64, 0.77, 0.01],
        [0.4, 0.86, 0.93], [1.0, 0.38, 0.67], [0.95, 0.5, 0.59], [0.0, 0.23, 0.99], [0.06, 0.62, 0.96],
        [0.57, 0.98, 0.04], [0.09, 0.93, 0.88], [0.85, 0.29, 0.92], [0.97, 0.58, 0.11], [1.0, 0.78, 0.45],
        [0.81, 0.44, 1.0], [0.93, 0.72, 0.65], [0.72, 0.94, 0.23], [1.0, 0.95, 0.22], [0.67, 0.75, 0.13],
        [0.47, 0.44, 0.93], [0.75, 0.67, 0.03], [0.74, 0.35, 0.81], [0.51, 0.69, 0.84], [0.9, 0.13, 1.0],
        [0.61, 0.53, 0.93], [1.0, 0.55, 0.58], [0.0, 0.64, 0.94], [0.9, 0.14, 0.42], [0.85, 0.22, 0.68],
        [0.96, 0.87, 0.08], [0.8, 0.13, 0.84], [0.72, 0.75, 0.07], [0.52, 0.56, 0.93], [0.15, 1.0, 0.97],
        [0.93, 0.41, 0.83], [0.75, 0.05, 0.85], [0.84, 0.51, 0.49], [0.14, 0.74, 0.67], [0.07, 0.99, 0.16],
        [0.79, 1.0, 0.53], [0.0, 0.81, 0.69], [0.62, 0.05, 0.83], [0.59, 0.9, 0.24], [0.85, 0.64, 0.15],
        [1.0, 0.68, 0.7], [0.78, 0.82, 0.16], [0.61, 0.75, 0.93], [0.54, 0.07, 0.93], [1.0, 0.74, 0.13],
        [0.4, 0.01, 0.99], [0.81, 0.62, 0.28], [0.86, 0.58, 0.45], [0.72, 0.13, 0.77], [0.45, 0.66, 0.73],
        [0.39, 0.61, 0.8], [0.86, 0.26, 0.44], [0.8, 0.7, 0.45], [0.72, 0.89, 0.35], [0.98, 0.24, 0.44],
        [0.59, 0.77, 0.27], [1.0, 0.11, 0.48], [0.46, 0.83, 0.3], [0.18, 0.67, 0.81], [0.54, 0.98, 0.77],
        [0.99, 0.23, 0.93], [0.91, 0.46, 0.95], [0.26, 0.16, 0.95], [0.69, 0.28, 0.77], [0.49, 0.35, 0.91],
        [0.63, 0.22, 0.74], [0.96, 0.78, 0.07], [0.94, 0.04, 0.99], [0.54, 1.0, 0.65], [0.28, 0.29, 0.92]
    ]

    if color_id is None:
        return [1.0, 1.0, 1.0]

    color_id = color_id % len(colors)
    return colors[color_id]


def get_residue_color(residue_rank: int) -> List[float]:
    """
    Returns a distinct RGB color for a residue type, offset from the tunnel-cluster palette so
    residue event paths are visually separable from the enclosing supercluster.
    :param residue_rank: 0-based rank of the residue (sorted alphabetically among all residues in the set)
    :return: RGB color in PyMOL format
    """

    colors = [
        [0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
        [0.71, 0.71, 0.97], [0.5, 0.99, 0.42], [0.99, 0.5, 0.42], [0.21, 0.5, 0.85], [0.5, 0.06, 0.87],
        [0.0, 0.87, 0.5], [0.86, 0.01, 0.5], [0.58, 0.82, 0.0], [0.89, 0.89, 0.6], [1.0, 0.41, 0.84], [0.41, 1.0, 0.85],
        [0.87, 0.49, 0.0], [0.62, 0.42, 0.69], [0.66, 0.67, 0.35], [0.4, 0.71, 0.62], [0.32, 1.0, 0.13],
        [0.72, 0.3, 1.0], [1.0, 0.82, 0.28], [0.96, 0.21, 0.25], [0.24, 0.22, 1.0], [0.0, 0.74, 0.79], [0.4, 0.71, 1.0],
        [0.75, 1.0, 0.2], [0.79, 0.07, 0.79], [0.87, 0.29, 0.6], [0.14, 1.0, 0.73], [0.45, 0.42, 1.0],
        [0.86, 0.59, 0.66], [0.61, 0.89, 0.66], [0.0, 0.36, 0.99], [0.86, 0.77, 0.03], [0.23, 0.99, 0.39],
        [1.0, 0.26, 0.0], [0.0, 0.98, 0.27], [0.15, 0.79, 1.0], [0.53, 0.62, 0.81], [0.88, 0.61, 0.22],
        [0.75, 0.47, 0.47], [0.73, 0.89, 0.42], [0.34, 0.94, 0.6], [0.77, 0.49, 0.87], [0.49, 0.85, 0.22],
        [0.69, 0.0, 1.0], [1.0, 0.1, 0.68], [0.24, 0.78, 0.78], [0.94, 0.23, 0.99], [0.87, 0.7, 0.46], [0.01, 0.6, 1.0],
        [0.33, 0.0, 1.0], [1.0, 0.42, 0.17], [0.61, 0.89, 0.89], [0.99, 0.0, 0.32], [0.22, 0.99, 0.96],
    ]

    offset = 3  # start at cyan, yellow, magenta — distinct from the first SC colors (blue, green, red)
    color_id = (offset + residue_rank) % len(colors)
    return colors[color_id]


def convert_coords2cgo(coords: np.ndarray, color_id: int | None, rgb: List[float] | None = None) -> List[float]:
    """
    Converts xyz-coordinates of sequence of points such as trace and tunnel to Pymol compiled graphics object(CGO)
    :param coords: xyz-coordinates of points
    :param color_id: Pymol color id
    :param rgb: explicit RGB color to bake into the CGO; overrides color_id when given (used to bake the
                residue-event colors from get_residue_color directly into bundled event CGOs)
    :return: CGO of the points
    """

    # using CGO codes defined in Pymol2
    line_strip = 3.0
    begin = 2.0
    end = 3.0
    vertex = 4.0
    color = 6.0

    cgo = [begin, line_strip, color, *(rgb if rgb is not None else get_caver_color(color_id))]
    for xyz in coords:
        cgo.extend([vertex, xyz[0], xyz[1], xyz[2]])

    cgo.append(end)

    return cgo


def _get_boundaries(spheres: List[Tuple[np.ndarray, float]]) -> Tuple[Tuple[float, float], Tuple[float, float],
                                                                    Tuple[float, float]]:
    """
    Calculates the boundaries for the box enclosing the set of spheres
    :param spheres: list of spheres to generate their enclosing box
    :return: ranges of boundary coordinates
    """

    max_x, max_y, max_z = -999.0, -999.0, -999.0
    min_x, min_y, min_z = 999.0, 999.0, 999.0
    max_r = 0.0
    for sphere in spheres:
        if sphere[0][0] > max_x:
            max_x = sphere[0][0]
            if sphere[1] > max_r:
                max_r = sphere[1]
        if sphere[0][0] < min_x:
            min_x = sphere[0][0]
            if sphere[1] > max_r:
                max_r = sphere[1]
        if sphere[0][1] > max_y:
            max_y = sphere[0][1]
            if sphere[1] > max_r:
                max_r = sphere[1]
        if sphere[0][1] < min_y:
            min_y = sphere[0][1]
            if sphere[1] > max_r:
                max_r = sphere[1]
        if sphere[0][2] > max_z:
            max_z = sphere[0][2]
            if sphere[1] > max_r:
                max_r = sphere[1]
        if sphere[0][2] < min_z:
            min_z = sphere[0][2]
            if sphere[1] > max_r:
                max_r = sphere[1]
    # For the limits is necessary to take into account the sphere radius
    return ((min_x - max_r - 0.1, max_x + max_r),
            (min_y - max_r - 0.1, max_y + max_r),
            (min_z - max_r - 0.1, max_z + max_r))


def _build_grid(spheres: List[Tuple[np.ndarray, float]], x_points: np.ndarray, y_points: np.ndarray
                , z_points: np.ndarray) -> np.ndarray:
    """
    For each point in the grid we check if is inside the sphere and if so, store the data in the grid
    :param spheres: list of spheres to generate their grid
    :param x_points: evenly spaced x-coordinates in the box encasing the spheres
    :param y_points: evenly spaced y-coordinates in the box encasing the spheres
    :param z_points: evenly spaced z-coordinates in the box encasing the spheres
    :return: grid of points inside spheres
    """

    grid = np.zeros([x_points.shape[0], y_points.shape[0], z_points.shape[0]])
    for i, xp in enumerate(x_points):
        for j, yp in enumerate(y_points):
            for k, zp in enumerate(z_points):
                for sphere in spheres:
                    if abs(sphere[0][0] - xp) > sphere[1]:
                        continue
                    if abs(sphere[0][1] - yp) > sphere[1]:
                        continue
                    if abs(sphere[0][2] - zp) > sphere[1]:
                        continue
                    dist = np.linalg.norm(sphere[0] - [xp, yp, zp])
                    if dist < sphere[1]:
                        grid[i, j, k] = sphere[1] / dist
    return grid


def _get_mesh(grid: np.ndarray, x_points: np.ndarray, y_points: np.ndarray, z_points: np.ndarray,
              isolevel: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use marching cubes to get the triangles and coordinates of the surface.
    :param grid: grid of points inside spheres
    :param x_points: evenly spaced x-coordinates in the box encasing the spheres
    :param y_points: evenly spaced y-coordinates in the box encasing the spheres
    :param z_points: evenly spaced z-coordinates in the box encasing the spheres
    :param isolevel: for the mesh generation, values between 0.0 and 1.0
    :return: arrays of vertices, normals and triangles for the mesh
    """

    try:
        import mcubes  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("PyMCubes is required for mesh generation. Install it with: pip install PyMCubes")

    grid_vertices, triangles = mcubes.marching_cubes(grid, isolevel)
    grid_vertices = grid_vertices.astype(int)
    vertices = np.zeros(grid_vertices.shape, dtype=float)
    for i, gv in enumerate(grid_vertices):
        vertices[i] = x_points[gv[0]], y_points[gv[1]], z_points[gv[2]]
    normals = np.zeros(grid_vertices.shape, dtype=float)
    normals[:, 2] = 0.01
    for i, t in enumerate(triangles):
        v1 = vertices[t[1]] - vertices[t[0]]
        v2 = vertices[t[2]] - vertices[t[0]]
        n = np.cross(v1, v2)
        normals[t[0]] += n
        normals[t[1]] += n
        normals[t[2]] += n
    mods = np.linalg.norm(normals, axis=1)
    for i in range(normals.shape[0]):
        normals[i] /= mods[i]
    return vertices, normals, triangles


def convert_spheres2cgo_surface(spheres: List[Tuple[np.ndarray, float]], color_id: int | None, resolution: float = 0.5) -> List[float]:
    """
    Converts spheres (tuples that contain XYZ coords and radius) to surface complied graphics object for Pymol.
    :param spheres: list of spheres to generate their approximate surface
    :param color_id: Pymol color id
    :param resolution: surface grid resolution in Angstroms
    :return: CGO of spheres surface
    """

    # First we need to get the boundaries for the box of the grid
    x_lims, y_lims, z_lims = _get_boundaries(spheres)

    # Then build the grid with the desired resolution
    x_points = np.arange(x_lims[0], x_lims[1]+resolution, resolution)
    y_points = np.arange(y_lims[0], y_lims[1]+resolution, resolution)
    z_points = np.arange(z_lims[0], z_lims[1]+resolution, resolution)
    grid = _build_grid(spheres, x_points, y_points, z_points)
    #  Get surface mesh
    vertices, normals, triangles = _get_mesh(grid, x_points, y_points, z_points)

    # Create and populate the cgo to use in pymol
    cgo_obj = [2.0, 4.0]
    cgo_obj.extend([6.0, *get_caver_color(color_id)])
    for triangle in triangles:
        for t in triangle:
            cgo_obj.append(5.0)
            cgo_obj.append(normals[t, 0])
            cgo_obj.append(normals[t, 1])
            cgo_obj.append(normals[t, 2])
            cgo_obj.append(4.0)
            cgo_obj.append(vertices[t, 0])
            cgo_obj.append(vertices[t, 1])
            cgo_obj.append(vertices[t, 2])
    cgo_obj.append(3.0)

    return cgo_obj


def test_file(filename: str):
    """
    Tests if file exists
    :param filename: path to the file to test
    """

    if not os.path.exists(filename):
        raise FileNotFoundError("Input file '{}' does not exist!".format(filename))


def node_labels_split(node_label: str) -> Tuple[int, int]:
    """
    Returns layer ID and cluster ID, useful for sorting of nodes labels
    :param node_label: node label formatted as {layer}_{cluster_id} or SP for starting point
    :return: sortable tuple for ordering by layer and then by cluster id
    """

    try:
        return int(node_label.split("_")[0]), int(node_label.split("_")[1])
    except ValueError:  # cannot be split
        return -99, -99


class SnapshotFrameMap:
    """
    Maps between AQUA-DUCT / source-trajectory frame indices and CAVER snapshot IDs.

    Historically 'snapshots_per_simulation' was overloaded into three roles that only coincide when
    CAVER analysed every frame and numbered snapshots densely 1..N: (A) the set of valid snapshot IDs,
    (B) the count of analysed snapshots used for normalisation, and (C) the frame<->snapshot
    correspondence. This object decouples them so CAVER sparsity (every Nth frame) and mismatched
    AQUA-DUCT/CAVER frame domains are handled robustly.

    CAVER labels snapshots one of two ways, *detected from the observed IDs* rather than configured (so
    nothing has to be stored or persisted across stages - each call site passes the snapshot IDs it has
    already loaded):
      - 'sequential': dense IDs 1..N - CAVER consumed a pre-thinned trajectory. An event frame f maps to
        snapshot f // stride + offset, where stride is the event-frames-per-snapshot ratio
        (caver_snapshot_stride; e.g. a 20000-frame trajectory thinned to 1000 snapshots has stride 20).
      - 'by_frame': IDs are the original frame numbers, spaced by the sampling stride. An event frame f
        maps to snapshot f + offset; whether that snapshot actually exists is left to the caller's
        tunnel-presence check.
    The two are told apart by the gcd of the gaps between sorted observed IDs (1 => sequential, >=2 =>
    by_frame with that stride); a missing snapshot only ever removes an ID, leaving the gcd intact. When
    no observed IDs are supplied the map falls back to 'sequential', which at stride 1 reproduces the
    legacy 'frame + caver_traj_offset' behaviour exactly.
    """

    # parameters keys under which the resolved labelling is cached so it is derived once (at stage 2 /
    # the first stage-9 process) and then carried along on the parameters dict - to local workers via the
    # pool initializer and across a resume via the checkpoint carry-forward - rather than re-derived
    MODE_KEY = "_caver_snapshot_by_frame"
    ID_STRIDE_KEY = "_caver_snapshot_id_stride"

    def __init__(self, num_snapshots: int, offset: int = 1, stride: int = 1,
                 by_frame: bool = False, id_stride: int | None = None):
        self.num_snapshots = int(num_snapshots)
        self.offset = int(offset)
        self.stride = max(1, int(stride))
        self.by_frame = bool(by_frame)
        # spacing between consecutive snapshot IDs when enumerating the analysed-snapshot domain
        self.id_stride = int(id_stride) if id_stride else (self.stride if self.by_frame else 1)

    @classmethod
    def from_parameters(cls, parameters: dict, observed_ids=None) -> "SnapshotFrameMap":
        """
        Build a map from job parameters. When ``observed_ids`` (the CAVER snapshot IDs actually present in
        the data the caller is processing) is given, the labelling is detected from them; otherwise a
        previously resolved labelling cached on ``parameters`` (see resolve_into) is used, falling back to
        the back-compatible 'sequential' labelling with the configured stride.
        """

        num = parameters["snapshots_per_simulation"]
        offset = parameters.get("caver_traj_offset", 1)
        stride = parameters.get("caver_snapshot_stride") or 1
        if observed_ids:
            by_frame, id_stride = cls._detect(observed_ids)
            return cls(num, offset, stride, by_frame=by_frame, id_stride=id_stride)
        if parameters.get(cls.MODE_KEY) is not None:
            return cls(num, offset, stride, by_frame=parameters[cls.MODE_KEY],
                       id_stride=parameters.get(cls.ID_STRIDE_KEY))
        return cls(num, offset, stride, by_frame=False, id_stride=1)

    @classmethod
    def resolve_into(cls, parameters: dict, observed_ids) -> "SnapshotFrameMap":
        """
        Detect the labelling from observed CAVER snapshot IDs and cache it on ``parameters`` (idempotent:
        an already-cached labelling is kept). Storing it on the parameters dict lets it travel along the
        existing channels - to multiprocessing workers via the pool initializer and across a resume via
        the checkpoint carry-forward - so the data is scanned once rather than per event.
        """

        if parameters.get(cls.MODE_KEY) is not None:
            return cls.from_parameters(parameters)
        snapshot_map = cls.from_parameters(parameters, observed_ids=observed_ids)
        parameters[cls.MODE_KEY] = snapshot_map.by_frame
        parameters[cls.ID_STRIDE_KEY] = snapshot_map.id_stride
        return snapshot_map

    @staticmethod
    def _gap_gcd(observed_ids) -> int:
        ids = sorted({int(i) for i in observed_ids})
        if len(ids) < 2:
            return 1
        diffs = np.diff(ids)
        diffs = diffs[diffs > 0]
        if diffs.size == 0:
            return 1
        return int(np.gcd.reduce(diffs))

    @classmethod
    def _detect(cls, observed_ids) -> Tuple[bool, int]:
        """
        Infer (by_frame, id_stride) purely from the spacing of the observed snapshot IDs. id_stride is the
        gap between consecutive ID *labels*, not the event-frames-per-snapshot stride (that is the user's
        caver_snapshot_stride, carried separately in self.stride/map_stride):
          - gap >= 2 => 'by_frame': IDs are frame numbers spaced by the sampling stride, so id_stride = gap
          - gap == 1 => 'sequential': dense IDs 1,2,3,..., so id_stride = 1 (the stride lives in map_stride)
        """

        gap = cls._gap_gcd(observed_ids)
        if gap >= 2:
            return True, gap
        return False, 1

    @property
    def map_stride(self) -> int:
        """Event-frames per snapshot used in the frame<->snapshot formula (1 for by_frame)."""

        return 1 if self.by_frame else self.stride

    def frame_to_snap(self, frame: int, interpolate: bool = False) -> int | None:
        """
        Snapshot ID for an event/source-trajectory frame.

        By default (interpolate=False) a frame that does not coincide with an analysed snapshot returns
        None: CAVER computed no tunnel at that frame, so it must not be matched against a tunnel from a
        different time. This makes the two labellings ('sequential' and 'by_frame') of the same physical
        sampling behave identically - only frames on the analysed grid match, the rest count as 'no tunnel
        in this frame'. With interpolate=True the *nearest* analysed snapshot is returned instead (clamped
        into the domain), for the occasional call site that needs every frame to resolve to some snapshot.

        At the default stride 1 every frame is on the grid, so both modes reduce to the legacy
        'frame + offset' mapping.
        """

        frame = int(frame)
        # event-frame spacing between consecutive analysed snapshots (= the sampling stride): exactly one
        # of map_stride / id_stride carries the stride and the other is 1, so their product is the stride
        frame_stride = self.map_stride * self.id_stride
        if interpolate:
            # nearest analysed snapshot (deterministic half-up rounding), clamped into the analysed domain
            idx = (frame + frame_stride // 2) // frame_stride
            idx = min(max(idx, 0), self.num_snapshots - 1)
        else:
            if frame % frame_stride != 0:
                return None  # frame lies between analysed snapshots - no tunnel was computed here
            idx = frame // frame_stride
            if idx < 0 or idx >= self.num_snapshots:
                return None  # frame outside the analysed-snapshot domain
        return self.offset + idx * self.id_stride

    def snap_to_frame(self, snap_id: int) -> int:
        """Representative source-trajectory frame for a snapshot ID (inverse of frame_to_snap)."""

        return (int(snap_id) - self.offset) * self.map_stride

    def snapshot_ids(self) -> List[int]:
        """The analysed-snapshot ID domain - replaces the legacy ``range(1, N+1)``."""

        return [self.offset + i * self.id_stride for i in range(self.num_snapshots)]

    @property
    def num_analyzed_snapshots(self) -> int:
        """Number of analysed CAVER snapshots - the correct denominator for occupancy normalisation."""

        return self.num_snapshots

    @staticmethod
    def derive_stride(event_domain_length: int, num_snapshots: int) -> Tuple[int, bool]:
        """
        Stride implied by an AQUA-DUCT frame window vs the CAVER snapshot count, with a flag marking
        whether the division was exact. Used to validate/inform caver_snapshot_stride.
        """

        if num_snapshots <= 0:
            return 1, False
        stride = event_domain_length // num_snapshots
        return max(1, stride), (event_domain_length % num_snapshots == 0)


def parse_aquaduct_frames_window(summary_text: List[str]) -> int | None:
    """
    Parse the number of analyzed source-trajectory frames from an AQUA-DUCT summary's
    "Frames window: <start>:<end> step <step>" line - the authoritative event frame-domain length used to
    validate caver_snapshot_stride. Returns None when the line is absent (older AQUA-DUCT outputs) so the
    caller can fall back gracefully.
    :param summary_text: lines of the AQUA-DUCT 5_analysis_results.txt summary
    :return: number of frames in the analyzed window, or None if not present
    """

    for line in summary_text:
        match = search(r"Frames window:\s*(\d+):(\d+)\s+step\s+(\d+)", line)
        if match:
            start, end, step = (int(match.group(i)) for i in (1, 2, 3))
            if step <= 0:
                return None
            return (end - start) // step + 1
    return None


def get_filepath(root_folder: str = ".", pattern: str = "*") -> str:
    """
    Get full and unique path to file in root folder specified by file pattern (may include subfolders)
    :param root_folder: folder to search in
    :param pattern: search patter
    :return: file path
    """

    file_path = [*Path(root_folder).glob(pattern)]
    if len(file_path) != 1 or not file_path[0].exists():
        raise RuntimeError("File '{}' does not exist in '{}' or more then 1 file "
                           "can be matched!".format(pattern, root_folder))

    return file_path[0].as_posix()


def reweighting_file_parser(engine: str = "gamd"):
    # TODO - get file, return tunnel weights
    pass


def set_paths_from_package_root(*args)-> str:
    """
    Set paths leading from root, mainly for test suite 
    
    :param args: folders to include in path
    :return: full path relative to the root of installed transport_tools package
    """

    import transport_tools
    # Get the package directory location
    package_dir = os.path.dirname(transport_tools.__file__)
    root = os.path.join(package_dir, *args)

    return root

def splitall(path2split: str) -> List[str]:
    """
    Get all sub-parts of a file or directory path. Credited to Trent Mick.
    :param path2split: path to process
    :return: list of subpart of the path
    """

    all_parts = list()

    while True:
        parts = os.path.split(path2split)
        if parts[0] == path2split:  # sentinel for absolute paths
            all_parts.insert(0, parts[0])
            break
        elif parts[1] == path2split:  # sentinel for relative paths
            all_parts.insert(0, parts[1])
            break
        else:
            path2split = parts[0]
            all_parts.insert(0, parts[1])

    return all_parts


def path_loader_string(path: str) -> str:
    """
    Converts a string with a path to a string with an os.path.join loader for a cross-platform compatibility.
    :param path: path to convert
    :return: os path loader
    """

    all_parts = list()
    for path_segment in splitall(path):
        all_parts.append("'{}'".format(path_segment))

    return "os.path.join({})".format(", ".join(all_parts))

def prep_test_config(root: str, output_dir: str,
                     source_filename: str = "tmp_config.ini",
                     simulations_subdir: str = "simulations",
                     param_overrides: dict | None = None):
    """
    Prepare a test configuration file by updating paths in the template.

    Reads the source config from the test data directory and writes a new config.ini
    in the test output directory with paths rewritten to absolute test-data paths.

    Parameters
    ----------
    root : str
        Path to the test data directory containing the source config and simulations folder
    output_dir : str
        Path to the test output directory where config.ini will be created
    source_filename : str
        Filename of the source config within root (default: "tmp_config.ini")
    simulations_subdir : str
        Subdirectory of root used as fallback substitution for legacy parameters not
        present in param_overrides (default: "simulations")
    param_overrides : dict | None
        Per-parameter explicit replacement values. Takes precedence over the fallback
        substitution. Use this when different parameters need different paths
        (e.g. caver and aquaduct in separate subdirectories), or to inject values for
        parameters outside the default legacy set.
    """
    in_config_file = os.path.join(root, source_filename)
    out_config_file = os.path.join(output_dir, "config.ini")
    overrides = param_overrides or {}
    legacy_parameters = {"caver_results_path", "aquaduct_results_path", "trajectory_path"}
    update_parameters = legacy_parameters | set(overrides.keys())
    fallback = os.path.join(root, simulations_subdir)

    with open(in_config_file) as in_stream, open(out_config_file, "w") as out_stream:
        for line in in_stream.readlines():
            for param in update_parameters:
                if param in line:
                    line = "{} = {}\n".format(param, overrides.get(param, fallback))
            out_stream.write(line)


 # Test utilities for file/folder comparisons

def focus_pdbfile(file_lines: List[str]) -> List[str]:
    """
    Focus PDB file comparison on atom lines only, skipping version-dependent formatting.

    :param file_lines: list of lines from PDB file
    :return: list of focused lines
    """
    focused_filelines = list()
    for file_line in file_lines:
        # skip over version dependent formating of PDB
        if file_line.startswith("REMARK") or file_line.startswith("ENDMDL") or file_line.startswith("MODEL   "):
            continue
        # make chain id blank to avoid their version dependent treatment
        if file_line.startswith("ATOM") or file_line.startswith("HETATM"):
            file_line = file_line[:21] + " " + file_line[22:]
        if file_line.startswith("TER") and len(file_line) > 21:
            file_line = file_line[:21] + " " + (file_line[22:] if len(file_line) > 22 else "")
        focused_filelines.append(file_line.split())

    return focused_filelines

def compare_test_files(out_file: str, res_file: str, test_case, precision_places: int = 3) -> None:
    """
    Compare two files for testing purposes with support for various file types.

    Supports:
    - Pickled files (.dump, .dump.gz)
    - Gzipped text files (.gz)
    - NumPy arrays (.npy)
    - PDB files (.pdb) with version-independent comparison
    - CSV files (.csv) with numeric tolerance
    - Plain text files

    :param out_file: path to output file from test
    :param res_file: path to reference/expected file
    :param test_case: unittest.TestCase instance for assertions
    :param precision_places: how many decimals to use for comparisons
    """
    import pickle
    import gzip
    from sys import maxsize

    np.set_printoptions(threshold=maxsize)

    res_lines = out_lines = None
    out_mat = res_mat = None

    # Load files based on type
    if res_file.endswith(".dump.gz"):
        with gzip.open(res_file, 'rb') as res_in, gzip.open(out_file, 'rb') as out_in:
            res_lines = pickle.load(res_in)
            out_lines = pickle.load(out_in)
    elif ".dump" in res_file:
        with open(res_file, "rb") as res_in, open(out_file, "rb") as out_in:
            res_lines = pickle.load(res_in)
            out_lines = pickle.load(out_in)
    elif ".gz" in res_file:
        # Try text mode first (rt), fall back to binary (r) for compatibility
        try:
            with gzip.open(res_file, 'rt') as res_in, gzip.open(out_file, 'rt') as out_in:
                res_lines = res_in.readlines()
                out_lines = out_in.readlines()
        except:
            with gzip.open(res_file, 'rb') as res_in, gzip.open(out_file, 'rb') as out_in:
                res_lines = [line.decode('utf-8') for line in res_in.readlines()]
                out_lines = [line.decode('utf-8') for line in out_in.readlines()]
    elif ".npy" in res_file:
        res_mat = np.load(res_file)
        out_mat = np.load(out_file)
    else:
        with open(res_file, "r") as res_in, open(out_file, "r") as out_in:
            res_lines = res_in.readlines()
            out_lines = out_in.readlines()

    # For pdb files, focus comparison on atoms only
    if ".pdb" in res_file:
        if res_lines is not None and isinstance(res_lines, list):
            res_lines = focus_pdbfile(res_lines)
        if out_lines is not None and isinstance(out_lines, list):
            out_lines = focus_pdbfile(out_lines)

    # Compare content
    if res_lines is not None:
        assert out_lines is not None, "out_lines should not be None when res_lines is not None"
        if isinstance(res_lines, np.ndarray):
            test_case.assertTrue(np.allclose(out_lines, res_lines, atol=10**(-precision_places)),
                               msg="In files '{}' and '{}':".format(out_file, res_file))
        else:
            test_case.assertTrue(len(res_lines) == len(out_lines),
                               msg="Different length of files '{}' and '{}':".format(out_file, res_file))
            for res_line, out_line in zip(res_lines, out_lines):
                if isinstance(res_line, list) or isinstance(res_line, tuple):
                    test_case.assertTrue(len(res_line) == len(out_line),
                                       msg="Different length of lists {} and {}\n "
                                           "in files '{}' and '{}':".format(res_line, out_line, out_file, res_file))
                    for res_item, out_item in zip(res_line, out_line):
                        try:
                            test_case.assertAlmostEqual(float(out_item), float(res_item), places=precision_places,
                                                      msg="In files '{}' and '{}':".format(out_file, res_file))
                        except (ValueError, TypeError):
                            test_case.assertEqual(out_item, res_item,
                                                msg="In files '{}' and '{}':".format(out_file, res_file))
                elif ".csv" in res_file:
                    # Handle CSV files with field-by-field comparison
                    for out_item, res_item in zip(out_line.split(","), res_line.split(",")):
                        try:
                            test_case.assertAlmostEqual(float(out_item), float(res_item), places=precision_places,
                                                      msg="In files '{}' and '{}':".format(out_file, res_file))
                        except ValueError:
                            test_case.assertEqual(out_item, res_item,
                                                msg="In files '{}' and '{}':".format(out_file, res_file))
                else:
                    try:
                        test_case.assertAlmostEqual(float(out_line), float(res_line), places=precision_places,
                                                  msg="In files '{}' and '{}':".format(out_file, res_file))
                    except (ValueError, TypeError):
                        test_case.assertEqual(out_line, res_line,
                                            msg="In files '{}' and '{}':".format(out_file, res_file))
    else:
        assert res_mat is not None and out_mat is not None, "Matrices should not be None"
        test_case.assertTrue(np.allclose(out_mat, res_mat, atol=10**(-precision_places)),
                           msg="In files '{}' and '{}':".format(out_file, res_file))


def compare_test_folders(saved_outputs_dir: str, results_dir: str, test_case,
                        pattern: str = ".+", **compare_kwargs) -> None:
    """
    Compare contents of two folders recursively for testing purposes.

    :param saved_outputs_dir: path to reference/expected output directory
    :param results_dir: path to test output directory
    :param test_case: unittest.TestCase instance for assertions
    :param pattern: optional regex pattern to filter files (default: match all)
    :param compare_kwargs: additional keyword arguments to pass to compare_test_files
    """
    from re import search
    out_files = list()
    results_files = list()
    
    for _file in sorted(os.listdir(results_dir)):
        if not search(pattern, _file):
            continue
        results_files.append(_file)

    for _file in sorted(os.listdir(saved_outputs_dir)):
        if not search(pattern, _file):
            continue
        out_files.append(_file)

    test_case.assertEqual(out_files, results_files,
                        msg="In folders '{}' and '{}':".format(saved_outputs_dir, results_dir))

    for res_file, out_file in zip(results_files, out_files):
        res_file = os.path.join(results_dir, res_file)
        out_file = os.path.join(saved_outputs_dir, out_file)
        if os.path.isfile(res_file) and os.path.isfile(out_file):
            compare_test_files(out_file, res_file, test_case, **compare_kwargs)