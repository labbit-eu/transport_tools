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

__version__ = '0.9.8'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

"""
Stress test comparing the 'local' (CPU pool) and 'cuda' stage-4 distance backends on the
real mixed_events fixture (100 clusters, 4,950 pairs, 2,405 nodes) - the same dataset used
for the numbers in benchmarks/README.md. Skips automatically when no CUDA-capable cupy
installation is available, so it is safe to leave in the normal test run.
"""

import os
import time

import numpy as np
import pytest

from transport_tools.libs.config import AnalysisConfig
from transport_tools.libs.tools import load_checkpoint


def _cuda_available() -> bool:
    """True iff cupy imports and reports at least one CUDA-capable device; used to skip this
    test in environments without a GPU (e.g. CI) instead of failing it."""

    try:
        import cupy as cp
    except (ImportError, OSError):
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() >= 1
    except Exception:
        return False


_FIXTURE_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "data", "saved_outputs_mixed_events",
)
_CHECKPOINT = os.path.join(_FIXTURE_ROOT, "_internal", "checkpoints", "stage003.dump")


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA-capable cupy installation available")
def test_stage4_cuda_matches_local(tmp_path):
    """
    Run TransportProcesses.compute_tunnel_clusters_distances() twice from the same stage-3
    checkpoint (once with stage04_backend='local', once 'cuda') and compare the resulting
    condensed distance vectors and wall-clock time. Acts as both a smoke test for the CUDA code
    path (it does exercise a real GPU end to end, unlike the mocked-cupy unit tests in
    test_geometry.py) and a live benchmark, printed via -s.
    :param tmp_path: pytest's built-in per-test temporary directory fixture
    """

    timings = {}
    condensed = {}
    # AnalysisConfig() without a file never builds self.parameters (only file-backed loads do -
    # see AnalysisConfig.__init__); replicate its section merge order to get full current defaults
    _defaults_cfg = AnalysisConfig()
    _current_defaults = {}
    for _section in (_defaults_cfg.calculations_settings, _defaults_cfg.output_settings,
                     _defaults_cfg.input_paths, _defaults_cfg.output_paths,
                     _defaults_cfg.advanced_settings, _defaults_cfg.internal_settings):
        _current_defaults.update(_section)

    for backend in ("local", "cuda"):
        mol_system = load_checkpoint(_CHECKPOINT)
        # this checkpoint predates some current parameters (e.g. worker_task_timeout_s) - fill
        # in anything missing from the current defaults without disturbing what it already has
        mol_system.parameters = {**_current_defaults, **mol_system.parameters}
        mol_system.parameters["stage04_backend"] = backend
        # the checkpoint's output-side paths were resolved relative to the *original* run's
        # (now gone) output folder, so the read paths stage-4 needs (transformations, layered
        # network dumps) must be re-pointed at this fixture's own tree; only clustering_folder
        # (the write target) is redirected to an isolated tmp dir so we never touch fixture data
        mol_system.parameters["transformation_folder"] = os.path.join(_FIXTURE_ROOT, "_internal", "transformations")
        mol_system.parameters["layered_caver_network_data_path"] = os.path.join(
            _FIXTURE_ROOT, "_internal", "layered_data", "caver")
        mol_system.parameters["clustering_folder"] = str(tmp_path / backend)

        start = time.perf_counter()
        mol_system.compute_tunnel_clusters_distances()
        timings[backend] = time.perf_counter() - start
        condensed[backend] = np.load(mol_system._condensed_distances_path())

    speedup = timings["local"] / timings["cuda"]
    max_diff = float(np.nanmax(np.abs(condensed["local"] - condensed["cuda"])))
    print(f"\nstage-4 local={timings['local']:.3f}s cuda={timings['cuda']:.3f}s "
         f"speedup={speedup:.2f}x max_abs_diff={max_diff:.4f}")

    # a prior manual run on this same fixture showed up to ~0.0422 max abs diff (see
    # benchmarks/README.md); this is a regression guard against a much larger mismatch,
    # not a strict bit-exactness check
    assert max_diff < 0.5
