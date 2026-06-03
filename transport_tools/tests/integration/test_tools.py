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

import unittest
import os
import pytest
from transport_tools.libs.utils import set_paths_from_package_root, prep_test_config, compare_test_folders, compare_test_files
from transport_tools.libs.tools import load_checkpoint, define_filters, TransportProcesses, save_checkpoint

class TestTransportProcesses(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _pytest_info(self, request):
        """Capture pytest request for failure detection"""
        self._request = request

    @classmethod
    def setUpClass(cls):
        from transport_tools.libs.config import AnalysisConfig

        cls._test_failed = False # Initialize flag to track failures
        cls.maxDiff = None
        cls.results = "test_results"
        cls.root = set_paths_from_package_root("tests", "data")
        cls.out_path = set_paths_from_package_root("tests", "test_results", "TestTransportProcesses")
        os.makedirs(cls.out_path, exist_ok=True)
        prep_test_config(cls.root, cls.out_path)
        
        cls.config = AnalysisConfig(os.path.join(cls.out_path, "config.ini"), logging=False)
        print(cls.config)
        cls.config.set_parameter("output_path", cls.out_path)
        cls.config.set_parameter("slurm_poll_wait_seconds", 1)
        os.makedirs(os.path.join(cls.out_path, "temp"), exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        from shutil import rmtree
        # Check if any test failed and keep results
        if cls._test_failed:
            return

        # Config file is now in the output directory, so it's removed with rmtree
        rmtree(cls.out_path)

    def tearDown(self):
        """
        Called after each test method - track if any test failed
        """
        # Check if any test has failed
        # Works with both unittest and pytest
        if hasattr(self, '_outcome'):
            # For unittest (Python 3.11+)
            if hasattr(self._outcome, 'result'):  # type: ignore[attr-defined]
                result_errors = getattr(self._outcome.result, 'errors', [])  # type: ignore[attr-defined]
                result_failures = getattr(self._outcome.result, 'failures', [])  # type: ignore[attr-defined]
                if result_errors or result_failures:
                    self.__class__._test_failed = True

        # For pytest - check using pytest's request fixture if available
        # This is set by pytest when running tests
        if hasattr(self, '_request') and hasattr(self._request, 'node'):
            # Check if this test or any previous test in the session has failed
            if self._request.session.testsfailed > 0:
                self.__class__._test_failed = True

    def setUp(self):
        self.saved_data = os.path.join(TestTransportProcesses.root, "saved_outputs")
        self.out_path = TestTransportProcesses.out_path

    def _get_dumpfile(self, stage: int) -> str:
        return os.path.join(self.out_path, "temp", "mol_system_{}.dump".format(stage))

    def test_01compute_transformations(self):
        mol_system = TransportProcesses(TestTransportProcesses.config)
        mol_system.compute_transformations()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "transformations"),
                              os.path.join(self.out_path, "_internal", "transformations"), self)
        compare_test_folders(os.path.join(self.saved_data, "_internal", "transformations", "aquaduct"),
                              os.path.join(self.out_path, "_internal", "transformations", "aquaduct"), self)
        compare_test_folders(os.path.join(self.saved_data, "_internal", "transformations", "caver"),
                              os.path.join(self.out_path, "_internal", "transformations", "caver"), self)
        save_checkpoint(mol_system, self._get_dumpfile(1), overwrite=True)

    def test_02process_tunnel_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(1))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.process_tunnel_networks()
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"), self)
        save_checkpoint(mol_system, self._get_dumpfile(2), overwrite=True)

    def test_02process_tunnel_networks_slurm(self):
        """Same checks as test_02process_tunnel_networks, but is computed via the SLURM
        backend (submitit). Runs only when a SLURM environment with submitit is detected."""
        import shutil
        import glob
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(1))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        # The SLURM shards rebuild the analysis from the config file, so it must carry an
        # absolute output_path matching this test run (the shared test config uses a relative
        # one) and the same slurm_root_folder as the launcher - kept outside the layered_data
        # folders compared below.
        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage02.ini")
        with open(os.path.join(self.out_path, "config.ini")) as in_stream, \
                open(slurm_config, "w") as out_stream:
            for line in in_stream:
                if line.startswith("output_path"):
                    line = "output_path = {}\n".format(self.out_path)
                out_stream.write(line)
                if line.strip() == "[CALCULATIONS_SETTINGS]":
                    out_stream.write("slurm_root_folder = {}\n".format(slurm_root_folder))

        mol_system.config_file = slurm_config

        mol_system.parameters["stage02_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # Remove the outputs that test_02 (local backend) already wrote so that a silent
        # no-op in the SLURM path would cause compare_test_folders to fail rather than
        # silently pass against stale local-backend files.
        for _stale_dir in [
            os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"),
        ]:
            if os.path.isdir(_stale_dir):
                shutil.rmtree(_stale_dir)

        mol_system.process_tunnel_networks()
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"), self)

    def test_02process_tunnel_networks_slurm_resume(self):
        """Resume-path coverage for the stage-2 SLURM backend: after
        test_02process_tunnel_networks_slurm has populated the slurm folder with per-shard pickles,
        manifests, and side-effect tarballs, wipe the user-facing visualisation folder and
        re-run process_tunnel_networks. The framework must restore the
        wiped files from the per-shard tarball backups WITHOUT resubmitting any SLURM job
        (which would re-run the per-cluster sklearn stack). Verified by:
          1) the restored output matches the saved reference data byte-for-byte (same as the
             primary slurm test)
          2) submitit's per-task log files in the slurm folder do not grow (no new array job
             was submitted)
        Runs only when a SLURM environment with submitit is detected; depends on the primary
        slurm test having run first to populate the cache.
        """

        import shutil
        import glob
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(1))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage02.ini")
        stage_folder = os.path.join(slurm_root_folder, "stage02_tunnel_networks")
        shard_pickles = sorted(glob.glob(os.path.join(stage_folder, "shard_result_*.pkl")))
        if not shard_pickles:
            self.skipTest("primary stage-2 slurm test did not populate the cache")
        mol_system.config_file = slurm_config

        mol_system.parameters["stage02_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # Snapshot the submitit log files before the resume call; if the framework correctly
        # restores from backup, no new SLURM job is submitted and the set must be unchanged
        # afterwards. Captures both *_log.out and *_log.err which submitit creates per task.
        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        # Wipe the visualisation folder - the side-effect files live
        # here; the framework must restore them from the per-shard tar.gz backups in
        # slurm_root_folder/stage02_tunnel_networks/. Internal network_data folder is also
        # wiped to ensure the launcher's assembly path also re-runs.
        for _wipe_dir in [
            os.path.join(self.out_path, "_internal", "network_data", "caver"),
            os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.process_tunnel_networks()
        # 1) parity check: same comparisons as the primary slurm test - restored side-effects
        #    must be byte-identical to the original outputs
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"), self)
        
        # 2) no-resubmission check: submitit log file set must be unchanged - any new files
        #    would indicate a fresh array submission, which means the restore path did not
        #    kick in. This is the strongest evidence of "reconstruction without rerun".
        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertEqual(logs_before, logs_after,
                          msg="resume path should restore from backup without submitting a "
                              "new SLURM array job; new submitit log files indicate a "
                              "resubmission happened")        

    def test_02process_tunnel_networks_slurm_restart(self):
        """Restart-path coverage for the  stage-2 SLURM backend: companion to the resume test
        above. Exercises the fallback path where the side-effects backup is ALSO unrecoverable
        (or the per-shard pickle itself is missing), so the framework must invalidate the
        cached shard and resubmit a fresh SLURM array job. Two scenarios are exercised:
          a) per-shard pickle deleted - the missing-shard check immediately treats it as
             missing and the launcher resubmits without even consulting the manifest.
          b) side-effects tar.gz deleted + visualisation files wiped - the manifest is still
             present so the resume check finds expected files missing, attempts restore,
             fails (no archive), invalidates the cache, and resubmits.
        Both scenarios end with a parity check against the saved reference output. Runs only
        when a SLURM environment with submitit is detected and the primary slurm test has
        populated the cache.
        """

        import shutil
        import glob
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(1))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage02.ini")
        stage_folder = os.path.join(slurm_root_folder, "stage02_tunnel_networks")
        shard_pickles = sorted(glob.glob(os.path.join(stage_folder, "shard_result_*.pkl")))
        if not shard_pickles:
            self.skipTest("primary stage-2 slurm test did not populate the cache")
        mol_system.config_file = slurm_config

        mol_system.parameters["stage02_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # ---- Scenario (a): delete one per-shard pickle. The missing-shard check sees it
        # gone immediately and the launcher must resubmit a fresh SLURM array.
        os.remove(shard_pickles[0])

        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        for _wipe_dir in [
            os.path.join(self.out_path, "_internal", "network_data", "caver"),
            os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.process_tunnel_networks()

        # parity: a fresh resubmission must produce the same output as the original run
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"), self)
        
        # logs grew: new submission happened (the resume path would have left logs unchanged)
        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertGreater(len(logs_after), len(logs_before),
                            msg="restart scenario (a): deleting a per-shard pickle should "
                                "force a SLURM resubmission, but the submitit log-file set "
                                "did not grow")    

        # ---- Scenario (b): delete the side-effects backup tar.gz for one shard, then wipe
        # the visualisation folder. Manifest still indicates files are expected, but restore
        # cannot find the archive, so the cache is invalidated and the shard is resubmitted.
        from transport_tools.libs.slurm import TunnelNetworksShardStage
        archive_path = TunnelNetworksShardStage.shard_side_effects_archive_path(stage_folder, 0)
        if os.path.exists(archive_path):
            os.remove(archive_path)

        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        for _wipe_dir in [
            os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.process_tunnel_networks()

        # parity: a fresh resubmission must produce the same output as the original run
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "caver", "md1"), self)
        
        # logs grew: new submission happened (the resume path would have left logs unchanged)
        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        self.assertGreater(len(logs_after), len(logs_before),
                            msg="restart scenario (b): deleting a side-effects archive + "
                                "wiping the vis folder should force a SLURM resubmission, "
                                "but the submitit log-file set did not grow")


    def test_03create_layered_description4tunnel_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(2))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.create_layered_description4tunnel_networks()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "layered_data", "caver"),
                              os.path.join(self.out_path, "_internal", "layered_data", "caver"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1",
                                           "nodes"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1",
                                           "nodes"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1",
                                           "paths"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1",
                                           "paths"), self)
        save_checkpoint(mol_system, self._get_dumpfile(3), overwrite=True)

    def test_03create_layered_description4tunnel_networks_slurm(self):
        """Same checks as test_03create_layered..., but stage-3 layering is computed via the SLURM
        backend (submitit). Runs only when a SLURM environment with submitit is detected."""
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(2))
        except FileNotFoundError:
            self.skipTest("previous test not finished")


        # The SLURM shards rebuild the analysis from the config file, so it must carry an
        # absolute output_path matching this test run (the shared test config uses a relative
        # one) and the same slurm_root_folder as the launcher - kept outside the layered_data
        # folders compared below.
        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage03.ini")
        with open(os.path.join(self.out_path, "config.ini")) as in_stream, \
                open(slurm_config, "w") as out_stream:
            for line in in_stream:
                if line.startswith("output_path"):
                    line = "output_path = {}\n".format(self.out_path)
                out_stream.write(line)
                if line.strip() == "[CALCULATIONS_SETTINGS]":
                    out_stream.write("slurm_root_folder = {}\n".format(slurm_root_folder))
        mol_system.config_file = slurm_config

        mol_system.parameters["stage03_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # Remove the outputs that test_03 (local backend) already wrote so that a silent
        # no-op in the SLURM path would cause compare_test_folders to fail rather than
        # silently pass against stale local-backend files.
        for _stale_dir in [
            os.path.join(self.out_path, "_internal", "layered_data", "caver"),
            os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"),
        ]:
            if os.path.isdir(_stale_dir):
                shutil.rmtree(_stale_dir)

        mol_system.create_layered_description4tunnel_networks()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "layered_data", "caver"),
                              os.path.join(self.out_path, "_internal", "layered_data", "caver"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"),
                              self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1",
                                           "nodes"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1",
                                           "nodes"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1",
                                           "paths"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1",
                                           "paths"), self)

    def test_03create_layered_description4tunnel_networks_slurm_resume(self):
        """Resume-path coverage for the stage-3 SLURM backend: after
        test_03create_layered..._slurm has populated the slurm folder with per-shard pickles,
        manifests, and side-effect tarballs, wipe the user-facing visualisation folder and
        re-run create_layered_description4tunnel_networks. The framework must restore the
        wiped files from the per-shard tarball backups WITHOUT resubmitting any SLURM job
        (which would re-run the per-cluster sklearn stack). Verified by:
          1) the restored output matches the saved reference data byte-for-byte (same as the
             primary slurm test)
          2) submitit's per-task log files in the slurm folder do not grow (no new array job
             was submitted)
        Runs only when a SLURM environment with submitit is detected; depends on the primary
        slurm test having run first to populate the cache.
        """
        import glob
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(2))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        # Re-use the same slurm folder + config as the primary slurm test so we hit the
        # populated cache. If the primary test did not run (e.g. SLURM was unavailable),
        # the cache is empty and there is nothing to validate here either.
        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage03.ini")
        stage_folder = os.path.join(slurm_root_folder, "stage03_tunnel_layering")
        shard_pickles = glob.glob(os.path.join(slurm_root_folder, "stage03_tunnel_layering",
                                                "shard_result_*.pkl"))
        if not shard_pickles:
            self.skipTest("primary stage-3 slurm test did not populate the cache")
        mol_system.config_file = slurm_config

        mol_system.parameters["stage03_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # Snapshot the submitit log files before the resume call; if the framework correctly
        # restores from backup, no new SLURM job is submitted and the set must be unchanged
        # afterwards. Captures both *_log.out and *_log.err which submitit creates per task.
        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        # Wipe the visualisation folder - the side-effect files (Cluster_*.py, nodes/) live
        # here; the framework must restore them from the per-shard tar.gz backups in
        # slurm_root_folder/stage03_tunnel_layering/. Internal layered_data folder is also
        # wiped to ensure the launcher's assembly path also re-runs.
        for _wipe_dir in [
            os.path.join(self.out_path, "_internal", "layered_data", "caver"),
            os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.create_layered_description4tunnel_networks()

        # 1) parity check: same comparisons as the primary slurm test - restored side-effects
        #    must be byte-identical to the original outputs
        compare_test_folders(os.path.join(self.saved_data, "_internal", "layered_data", "caver"),
                              os.path.join(self.out_path, "_internal", "layered_data", "caver"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"),
                              self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1",
                                           "nodes"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1",
                                           "nodes"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1",
                                           "paths"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1",
                                           "paths"), self)

        # 2) no-resubmission check: submitit log file set must be unchanged - any new files
        #    would indicate a fresh array submission, which means the restore path did not
        #    kick in. This is the strongest evidence of "reconstruction without rerun".
        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertEqual(logs_before, logs_after,
                          msg="resume path should restore from backup without submitting a "
                              "new SLURM array job; new submitit log files indicate a "
                              "resubmission happened")

    def test_03create_layered_description4tunnel_networks_slurm_restart(self):
        """Restart-path coverage for the stage-3 SLURM backend: companion to the resume test
        above. Exercises the fallback path where the side-effects backup is ALSO unrecoverable
        (or the per-shard pickle itself is missing), so the framework must invalidate the
        cached shard and resubmit a fresh SLURM array job. Two scenarios are exercised:
          a) per-shard pickle deleted - the missing-shard check immediately treats it as
             missing and the launcher resubmits without even consulting the manifest.
          b) side-effects tar.gz deleted + visualisation files wiped - the manifest is still
             present so the resume check finds expected files missing, attempts restore,
             fails (no archive), invalidates the cache, and resubmits.
        Both scenarios end with a parity check against the saved reference output. Runs only
        when a SLURM environment with submitit is detected and the primary slurm test has
        populated the cache.
        """
        import glob
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(2))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage03.ini")
        stage_folder = os.path.join(slurm_root_folder, "stage03_tunnel_layering")
        shard_pickles = sorted(glob.glob(os.path.join(stage_folder, "shard_result_*.pkl")))
        if not shard_pickles:
            self.skipTest("primary stage-3 slurm test did not populate the cache")
        mol_system.config_file = slurm_config

        mol_system.parameters["stage03_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # ---- Scenario (a): delete one per-shard pickle. The missing-shard check sees it
        # gone immediately and the launcher must resubmit a fresh SLURM array.
        os.remove(shard_pickles[0])

        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        for _wipe_dir in [
            os.path.join(self.out_path, "_internal", "layered_data", "caver"),
            os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.create_layered_description4tunnel_networks()

        # parity: a fresh resubmission must produce the same output as the original run
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"),
                              self)

        # logs grew: new submission happened (the resume path would have left logs unchanged)
        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertGreater(len(logs_after), len(logs_before),
                            msg="restart scenario (a): deleting a per-shard pickle should "
                                "force a SLURM resubmission, but the submitit log-file set "
                                "did not grow")

        # ---- Scenario (b): delete the side-effects backup tar.gz for one shard, then wipe
        # the visualisation folder. Manifest still indicates files are expected, but restore
        # cannot find the archive, so the cache is invalidated and the shard is resubmitted.
        from transport_tools.libs.slurm import TunnelLayeringShardStage
        archive_path = TunnelLayeringShardStage.shard_side_effects_archive_path(stage_folder, 1)
        if os.path.exists(archive_path):
            os.remove(archive_path)

        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        for _wipe_dir in [
            os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.create_layered_description4tunnel_networks()

        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "caver", "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "caver", "md1"),
                              self)

        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertGreater(len(logs_after), len(logs_before),
                            msg="restart scenario (b): deleting a side-effects archive + "
                                "wiping the vis folder should force a SLURM resubmission, "
                                "but the submitit log-file set did not grow")

    def test_04distance_shard_consistency(self):
        """SLURM-shard distance calculation must reproduce the local distance matrix exactly"""
        import numpy as np

        try:
            mol_system = load_checkpoint(self._get_dumpfile(3))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        from scipy.spatial.distance import squareform

        cluster_specifications, path_sets, _ = mol_system._assemble_cluster_path_sets(load_path_sets=True)
        local_condensed = mol_system._compute_intercluster_distances(cluster_specifications, path_sets)

        # rebuild the matrix from sharded calculations, exactly as the SLURM backend would
        num_shards = 3
        num_clusters = len(cluster_specifications)
        sharded_matrix = np.full((num_clusters, num_clusters), np.inf)
        for shard_id in range(num_shards):
            for id1, id2, distance in mol_system.compute_distance_shard(num_shards, shard_id):
                sharded_matrix[int(id1), int(id2)] = distance
        for cls1 in range(num_clusters):
            sharded_matrix[cls1, cls1] = 0
            for cls2 in range(cls1 + 1, num_clusters):
                sharded_matrix[cls2, cls1] = sharded_matrix[cls1, cls2]

        self.assertFalse(np.isinf(sharded_matrix).any())
        self.assertTrue(np.array_equal(local_condensed, squareform(sharded_matrix, checks=False)))

    def _purge_super_cluster_outputs(self):
        """Remove the supercluster outputs of a merge/profiles/summary run while keeping the
        (linkage-independent) distance matrix, so a subsequent run starts from a clean state."""
        from shutil import rmtree

        for path in (os.path.join(self.out_path, "data", "super_clusters"),
                     os.path.join(self.out_path, "statistics"),
                     os.path.join(self.out_path, "_internal", "super_cluster_pathsets"),
                     os.path.join(self.out_path, "_internal", "super_cluster_profiles")):
            if os.path.isdir(path):
                rmtree(path)

    def _check_super_cluster_clustering_variant(self, mol_system, variant: str, param_overrides: dict):
        """Cluster the already-computed distances with the given non-default clustering configuration
        (linkage and/or method), generate the supercluster details + summary, golden-compare them
        against the per-variant fixtures, then delete the generated outputs so the default 'average'
        chain (test_05+) is unaffected. The distance matrix is reused, not recomputed."""
        for key, value in param_overrides.items():
            mol_system.parameters[key] = value
        mol_system.merge_tunnel_clusters2super_clusters()
        mol_system.create_super_cluster_profiles()
        mol_system.generate_super_cluster_summary(out_filename="1-initial_tunnels_summary.txt")

        fixtures = os.path.join(self.saved_data, "clustering_variants", variant)
        compare_test_files(os.path.join(fixtures, "initial_super_cluster_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "initial_super_cluster_details.txt"), self)
        compare_test_files(os.path.join(fixtures, "1-initial_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "1-initial_tunnels_summary.txt"), self)
        self._purge_super_cluster_outputs()

    def test_04merge_tunnel_clusters2super_clusters(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(3))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        # distances are clustering-independent - compute them once and reuse for every variant below
        mol_system.compute_tunnel_clusters_distances()

        # validate the non-default clustering variants against their golden fixtures, deleting each
        # one's generated supercluster outputs afterwards so only the 'average' results remain for
        # the followup tests (the distance matrix is kept and reused, never recomputed)
        variants = (
            ("single",   {"clustering_method": "agglomerative", "clustering_linkage": "single"}),
            ("complete", {"clustering_method": "agglomerative", "clustering_linkage": "complete"}),
            ("ward",     {"clustering_method": "agglomerative", "clustering_linkage": "ward"}),
            ("hdbscan",  {"clustering_method": "hdbscan"}),
            # HDBSCAN with the partition scale decoupled from the selection scale (Phase A) plus the
            # global orphan->core consolidation pass (Phase B) - exercises both multi-scale features
            ("hdbscan_consolidated", {"clustering_method": "hdbscan",
                                      "hdbscan_clustering_partition_cutoff": 4.0,
                                      "consolidate_orphan_superclusters": True,
                                      "supercluster_core_min_size": 3,
                                      "orphan_assignment_cutoff": 3.0}),
        )
        for variant, param_overrides in variants:
            self._check_super_cluster_clustering_variant(mol_system, variant, param_overrides)

        # restore the multi-scale knobs to their defaults so the 'average' chain (and the downstream
        # stage-5 checkpoint feeding test_05+) is unaffected by the consolidated variant above
        for param, default in (("hdbscan_clustering_partition_cutoff", None),
                               ("hdbscan_cluster_selection_epsilon", None),
                               ("hdbscan_noise_merge_cutoff", None),
                               ("consolidate_orphan_superclusters", False),
                               ("orphan_assignment_cutoff", None)):
            mol_system.parameters[param] = default

        # default 'average' linkage: historical distance-matrix golden comparison + chain checkpoint
        mol_system.parameters["clustering_method"] = "agglomerative"
        mol_system.parameters["clustering_linkage"] = "average"
        mol_system.merge_tunnel_clusters2super_clusters()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "clustering"),
                              os.path.join(self.out_path, "_internal", "clustering"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "clustering"),
                              os.path.join(self.out_path, "data", "clustering"), self)
        save_checkpoint(mol_system, self._get_dumpfile(4), overwrite=True)

    def test_04merge_tunnel_clusters2super_clusters_slurm(self):
        """Same checks as test_04merge..., but the stage-4 distances are computed via the SLURM
        backend (submitit). Runs only when a SLURM environment with submitit is detected."""
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(3))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        # the SLURM shards rebuild the analysis from the config file, so it must carry an absolute
        # output_path matching this test run (the shared test config uses a relative one) and the
        # same slurm_root_folder as the launcher - kept outside the clustering folder compared below
        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage04.ini")
        with open(os.path.join(self.out_path, "config.ini")) as in_stream, \
                open(slurm_config, "w") as out_stream:
            for line in in_stream:
                if line.startswith("output_path"):
                    line = "output_path = {}\n".format(self.out_path)
                out_stream.write(line)
                if line.strip() == "[CALCULATIONS_SETTINGS]":
                    out_stream.write("slurm_root_folder = {}\n".format(slurm_root_folder))
        mol_system.config_file = slurm_config

        mol_system.parameters["stage04_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 3
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # Remove the outputs that test_04 (local backend) already wrote so that a silent
        # no-op in the SLURM path would cause compare_test_folders to fail rather than
        # silently pass against stale local-backend files.
        for _stale_dir in [
            os.path.join(self.out_path, "_internal", "clustering"),
            os.path.join(self.out_path, "data", "clustering"),
        ]:
            if os.path.isdir(_stale_dir):
                shutil.rmtree(_stale_dir)

        mol_system.compute_tunnel_clusters_distances()
        mol_system.merge_tunnel_clusters2super_clusters()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "clustering"),
                              os.path.join(self.out_path, "_internal", "clustering"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "clustering"),
                              os.path.join(self.out_path, "data", "clustering"), self)

    def test_04merge_tunnel_clusters2super_clusters_slurm_restart(self):
        """Restart-path coverage for the stage-4 SLURM backend (distance shards): exercises
        the fingerprint-only resume path by deleting a per-shard .npy and confirming the
        framework resubmits a fresh SLURM array job. Stage 4 has `has_side_effects=False` so
        there is no archive / manifest dimension - the distance matrix is the only artefact -
        but it is still important to verify that partial-cache deletion drives a correct
        resubmission. Runs only when SLURM + submitit are available and the primary stage-4
        slurm test has populated the cache.
        """
        import glob
        import shutil
        import numpy as np
        from scipy.spatial.distance import squareform
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(3))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage04.ini")
        stage_folder = os.path.join(slurm_root_folder, "stage04_distances")
        shard_results = sorted(glob.glob(os.path.join(stage_folder, "shard_result_*.npy")))
        if not shard_results:
            self.skipTest("primary stage-4 slurm test did not populate the cache")
        mol_system.config_file = slurm_config

        mol_system.parameters["stage04_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 3
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # delete one per-shard .npy; the missing-shard check must detect it and resubmit
        os.remove(shard_results[0])
        # wipe the downstream assembly outputs so a silent no-op would fail comparison
        for _wipe_dir in [
            os.path.join(self.out_path, "_internal", "clustering"),
            os.path.join(self.out_path, "data", "clustering"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        mol_system.compute_tunnel_clusters_distances()
        mol_system.merge_tunnel_clusters2super_clusters()

        # parity check against the saved reference matrix
        compare_test_folders(os.path.join(self.saved_data, "_internal", "clustering"),
                              os.path.join(self.out_path, "_internal", "clustering"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "clustering"),
                              os.path.join(self.out_path, "data", "clustering"), self)

        # submission happened: log set grew
        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertGreater(len(logs_after), len(logs_before),
                            msg="restart: deleting a per-shard .npy should force a SLURM "
                                "resubmission, but the submitit log-file set did not grow")

        # all per-shard .npy back on disk after resubmission
        rebuilt = sorted(glob.glob(os.path.join(stage_folder, "shard_result_*.npy")))
        self.assertEqual(len(rebuilt), 3,
                          msg="restart: expected 3 per-shard .npy files after resubmission")

    def test_05create_super_cluster_profiles(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(4))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.create_super_cluster_profiles()
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "initial_super_cluster_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "initial_super_cluster_details.txt"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "initial"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "initial"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "initial"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "initial"), self)
        save_checkpoint(mol_system, self._get_dumpfile(5), overwrite=True)

    def test_06generate_super_cluster_summary(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(5))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.generate_super_cluster_summary(out_filename="1-initial_tunnels_summary.txt")
        compare_test_files(os.path.join(self.saved_data, "statistics", "1-initial_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "1-initial_tunnels_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "statistics",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "1-initial_tunnels_summary_bottleneck_residues.txt"), self)

        save_checkpoint(mol_system, self._get_dumpfile(6), overwrite=True)

    def test_07save_super_clusters_visualization(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(6))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.save_super_clusters_visualization(script_name="visualize_tunnels.py")
        compare_test_files(os.path.join(self.saved_data, "visualization", "visualize_tunnels.py"),
                            os.path.join(self.out_path, "visualization", "visualize_tunnels.py"), self)
        compare_test_files(os.path.join(self.saved_data, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels.py"),
                            os.path.join(self.out_path, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels.py"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "super_cluster_CGOs"),
                              os.path.join(self.out_path, "visualization", "sources", "super_cluster_CGOs"), self,
                              pattern=r'.+pathset_1\.dump')
        save_checkpoint(mol_system, self._get_dumpfile(7), overwrite=True)

    def test_08filter_super_cluster_profiles(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(7))
        except FileNotFoundError:
            self.skipTest("previous test not finished")
        mol_system.filter_super_cluster_profiles(min_length=5, min_bottleneck_radius=1, min_avg_snapshots_num=-1,
                                                 min_sims_num=-1)
        mol_system.generate_super_cluster_summary(out_filename="2-filtered_tunnels_summary.txt")
        mol_system.save_super_clusters_visualization(script_name="visualize_tunnels_filtered.py")
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details1.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details1.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels_filtered.py"),
                            os.path.join(self.out_path, "visualization", "comparative_analysis", "md1",
                                         "visualize_tunnels_filtered.py"), self)
        compare_test_files(os.path.join(self.saved_data, "visualization", "visualize_tunnels_filtered.py"),
                            os.path.join(self.out_path, "visualization", "visualize_tunnels_filtered.py"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "super_cluster_CGOs"),
                              os.path.join(self.out_path, "visualization", "sources", "super_cluster_CGOs"), self,
                              pattern=r'.+pathset_2\.dump')

        compare_test_files(os.path.join(self.saved_data, "statistics", "2-filtered_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "2-filtered_tunnels_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "statistics",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "2-filtered_tunnels_summary_bottleneck_residues.txt"), self)

        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "filtered01"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "filtered01"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "filtered01"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "filtered01"), self)
        save_checkpoint(mol_system, self._get_dumpfile(8), overwrite=True)

    def test_09process_aquaduct_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(8))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.process_aquaduct_networks()
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"), self)

        save_checkpoint(mol_system, self._get_dumpfile(9), overwrite=True)

    def test_09process_aquaduct_networks_slurm(self):
        """Same checks as test_09process_aquaduct_networks, but is computed via the SLURM
        backend (submitit). Runs only when a SLURM environment with submitit is detected."""
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(8))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        # The SLURM shards rebuild the analysis from the config file, so it must carry an
        # absolute output_path matching this test run (the shared test config uses a relative
        # one) and the same slurm_root_folder as the launcher - kept outside the network_data
        # folders compared below.
        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage07.ini")
        with open(os.path.join(self.out_path, "config.ini")) as in_stream, \
                open(slurm_config, "w") as out_stream:
            for line in in_stream:
                if line.startswith("output_path"):
                    line = "output_path = {}\n".format(self.out_path)
                out_stream.write(line)
                if line.strip() == "[CALCULATIONS_SETTINGS]":
                    out_stream.write("slurm_root_folder = {}\n".format(slurm_root_folder))

        mol_system.config_file = slurm_config

        mol_system.parameters["stage07_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # Remove the outputs that test_09 (local backend) already wrote so that a silent
        # no-op in the SLURM path would cause compare_test_folders to fail rather than
        # silently pass against stale local-backend files.
        for _stale_dir in [
            os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct", "md1"),
        ]:
            if os.path.isdir(_stale_dir):
                shutil.rmtree(_stale_dir)

        mol_system.process_aquaduct_networks()
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"), self)

    def test_09process_aquaduct_networks_slurm_resume(self):
        """Resume-path coverage for the stage-7 SLURM backend: after
        test_09process_aquaduct_networks_slurm has populated the slurm folder with per-shard
        pickles, manifests, and side-effect tarballs, wipe the user-facing visualisation
        folder and re-run process_aquaduct_networks. The framework must restore the wiped
        files from the per-shard tarball backups WITHOUT resubmitting any SLURM job (which
        would re-extract the AQUA-DUCT tarballs). Verified by:
          1) the restored output matches the saved reference data byte-for-byte (same as the
             primary slurm test)
          2) submitit's per-task log files in the slurm folder do not grow (no new array job
             was submitted)
        Runs only when a SLURM environment with submitit is detected; depends on the primary
        slurm test having run first to populate the cache.
        """

        import shutil
        import glob
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(8))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage07.ini")
        stage_folder = os.path.join(slurm_root_folder, "stage07_aquaduct_networks")
        shard_pickles = sorted(glob.glob(os.path.join(stage_folder, "shard_result_*.pkl")))
        if not shard_pickles:
            self.skipTest("primary stage-7 slurm test did not populate the cache")
        mol_system.config_file = slurm_config

        mol_system.parameters["stage07_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # Snapshot the submitit log files before the resume call; if the framework correctly
        # restores from backup, no new SLURM job is submitted and the set must be unchanged
        # afterwards. Captures both *_log.out and *_log.err which submitit creates per task.
        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        # Wipe the visualisation folder + internal network_data folder - the side-effect
        # files live in both; the framework must restore them from the per-shard tar.gz
        # backups in slurm_root_folder/stage07_aquaduct_networks/.
        for _wipe_dir in [
            os.path.join(self.out_path, "_internal", "network_data", "aquaduct"),
            os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.process_aquaduct_networks()
        # 1) parity check: same comparisons as the primary slurm test - restored side-effects
        #    must be byte-identical to the original outputs
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"), self)

        # 2) no-resubmission check: submitit log file set must be unchanged - any new files
        #    would indicate a fresh array submission, which means the restore path did not
        #    kick in. This is the strongest evidence of "reconstruction without rerun".
        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertEqual(logs_before, logs_after,
                          msg="resume path should restore from backup without submitting a "
                              "new SLURM array job; new submitit log files indicate a "
                              "resubmission happened")

    def test_09process_aquaduct_networks_slurm_restart(self):
        """Restart-path coverage for the stage-7 SLURM backend: companion to the resume test
        above. Exercises the fallback path where the side-effects backup is ALSO unrecoverable
        (or the per-shard pickle itself is missing), so the framework must invalidate the
        cached shard and resubmit a fresh SLURM array job. Two scenarios are exercised:
          a) per-shard pickle deleted - the missing-shard check immediately treats it as
             missing and the launcher resubmits without even consulting the manifest.
          b) side-effects tar.gz deleted + visualisation files wiped - the manifest is still
             present so the resume check finds expected files missing, attempts restore,
             fails (no archive), invalidates the cache, and resubmits.
        Both scenarios end with a parity check against the saved reference output. Runs only
        when a SLURM environment with submitit is detected and the primary slurm test has
        populated the cache.
        """

        import shutil
        import glob
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(8))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage07.ini")
        stage_folder = os.path.join(slurm_root_folder, "stage07_aquaduct_networks")
        shard_pickles = sorted(glob.glob(os.path.join(stage_folder, "shard_result_*.pkl")))
        if not shard_pickles:
            self.skipTest("primary stage-7 slurm test did not populate the cache")
        mol_system.config_file = slurm_config

        mol_system.parameters["stage07_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # ---- Scenario (a): delete one per-shard pickle. The missing-shard check sees it
        # gone immediately and the launcher must resubmit a fresh SLURM array.
        os.remove(shard_pickles[0])

        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        for _wipe_dir in [
            os.path.join(self.out_path, "_internal", "network_data", "aquaduct"),
            os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.process_aquaduct_networks()

        # parity: a fresh resubmission must produce the same output as the original run
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"), self)

        # logs grew: new submission happened (the resume path would have left logs unchanged)
        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertGreater(len(logs_after), len(logs_before),
                            msg="restart scenario (a): deleting a per-shard pickle should "
                                "force a SLURM resubmission, but the submitit log-file set "
                                "did not grow")

        # ---- Scenario (b): delete the side-effects backup tar.gz for one shard, then wipe
        # the visualisation folder. Manifest still indicates files are expected, but restore
        # cannot find the archive, so the cache is invalidated and the shard is resubmitted.
        from transport_tools.libs.slurm import AquaductNetworksShardStage
        archive_path = AquaductNetworksShardStage.shard_side_effects_archive_path(
            stage_folder, 0)
        if os.path.exists(archive_path):
            os.remove(archive_path)

        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        for _wipe_dir in [
            os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.process_aquaduct_networks()

        # parity: a fresh resubmission must produce the same output as the original run
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "network_data", "aquaduct",
                                           "md1"), self)

        # logs grew: new submission happened (the resume path would have left logs unchanged)
        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertGreater(len(logs_after), len(logs_before),
                            msg="restart scenario (b): deleting a side-effects archive + "
                                "wiping the vis folder should force a SLURM resubmission, "
                                "but the submitit log-file set did not grow")

    def test_10create_layered_description4aquaduct_networks(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(9))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.create_layered_description4aquaduct_networks()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "layered_data", "aquaduct"),
                              os.path.join(self.out_path, "_internal", "layered_data", "aquaduct"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "nodes"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1",
                                           "nodes"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "paths"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1",
                                           "paths"), self)

        save_checkpoint(mol_system, self._get_dumpfile(10), overwrite=True)

    def test_10create_layered_description4aquaduct_networks_slurm(self):
        """Same checks as test_10create_layered_description4aquaduct_networks, but stage-8
        layering is computed via the SLURM backend (submitit). Runs only when a SLURM
        environment with submitit is detected. Mirrors test_03create_layered..._slurm."""
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(9))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        # The SLURM shards rebuild the analysis from the config file, so it must carry an
        # absolute output_path matching this test run (the shared test config uses a relative
        # one) and the same slurm_root_folder as the launcher - kept outside the layered_data
        # folders compared below.
        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage08.ini")
        with open(os.path.join(self.out_path, "config.ini")) as in_stream, \
                open(slurm_config, "w") as out_stream:
            for line in in_stream:
                if line.startswith("output_path"):
                    line = "output_path = {}\n".format(self.out_path)
                out_stream.write(line)
                if line.strip() == "[CALCULATIONS_SETTINGS]":
                    out_stream.write("slurm_root_folder = {}\n".format(slurm_root_folder))
        mol_system.config_file = slurm_config

        mol_system.parameters["stage08_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # Remove the outputs that test_10 (local backend) already wrote so that a silent
        # no-op in the SLURM path would cause compare_test_folders to fail rather than
        # silently pass against stale local-backend files.
        for _stale_dir in [
            os.path.join(self.out_path, "_internal", "layered_data", "aquaduct"),
            os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1"),
        ]:
            if os.path.isdir(_stale_dir):
                shutil.rmtree(_stale_dir)

        mol_system.create_layered_description4aquaduct_networks()
        compare_test_folders(os.path.join(self.saved_data, "_internal", "layered_data", "aquaduct"),
                              os.path.join(self.out_path, "_internal", "layered_data", "aquaduct"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "nodes"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1",
                                           "nodes"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "paths"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1",
                                           "paths"), self)

    def test_10create_layered_description4aquaduct_networks_slurm_resume(self):
        """Resume-path coverage for the stage-8 SLURM backend: mirror of
        test_03create_layered_description4tunnel_networks_slurm_resume but for aquaduct
        layering. Verifies the framework restores per-event .py + nodes/ side-effects from
        the per-shard tar.gz backup without resubmitting any SLURM job. Runs only when a
        SLURM environment with submitit is detected and the primary stage-8 slurm test has
        already populated the cache.
        """
        import glob
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(9))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage08.ini")
        stage_folder = os.path.join(slurm_root_folder, "stage08_aquaduct_layering")
        shard_pickles = glob.glob(os.path.join(slurm_root_folder, "stage08_aquaduct_layering",
                                                "shard_result_*.pkl"))
        if not shard_pickles:
            self.skipTest("primary stage-8 slurm test did not populate the cache")
        mol_system.config_file = slurm_config

        mol_system.parameters["stage08_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # Snapshot the submitit log-file set before the resume call; if the framework
        # correctly restores from backup, no new SLURM job is submitted and the set must be
        # unchanged afterwards.
        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        # Wipe the visualisation folder + the internal layered_data folder; the side-effects
        # tar.gz backup lives inside slurm_root_folder and must restore the per-event .py +
        # nodes/ files without re-running the per-event sklearn stack on any SLURM node.
        for _wipe_dir in [
            os.path.join(self.out_path, "_internal", "layered_data", "aquaduct"),
            os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.create_layered_description4aquaduct_networks()

        compare_test_folders(os.path.join(self.saved_data, "_internal", "layered_data", "aquaduct"),
                              os.path.join(self.out_path, "_internal", "layered_data", "aquaduct"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "nodes"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1",
                                           "nodes"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1", "paths"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1",
                                           "paths"), self)

        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertEqual(logs_before, logs_after,
                          msg="resume path should restore from backup without submitting a "
                              "new SLURM array job; new submitit log files indicate a "
                              "resubmission happened")

    def test_10create_layered_description4aquaduct_networks_slurm_restart(self):
        """Restart-path coverage for the stage-8 SLURM backend: companion to the stage-8
        resume test above. Mirrors test_03..._slurm_restart for aquaduct layering -
        exercises both the missing-pickle path (scenario a) and the missing-archive +
        wiped-files path (scenario b), each ending with a parity check and the assertion
        that submitit logged a new array submission. Runs only when a SLURM environment
        with submitit is detected and the primary stage-8 slurm test has populated the
        cache.
        """
        import glob
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(9))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage08.ini")
        stage_folder = os.path.join(slurm_root_folder, "stage08_aquaduct_layering")
        shard_pickles = sorted(glob.glob(os.path.join(stage_folder, "shard_result_*.pkl")))
        if not shard_pickles:
            self.skipTest("primary stage-8 slurm test did not populate the cache")
        mol_system.config_file = slurm_config

        mol_system.parameters["stage08_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # ---- Scenario (a): delete one per-shard pickle, force resubmission
        os.remove(shard_pickles[0])

        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        for _wipe_dir in [
            os.path.join(self.out_path, "_internal", "layered_data", "aquaduct"),
            os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.create_layered_description4aquaduct_networks()

        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"), self)

        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertGreater(len(logs_after), len(logs_before),
                            msg="restart scenario (a): deleting a per-shard pickle should "
                                "force a SLURM resubmission, but the submitit log-file set "
                                "did not grow")

        # ---- Scenario (b): delete the side-effects backup tar.gz, wipe vis folder
        from transport_tools.libs.slurm import AquaductLayeringShardStage
        archive_path = AquaductLayeringShardStage.shard_side_effects_archive_path(stage_folder, 1)
        if os.path.exists(archive_path):
            os.remove(archive_path)

        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        for _wipe_dir in [
            os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct", "md1"),
        ]:
            if os.path.isdir(_wipe_dir):
                shutil.rmtree(_wipe_dir)

        mol_system.create_layered_description4aquaduct_networks()

        compare_test_folders(os.path.join(self.saved_data, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"),
                              os.path.join(self.out_path, "visualization", "sources", "layered_data", "aquaduct",
                                           "md1"), self)

        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertGreater(len(logs_after), len(logs_before),
                            msg="restart scenario (b): deleting a side-effects archive + "
                                "wiping the vis folder should force a SLURM resubmission, "
                                "but the submitit log-file set did not grow")

    def test_11assign_transport_events(self):
        try:
            mol_system = load_checkpoint(self._get_dumpfile(10))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        mol_system.assign_transport_events()
        mol_system.save_super_clusters_visualization(script_name="visualize_events.py")
        mol_system.generate_super_cluster_summary(out_filename="3-initial_events_summary.txt")
        mol_system.filter_super_cluster_profiles(min_length=5, min_avg_snapshots_num=-1, min_sims_num=1,
                                                 min_total_events=1)
        mol_system.save_super_clusters_visualization(script_name="visualize_events_filtered.py")
        mol_system.generate_super_cluster_summary(out_filename="4-filtered_events_summary.txt")

        compare_test_files(os.path.join(self.saved_data, "visualization", "visualize_events.py"),
                            os.path.join(self.out_path, "visualization", "visualize_events.py"),self)
        compare_test_files(os.path.join(self.saved_data, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events.py"),
                            os.path.join(self.out_path, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events.py"), self)
        compare_test_files(os.path.join(self.saved_data, "visualization", "visualize_events_filtered.py"),
                            os.path.join(self.out_path, "visualization", "visualize_events_filtered.py"),self)
        compare_test_files(os.path.join(self.saved_data, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events_filtered.py"),
                            os.path.join(self.out_path, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events_filtered.py"), self)

        compare_test_files(os.path.join(self.saved_data, "statistics", "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "3-initial_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "3-initial_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "4-filtered_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "statistics",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "3-initial_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "3-initial_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "4-filtered_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary_bottleneck_residues.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "initial_super_cluster_events_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "initial_super_cluster_events_details.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "outlier_transport_events_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "outlier_transport_events_details.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details2.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details2.txt"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "exact_matching_analysis", "md1"),
                              os.path.join(self.out_path, "data", "exact_matching_analysis", "md1"), self)

        compare_test_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_entry_sc2"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_entry_sc2"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc3"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc3"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc2"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc2"), self)

        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "filtered02"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "filtered02"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "filtered02"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "filtered02"), self)

        save_checkpoint(mol_system, self._get_dumpfile(11), overwrite=True)

    def test_11assign_transport_events_slurm(self):
        """Same checks as test_11assign_transport_events, but the stage-9 assignment is
        computed via the SLURM backend (submitit). Runs only when a SLURM environment with
        submitit is detected. Mirrors test_03create_layered..._slurm /
        test_10create_layered_description4aquaduct_networks_slurm: replays the full
        post-assignment pipeline (visualisations, summaries, filter pass) so the same set of
        artifacts the local test compares is regenerated under the SLURM path, removes the
        stale local-backend outputs first so a silent no-op in the SLURM path would fail the
        comparison rather than silently pass against stale files, and finally asserts the
        same set of files matches the saved reference data byte-for-byte.
        """
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(10))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        # The SLURM shards rebuild the analysis from the config file, so it must carry an
        # absolute output_path matching this test run (the shared test config uses a relative
        # one) and the same slurm_root_folder as the launcher - kept outside the super_clusters
        # folder compared below.
        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage09.ini")
        with open(os.path.join(self.out_path, "config.ini")) as in_stream, \
                open(slurm_config, "w") as out_stream:
            for line in in_stream:
                if line.startswith("output_path"):
                    line = "output_path = {}\n".format(self.out_path)
                out_stream.write(line)
                if line.strip() == "[CALCULATIONS_SETTINGS]":
                    out_stream.write("slurm_root_folder = {}\n".format(slurm_root_folder))
        mol_system.config_file = slurm_config

        mol_system.parameters["stage09_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # Remove the outputs that the local test_11 above already wrote so a silent no-op in
        # the SLURM path would cause the comparisons to fail rather than silently pass against
        # stale local-backend files.
        for _stale_dir in [
            os.path.join(self.out_path, "data", "super_clusters", "details"),
            os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "filtered02"),
            os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "filtered02"),
            os.path.join(self.out_path, "data", "exact_matching_analysis"),
            os.path.join(self.out_path, "visualization", "exact_matching_analysis"),
        ]:
            if os.path.isdir(_stale_dir):
                shutil.rmtree(_stale_dir)
        for _stale_file in [
            os.path.join(self.out_path, "visualization", "visualize_events.py"),
            os.path.join(self.out_path, "visualization", "visualize_events_filtered.py"),
            os.path.join(self.out_path, "visualization", "comparative_analysis", "md1",
                         "visualize_events.py"),
            os.path.join(self.out_path, "visualization", "comparative_analysis", "md1",
                         "visualize_events_filtered.py"),
            os.path.join(self.out_path, "statistics", "3-initial_events_summary.txt"),
            os.path.join(self.out_path, "statistics", "comparative_analysis",
                         "3-initial_events_summary.txt"),
            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                         "3-initial_events_summary.txt"),
            os.path.join(self.out_path, "statistics", "4-filtered_events_summary.txt"),
            os.path.join(self.out_path, "statistics", "comparative_analysis",
                         "4-filtered_events_summary.txt"),
            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                         "4-filtered_events_summary.txt"),
            os.path.join(self.out_path, "statistics",
                         "3-initial_events_summary_bottleneck_residues.txt"),
            os.path.join(self.out_path, "statistics", "comparative_analysis",
                         "3-initial_events_summary_bottleneck_residues.txt"),
            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                         "3-initial_events_summary_bottleneck_residues.txt"),
            os.path.join(self.out_path, "statistics",
                         "4-filtered_events_summary_bottleneck_residues.txt"),
            os.path.join(self.out_path, "statistics", "comparative_analysis",
                         "4-filtered_events_summary_bottleneck_residues.txt"),
            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                         "4-filtered_events_summary_bottleneck_residues.txt"),
        ]:
            if os.path.isfile(_stale_file):
                os.remove(_stale_file)

        # mirror the post-assignment pipeline the local test_11 runs, so the same set of
        # artifacts is regenerated under the SLURM path
        mol_system.assign_transport_events()
        mol_system.save_super_clusters_visualization(script_name="visualize_events.py")
        mol_system.generate_super_cluster_summary(out_filename="3-initial_events_summary.txt")
        mol_system.filter_super_cluster_profiles(min_length=5, min_avg_snapshots_num=-1, min_sims_num=1,
                                                 min_total_events=1)
        mol_system.save_super_clusters_visualization(script_name="visualize_events_filtered.py")
        mol_system.generate_super_cluster_summary(out_filename="4-filtered_events_summary.txt")

        # exact mirror of the comparison block in test_11assign_transport_events
        compare_test_files(os.path.join(self.saved_data, "visualization", "visualize_events.py"),
                            os.path.join(self.out_path, "visualization", "visualize_events.py"),self)
        compare_test_files(os.path.join(self.saved_data, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events.py"),
                            os.path.join(self.out_path, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events.py"), self)
        compare_test_files(os.path.join(self.saved_data, "visualization", "visualize_events_filtered.py"),
                            os.path.join(self.out_path, "visualization", "visualize_events_filtered.py"),self)
        compare_test_files(os.path.join(self.saved_data, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events_filtered.py"),
                            os.path.join(self.out_path, "visualization",  "comparative_analysis", "md1",
                                         "visualize_events_filtered.py"), self)

        compare_test_files(os.path.join(self.saved_data, "statistics", "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "3-initial_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "3-initial_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "4-filtered_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "statistics",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "3-initial_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "3-initial_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "3-initial_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics",
                                         "4-filtered_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis",
                                         "4-filtered_events_summary_bottleneck_residues.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary_bottleneck_residues.txt"),
                            os.path.join(self.out_path, "statistics", "comparative_analysis", "md1",
                                         "4-filtered_events_summary_bottleneck_residues.txt"), self)

        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "initial_super_cluster_events_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "initial_super_cluster_events_details.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "outlier_transport_events_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "outlier_transport_events_details.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details2.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "filtered_super_cluster_details2.txt"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "exact_matching_analysis", "md1"),
                              os.path.join(self.out_path, "data", "exact_matching_analysis", "md1"), self)

        compare_test_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_entry_sc2"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_entry_sc2"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc3"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc3"), self)
        compare_test_folders(os.path.join(self.saved_data, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc2"),
                              os.path.join(self.out_path, "visualization", "exact_matching_analysis", "md1",
                                           "wat_1_release_sc2"), self)

        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "CSV_profiles", "filtered02"),
                              os.path.join(self.out_path, "data", "super_clusters", "CSV_profiles", "filtered02"), self)
        compare_test_folders(os.path.join(self.saved_data, "data", "super_clusters", "bottlenecks", "filtered02"),
                              os.path.join(self.out_path, "data", "super_clusters", "bottlenecks", "filtered02"), self)

    def test_11assign_transport_events_slurm_restart(self):
        """Restart-path coverage for the stage-9 SLURM backend (event assignment): companion
        to test_11assign_transport_events_slurm. Exercises the basic resume optimisation by
        deleting one per-shard pickle and verifying the framework resubmits a fresh SLURM
        array job, and that the produced assignment state still matches the saved reference
        byte-for-byte (verified via the two text dumps directly produced by
        assign_transport_events).

        Note: stage 9 keeps `has_side_effects=False` for now - manifest support for 
        `_exact_event_tunnel_matching` side-effect files is deferred until a
        user request surfaces. So no "resume from backup" path applies here; only the
        restart-by-missing-pickle path. Runs only when SLURM + submitit are available and
        the primary stage-9 slurm test has populated the cache.
        """
        import glob
        import shutil
        from transport_tools.libs.slurm import submitit_available

        if shutil.which("sbatch") is None:
            self.skipTest("SLURM not available (sbatch not found)")
        if not submitit_available():
            self.skipTest("optional 'submitit' package not installed")

        try:
            mol_system = load_checkpoint(self._get_dumpfile(10))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        slurm_root_folder = os.path.join(self.out_path, "_slurm")
        slurm_config = os.path.join(self.out_path, "config_slurm_stage09.ini")
        stage_folder = os.path.join(slurm_root_folder, "stage09_event_assignment")
        shard_pickles = sorted(glob.glob(os.path.join(stage_folder, "shard_result_*.pkl")))
        if not shard_pickles:
            self.skipTest("primary stage-9 slurm test did not populate the cache")
        mol_system.config_file = slurm_config

        mol_system.parameters["stage09_backend"] = "slurm"
        mol_system.parameters["slurm_num_shards"] = 2
        mol_system.parameters["slurm_cpus_per_task"] = 2
        mol_system.parameters["slurm_timeout_min"] = 15
        mol_system.parameters["slurm_mem_gb"] = 2.0
        mol_system.parameters["slurm_root_folder"] = slurm_root_folder
        # keep the per-shard cache so the resume + restart tests in this chain have something
        # to operate on; default behaviour wipes the cache after a successful assembly
        mol_system.parameters["slurm_keep_shard_results"] = True

        # delete one per-shard pickle to force resubmission; the launcher's missing-shard
        # check immediately treats it as missing because the result file is the only artefact
        # stage 9 currently tracks (no manifest / archive plumbing yet)
        os.remove(shard_pickles[0])
        # wipe the assignment-output text dumps so a silent no-op would fail the comparison
        for _stale_file in [
            os.path.join(self.out_path, "data", "super_clusters", "details",
                         "initial_super_cluster_events_details.txt"),
            os.path.join(self.out_path, "data", "super_clusters", "details",
                         "outlier_transport_events_details.txt"),
        ]:
            if os.path.isfile(_stale_file):
                os.remove(_stale_file)

        logs_before = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                       set(glob.glob(os.path.join(stage_folder, "*_log.err")))

        mol_system.assign_transport_events()

        # parity: the per-supercluster + outlier event-details text dumps must match the
        # saved reference (these are the most direct artefacts of assign_transport_events)
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "initial_super_cluster_events_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "initial_super_cluster_events_details.txt"), self)
        compare_test_files(os.path.join(self.saved_data, "data", "super_clusters", "details",
                                         "outlier_transport_events_details.txt"),
                            os.path.join(self.out_path, "data", "super_clusters", "details",
                                         "outlier_transport_events_details.txt"), self)

        # submission happened: log set grew
        logs_after = set(glob.glob(os.path.join(stage_folder, "*_log.out"))) | \
                      set(glob.glob(os.path.join(stage_folder, "*_log.err")))
        self.assertGreater(len(logs_after), len(logs_before),
                            msg="restart: deleting a per-shard pickle should force a SLURM "
                                "resubmission, but the submitit log-file set did not grow")

    def test_12custom_analyses(self):
        from numpy import save

        try:
            mol_system = load_checkpoint(self._get_dumpfile(11))
        except FileNotFoundError:
            self.skipTest("previous test not finished")

        super_cluster_id = 1
        filters = define_filters(min_length=10, min_bottleneck_radius=1.4)

        visualization_output_folder = os.path.join(self.out_path, "customs", "static")
        mol_system.show_tunnels_passing_filter(super_cluster_id, filters, visualization_output_folder,
                                               start_snapshot=1, end_snapshot=100, trajectory=False)
        compare_test_folders(os.path.join(self.saved_data, "customs", "static"),
                              os.path.join(self.out_path, "customs", "static"), self)

        visualization_output_folder = os.path.join(self.out_path, "customs", "dynamics")
        md_labels = ["md1"]
        mol_system.show_tunnels_passing_filter(super_cluster_id, filters, visualization_output_folder,
                                               md_labels=md_labels, start_snapshot=1, end_snapshot=100, trajectory=True)
        compare_test_folders(os.path.join(self.saved_data, "customs", "dynamics"),
                              os.path.join(self.out_path, "customs", "dynamics"), self)

        filters = define_filters(min_length=10)
        super_cluster_id = None
        parameter = "bottleneck_radius"
        dataset = mol_system.get_property_time_evolution_data(parameter, active_filters=filters, sc_id=super_cluster_id)
        self.assertListEqual([28, 14, 12, 10, 27, 13, 26, 8, 25, 24, 3, 4, 18, 23, 22, 21, 2, 20, 6, 19, 9, 17, 16, 15, 7, 11, 1, 5],
                              [*dataset.keys()])
        save(os.path.join(self.out_path, "customs", "time_evolution_dataset.npy"), dataset[1]["md1"])
        compare_test_files(os.path.join(self.saved_data, "customs", "time_evolution_dataset.npy"),
                            os.path.join(self.out_path, "customs", "time_evolution_dataset.npy"), self)


if __name__ == "__main__":
    unittest.main()
