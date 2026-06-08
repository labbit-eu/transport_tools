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

import os
import sys
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError
from configparser import ConfigParser
from typing import List, Any, Tuple
from logging import getLogger
from transport_tools.libs.ui import initiate_tools
from transport_tools.libs.utils import get_filepath

logger = getLogger(__name__)


@dataclass(frozen=True)
class ShardContext:
    """
    Execution context for a config load. The launcher uses the default (is_shard=False);
    a SLURM shard rebuild passes is_shard=True. It carries the on/off switches that depend on
    *where* the config is being rebuilt rather than *which* parameters it holds - so validation
    that is meaningful on the launcher but inert on a compute node can be skipped on shards.

    :param is_shard: True when the config is rebuilt inside a SLURM array task (see
        slurm._rebuild_mol_system). On a shard 'num_cpus' is inert - every compute_*_shard sizes
        its inner pool from 'slurm_cpus_per_task', never 'num_cpus' - and the compute node's CPU
        count is unrelated to the launcher's, so the whole num_cpus validation block is skipped to
        avoid spuriously aborting the shard. Future shard-only switches belong here too.
    """

    is_shard: bool = False


class AnalysisConfig:
    def __init__(self, file2load_from: str | None = None, logging: bool = True,
                 context: ShardContext | None = None, active_stages: Tuple[int, int] | None = None):
        """
        Class storing the job configuration, also enables some parameter evaluation & completion
        :param file2load_from: INI file with configuration to load from
        :param logging: if logging should be set up
        :param context: execution context controlling where-dependent validation (see ShardContext).
            None => a normal local/launcher run (ShardContext() with is_shard=False).
        :param active_stages: inclusive (lo, hi) interval of pipeline stages this process will
            actually run. Stage-scoped setups and input-existence checks whose owning stage falls
            outside this interval are skipped (their state, when needed downstream on resume, comes
            from the checkpoint instead - see TransportProcesses.update_configuration). None => derive
            from 'start_from_stage'/'stop_after_stage' at config-load time (the normal launcher case,
            i.e. validate everything the run will touch). A SLURM shard passes its single stage, e.g.
            (4, 4), so only that stage's prerequisites are checked.
        """

        self.context = context or ShardContext()
        self._active_stages_override = active_stages
        # resolved (lo, hi) interval; finalised in _load_configuration once start/stop are parsed
        self._active_stages: Tuple[int, int] = active_stages if active_stages is not None else (1, 10)
        self.source_file = file2load_from
        self.calculations_settings = dict()
        self.output_settings = dict()
        self.input_paths = dict()
        self.output_paths = dict()
        self.advanced_settings = dict()
        self._logged_empty_aquaduct_folders: set = set()
        self.legacy_params = {
                                "min_tunnel_radius4clustering":"relevant_tunnel_min_radius",
                                "min_tunnel_length4clustering":"relevant_tunnel_min_length",
                                "max_tunnel_curvature4clustering":"relevant_tunnel_max_curvature"
                             }
        

        self._set_initial_sections_w_defaults()
        if file2load_from is not None:
            self._load_configuration(file2load_from)
            self.parameters: dict = self.calculations_settings.copy()
            self.parameters.update(self.output_settings)
            self.parameters.update(self.input_paths)
            self.parameters.update(self.output_paths)
            self.parameters.update(self.advanced_settings)
            self.parameters.update(self.internal_settings)
            if logging:
                initiate_tools(self.get_parameter("std_level"), self.get_parameter("log_level"), self.get_parameter("verbose_logging"),
                               self.get_parameter("logfile_path"))
            self._test_input_data()
            self._autocomplete_parameters()
            self._validate_parameter_values()

    def _set_initial_sections_w_defaults(self):
        """
        Generates initial dictionaries with default configuration of each section
        """

        # assemble input paths
        self.input_paths = {
            # CAVER
            "caver_results_path": None,
            "caver_results_folder_pattern": "*",
            "caver_results_relative_subfolder_path": None,
            "caver_relative_profile_file": None,
            "caver_relative_origin_file": None,
            "caver_relative_pdb_file": None,
            "caver_pdb_file_pattern": "*[0-9]*.pdb",
            "caver_relative_bottleneck_file": None,

            # AQUA-DUCT
            "aquaduct_results_path": None,
            "aquaduct_results_folder_pattern": "*",
            "aquaduct_results_relative_tarfile": None,
            "aquaduct_results_relative_summaryfile": None,

            # MD trajectories
            "trajectory_path": None,
            "trajectory_folder_pattern": "*",
            "trajectory_relative_file": None,
            "topology_relative_file": None
        }

        self.calculations_settings = {
            "start_from_stage": 1,
            "stop_after_stage": None,
            "num_cpus": None,  # set number of CPUs automatically using self._detect_set_num_cpu()
            "worker_task_timeout_s": 3600,  # per-task timeout for multiprocessing-Pool workers; on TimeoutError the offending task is logged and the pool terminated. Set to None to disable.
            "pdb_reference_structure": None,  # reference structure for alignment
            "layer_thickness": 1.5,  # thickness of concentric layered grid

            # Parsing of tunnel clusters from CAVER results
            "snapshots_per_simulation": None,  # number of snapshots used for CAVER calculation
            "caver_traj_offset": 1,  # difference in IDs of MD frames (from 0) and caver snapshots (often from 1)
            "snapshot_id_position": 1,  # location of IDs in snapshot filenames from CAVER split by snapshot_delimiter
            "snapshot_delimiter": ".",  # delimiter used for splitting snapshot filenames from CAVER to get IDs
            "process_bottleneck_residues": False,  # read file bottlenecks.csv

            # Clustering of tunnel clusters into superclusters
            "relevant_tunnel_cluster_min_size": 1, #filters too small cluster, applies to tunnel network processing, clusters with fewer RELEVANT tunnels are not layered at all
            "relevant_tunnel_min_radius": 0.75,  # filters on tunnel load, defines RELEVANT tunnels, irrelevant tunnels are not layered 
            "relevant_tunnel_min_length": 5.0,  # filters on tunnel load, defines RELEVANT tunnels, irrelevant tunnels are not layered 
            "relevant_tunnel_max_curvature": 2.0,  # filters on tunnel load, defines RELEVANT tunnels, irrelevant tunnels are not layered 

            "min_tunnel_radius4clustering": None,  # legacy name for relevant_tunnel_min_radius 
            "min_tunnel_length4clustering": None,  # legacy name for relevant_tunnel_min_length
            "max_tunnel_curvature4clustering": None,  # legacy name for relevant_tunnel_max_curvature

            #Global clustering settings
            "clustering_method": "agglomerative",  # 'agglomerative' (fastcluster linkage) or 'hdbscan' (density-based)
            "clustering_cutoff": 2.0,  # clustering of tunnel clusters to superclusters
            "calculate_exact_path_distances": True,  # request full calculation of distances even for very remote paths

            # Global, method-agnostic orphan consolidation: reassign every supercluster smaller than
            # supercluster_core_min_size to the nearest 'core' supercluster, stitching periphery the partition isolated
            # back into the major branches. Off by default.
            "consolidate_orphan_superclusters": False,  # enable the global orphan->core consolidation pass
            "supercluster_core_min_size": 5,  # min tunnel clusters for a supercluster to count as a core / assignment target
            "orphan_assignment_cutoff": None,  # max orphan-to-core distance allowed for absorption; None => uncapped

            #Agglomerative clustering settings:
            "clustering_linkage": "complete",  # linkage for agglomerative clustering (ignored when clustering_method='hdbscan')

            #HDBscan clustering settings
            "hdbscan_min_cluster_size": 2,  # smallest admissible supercluster when clustering_method='hdbscan'
            "hdbscan_min_samples": None,  # HDBSCAN conservativeness/noise sensitivity; None => equals hdbscan_min_cluster_size
            "hdbscan_cluster_selection_method": "eom",  # HDBSCAN flat-cluster extraction: 'eom' (fewer/larger) or 'leaf' (finer)
            # Decoupled clustering scales (None => inherit clustering_cutoff, preserving legacy behavior). They let a
            # single clustering pass be multi-scale: a generous partition/merge scale for context+assignment with a
            # tight selection scale for separating spatially overlapping branches.
            "hdbscan_clustering_partition_cutoff": None,  # sub-cutoff connected-components partition threshold (HDBSCAN only; agglomerative always partitions at its flat-cut for exactness). None => clustering_cutoff
            "hdbscan_cluster_selection_epsilon": None,  # HDBSCAN cluster_selection_epsilon (branch-separation fineness). None => clustering_cutoff
            "hdbscan_noise_merge_cutoff": None,  # per-component floor for absorbing HDBSCAN noise into the nearest in-component cluster. None => clustering_cutoff


            # SLURM execution backend (see SlurmShardStage framework in transport_tools.libs.slurm)
            "compute_backend": "local",  # global default backend: 'local' (multiprocessing) or 'slurm' (SLURM array jobs via submitit). Applied to every parallelized stage unless overridden by a stage-specific knob below.
            "stage02_backend": None,  # stage-2 (process_tunnel_networks) backend override; None => use compute_backend
            "stage03_backend": None,  # stage-3 (create_layered_description4tunnel_networks) backend override; None => use compute_backend
            "stage04_backend": None,  # stage-4 (compute_tunnel_clusters_distances) backend override; None => use compute_backend
            "stage07_backend": None,  # stage-7 (process_aquaduct_networks) backend override; None => use compute_backend
            "stage08_backend": None,  # stage-8 (create_layered_description4aquaduct_networks) backend override; None => use compute_backend
            "stage09_backend": None,  # stage-9 (assign_transport_events) backend override; None => use compute_backend

            # Shared SLURM submission parameters (apply to every stage that runs on SLURM)
            "slurm_partition": None,  # SLURM partition
            "slurm_account": None,  # SLURM account
            "slurm_timeout_min": 240,  # wall-time limit per array task, in minutes
            "slurm_mem_gb": 8.0,  # memory limit per array task, in GB
            "slurm_cpus_per_task": 4,  # CPU cores requested and used per array task
            "slurm_num_shards": None,  # number of array tasks; None => derived automatically from the job size
            "slurm_root_folder": None,  # shared parent dir for per-stage SLURM subfolders; None => <internal_folder>/_slurm. Each stage's shards land in <slurm_root_folder>/stage<NN>_<name>/ (e.g. stage04_distances/).
            "slurm_array_parallelism": None,  # optional cap on the number of array tasks running concurrently
            # The next three knobs configure DIFFERENT layers of an sbatch submission and
            # are NOT interchangeable. Picking the wrong layer is a frequent source of
            # confusion, so the intended split is:
            #
            #   slurm_additional_parameters  ->  #SBATCH header lines (allocation request).
            #                                    Tells the scheduler WHAT to allocate.
            #                                    Effective BEFORE the job starts running.
            #                                    Example: --qos, --constraint, --gres,
            #                                    --reservation, --exclude.
            #
            #   slurm_setup                  ->  shell commands inserted into the sbatch
            #                                    script BEFORE the srun line. Run ONCE per
            #                                    array task on the compute node, in the
            #                                    allocated shell, before the worker
            #                                    process exists. Use for environment
            #                                    bring-up (module load, conda/venv
            #                                    activation, export VAR=...).
            #
            #   slurm_srun_args              ->  extra command-line args appended to the
            #                                    srun invocation that actually launches
            #                                    the worker step. Influence HOW srun
            #                                    starts the worker (CPU/GPU binding, MPI
            #                                    layer, --exact, ...). Effective at job
            #                                    step launch time, after slurm_setup ran.
            "slurm_setup": None,  # comma-separated list of shell commands prepended to the sbatch script and executed once per array task before srun launches the worker (e.g. 'module load python/3.11, source ~/envs/tt/bin/activate'). MUST leave a working TransportTools env on PATH or the worker process will fail to import. None disables (no extra setup lines emitted).
            "slurm_srun_args": None,  # comma-separated extra args appended to the srun call that starts the worker job step (e.g. '--exact, --mpi=pmix'). The launcher ALWAYS guarantees '--cpu-bind=none' is present (prepended if missing) so a stage launched from inside a SLURM allocation does not fail with 'CPU binding outside of job step allocation'; any user-supplied '--cpu-bind=...' is honoured instead. None => default to just ['--cpu-bind=none'].
            "slurm_additional_parameters": None,  # comma-separated 'key=value' pairs forwarded verbatim as #SBATCH headers via submitit's slurm_additional_parameters mapping (e.g. 'qos=high, constraint=icelake, gres=gpu:1'). Use for cluster-specific allocation knobs (--qos, --constraint, --gres, --reservation, --exclude, ...) that are not exposed as dedicated slurm_* parameters here. Do NOT use this to override knobs that already have a dedicated parameter (slurm_partition, slurm_account, slurm_timeout_min => 'time', slurm_cpus_per_task, slurm_mem_gb, slurm_array_parallelism, ...) - those would silently shadow the dedicated knob via the additional #SBATCH header; _validate_parameter_values() rejects such collisions at config load but only when the override would actually take effect (i.e. the dedicated knob is also active). None disables.
            "slurm_poll_wait_seconds": 60, # period in seconds to wait for next iteration of job pooling
            "slurm_keep_shard_results": False,  # when False (default), the per-stage SLURM folder (shard pickles, manifests, side-effects tarballs, items.json, state.pkl, submitit per-job artifacts) is deleted as soon as the launcher's assembly succeeds, so the `_slurm/` tree does not grow without bound across many runs. Set to True to keep the cache after a successful run (useful for debugging shard outputs or for iterative re-runs that change only the assembly step). Resumability of an interrupted run is unaffected either way - cleanup only fires after the stage has completed successfully end-to-end.

            # Filters applied on superclusters before event assignment (-1 => inactive filter)
            "min_length": -1,  # filter on minimum tunnel length
            "max_length": -1,  # filter on maximum tunnel length
            "min_bottleneck_radius": -1,  # filter on minimum tunnel bottleneck radius
            "max_bottleneck_radius": -1,  # filter on maximum tunnel bottleneck radius
            "min_curvature": -1,  # filter on minimum tunnel curvature
            "max_curvature": -1,  # filter on maximum tunnel curvature
            "min_sims_num": -1,  # filter on presence in minimum number of MD simulations
            "min_snapshots_num": -1,  # filter on presence in minimum number of snapshots
            "min_avg_snapshots_num": -1,  # filter on presence in minimum number of snapshots on average

            # Processing of transport events from AQUA-DUCT results and their assignment to superclusters
            "aquaduct_traced_residues_filter": None,  # list of uppercase resnames to analyze; None = process all
            "aquaduct_allow_empty_folders": False,  # skip AQUA-DUCT folders missing tar/summary with a warning instead of erroring
            "event_min_distance": 6.0,  # closest distance of event from starting point to be processed
            "event_assignment_cutoff": 0.85,
            # event_assignment_cutoff represents minimal buriedness of transport event in a supercluster to be
            # assigned to that supercluster, [0.0-1.0]
            "ambiguous_event_assignment_resolution": "penetration_depth",
            # ambiguous_event_assignment_resolution decides how we assign events to multiple potential superclusters,
            # possible values:
            # 'penetration_depth' - how deep the event penetrates the supercluster
            # 'exact_matching' - matching of actual ligand transport event (all atoms from MD simulation) and real
            # tunnels existing in a given simulation during the event occurrence
            # 'assign2all' - assign event to all superclusters in which it is buried

            # Additional filters applied on superclusters after event assignment (-1 => inactive filter)
            "min_total_events": -1,  # filter on having minimum events assigned
            "min_entry_events": -1,  # filter on having minimum entry events assigned
            "min_release_events": -1  # filter on having minimum release events assigned
        }

        self.output_settings = {
            "output_path": "results",
            # Optional data generation
            "save_super_cluster_profiles_csvs": False,
            "save_distance_matrix_csv": False,
            # Optional visualization
            "visualize_super_cluster_volumes": False,
            "visualize_transformed_tunnels": False,
            "visualize_transformed_transport_events": False
        }

        self.advanced_settings = {
            # Calculations
            "random_seed": 4,
            "directional_cutoff": np.pi / 2,  # detect serious misalignments during clustering and assignment of events
            "aqauduct_ligand_effective_radius": 1.0,  # for overlap calculations
            "perform_exact_matching_analysis": False,
            "folder_pattern4exact_matching_analysis": "*",  # exact_matching_analysis is performed only for folders
            # matching this pattern
            "perform_comparative_analysis": False,
            "visualize_comparative_super_cluster_volumes": False,
            "comparative_groups_definition": None,
            # Format of group definition: group_name1: [folder1, folder2, ...]; group_name2: [folder1, folder2, ...]...

            # Finer control of outputs & logging
            "overwrite": False,
            "visualize_exact_matching_outcomes": False,
            "visualize_layered_clusters": False,  # mostly for debugging only
            "visualize_layered_events": False,  # mostly for debugging only
            "legacy_pymol_support": True,
            "trajectory_engine": "mdtraj",  # mdtraj or pytraj if installed additionally
            "logfilename": "transport_tools.log",
            "log_level": "info",
            "std_level": "info",
            "msms": None,  # path to binary of  https://ccsb.scripps.edu/msms/ program for faster volume visulaization
            "max_events_per_cluster4visualization": 1000
        }
        self.advanced_settings_defaults = self.advanced_settings.copy()

        # those cannot be loaded from config_file
        self.internal_settings = {
            # parsing related
            "aquaduct_results_pdb_filename": "molecule0_1.pdb",

            # calculations
            "clustering_max_num_rep_frag": 3,
            # clustering_max_num_rep_frag represents the number of fragments of layered path included in distance
            # calculation, 0 - all, 1 - longest, 2 - longest and shortest, >2 - add random intermediate fragments
            # when tested on five simulations of DhaA, results were essentially identical for more than 3 fragments
            "use_cluster_spread": True,  # distance calculations for event-SC assignment
            "sp_radius": 0.5,  # radius of starting point
            "tunnel_properties_quantile": 0.90,  # to use for calculation of representative radius of layered clusters
  
            # outputs
            "pickle_protocol": 4,
            "verbose_logging": False,
            "max_layered_points4visualization": 500  # to show points forming layered clusters,
        }

        self._set_output_paths("results")

        self.possibly_none_params = [
            "caver_results_path",
            "caver_results_relative_subfolder_path",
            "aquaduct_results_path",
            "aquaduct_results_relative_tarfile",
            "aquaduct_results_relative_summaryfile",
            "trajectory_path",
            "trajectory_relative_file",
            "topology_relative_file",
            "stop_after_stage",
            "num_cpus",
            "worker_task_timeout_s",
            "pdb_reference_structure",
            "snapshots_per_simulation",
            "min_tunnel_radius4clustering",
            "min_tunnel_length4clustering",
            "max_tunnel_curvature4clustering",
            "hdbscan_min_samples",
            "hdbscan_clustering_partition_cutoff",
            "hdbscan_cluster_selection_epsilon",
            "hdbscan_noise_merge_cutoff",
            "orphan_assignment_cutoff",
            "comparative_groups_definition",
            "aquaduct_traced_residues_filter",
            "msms",
            "stage02_backend",
            "stage03_backend",
            "stage04_backend",
            "stage07_backend",
            "stage08_backend",
            "stage09_backend",
            "slurm_partition",
            "slurm_account",
            "slurm_num_shards",
            "slurm_root_folder",
            "slurm_array_parallelism",
            "slurm_setup",
            "slurm_srun_args",
            "slurm_additional_parameters",
        ]

        self.boolean_params = [
            "process_bottleneck_residues",
            "consolidate_orphan_superclusters",
            "calculate_exact_path_distances",
            "use_cluster_spread",
            "perform_exact_matching_analysis",
            "verbose_logging",
            "overwrite",
            "perform_comparative_analysis",
            "save_super_cluster_profiles_csvs",
            "save_distance_matrix_csv",
            "visualize_super_cluster_volumes",
            "visualize_comparative_super_cluster_volumes",
            "visualize_transformed_tunnels",
            "visualize_transformed_transport_events",
            "visualize_layered_clusters",
            "visualize_layered_events",
            "visualize_exact_matching_outcomes",
            "legacy_pymol_support",
            "aquaduct_allow_empty_folders",
            "slurm_keep_shard_results",
        ]

        self.integer_params = [
            "start_from_stage",
            "stop_after_stage",
            "num_cpus",
            "worker_task_timeout_s",
            "snapshots_per_simulation",
            "caver_traj_offset",
            "snapshot_id_position",
            "relevant_tunnel_cluster_min_size",
            "min_sims_num",
            "min_snapshots_num",
            "min_total_events",
            "min_entry_events",
            "min_release_events",
            "random_seed",
            "clustering_max_num_rep_frag",
            "hdbscan_min_cluster_size",
            "hdbscan_min_samples",
            "supercluster_core_min_size",
            "max_layered_points4visualization",
            "max_events_per_cluster4visualization",
            "pickle_protocol",
            "slurm_timeout_min",
            "slurm_cpus_per_task",
            "slurm_num_shards",
            "slurm_array_parallelism",
            "slurm_poll_wait_seconds"
        ]

        self.float_params = [
            "layer_thickness",
            "relevant_tunnel_min_radius",
            "relevant_tunnel_min_length",
            "relevant_tunnel_max_curvature",
            "min_tunnel_radius4clustering",
            "min_tunnel_length4clustering",
            "max_tunnel_curvature4clustering",
            "clustering_cutoff",
            "hdbscan_clustering_partition_cutoff",
            "hdbscan_cluster_selection_epsilon",
            "hdbscan_noise_merge_cutoff",
            "orphan_assignment_cutoff",
            "min_length",
            "max_length",
            "min_bottleneck_radius",
            "max_bottleneck_radius",
            "min_curvature",
            "max_curvature",
            "min_avg_snapshots_num",
            "event_min_distance",
            "event_assignment_cutoff",
            "sp_radius",
            "tunnel_properties_quantile",
            "directional_cutoff",
            "aqauduct_ligand_effective_radius",
            "slurm_mem_gb"
        ]

        # parameters whose value is a list of strings, one per (non-empty) line in the INI file
        # parameters whose value is a comma-separated list of strings in the INI file,
        # auto-parsed into a list of stripped non-empty tokens at load time (e.g.
        # 'cmd1, cmd2' -> ['cmd1', 'cmd2']). Consistent with the other comma-separated
        # knobs in this file (aquaduct_results_path, aquaduct_traced_residues_filter, ...).
        self.list_params = [
            "slurm_setup",
            "slurm_srun_args",
        ]
        # parameters whose value is a comma-separated 'key=value' string in the INI file,
        # auto-parsed into a dict at load time (e.g. 'qos=high, constraint=icelake' ->
        # {'qos': 'high', 'constraint': 'icelake'}). Empty values are allowed (e.g. 'gres='
        # yields {'gres': ''}); a non-empty token without '=' raises ValueError so a typo
        # surfaces at config load instead of silently dropping the entry.
        self.dict_params = [
            "slurm_additional_parameters",
        ]

        # registry of per-stage SLURM backend override knobs. Each row is (knob_name, label) -
        # the label is only used in log messages and error reporting; the knob_name must match
        # a parameter declared above with default None. Add one row per SLURM-capable stage as
        # it lands; _validate_parameter_values() and resolve_stage_backend() walk this list.
        self._stage_backend_keys: List[Tuple[str, str]] = [
            ("stage02_backend", "stage 2 / tunnel networks"),
            ("stage03_backend", "stage 3 / tunnel layering"),
            ("stage04_backend", "stage 4 / distance shards"),
            ("stage07_backend", "stage 7 / aquaduct networks"),
            ("stage08_backend", "stage 8 / aquaduct layering"),
            ("stage09_backend", "stage 9 / event assignment"),
        ]

    def _load_configuration(self, path2configfile: str):
        """
        Load job configuration from INI formatted file
        :param path2configfile: INI file with job configuration
        """

        config = ConfigParser(allow_no_value=True)
        if not config.read(path2configfile):
            raise IOError("\nFile with job configuration \"{}\" could not be read".format(path2configfile))
        config_sections = {
            "input_paths": self.input_paths,
            "calculations_settings": self.calculations_settings,
            "output_settings": self.output_settings,
            "advanced_settings": self.advanced_settings,
            "internal_settings": self.internal_settings
        }

        forbid_autoactivation = list()
        for section in config.sections():
            config_dictionary = config_sections[str(section).lower()]
            for param_name in config[section].keys():
                if param_name not in config_dictionary.keys():
                    raise ValueError("\nUnknown parameter '{}' in section '{}' "
                                     "of configuration file {}.\n".format(param_name, section, path2configfile))

                # prepare value
                if str(config[section].get(param_name)) == "" or \
                        param_name in self.possibly_none_params and str(config[section].get(param_name)) == "None":
                    param_value = None
                else:
                    if param_name in self.boolean_params:
                        param_value = config[section].getboolean(param_name)
                    elif param_name in self.integer_params:
                        param_value = config[section].getint(param_name)
                    elif param_name in self.float_params:
                        param_value = config[section].getfloat(param_name)
                    elif param_name in self.list_params:
                        # comma-separated list of strings; matches the format used by the
                        # other list-typed knobs in this file (aquaduct_results_path,
                        # aquaduct_traced_residues_filter). Whitespace around tokens is
                        # stripped and empty tokens are dropped.
                        param_value = [token.strip() for token
                                       in str(config[section].get(param_name)).split(",")
                                       if token.strip()]
                    elif param_name in self.dict_params:
                        # comma-separated 'key=value' pairs -> dict. Each non-empty token
                        # must contain '=' (an empty value like 'gres=' is OK); a typo
                        # without '=' raises so the user sees the error at config load
                        # rather than three stages in.
                        param_value = {}
                        for token in str(config[section].get(param_name)).split(","):
                            token = token.strip()
                            if not token:
                                continue
                            if "=" not in token:
                                raise ValueError(
                                    "Parameter '{}' entries must be 'key=value' (got '{}'). "
                                    "Use a comma-separated list like 'qos=high, "
                                    "constraint=icelake'.".format(param_name, token))
                            key, _, value = token.partition("=")
                            param_value[key.strip()] = value.strip()
                    else:
                        param_value = str(config[section].get(param_name))

                if param_name == "visualize_super_cluster_volumes" and param_value is False:
                    forbid_autoactivation.append(param_name)
                if param_name == "trajectory_engine" and param_value == "mdtraj":
                    forbid_autoactivation.append(param_name)

                if param_name == "output_path":  # update also dependent paths
                    if param_value is None or param_value == "":
                        raise ValueError("\nParameter 'output_path' must be defined and cannot be empty")
                    if not isinstance(param_value, str):
                        raise TypeError("\nParameter 'output_path' must be a string")
                    self._set_output_paths(param_value)

                config_dictionary[param_name] = param_value

        self._resolve_active_stages()
        self._set_input_paths()
        self._autoactivate_functionalities(forbid_autoactivation)

    def _resolve_active_stages(self):
        """
        Finalise the inclusive (lo, hi) interval of pipeline stages this process will run, used to
        scope stage-specific setups and input-existence checks (see _runs_stage). An explicit
        'active_stages' passed to __init__ wins (e.g. a SLURM shard pinning its single stage);
        otherwise the interval is derived from the just-parsed 'start_from_stage'/'stop_after_stage'
        so the launcher validates everything the run will actually touch.
        """

        if self._active_stages_override is not None:
            self._active_stages = self._active_stages_override
            return
        lo = self.calculations_settings["start_from_stage"]
        hi = self.calculations_settings["stop_after_stage"]
        if hi is None:  # 'stop_after_stage' default (run to the end); 10 is the last pipeline stage
            hi = 10
        self._active_stages = (lo, hi)

    def _runs_stage(self, *stages: int) -> bool:
        """
        :param stages: one or more pipeline stage numbers
        :return: True if any of the given stages falls inside the active [lo, hi] interval, i.e. this
                 process will actually run it. Stage-scoped work whose owning stage(s) are all outside
                 the interval can be skipped.
        """

        lo, hi = self._active_stages
        return any(lo <= stage <= hi for stage in stages)

    def _autoactivate_functionalities(self, overridden_parameters: List[str]):
        """
        Activate optional functionalities if required modules are available, still can be overridden in a config file.
        :param overridden_parameters: list of parameter names which have been overridden
        """
        if not self.output_settings["visualize_super_cluster_volumes"] and \
                "visualize_super_cluster_volumes" not in overridden_parameters \
                and self.advanced_settings["msms"] is None:
            try:
                import mcubes  # type: ignore[import-untyped]
            except ModuleNotFoundError:
                pass
            else:
                logger.warning("'mcubes' module detected, provisionally enabling 'visualize_super_cluster_volumes'. "
                               "Should you like to override this action, please set 'visualize_super_cluster_volumes = "
                               "False' in 'OUTPUT_SETTINGS' section of your configuration file.")
                self.output_settings["visualize_super_cluster_volumes"] = True

        if self.advanced_settings["trajectory_engine"] != "pytraj" and "trajectory_engine" not in overridden_parameters:
            try:
                import pytraj  # type: ignore[import-untyped]
            except ModuleNotFoundError:
                pass
            else:
                logger.warning("'pytraj' module detected, provisionally enabling 'pytraj' as 'trajectory_engine'. "
                               "Should you like to override this action, please set 'trajectory_engine = mdtraj' in "
                               "'ADVANCED_SETTINGS' section of your configuration file.")
                self.advanced_settings["trajectory_engine"] = "pytraj"

    def _set_input_paths(self):
        """
        Complete and set input paths based on provided parameters
        """

        caver_subfolder_path = self.input_paths["caver_results_relative_subfolder_path"]
        self.input_paths["caver_relative_profile_file"] = os.path.join(caver_subfolder_path, "analysis",
                                                                       "tunnel_profiles.csv")
        self.input_paths["caver_relative_bottleneck_file"] = os.path.join(caver_subfolder_path, "analysis",
                                                                          "bottlenecks.csv")

        self.input_paths["caver_relative_origin_file"] = os.path.join(caver_subfolder_path, "data", "v_origins.pdb")

        if self.input_paths["aquaduct_results_path"] is not None:
            # to allow multiple sources with separate AquaDuct results as input (e.g. water and ligand if not process simultaneously)
            self.input_paths["aquaduct_results_path"] = [
                p.strip() for p in self.input_paths["aquaduct_results_path"].split(",") if p.strip()
            ]

        # Autodetecting the CAVER PDB filename globs every CAVER data/ folder. caver_relative_pdb_file
        # feeds the CAVER existence/parse path (stages 1-2) and the reference-structure detection, which
        # on the launcher runs only on a fresh stage-1 run but on a shard runs for the sharded alignment
        # stages (2 & 7, which have no checkpoint to restore the reference from). Resolve it whenever any
        # of those will run; on a later launcher resume the live network objects already hold their pdb
        # paths and the reference comes from the checkpoint, so the glob is skipped (see _runs_stage /
        # update_configuration).
        need_caver_pdb = self._runs_stage(1, 2) or (self.context.is_shard and self._runs_stage(7))
        if self.input_paths["caver_relative_pdb_file"] is None and need_caver_pdb:
            if not self.input_paths["caver_results_path"]:
                raise RuntimeError("\nParameter 'caver_results_path' must be defined and point to the folder with the"
                                   " data!")

            caver_pdb_filename = self._get_caver_filenames()
            if not caver_pdb_filename:
                msg = "\nCould not detect PDB file of protein in CAVER data folder! Please make sure your {}/data " \
                      "folders contain the PDB file with structure having the same filename and that it can be matched " \
                      "with the following pattern {}!".format(self.input_paths["caver_results_relative_subfolder_path"], 
                      self.input_paths["caver_pdb_file_pattern"])
                raise RuntimeError(msg)

            self.input_paths["caver_relative_pdb_file"] = os.path.join(caver_subfolder_path, "data", caver_pdb_filename)

    def _set_output_paths(self, output_path: str):
        """
        Set paths to output subfolders based on root output path
        :param output_path: root output path
        """

        # definitions of folder names
        caver_foldername = "caver"
        aquaduct_foldername = "aquaduct"
        network_data_foldername = "network_data"
        layered_data_foldername = "layered_data"
        internal_foldername = "_internal"
        data_foldername = "data"
        vis_foldername = "visualization"
        internal_folder = os.path.join(output_path, internal_foldername)
        data_folder = os.path.join(output_path, data_foldername)
        vis_folder = os.path.join(output_path, vis_foldername)

        self.output_paths = {
            "caver_foldername": caver_foldername,
            "aquaduct_foldername": aquaduct_foldername,
            "internal_folder": internal_folder,
            "data_folder": data_folder,
            "visualization_folder": vis_folder,
            "statistics_folder": os.path.join(output_path, "statistics"),
            "checkpoints_folder": os.path.join(internal_folder, "checkpoints"),
            "logfile_path": os.path.join(output_path, self.advanced_settings["logfilename"]),

            # folder structure for internal data
            "transformation_folder": os.path.join(internal_folder, "transformations"),
            "clustering_folder": os.path.join(internal_folder, "clustering"),

            "orig_caver_network_data_path": os.path.join(internal_folder, network_data_foldername, caver_foldername),
            "orig_aquaduct_network_data_path": os.path.join(internal_folder, network_data_foldername,
                                                            aquaduct_foldername),

            "layered_caver_network_data_path": os.path.join(internal_folder, layered_data_foldername, caver_foldername),
            "layered_aquaduct_network_data_path": os.path.join(internal_folder, layered_data_foldername,
                                                               aquaduct_foldername),
            "super_cluster_profiles_folder": os.path.join(internal_folder, "super_cluster_profiles"),
            "super_cluster_path_set_folder": os.path.join(internal_folder, "super_cluster_pathsets"),

            # folder structure for data
            "distance_matrix_csv_file": os.path.join(data_folder, "clustering", "tunnel_clusters_distance_matrix.csv"),
            "super_cluster_csv_folder": os.path.join(data_folder, "super_clusters", "CSV_profiles"),
            "super_cluster_details_folder": os.path.join(data_folder, "super_clusters", "details"),
            "super_cluster_bottleneck_folder": os.path.join(data_folder, "super_clusters", "bottlenecks"),
            "exact_matching_details_folder": os.path.join(data_folder, "exact_matching_analysis"),

            # folder structure for visualization
            "orig_caver_vis_path": os.path.join(vis_folder, "sources", network_data_foldername, caver_foldername),
            "orig_aquaduct_vis_path": os.path.join(vis_folder, "sources", network_data_foldername, aquaduct_foldername),
            "layered_caver_vis_path": os.path.join(vis_folder, "sources", layered_data_foldername, caver_foldername),
            "layered_aquaduct_vis_path": os.path.join(vis_folder, "sources", layered_data_foldername,
                                                      aquaduct_foldername),
            "super_cluster_vis_path": os.path.join(vis_folder, "sources", "super_cluster_CGOs"),
            "exact_matching_vis_path": os.path.join(vis_folder, "exact_matching_analysis")
        }

    def _autocomplete_parameters(self):
        """
        Completing parameters that can be derived from existing ones, or otherwise
        """

        for leg_par, new_par in self.legacy_params.items():
            if self.parameters[leg_par] is not None:
                logger.warning("\nLegacy parameter '{}' in use, this will be deperciated soon, use parameter '{}' instead!".format(leg_par, new_par))
                self.parameters[new_par] = self.parameters[leg_par]

        if self.parameters["stop_after_stage"] is None:
            self.parameters["stop_after_stage"] = sys.maxsize

        if self.parameters["num_cpus"] is None:
            self._detect_set_num_cpu()

        # Reference-structure autodetection globs the CAVER folders. The resolved reference is consumed
        # by stages 1, 2 and 7 (transformations, tunnel-network and aquaduct-network alignment) and is
        # baked into every checkpoint once resolved on the initial run. On the LAUNCHER we therefore
        # re-derive it only on a fresh run that includes stage 1; a resume into a later stage inherits
        # it from the checkpoint (preserved in update_configuration). A SHARD has no checkpoint, so it
        # must self-derive the reference whenever its own stage consumes it (the sharded alignment
        # stages 2 and 7).
        if self.context.is_shard:
            need_pdb_reference = self._runs_stage(2, 7)
        else:
            need_pdb_reference = self._runs_stage(1)
        if self.parameters["pdb_reference_structure"] is None and need_pdb_reference:
            self._detect_set_pdb_reference_structure()

        # Snapshot autodetection opens every input trajectory - by far the most expensive setup. The
        # value is consumed only by launcher-side stages (supercluster build/filtering and their
        # visualisation); no sharded compute method reads it. On the LAUNCHER it is resolved on the
        # initial run (stage 1 present) and inherited from the checkpoint on every later resume, so a
        # resume never re-reads the trajectories. A SHARD never needs it. When the initial run does
        # require it and it is neither given nor detectable, the original hard error still fires.
        need_snapshots = (not self.context.is_shard) and self._runs_stage(1)
        if self.parameters["snapshots_per_simulation"] is None and need_snapshots:
            if self.parameters["trajectory_path"] is not None:
                self._detect_set_num_snapshots()
            else:
                raise RuntimeError("\nParameter 'snapshots_per_simulation' must be defined or access to input "
                                   "trajectories must be provided to allow auto-detection.")

        if self.parameters["perform_comparative_analysis"] \
                and self.parameters["comparative_groups_definition"] is not None:
            self._parse_group_definition()

        if isinstance(self.parameters["aquaduct_traced_residues_filter"], str):
            # parse the resides into list
            self.parameters["aquaduct_traced_residues_filter"] = [
                r.strip().upper() for r in self.parameters["aquaduct_traced_residues_filter"].split(",")
                if r.strip()
            ]

        if self.parameters["legacy_pymol_support"]:
            self.parameters["pickle_protocol"] = 2
            self.internal_settings["pickle_protocol"] = 2

        if self.parameters["slurm_root_folder"] is None:
            self.parameters["slurm_root_folder"] = os.path.join(self.parameters["internal_folder"], "_slurm")

    def _test_parameter_sanity(self, param_name: str, min_val: int | float, max_val: int | float):
        """
        Checking if value assigned to parameter is within allowed range (if not raising error),
        and also prints warning for suboptimal values (however, job continues in this case)
        :param param_name: parameter to test
        :param min_val: min value from the allowed range
        :param max_val: max value from the allowed range
        """

        if not min_val <= self.parameters[param_name] <= max_val:
            raise ValueError("\nThe value of parameter '{}' must be within <{},{}> but currently is: '{}'. "
                             "Please, consider consulting the user guide.".format(param_name, min_val, max_val,
                                                                                  self.parameters[param_name]))

        suboptimal_params_borders = {
            "layer_thickness": (1, 5),
            "relevant_tunnel_min_radius": (0.5, 2),
            "relevant_tunnel_min_length": (3, 10),
            "relevant_tunnel_max_curvature": (2, 5),
            "clustering_cutoff": (1, 3),
            "event_min_distance": (5, 10),
            "event_assignment_cutoff": (0.5, 1),
            "max_events_per_cluster4visualization": (100, 5000)
        }

        if param_name == "clustering_max_num_rep_frag" and self.parameters[param_name] == 1:
            logger.warning("\nParameter '{}' seems to be too small to perform correct calculation. "
                           "PLEASE, consider consulting the user guide.".format(param_name))

        if param_name in suboptimal_params_borders.keys():
            if self.parameters[param_name] > suboptimal_params_borders[param_name][1]:
                logger.warning("\nParameter '{}' seems to be too large to perform correct calculation. "
                               "PLEASE, consider consulting the user guide.".format(param_name))
            if self.parameters[param_name] < suboptimal_params_borders[param_name][0]:
                logger.warning("\nParameter '{}' seems to be too small to perform correct calculation. "
                               "PLEASE, consider consulting the user guide.".format(param_name))

    def _validate_parameter_values(self):
        """
        Tests proper values of input parameters
        """

        # The whole num_cpus block is launcher-only: on a SLURM shard num_cpus is inert (the inner
        # pool is sized from slurm_cpus_per_task) and this compute node's core count is unrelated to
        # the launcher's, so none of the three checks are meaningful there (see ShardContext).
        if not self.context.is_shard and self.parameters["num_cpus"] is not None:
            if self.parameters["num_cpus"] > os.cpu_count():
                raise ValueError("\nThe number of CPU({:d}) specified for parallel processing is larger than the number"
                                 " of CPU({:d}) reported as available on this computer.\n Decrease the value of "
                                 "parameter 'num_cpus'.".format(self.parameters["num_cpus"], os.cpu_count()))
            if self.parameters["num_cpus"] < 1:
                raise ValueError("\nThe number of CPU({:d}) specified for parallel processing is smaller than 1"
                                 "\n Increase the value of parameter 'num_cpus'.")

            if self.parameters["num_cpus"] > 10:
                logger.warning("\nThe number of CPU{:d}) specified for parallel processing is quite large, which might "
                               "actually hamper the performance due to IO restrictions.\n Consider decreasing the value"
                               " of parameter 'num_cpus'".format(self.parameters["num_cpus"]))

        if self.parameters["worker_task_timeout_s"] is not None:
            self._test_parameter_sanity("worker_task_timeout_s", 1, sys.maxsize)
            if self.parameters["worker_task_timeout_s"] < 600:
                logger.warning("\nParameter 'worker_task_timeout_s' is set to a very small value ({:d}s); any worker "
                               "task that legitimately takes longer will abort the run. Set to None to disable the "
                               "timeout.".format(self.parameters["worker_task_timeout_s"]))

        if self.parameters["slurm_poll_wait_seconds"] > 600:
                logger.warning("\nParameter 'slurm_poll_wait_seconds' is set to a rather large value ({:d}s); " \
                                "possibly compromising effective job processing".format(self.parameters["slurm_poll_wait_seconds"]))

        valid_clustering_methods = ("agglomerative", "hdbscan")
        if self.parameters["clustering_method"] not in valid_clustering_methods:
            raise ValueError("\nUnsupported value '{}' for 'clustering_method' parameter.\n Valid options are "
                             "'{}'".format(self.parameters["clustering_method"], valid_clustering_methods))

        import fastcluster
        valid_linkage = fastcluster.mthidx.copy()

        if self.parameters["clustering_linkage"] not in valid_linkage.keys():
            raise ValueError("\nUnsupported linkage type '{}' specified in 'clustering_linkage' parameter.\n Valid "
                             "options are '{}'".format(self.parameters["clustering_linkage"], valid_linkage.keys()))

        if self.parameters["clustering_method"] == "agglomerative" and self.parameters["clustering_linkage"] == "single":
            logger.warning("\nUse of 'single' for 'cluster_linkage' parameter is discouraged as it often leads to "
                           "rather poorly defined superclusters. PLEASE, consider using 'average' or 'complete'.")

        if self.parameters["clustering_method"] == "hdbscan":
            valid_selection_methods = ("eom", "leaf")
            if self.parameters["hdbscan_cluster_selection_method"] not in valid_selection_methods:
                raise ValueError("\nUnsupported value '{}' for 'hdbscan_cluster_selection_method' parameter.\n Valid "
                                 "options are '{}'".format(self.parameters["hdbscan_cluster_selection_method"],
                                                           valid_selection_methods))
            self._test_parameter_sanity("hdbscan_min_cluster_size", 2, sys.maxsize)
            if self.parameters["hdbscan_min_samples"] is not None:
                self._test_parameter_sanity("hdbscan_min_samples", 1, sys.maxsize)

        # decoupled clustering scales: each None inherits clustering_cutoff (legacy behavior); when set they must be
        # non-negative distances. They only take effect for clustering_method='hdbscan'.
        for cutoff_param in ("hdbscan_clustering_partition_cutoff", "hdbscan_cluster_selection_epsilon",
                             "hdbscan_noise_merge_cutoff"):
            if self.parameters[cutoff_param] is not None:
                self._test_parameter_sanity(cutoff_param, 0.0, sys.maxsize)

        # global orphan->core consolidation (method-agnostic post-pass)
        if self.parameters["consolidate_orphan_superclusters"]:
            self._test_parameter_sanity("supercluster_core_min_size", 2, sys.maxsize)
            if self.parameters["orphan_assignment_cutoff"] is not None:
                self._test_parameter_sanity("orphan_assignment_cutoff", 0.0, sys.maxsize)

        # validate compute_backend + per-stage overrides; resolve each stage's effective backend.
        # _stage_backend_keys lists every (stage_knob_name, human_label) recognised by the
        # SlurmShardStage framework. Adding a new SLURM-capable stage means adding both its
        # config knob above (default None) and a row here, with no other validation changes.
        valid_backends = ("local", "slurm")
        if str(self.parameters["compute_backend"]).lower() not in valid_backends:
            raise ValueError("\nUnsupported value '{}' for 'compute_backend' parameter.\n Valid options are "
                             "'local' and 'slurm'.".format(self.parameters["compute_backend"]))

        any_slurm = str(self.parameters["compute_backend"]).lower() == "slurm"
        for stage_knob, _label in self._stage_backend_keys:
            value = self.parameters.get(stage_knob)
            if value is None:
                continue
            if str(value).lower() not in valid_backends:
                raise ValueError("\nUnsupported value '{}' for '{}' parameter.\n Valid options are "
                                 "'local' and 'slurm'.".format(value, stage_knob))
            if str(value).lower() == "slurm":
                any_slurm = True

        if any_slurm:
            # any stage running on SLURM needs the optional submitit package; fail at config
            # load time (not three stages in) so the user can install it before launching
            from transport_tools.libs.slurm import submitit_available
            if not submitit_available():
                raise RuntimeError("\nA SLURM backend is enabled ('compute_backend = slurm' or a "
                                   "stage-specific '*_backend = slurm'), but the optional "
                                   "'submitit' package is not installed.\nInstall it with "
                                   "'pip install submitit' or 'conda install -c conda-forge "
                                   "submitit', or set the backend to 'local'.")
            self._test_parameter_sanity("slurm_timeout_min", 1, sys.maxsize)
            self._test_parameter_sanity("slurm_cpus_per_task", 1, sys.maxsize)
            if self.parameters["slurm_mem_gb"] <= 0:
                raise ValueError("\nParameter 'slurm_mem_gb' must be a positive number.")
            if self.parameters["slurm_num_shards"] is not None:
                self._test_parameter_sanity("slurm_num_shards", 1, sys.maxsize)
            if self.parameters["slurm_array_parallelism"] is not None:
                self._test_parameter_sanity("slurm_array_parallelism", 1, sys.maxsize)
            # Reject only ACTUAL overrides: an entry in slurm_additional_parameters is
            # flagged only when the same knob is also being passed to submitit through a
            # dedicated parameter (so both #SBATCH headers would land in the sbatch
            # script and the additional_parameters entry would silently win - confusing
            # the user into thinking the dedicated knob is being respected).
            # Hypothetical shadowing (e.g. 'partition=foo' when slurm_partition is None)
            # is allowed - no override actually happens in that case.
            extra = self.parameters.get("slurm_additional_parameters") or {}
            if extra:
                # knobs the launcher ALWAYS passes to submitit in executor_params() -
                # any collision here is always a real override
                always_active = {
                    "time": "slurm_timeout_min",
                    "timeout_min": "slurm_timeout_min",
                    "cpus_per_task": "slurm_cpus_per_task",
                    "cpus-per-task": "slurm_cpus_per_task",
                    "mem": "slurm_mem_gb",
                    "mem_gb": "slurm_mem_gb",
                    "nodes": "(fixed to 1 by the launcher)",
                    "tasks_per_node": "(fixed to 1 by the launcher)",
                    "ntasks_per_node": "(fixed to 1 by the launcher)",
                    "ntasks-per-node": "(fixed to 1 by the launcher)",
                    "name": "(set automatically to the stage name)",
                    "job_name": "(set automatically to the stage name)",
                    "job-name": "(set automatically to the stage name)",
                    "array": "(set automatically from slurm_num_shards)",
                    # srun_args is always passed (defaults to ['--cpu-bind=none'])
                    "srun_args": "slurm_srun_args",
                }
                # knobs only forwarded when the matching dedicated parameter is set -
                # collide only when that dedicated parameter is non-default
                conditional = {
                    "partition": ("slurm_partition",
                                  self.parameters.get("slurm_partition") is not None),
                    "account": ("slurm_account",
                                self.parameters.get("slurm_account") is not None),
                    "array_parallelism": ("slurm_array_parallelism",
                                          self.parameters.get("slurm_array_parallelism") is not None),
                    "setup": ("slurm_setup",
                              bool(self.parameters.get("slurm_setup"))),
                }
                collisions = []
                for key in extra:
                    norm = str(key).strip().lower()
                    if norm in always_active:
                        collisions.append((key, always_active[norm]))
                    elif norm in conditional and conditional[norm][1]:
                        collisions.append((key, conditional[norm][0]))
                if collisions:
                    lines = ", ".join("'{}' (use '{}' instead)".format(k, target)
                                      for k, target in collisions)
                    raise ValueError("\nParameter 'slurm_additional_parameters' contains "
                                     "entries that would override values already set "
                                     "via dedicated configuration knobs: {}.\nRemove "
                                     "them from slurm_additional_parameters and use the "
                                     "dedicated knob instead.".format(lines))

        ambiguous_assignment_resolution_methods = ["exact_matching", "penetration_depth", "assign2all"]
        if self.parameters["ambiguous_event_assignment_resolution"] not in ambiguous_assignment_resolution_methods:
            raise ValueError("\nUnsupported method for resolution of ambiguous assignments '{}' specified in "
                             "'ambiguous_event_assignment_resolution' parameter.\n Valid options are "
                             "'{}'".format(self.parameters["ambiguous_event_assignment_resolution"],
                                           ambiguous_assignment_resolution_methods))

        self._test_parameter_sanity("start_from_stage", 0, 10)
        self._test_parameter_sanity("stop_after_stage", 1, sys.maxsize)
        if self.parameters["stop_after_stage"] < self.parameters["start_from_stage"]:
            raise ValueError("\nParameter 'stop_after_stage' must be >= 'start_from_stage' to perform any calculation.")
        self._test_parameter_sanity("layer_thickness", 0.9, sys.maxsize)
        self._test_parameter_sanity("snapshots_per_simulation", 1, sys.maxsize)
        self._test_parameter_sanity("caver_traj_offset", 0, 1)
        self._test_parameter_sanity("snapshot_id_position", 0, sys.maxsize)
        self._test_parameter_sanity("relevant_tunnel_cluster_min_size", 1, sys.maxsize)
        self._test_parameter_sanity("relevant_tunnel_min_radius", 0, sys.maxsize)
        self._test_parameter_sanity("relevant_tunnel_min_length", 0, sys.maxsize)
        self._test_parameter_sanity("relevant_tunnel_max_curvature", 1, sys.maxsize)
        self._test_parameter_sanity("clustering_cutoff", 0, sys.maxsize)
        self._test_parameter_sanity("event_min_distance", 0, sys.maxsize)
        self._test_parameter_sanity("event_assignment_cutoff", 0, 1)
        self._test_parameter_sanity("clustering_max_num_rep_frag", 0, sys.maxsize)
        self._test_parameter_sanity("max_events_per_cluster4visualization", 0, sys.maxsize)
        self._test_parameter_sanity("slurm_poll_wait_seconds", 1, sys.maxsize)


        # The aquaduct_traced_residues_filter shape check needs no filesystem access - keep it always.
        if self.parameters["aquaduct_traced_residues_filter"] is not None:
            if not isinstance(self.parameters["aquaduct_traced_residues_filter"], list) \
                    or not self.parameters["aquaduct_traced_residues_filter"]:
                raise ValueError("\nParameter 'aquaduct_traced_residues_filter' must be a non-empty "
                                 "comma-separated list of residue names (e.g. 'WAT, O2').")

        # The cheap exact-matching config guards are meaningful only when event assignment (stage 9)
        # will actually run; skip them otherwise so a tunnels-only rerun is not constrained by event
        # parameters it never reaches.
        if self._runs_stage(9) and self.parameters["aquaduct_results_path"]:
            if self.parameters["visualize_exact_matching_outcomes"] \
                    and not self.parameters["perform_exact_matching_analysis"]:
                logger.warning("\nUse of 'visualize_exact_matching_outcomes = True' has no effect when "
                               "'perform_exact_matching_analysis = False'.")

            if not self.parameters["trajectory_path"] and \
                    (self.parameters["perform_exact_matching_analysis"] or
                     self.parameters["ambiguous_event_assignment_resolution"] == "exact_matching"):
                raise ValueError("\nRequested Exact matching of events requires requires access to source MD "
                                 "trajectories specified by 'trajectory_path' parameter. PLEASE, consider consulting "
                                 "the user guide.")

        # The comparative-group membership check and the exact-matching data-coverage warnings both
        # walk the input folders via get_input_folders(); each is only meaningful for the stage(s)
        # that consume that result - comparative output (5-6, 9-10) and event assignment / exact
        # matching (9). Skip the glob entirely when neither will run for this invocation.
        check_comparative = (self._runs_stage(5, 6, 9, 10)
                             and self.parameters["perform_comparative_analysis"]
                             and self.parameters["comparative_groups_definition"] is not None)
        check_exact_matching = self._runs_stage(9) and bool(self.parameters["aquaduct_results_path"])

        if check_comparative or check_exact_matching:
            caver_paths, traj_paths, aquaduct_paths = self.get_input_folders()

            if check_comparative:
                membership = dict()
                for group, md_labels in self.parameters["comparative_groups_definition"].items():
                    for md_label in md_labels:
                        if md_label not in caver_paths:
                            raise ValueError("\nFolder (pattern) '{}' specified in the group '{}' from "
                                             "'comparative_groups_definition' parameter does not match any detected "
                                             "input folder".format(md_label, group))

                        if md_label in membership.keys() and group != membership[md_label]:
                            raise ValueError("\nFolder (pattern) '{}' is assigned into multiple groups (at least to "
                                             "'{}' and '{}') in 'comparative_groups_definition' "
                                             "parameter but must be unique!".format(md_label, group,
                                                                                    membership[md_label]))
                        membership[md_label] = group

            if check_exact_matching:
                # we need to have corresponding caver and trajectory data for each aquaduct data otherwise exact analyses/matching will not work!
                all_aquaduct_md_labels = sorted({md for mds in aquaduct_paths.values() for md in mds})
                exact_matching_data_mismatch = list()
                for path2assign in all_aquaduct_md_labels:
                    if path2assign not in caver_paths or path2assign not in traj_paths:
                        exact_matching_data_mismatch.append(path2assign)
                exact_matching_data_coverage = int((len(all_aquaduct_md_labels) - len(exact_matching_data_mismatch))
                                                   * 100 / len(all_aquaduct_md_labels))

                if self.parameters["ambiguous_event_assignment_resolution"] == "exact_matching" \
                        and exact_matching_data_coverage < 90:
                    logger.warning("\nUsing 'exact_matching' for 'ambiguous_event_assignment_resolution' with low "
                                   "coverage ({:d}%) by matching tunnel data will result to a large number of events "
                                   "assigned among outliers. PLEASE, consider consulting "
                                   "the user guide.".format(exact_matching_data_coverage))

                if not self.parameters["ambiguous_event_assignment_resolution"] == "exact_matching" \
                        and self.parameters["stop_after_stage"] >= 8 \
                        and exact_matching_data_coverage >= 90:
                    logger.warning("\nSince it seems that source MD trajectories as well as tunnel data (tunnel data "
                                   "coverage = {:d}%) are available,\n consider using 'exact_matching' for "
                                   "'ambiguous_event_assignment_resolution' parameter to provide the most reliable "
                                   "assignment of events to supercluster at a cost of only mild increase in "
                                   "computation time.".format(exact_matching_data_coverage))

    def _test_input_data(self):
        """
        Tests existence of input files as specified by input parameters
        """

        if not self.parameters["caver_results_path"]:
            raise RuntimeError("\nParameter 'caver_results_path' must be defined and point to the folder "
                               "with the data!")

        if not self.parameters["caver_results_folder_pattern"]:
            raise RuntimeError("\nParameter 'caver_results_folder_pattern' must be defined")

        if not self.parameters["caver_results_relative_subfolder_path"]:
            raise RuntimeError("\nParameter 'caver_results_relative_subfolder_path' must be defined")

        if self.parameters["pdb_reference_structure"] is not None \
                and not os.path.exists(self.parameters["pdb_reference_structure"]):
            raise RuntimeError("\nParameter 'pdb_reference_structure' must point to existing file!.")

        if self.parameters["trajectory_path"]:
            if not self.parameters["trajectory_folder_pattern"]:
                raise ValueError("\nWhen 'trajectory_path' parameter is used, the corresponding folder pattern has to"
                                 " be specified through the 'trajectory_folder_pattern' parameter")
            if not self.parameters["topology_relative_file"] or not self.parameters["trajectory_relative_file"]:
                raise ValueError("\nWhen 'trajectory_path' parameter is used, the corresponding relative filenames for"
                                 " the trajectory and matching topology have to be specified through the "
                                 "'trajectory_relative_file' and 'topology_relative_file' parameters")

        if self.parameters["aquaduct_results_path"]:
            if not self.parameters["aquaduct_results_folder_pattern"]:
                raise ValueError("\nWhen 'aquaduct_results_path' parameter is used, corresponding folder patterns has "
                                 "to be specified through the 'aquaduct_results_folder_pattern' parameter")

            # in aquaduct_results_path, there are folders for matching aquaduct_results_folder_pattern, in which
            # 6_visualize_results.tar.gz and 5_analysis_results.txt files must be located

            if not self.parameters["aquaduct_results_relative_tarfile"]:
                raise ValueError("\nWhen 'aquaduct_results_path' is used, the corresponding relative filename for the "
                                 "AQUA-DUCT tar.gz file with visualization scripts have to be specified through "
                                 "the 'aquaduct_results_relative_tarfile' parameter")

            if not self.parameters["aquaduct_results_relative_summaryfile"]:
                raise ValueError("\nWhen 'aquaduct_results_path' is used, the corresponding relative filename for the "
                                 "AQUA-DUCT summary file have to be specified through "
                                 "the 'aquaduct_results_relative_summaryfile' parameter")

        # Each filesystem-existence sweep is scoped to the stage(s) that consume that input: CAVER
        # data is read by stages 1-2, MD trajectories by parsing/alignment (2) and exact matching (9),
        # AQUA-DUCT archives by stage 7. On a resume past those stages the data is already parsed into
        # the checkpoint, so the (often large) input trees are not re-walked here - and input archived
        # or moved after parsing no longer spuriously fails a late-stage rerun. The cheap
        # parameter-presence guards above stay unconditional; get_input_folders() (the actual glob) is
        # only invoked when at least one sweep below will run.
        test_caver = self._runs_stage(1, 2)
        test_trajectory = self._runs_stage(2, 9) and bool(self.parameters["trajectory_path"])
        test_aquaduct = self._runs_stage(7) and bool(self.parameters["aquaduct_results_path"])

        if test_caver or test_trajectory or test_aquaduct:
            caver_paths, traj_paths, aquaduct_paths = self.get_input_folders()

            # testing inputs of essential CAVER data
            if test_caver:
                if not caver_paths:
                    parameters2check = ("caver_results_path", "caver_results_folder_pattern")
                    raise FileNotFoundError("\nNo folders matching pattern '{}' can be found in {}! Please check if "
                                            "'{}' parameters are defined "
                                            "correctly".format(self.parameters["caver_results_folder_pattern"],
                                                               self.parameters["caver_results_path"], parameters2check))
                caver_keys2test = ["caver_relative_profile_file", "caver_relative_origin_file",
                                   "caver_relative_pdb_file"]
                missing_file_reports = self._test_input_files(caver_paths, caver_keys2test,
                                                              root_folder=self.parameters["caver_results_path"])
                if missing_file_reports:
                    raise FileNotFoundError(missing_file_reports)

                if self.parameters["process_bottleneck_residues"]:
                    missing_file_reports = self._test_input_files(caver_paths, ["caver_relative_bottleneck_file"],
                                                                  root_folder=self.parameters["caver_results_path"])
                    if missing_file_reports:
                        missing_file_reports += "\n\nAlternatively, if the file(s) should not be processed, " \
                                                "set 'process_bottleneck_residues' parameter to 'False'\n"
                        raise FileNotFoundError(missing_file_reports)

            if test_trajectory:
                # test presence of trajectory data
                traj_keys2test = ["trajectory_relative_file", "topology_relative_file"]
                if not traj_paths:
                    parameters2check = ("trajectory_path", "trajectory_folder_pattern")
                    raise FileNotFoundError("\nNo folders matching pattern '{}' were found in {}! Please check if '{}' "
                                            "parameters are defined "
                                            "correctly".format(self.parameters["trajectory_folder_pattern"],
                                                               self.parameters["trajectory_path"], parameters2check))

                missing_file_reports = self._test_input_files(traj_paths, traj_keys2test,
                                                              root_folder=self.parameters["trajectory_path"])
                if missing_file_reports:
                    raise FileNotFoundError(missing_file_reports)

            if test_aquaduct:
                # test presence of AQUA-DUCT data
                aquaduct_keys2test = ["aquaduct_results_relative_tarfile", "aquaduct_results_relative_summaryfile"]
                if not any(aquaduct_paths.values()):
                    parameters2check = ("aquaduct_results_path", "aquaduct_results_folder_pattern")
                    raise FileNotFoundError("\nNo folders matching pattern '{}' can be found in any of {}! Please check "
                                            "if '{}' parameters are defined "
                                            "correctly".format(self.parameters["aquaduct_results_folder_pattern"],
                                                               self.parameters["aquaduct_results_path"],
                                                               parameters2check))

                missing_file_reports = ""
                for root_path, md_labels in aquaduct_paths.items():
                    missing_file_reports += self._test_input_files(md_labels, aquaduct_keys2test, root_folder=root_path)
                if missing_file_reports:
                    raise FileNotFoundError(missing_file_reports)

        if self.parameters["visualize_comparative_super_cluster_volumes"]:
            if not self.parameters["visualize_super_cluster_volumes"]:
                raise ValueError("\nWhen 'visualize_comparative_super_cluster_volumes' parameter is enabled, "
                                 "'visualize_super_cluster_volumes' must be enabled too.")

    def _detect_set_pdb_reference_structure(self):
        """
        Sets a reference PDB structure as the caver PDB file form the first CAVER results folder
        """

        search_pattern = os.path.join(self.parameters["caver_results_folder_pattern"],
                                      self.parameters["caver_relative_pdb_file"])
        ref_structure = sorted([*Path(self.parameters["caver_results_path"]).glob(search_pattern)])[0].as_posix()
        self.parameters["pdb_reference_structure"] = ref_structure
        self.calculations_settings["pdb_reference_structure"] = ref_structure

    def _detect_set_num_cpu(self):
        """
        Detecting and setting of number of CPUs used for job parallelization
        """

        cpu_os = os.cpu_count()
        if cpu_os is None:
            # Cannot detect CPU count, default to 1
            self.parameters["num_cpus"] = 1
        elif cpu_os > 4:
            # many cpus => keep 2 for system
            self.parameters["num_cpus"] = cpu_os - 2
        elif cpu_os >= 2:
            self.parameters["num_cpus"] = cpu_os - 1
            # 2-4 cpus => keep 1 for system
        else:
            self.parameters["num_cpus"] = 1

        self.calculations_settings["num_cpus"] = self.parameters["num_cpus"]

    def _detect_set_num_snapshots(self):
        """
        Reads info on number of snapshots/frames in simulations and if same in all simulations,
        set the number of snapshots
        """
        import mdtraj

        if self.parameters["trajectory_engine"] == "pytraj":
            try:
                import pytraj # type: ignore[import-untyped]
            except ModuleNotFoundError:
                raise RuntimeError("Requested to use 'pytaj' as 'trajectory_engine' but pytraj package cannot be "
                                   "imported. Please check that it is properly installed in the current environment.")

        folders_trajectory = Path(self.parameters["trajectory_path"]).glob(self.parameters["trajectory_folder_pattern"])
        n_frames = set()
        for folder_path in folders_trajectory:
            if not folder_path.is_dir():
                continue
            trajname = get_filepath(folder_path.as_posix(), self.input_paths["trajectory_relative_file"])
            topname = get_filepath(folder_path.as_posix(), self.input_paths["topology_relative_file"])

            if self.parameters["trajectory_engine"] == "pytraj":
                md_traj = pytraj.iterload(trajname, topname) # type: ignore[import-untyped]
                n_frames.add(md_traj.n_frames)

            else:
                traj_frames = 0

                for traj in mdtraj.iterload(trajname, top=topname, chunk=1000):
                    traj_frames += traj.n_frames
                n_frames.add(traj_frames)

        if len(n_frames) == 1:  # same num frames exists in all locations
            num_snaps = n_frames.pop()
            self.parameters["snapshots_per_simulation"] = num_snaps
            self.calculations_settings["snapshots_per_simulation"] = num_snaps
        else:  # different or none => cannot define the value unambiguously
            raise RuntimeError("Could not detected number of frames (or they differ among the trajectories)! "
                               "Please define number of snapshots per simulation using 'snapshots_per_simulation' "
                               "parameter.")

    def _parse_group_definition(self):
        """
        Transform string with group definition to the dictionary
        Format of group definition: group_name1: [folder1, folder2, ...]; group_name2: [folder1, folder2, ...]...
        """
        from string import punctuation

        group_def = dict()
        md_folders = list()
        for group in self.parameters["comparative_groups_definition"].split(';'):
            name, folders = group.split(':')
            name = name.strip()
            folders = folders.strip()[1:-1]
            if '[' in folders and ']' in folders:
                open_id = folders.index('[')
                close_id = folders.index(']')
                new_delim = "+"
                for new_delim in (set(punctuation) - {'!', '?', '*', '[', ']'}):  # avoid glob active chars
                    if new_delim not in folders[open_id + 1:close_id]:
                        break
                folders = folders[:open_id] + folders[open_id:close_id].replace(',', new_delim) + folders[close_id:]
                folders = [folder.replace(new_delim, ',') for folder in folders.split(",")]
            else:
                folders = folders.split(",")

            group_def[name] = set()
            for folder_pattern in folders:
                folder_pattern = folder_pattern.strip()
                matching_folders = [a.name for a in Path(self.parameters["caver_results_path"]).glob(folder_pattern)
                                    if a.is_dir()]
                if not matching_folders:
                    raise ValueError("\nFolder (pattern) '{}' specified in the group '{}' from "
                                     "'comparative_groups_definition' parameter does not match any detected "
                                     "input folder".format(folder_pattern, name))
                md_folders.extend(matching_folders)
                group_def[name].update(matching_folders)

        for group in group_def.keys():
            if group in md_folders:
                raise ValueError("\nName of the group '{}' is the same as one of the input folders. Please use "
                                 "a different unique name.".format(group))

        self.parameters["comparative_groups_definition"] = group_def
        self.advanced_settings["comparative_groups_definition"] = group_def

    def _get_caver_filenames(self) -> str:
        """
        Check if files with same filenames matching the pdbfile_pattern can be found in analyzed folders
        :return: file_name
        """

        folders_caver = Path(self.input_paths["caver_results_path"]).glob(self.input_paths["caver_results_folder_pattern"])
        irrelevant_pdb_files = ["start_zone.pdb", "end_zone.pdb", "v_origins.pdb", "origins.pdb"]
        filenames = set()
        for folder_path in folders_caver:
            if not folder_path.is_dir():
                continue
            _search = os.path.join(self.input_paths["caver_results_relative_subfolder_path"], "data",
                                   self.input_paths["caver_pdb_file_pattern"])
            for item in folder_path.glob(_search):
                if item.is_file() and os.path.basename(item) not in irrelevant_pdb_files:
                    filenames.add(item.name)

        if len(filenames) == 1:  # same file exists in all locations
            return filenames.pop()
        elif len(filenames) == 0:  # none
            print("Cannot match any file with 'caver_pdb_file_pattern' in the given paths")
            return ""
        else: # multiple different filenames found => cannot define the name unambigiously
            print("Multiple different files matching 'caver_pdb_file_pattern' in the given paths were found. "
                  "Cannot define the filename unambigiously")
            return ""

    def _test_input_files(self, analyzed_folders: List[str], keys2test: List[str], root_folder: str = ".") -> str:
        """
        Check if files specified by their keys2test exists in given folders
        :param analyzed_folders:  folders where to search for files
        :param keys2test: keys with relative file paths
        :param root_folder: root folder where the analyzed folders are
        :return: info on missing files
        """

        warns = ""
        for key2test in keys2test:
            missing_files = list()
            for folder_path in analyzed_folders:
                file_path = [*Path(os.path.join(root_folder, folder_path)).glob(self.input_paths[key2test])]
                if len(file_path) != 1 or not file_path[0].exists():
                    missing_files.append(os.path.join(root_folder, folder_path, self.input_paths[key2test]))
            if missing_files:
                warns += "\nFollowing required input file(s) does not exist or more then 1 file can be matched!\n " \
                         "Please check if '{}' parameter is defined correctly:\n{}".format(key2test,
                                                                                           "\n".join(missing_files))
        return warns

    def __str__(self):
        msg = "\n#=== Version and dependencies ===\n"
        msg += "TransportTools library version {}\n".format(__version__)
        msg += "\nEmploying:\n"
        msg += "python {}.{}.{}\n".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
        msg += "numpy {}\n".format(version('numpy'))
        msg += "scikit-learn {}\n".format(version('scikit-learn'))
        msg += "scipy {}\n".format(version('scipy'))
        msg += "biopython {}\n".format(version('biopython'))
        msg += "fastcluster {}\n".format(version('fastcluster'))
        msg += "hdbscan {}\n".format(version('hdbscan'))
        msg += "MDtraj {}\n".format(version('mdtraj'))
        if self.parameters["visualize_super_cluster_volumes"]:
            if self.parameters["msms"] is None:
                try:
                    msg += "mcubes {}\n".format(version('pymcubes'))
                except PackageNotFoundError:
                    raise RuntimeError("Requested to 'visualize_super_cluster_volumes' but required 'mcubes' module "
                                       "cannot be imported. Please ensure that it is properly installed in the current "
                                       "environment--for example by running 'pip install --upgrade PyMCubes'.")
            else:
                import subprocess
                try:
                    result = subprocess.run(self.parameters["msms"], capture_output=True, text=True).stdout
                    msg += "msms {}\n".format(result.split()[1])
                except FileNotFoundError:
                    raise RuntimeError("The program msms was not found at '{}' as specified by 'msms' parameter in "
                                       "'ADVANCED_SETTINGS' section of your "
                                       "configuration file.".format(self.parameters["msms"]))

        if self.parameters["trajectory_engine"] == "pytraj":
            try:
                import pytraj # type: ignore[import-untyped]
                msg += "pytraj {}\n".format(version('pytraj'))
                msg += "cpptraj {}\n".format(pytraj.__cpptraj_internal_version__)
            except (PackageNotFoundError, ModuleNotFoundError):
                raise RuntimeError("Requested to use 'pytraj' as 'trajectory_engine' but 'pytraj' module cannot be "
                                   "imported. Please ensure that it is properly installed in the current "
                                   "environment--for example by running 'conda install ambertools -c conda-forge'.")
        msg += "\nJob configuration loaded from file: '{}'\n".format(self.source_file)
        # report submitit's version when any stage is set to run on SLURM; the validation in
        # _validate_parameter_values() has already guaranteed it is importable in that case
        any_slurm = str(self.parameters.get("compute_backend", "local")).lower() == "slurm" or any(
            str(self.parameters.get(knob, None) or "local").lower() == "slurm"
            for knob, _ in getattr(self, "_stage_backend_keys", []))
        if any_slurm:
            try:
                import submitit  # type: ignore[import-untyped]
                msg += "submitit {}\n".format(version('submitit'))
            except (PackageNotFoundError, ModuleNotFoundError):
                raise RuntimeError("Requested to use SLURM via submitit but required module cannot be imported.\n"
                                   " Please ensure that it is properly installed in the current environment,"
                                   "for example by running 'conda install -c conda-forge submitit'.")

        msg += "#=== General calculations settings ===\n"
        for name, values in self.calculations_settings.items():
            msg += " {} = {}\n".format(name, str(values))

        msg += "\n#=== Output settings ===\n"
        for name, values in self.output_settings.items():
            msg += " {} = {}\n".format(name, str(values))

        msg += "\n#=== Input paths ===\n"
        for name, values in self.input_paths.items():
            msg += " {} = {}\n".format(name, str(values))

        msg += "\n#=== Output paths ===\n"
        for name, values in self.output_paths.items():
            msg += " {} = {}\n".format(name, str(values))

        if self.advanced_settings != self.advanced_settings_defaults:
            msg += "\n#=== Advanced settings ===\n"
            for name, values in self.advanced_settings.items():
                if values != self.advanced_settings_defaults[name]:
                    msg += " {} = {}\n".format(name, str(values))
        return msg

    def report_updates(self, old_parameters: dict):
        """
        Logs information on modified parameters upon restart
        :param old_parameters: previous parameters to compare with
        """

        header = "\nJob configuration UPDATED from file: '{}'\n".format(self.source_file)
        msg = ""
        section_head = "#=== General calculations settings ===\n"
        section_msg = ""
        for name, values in self.calculations_settings.items():
            if name in old_parameters.keys() and values == old_parameters[name]:
                continue
            section_msg += " {} = {}\n".format(name, str(values))
        if section_msg:
            msg += section_head + section_msg

        section_head = "\n#=== Output settings ===\n"
        section_msg = ""
        for name, values in self.output_settings.items():
            if name in old_parameters.keys() and values == old_parameters[name]:
                continue
            section_msg += " {} = {}\n".format(name, str(values))
        if section_msg:
            msg += section_head + section_msg

        section_head = "\n#=== Input paths ===\n"
        section_msg = ""
        for name, values in self.input_paths.items():
            if name in old_parameters.keys() and values == old_parameters[name]:
                continue
            section_msg += " {} = {}\n".format(name, str(values))
        if section_msg:
            msg += section_head + section_msg

        section_head = "\n#=== Output paths ===\n"
        section_msg = ""
        for name, values in self.output_paths.items():
            if name in old_parameters.keys() and values == old_parameters[name]:
                continue
            section_msg += " {} = {}\n".format(name, str(values))
        if section_msg:
            msg += section_head + section_msg

        section_head = "\n#=== Advanced settings ===\n"
        section_msg = ""
        for name, values in self.advanced_settings.items():
            if name in old_parameters.keys() and values == old_parameters[name]:
                continue
            section_msg += " {} = {}\n".format(name, str(values))
        if section_msg:
            msg += section_head + section_msg

        if msg:  # only report when parameters actually changed on this (re)start
            logger.warning(header + msg)

    def get_parameter(self, par_name: str):
        """
        Return values for query parameter
        :param par_name: parameter name
        """

        return self.parameters[par_name]

    def resolve_stage_backend(self, stage_knob: str) -> str:
        """
        Resolve a stage's effective execution backend ('local' or 'slurm') from the per-stage
        override knob (if set) and the global compute_backend default. Used by callers that
        need to dispatch between a local Pool and the SLURM backend for a given parallelized
        stage; the knob name must match one of the rows registered in `self._stage_backend_keys`.
        :param stage_knob: name of the per-stage backend parameter (e.g. 'stage04_backend')
        :return: lowercase 'local' or 'slurm'
        """

        registered = {knob for knob, _ in self._stage_backend_keys}
        if stage_knob not in registered:
            raise KeyError("Unknown stage backend knob '{}'; registered knobs are {}".format(
                stage_knob, sorted(registered)))
        override = self.parameters.get(stage_knob)
        if override is not None:
            return str(override).lower()
        return str(self.parameters["compute_backend"]).lower()

    def get_filters(self) -> Tuple[float, float, float, float, float, float, int, int, float, int, int, int]:
        """
        Collect filter values properly ordered to act as argument for filter_super_cluster_profiles and define_filters
        :return: tuple of filter values (min_length, max_length, min_bottleneck_radius, max_bottleneck_radius,
                 min_curvature, max_curvature, min_sims_num, min_snapshots_num, min_avg_snapshots_num,
                 min_total_events, min_entry_events, min_release_events)
        """

        filter_names = ["min_length", "max_length", "min_bottleneck_radius", "max_bottleneck_radius", "min_curvature",
                        "max_curvature", "min_sims_num", "min_snapshots_num", "min_avg_snapshots_num",
                        "min_total_events", "min_entry_events", "min_release_events"]
        filters = []
        for name in filter_names:
            value = self.get_parameter(name)
            # Ensure integer parameters are cast to int for type safety
            if name in ["min_sims_num", "min_snapshots_num", "min_total_events", "min_entry_events", "min_release_events"]:
                filters.append(int(value))
            else:
                filters.append(float(value))

        return tuple(filters)

    def set_parameter(self, par_name: str, value: Any):
        """
        Set new value for query parameter
        :param par_name: parameter name
        :param value: new value of the par_name parameter
        """

        self.parameters[par_name] = value
        if par_name == "output_path":  # update also dependent paths
            self._set_output_paths(value)
            self.parameters.update(self.output_paths)

    def get_parameters(self) -> dict:
        """
        Return compiled parameters of the job
        """

        return self.parameters.copy()

    def get_input_folders(self) -> Tuple[List[str], List[str], dict]:
        """
        Enumerates source folders with CAVER data, MD trajectories and AQUA-DUCT data separately
        :return: list of source folders with CAVER data, MD trajectories, and dict mapping each
                 aquaduct_results_path to its sorted list of MD simulation folder names
        """

        folders_caver = [a.name for a in Path(self.parameters["caver_results_path"]).glob(self.parameters["caver_results_folder_pattern"]) if a.is_dir()]

        if self.parameters["trajectory_path"] is None:
            folders_trajectory = []
        else:
            folders_trajectory = [a.name for a in Path(self.parameters["trajectory_path"]).glob(self.parameters["trajectory_folder_pattern"]) if a.is_dir()]

        if self.parameters["aquaduct_results_path"] is None:
            folders_aquaduct = {}
        else:
            folders_aquaduct = {
                root_path: sorted(a.name for a in Path(root_path).glob(
                    self.parameters["aquaduct_results_folder_pattern"]) if a.is_dir())
                for root_path in self.parameters["aquaduct_results_path"]
            }

            if self.parameters.get("aquaduct_allow_empty_folders"):
                required_patterns = [
                    self.parameters["aquaduct_results_relative_tarfile"],
                    self.parameters["aquaduct_results_relative_summaryfile"],
                ]
                pruned: dict = {}
                for root_path, md_labels in folders_aquaduct.items():
                    kept = []
                    for md_label in md_labels:
                        folder = Path(root_path) / md_label
                        if all(len([*folder.glob(pat)]) == 1 for pat in required_patterns):
                            kept.append(md_label)
                        else:
                            key = os.path.join(root_path, md_label)
                            if key not in self._logged_empty_aquaduct_folders:
                                logger.warning("AQUA-DUCT folder '{}' is missing required tar/summary files; "
                                               "skipping it due to 'aquaduct_allow_empty_folders = True'.".format(key))
                                self._logged_empty_aquaduct_folders.add(key)
                    pruned[root_path] = kept
                folders_aquaduct = pruned

        return sorted(folders_caver), sorted(folders_trajectory), folders_aquaduct

    def get_reference_pdb_file(self) -> str:
        """
        Returns path to the reference PDB file that will define transformations for the job,
        unless user specified, it is the file from first CAVER folder
        """

        return self.parameters["pdb_reference_structure"]

    def write_template_file(self, filepath: str, advanced: bool = False):
        """
        Save job configuration template with default values to a file
        :param filepath: path to INI file to which the configuration template will be saved
        :param advanced: write also advanced section
        """

        config_sections = {
            "input_paths": self.input_paths,
            "calculations_settings": self.calculations_settings,
            "output_settings": self.output_settings,
        }
        if advanced:
            config_sections.update({"advanced_settings": self.advanced_settings})

        commentaries = {
            "caver_results_path": "# CAVER results",
            "aquaduct_results_path": "# AQUA-DUCT results",
            "trajectory_path": "# Source MD trajectories",
            "snapshots_per_simulation": "# Parsing of tunnel clusters from CAVER results",
            "relevant_tunnel_cluster_min_size": "# Filtreing of tunnels and clusters before layering",
            "clustering_method": "# Clustering of tunnel clusters into superclusters - global and agglomerative settings",
            "hdbscan_min_cluster_size": "# HDBscan clustering settings",
            "compute_backend": "# SLURM execution backend "
                               "(the 'slurm' backend requires the optional 'submitit' package)",
            "min_length": "# Filters applied on superclusters before event assignment (-1 => inactive filter)",
            "event_min_distance": "# Processing of transport events from AQUA-DUCT results, "
                                  "and their assignment to superclusters",
            "min_total_events": "# Additional filters applied on superclusters after event assignment "
                                "(-1 => inactive filter)",
            "save_super_cluster_profiles_csvs": "# Optional data generation",
            "visualize_super_cluster_volumes": "# Optional visualization",
            "random_seed": "# Calculations",
            "visualize_exact_matching_outcomes": "# Finer control of outputs & logging"
        }

        with open(filepath, "w") as out_stream:
            for section_name, section_dict in config_sections.items():
                out_stream.write("[{}]\n".format(section_name.upper()))
                for name, value in section_dict.items():
                    if name in commentaries.keys():
                        out_stream.write("{}\n".format(commentaries[name]))
                    if not name.startswith("caver_relative_") and name not in self.legacy_params :  # these are autocompleted here or legacy => do not show to users
                        if name in self.float_params and value is not None:
                            out_stream.write("{} = {:.2f}\n".format(name, value))
                        else:
                            out_stream.write("{} = {}\n".format(name, value))
                out_stream.write("\n")
