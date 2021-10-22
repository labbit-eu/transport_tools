# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2021  Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej
# Surpeta, Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal  <janbre@amu.edu.pl>
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

__version__ = '0.9.0'
__author__ = 'Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta, ' \
             'Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal'
__mail__ = 'janbre@amu.edu.pl'

import os
import sys
import numpy as np
from pathlib import Path
from pkg_resources import get_distribution, DistributionNotFound
from configparser import ConfigParser
from typing import List, Any, Optional, Union
from logging import getLogger
from transport_tools.libs.ui import initiate_tools
from transport_tools.libs.utils import get_filepath

logger = getLogger(__name__)


class AnalysisConfig:
    def __init__(self, file2load_from: Optional[str] = None, logging: bool = True):
        """
        Class storing the job configuration, also enables some parameter evaluation & completion
        :param file2load_from: INI file with configuration to load from
        :param logging: if logging should be set up
        """

        self.source_file = file2load_from
        self.calculations_settings = dict()
        self.output_settings = dict()
        self.input_paths = dict()
        self.output_paths = dict()
        self.advanced_settings = dict()

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
                initiate_tools(self.get_parameter("log_level"), self.get_parameter("verbose_logging"),
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
            "pdb_reference_structure": None,  # reference structure for alignment
            "layer_thickness": 1.5,  # thickness of concentric layered grid

            # Parsing of tunnel clusters from CAVER results
            "snapshots_per_simulation": None,  # number of snapshots used for CAVER calculation
            "caver_traj_offset": 1,  # difference in IDs of MD frames (from 0) and caver snapshots (often from 1)
            "snapshot_id_position": 1,  # location of IDs in snapshot filenames from CAVER split by snapshot_delimiter
            "snapshot_delimiter": ".",  # delimiter used for splitting snapshot filenames from CAVER to get IDs
            "process_bottleneck_residues": False,  # read file bottlenecks.csv

            # Clustering of tunnel clusters into superclusters
            "min_tunnel_radius4clustering": 0.75,  # filters on tunnel load, applies for layering only
            "min_tunnel_length4clustering": 5.0,  # filters on tunnel load, applies for layering only
            "max_tunnel_curvature4clustering": 2.0,  # filters on tunnel load, applies for layering only
            "clustering_linkage": "complete",  # for agglomerative clustering
            "clustering_cutoff": 2.0,  # clustering of tunnel clusters to superclusters

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
            "n_jobs_per_cpu_batch": 100000,  # 100000 jobs per cpu should be reasonable for 4 CPUs and 10 GB RAM

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
            "pdb_reference_structure",
            "snapshots_per_simulation",
            "comparative_groups_definition"
        ]

        self.boolean_params = [
            "process_bottleneck_residues",
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
            "legacy_pymol_support"
        ]

        self.integer_params = [
            "start_from_stage",
            "stop_after_stage",
            "num_cpus",
            "snapshots_per_simulation",
            "caver_traj_offset",
            "snapshot_id_position",
            "min_sims_num",
            "min_snapshots_num",
            "min_total_events",
            "min_entry_events",
            "min_release_events",
            "random_seed",
            "clustering_max_num_rep_frag",
            "n_jobs_per_cpu_batch",
            "max_layered_points4visualization",
            "max_events_per_cluster4visualization",
            "pickle_protocol"
        ]

        self.float_params = [
            "layer_thickness",
            "min_tunnel_radius4clustering",
            "min_tunnel_length4clustering",
            "max_tunnel_curvature4clustering",
            "clustering_cutoff",
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
            "aqauduct_ligand_effective_radius"
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
                    else:
                        param_value = str(config[section].get(param_name))

                if param_name == "visualize_super_cluster_volumes" and param_value is False:
                    forbid_autoactivation.append(param_name)
                if param_name == "trajectory_engine" and param_value == "mdtraj":
                    forbid_autoactivation.append(param_name)

                if param_name == "output_path":  # update also dependent paths
                    self._set_output_paths(param_value)

                config_dictionary[param_name] = param_value

        self._set_input_paths()
        self._autoactivate_functionalities(forbid_autoactivation)

    def _autoactivate_functionalities(self, overridden_parameters: List[str]):
        """
        Activate optional functionalities if required modules are available, still can be overridden in a config file.
        :param overridden_parameters: list of parameter names which have been overridden
        """
        if not self.output_settings["visualize_super_cluster_volumes"] and \
                "visualize_super_cluster_volumes" not in overridden_parameters:
            try:
                import mcubes
            except ModuleNotFoundError:
                pass
            else:
                logger.warning("'mcubes' module detected, provisionally enabling 'visualize_super_cluster_volumes'. "
                               "Should you like to override this action, please set 'visualize_super_cluster_volumes = "
                               "False' in 'OUTPUT_SETTINGS' section of your configuration file.")
                self.output_settings["visualize_super_cluster_volumes"] = True

        if self.advanced_settings["trajectory_engine"] != "pytraj" and "trajectory_engine" not in overridden_parameters:
            try:
                import pytraj
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

        if self.input_paths["caver_relative_pdb_file"] is None:
            if not self.input_paths["caver_results_path"]:
                raise RuntimeError("\nParameter 'caver_results_path' must be defined and point to the folder with the"
                                   " data!")

            caver_pdb_filename = self._get_caver_filenames()
            if not caver_pdb_filename:
                raise RuntimeError("\nCould not detected PDB file of protein in CAVER data folder! Please define its "
                                   "relative path in 'caver_relative_pdb_file' parameter.")

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

        if self.parameters["stop_after_stage"] is None:
            self.parameters["stop_after_stage"] = sys.maxsize

        if self.parameters["num_cpus"] is None:
            self._detect_set_num_cpu()

        if self.parameters["pdb_reference_structure"] is None:
            self._detect_set_pdb_reference_structure()

        if self.parameters["snapshots_per_simulation"] is None:
            if self.parameters["trajectory_path"] is not None:
                self._detect_set_num_snapshots()
            else:
                raise RuntimeError("\nParameter 'snapshots_per_simulation' must be defined or access to input "
                                   "trajectories must be provided to allow auto-detection.")

        if self.parameters["perform_comparative_analysis"] \
                and self.parameters["comparative_groups_definition"] is not None:
            self._parse_group_definition()

        if self.parameters["legacy_pymol_support"]:
            self.parameters["pickle_protocol"] = 2
            self.internal_settings["pickle_protocol"] = 2

    def _test_parameter_sanity(self, param_name: str, min_val: int or float, max_val: int or float):
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
            "min_tunnel_radius4clustering": (0.5, 2),
            "min_tunnel_length4clustering": (3, 10),
            "max_tunnel_curvature4clustering": (2, 5),
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

        if self.parameters["num_cpus"] is not None:
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

        import fastcluster
        valid_linkage = fastcluster.mthidx.copy()

        if self.parameters["clustering_linkage"] not in valid_linkage.keys():
            raise ValueError("\nUnsupported linkage type '{}' specified in 'clustering_linkage' parameter.\n Valid "
                             "options are '{}'".format(self.parameters["clustering_linkage"], valid_linkage.keys()))

        if self.parameters["clustering_linkage"] == "single":
            logger.warning("\nUse of 'single' for 'cluster_linkage' parameter is discouraged as it often leads to "
                           "rather poorly defined superclusters. PLEASE, consider using 'average' or 'complete'.")

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
        self._test_parameter_sanity("min_tunnel_radius4clustering", 0, sys.maxsize)
        self._test_parameter_sanity("min_tunnel_length4clustering", 0, sys.maxsize)
        self._test_parameter_sanity("max_tunnel_curvature4clustering", 1, sys.maxsize)
        self._test_parameter_sanity("clustering_cutoff", 0, sys.maxsize)
        self._test_parameter_sanity("event_min_distance", 0, sys.maxsize)
        self._test_parameter_sanity("event_assignment_cutoff", 0, 1)
        self._test_parameter_sanity("clustering_max_num_rep_frag", 0, sys.maxsize)
        self._test_parameter_sanity("max_events_per_cluster4visualization", 0, sys.maxsize)

        caver_paths, traj_paths, aquaduct_paths = self.get_input_folders()
        if self.parameters["perform_comparative_analysis"] \
                and self.parameters["comparative_groups_definition"] is not None:
            membership = dict()
            for group, md_labels in self.parameters["comparative_groups_definition"].items():
                for md_label in md_labels:
                    if md_label not in caver_paths:
                        raise ValueError("\nFolder (pattern) '{}' specified in the group '{}' from "
                                         "'comparative_groups_definition' parameter does not match any detected "
                                         "input folder".format(md_label, group))

                    if md_label in membership.keys() and group != membership[md_label]:
                        raise ValueError("\nFolder (pattern) '{}' is assigned into multiple groups (at least to '{}' "
                                         "and '{}') in 'comparative_groups_definition' "
                                         "parameter but must be unique!".format(md_label, group, membership[md_label]))
                    membership[md_label] = group

        if self.parameters["aquaduct_results_path"]:
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

            # we need to have caver and aquduct data from same MDs or exact analyses/matching will not work!
            exact_matching_data_mismatch = list()
            for path2assign in aquaduct_paths:
                if path2assign not in caver_paths or path2assign not in traj_paths:
                    exact_matching_data_mismatch.append(path2assign)
            exact_matching_data_coverage = int((len(aquaduct_paths) - len(exact_matching_data_mismatch)) * 100 /
                                               len(aquaduct_paths))

            if self.parameters["ambiguous_event_assignment_resolution"] == "exact_matching" \
                    and exact_matching_data_coverage < 90:
                logger.warning("\nUsing 'exact_matching' for 'ambiguous_event_assignment_resolution' with low coverage "
                               "({:d}%) by matching tunnel data will result to a large number of events assigned among "
                               "outliers. PLEASE, consider consulting "
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

            # in aquaduct_results_path, there are folders for MD matching aquaduct_results_folder_pattern, in which
            # 6_visualize_results.tar.gz and 5_analysis_results.txt files must be located

            if not self.parameters["aquaduct_results_relative_tarfile"]:
                raise ValueError("\nWhen 'aquaduct_results_path' is used, the corresponding relative filename for the "
                                 "AQUA-DUCT tar.gz file with visualization scripts have to be specified through "
                                 "the 'aquaduct_results_relative_tarfile' parameter")

            if not self.parameters["aquaduct_results_relative_summaryfile"]:
                raise ValueError("\nWhen 'aquaduct_results_path' is used, the corresponding relative filename for the "
                                 "AQUA-DUCT summary file have to be specified through "
                                 "the 'aquaduct_results_relative_summaryfile' parameter")

        caver_paths, traj_paths, aquaduct_paths = self.get_input_folders()

        # testing inputs of essential CAVER data
        if not caver_paths:
            parameters2check = ("caver_results_path", "caver_results_folder_pattern")
            raise FileNotFoundError("\nNo folders matching pattern '{}' can be found in {}! Please check if '{}' "
                                    "parameters are defined "
                                    "correctly".format(self.parameters["caver_results_folder_pattern"],
                                                       self.parameters["caver_results_path"], parameters2check))
        caver_keys2test = ["caver_relative_profile_file", "caver_relative_origin_file", "caver_relative_pdb_file"]
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

        if self.parameters["trajectory_path"]:
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

        if self.parameters["aquaduct_results_path"]:
            # test presence of AQUA-DUCT data
            aquaduct_keys2test = ["aquaduct_results_relative_tarfile", "aquaduct_results_relative_summaryfile"]
            if not aquaduct_paths:
                parameters2check = ("trajectory_path", "trajectory_folder_pattern")
                raise FileNotFoundError("\nNo folders matching pattern '{}' can be found in {}! Please check if '{}'"
                                        " parameters are defined "
                                        "correctly".format(self.parameters["aquaduct_results_folder_pattern"],
                                                           self.parameters["aquaduct_results_path"], parameters2check))

            missing_file_reports = self._test_input_files(aquaduct_paths, aquaduct_keys2test,
                                                          root_folder=self.parameters["aquaduct_results_path"])
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
        if cpu_os > 4:
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

        if self.parameters["trajectory_engine"] == "pytraj":
            try:
                import pytraj
            except ModuleNotFoundError:
                raise RuntimeError("Requested to use 'pytaj' as 'trajectory_engine' but pytraj package cannot be "
                                   "imported. Please check that it is properly installed in the current environment.")
        else:
            import mdtraj

        folders_trajectory = Path(self.parameters["trajectory_path"]).glob(self.parameters["trajectory_folder_pattern"])
        n_frames = set()
        for folder_path in folders_trajectory:
            if not folder_path.is_dir():
                continue
            trajname = get_filepath(folder_path.as_posix(), self.input_paths["trajectory_relative_file"])
            topname = get_filepath(folder_path.as_posix(), self.input_paths["topology_relative_file"])

            if self.parameters["trajectory_engine"] == "pytraj":
                md_traj = pytraj.iterload(trajname, topname)
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
        Check if files with same filenames matching the pattern can be found in analyzed folders
        :return: file_name
        """

        folders_caver = Path(self.input_paths["caver_results_path"]).glob(self.input_paths["caver_results_folder_pattern"])
        filenames = set()
        for folder_path in folders_caver:
            if not folder_path.is_dir():
                continue
            _search = os.path.join(self.input_paths["caver_results_relative_subfolder_path"], "data", "*[0-9]*.pdb")
            for item in folder_path.glob(_search):
                if item.is_file():
                    filenames.add(item.name)

        if len(filenames) == 1:  # same file exists in all locations
            return filenames.pop()
        else:  # different or none => cannot define the name unambiguously
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
        msg += "numpy {}\n".format(get_distribution('numpy').version)
        msg += "scikit-learn {}\n".format(get_distribution('scikit-learn').version)
        msg += "scipy {}\n".format(get_distribution('scipy').version)
        msg += "biopython {}\n".format(get_distribution('biopython').version)
        msg += "fastcluster {}\n".format(get_distribution('fastcluster').version)
        msg += "hdbscan {}\n".format(get_distribution('hdbscan').version)
        msg += "MDtraj {}\n".format(get_distribution('mdtraj').version)
        if self.parameters["visualize_super_cluster_volumes"]:
            try:
                msg += "mcubes {}\n".format(get_distribution('pymcubes').version)
            except DistributionNotFound:
                raise RuntimeError("Requested to 'visualize_super_cluster_volumes' but required 'mcubes' module "
                                   "cannot be imported. Please ensure that it is properly installed in the current "
                                   "environment--for example by running 'pip install --upgrade PyMCubes'.")
        if self.parameters["trajectory_engine"] == "pytraj":
            try:
                import pytraj
                msg += "pytraj {}\n".format(get_distribution('pytraj').version)
                msg += "cpptraj {}\n".format(pytraj.__cpptraj_internal_version__)
            except (DistributionNotFound, ModuleNotFoundError):
                raise RuntimeError("Requested to use 'pytraj' as 'trajectory_engine' but 'pytraj' module cannot be "
                                   "imported. Please ensure that it is properly installed in the current "
                                   "environment--for example by running 'conda install ambertools -c conda-forge'.")
        msg += "\nJob configuration loaded from file: '{}'\n".format(self.source_file)
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

        msg = "\nJob configuration UPDATED from file: '{}'\n".format(self.source_file)
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

        if self.advanced_settings != self.advanced_settings_defaults:
            section_head = "\n#=== Advanced settings ===\n"
            section_msg = ""
            for name, values in self.advanced_settings.items():
                if values != self.advanced_settings_defaults[name]:
                    if name in old_parameters.keys() and values == old_parameters[name]:
                        continue
                    section_msg += " {} = {}\n".format(name, str(values))

            if section_msg:
                msg += section_head + section_msg

        logger.debug(msg)

    def get_parameter(self, par_name: str):
        """
        Return values for query parameter
        :param par_name: parameter name
        """

        return self.parameters[par_name]

    def get_filters(self) -> List[Union[float, int]]:
        """
        Collect filter values properly ordered to act as argument for filter_super_cluster_profiles and define_filters
        :return: list of filter values
        """

        filter_names = ["min_length", "max_length", "min_bottleneck_radius", "max_bottleneck_radius", "min_curvature",
                        "max_curvature", "min_sims_num", "min_snapshots_num", "min_avg_snapshots_num",
                        "min_total_events", "min_entry_events", "min_release_events"]
        filters = list()
        for name in filter_names:
            filters.append(self.get_parameter(name))

        return filters

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

    def get_input_folders(self) -> (List[str], List[str], List[str]):
        """
        Enumerates source folders with CAVER data, MD trajectories and AQUA-DUCT data separately
        :return: list of source folders with CAVER data, MD trajectories, and AQUA-DUCT data, respectively
        """

        folders_caver = [a.name for a in Path(self.parameters["caver_results_path"]).glob(self.parameters["caver_results_folder_pattern"]) if a.is_dir()]

        if self.parameters["trajectory_path"] is None:
            folders_trajectory = []
        else:
            folders_trajectory = [a.name for a in Path(self.parameters["trajectory_path"]).glob(self.parameters["trajectory_folder_pattern"]) if a.is_dir()]

        if self.parameters["aquaduct_results_path"] is None:
            folders_aquaduct = []
        else:
            folders_aquaduct = [a.name for a in Path(self.parameters["aquaduct_results_path"]).glob(self.parameters["aquaduct_results_folder_pattern"]) if a.is_dir()]

        return sorted(folders_caver), sorted(folders_trajectory), sorted(folders_aquaduct)

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
            "min_tunnel_radius4clustering": "# Clustering of tunnel clusters into superclusters",
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
                    if not name.startswith("caver_relative_"):  # these are autocompleted here => do not show to users
                        if name in self.float_params:
                            out_stream.write("{} = {:.2f}\n".format(name, value))
                        else:
                            out_stream.write("{} = {}\n".format(name, value))
                out_stream.write("\n")
