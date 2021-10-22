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
import pickle
import numpy as np
import fastcluster
from typing import List, Dict, Tuple, Union, Optional
from scipy.cluster.hierarchy import fcluster
from multiprocessing import Pool
from transport_tools.libs.config import AnalysisConfig
from logging import getLogger
from transport_tools.libs.ui import progressbar, TimeProcess, process_count
from transport_tools.libs import utils
from transport_tools.libs.networks import TunnelNetwork, AquaductNetwork, SuperCluster, define_filters, TunnelCluster, \
    subsample_events, get_md_membership4groups
from transport_tools.libs.geometry import LayeredPathSet, average_starting_point
from transport_tools.libs.protein_files import TrajectoryTT, TrajectoryFactory, get_transform_matrix, \
    transform_pdb_file, get_general_rot_mat_from_2_ca_atoms, transform_aquaduct


logger = getLogger(__name__)


class OutlierTransportEvents:
    def __init__(self, parameters: dict):
        """
        Class for storing information on transport events that cannot be assigned to any supercluster, & their reporting
        :param parameters: job configuration parameters
        """

        self.parameters = parameters
        # info on transport events assigned to this SC
        # event_type -> md_label -> list of events and their residue info
        self.transport_events: Dict[str, Dict[str, List[Tuple[str, Tuple[str, Tuple[int, int]]]]]] = dict()
        self.transport_events_global: Dict[str, Dict[str, List[Tuple[str, Tuple[str, Tuple[int, int]]]]]] = dict()

        self.num_events: Dict[str, Dict[str, int]] = {"overall": {"entry": 0, "release": 0}}

    def exist(self, md_label: str = "overall") -> bool:
        """
        Tests if some events were unassigned hence considered as outlying
        :param md_label: name of folder with the MD simulation data that contain this transport event
        """

        if md_label in self.num_events.keys() and (self.num_events[md_label]["entry"] > 0
                                                   or self.num_events[md_label]["release"] > 0):
            return True
        else:
            return False

    def count_events(self, md_label: str = "overall") -> (int, int, int):
        """
        Counts number of all unassigned events (total, entries, releases)
        :param md_label: name of folder with the MD simulation data that contain this transport event
        """

        if md_label not in self.num_events.keys():
            return 0, 0, 0

        return self.num_events[md_label]["entry"] + self.num_events[md_label]["release"], \
               self.num_events[md_label]["entry"], self.num_events[md_label]["release"]

    def add_transport_event(self, md_label: str, path_id: str, event_type: str,
                            traced_event: Tuple[str, Tuple[int, int]], globally_unassigned: bool = True):
        """
        Store information on unassigned transport event
        :param md_label: name of folder with the MD simulation data that contain this transport event
        :param path_id: AQUA-DUCT path ID of this event
        :param event_type: type(release or entry) of this event
        :param traced_event: tuple containing identity of ligand responsible for this event,
                                and beginning and last frames of the event
        :param globally_unassigned: if the event is outlier for all simulations or just md_label
        """

        if event_type not in self.transport_events.keys():
            self.transport_events[event_type] = dict()

        if md_label not in self.transport_events[event_type].keys():
            self.transport_events[event_type][md_label] = list()

        self.transport_events[event_type][md_label].append((path_id, traced_event))

        stat_md_label = md_label
        if self.parameters["perform_comparative_analysis"] \
                and self.parameters["comparative_groups_definition"] is not None:
            membership = get_md_membership4groups(self.parameters["comparative_groups_definition"])
            if md_label in membership.keys():
                stat_md_label = membership[md_label]

        if stat_md_label not in self.num_events.keys():
            self.num_events[stat_md_label] = {"entry": 0, "release": 0}
        self.num_events[stat_md_label][event_type] += 1

        if globally_unassigned:
            self.num_events["overall"][event_type] += 1
            if event_type not in self.transport_events_global.keys():
                self.transport_events_global[event_type] = dict()

            if md_label not in self.transport_events_global[event_type].keys():
                self.transport_events_global[event_type][md_label] = list()

            self.transport_events_global[event_type][md_label].append((path_id, traced_event))

    def report_events_details(self, filename: str):
        """
        Save file with detailed information about unassigned transport events
        :param filename: output filename
        """

        details_file = os.path.join(self.parameters["super_cluster_details_folder"], filename)
        os.makedirs(os.path.dirname(details_file), exist_ok=True)
        with open(details_file, "w") as out_stream:
            out_stream.write("Unassigned transport events:\n")
            out_stream.write("Number of entry events = {:d}\n".format(self.num_events["overall"]["entry"]))
            out_stream.write("Number of release events = {:d}\n".format(self.num_events["overall"]["release"]))

            for event_type in sorted(self.transport_events_global.keys()):
                out_stream.write("{}: (from Simulation: AQUA-DUCT ID, (Resname:Residue), "
                                 "start_frame->end_frame; ... )\n".format(event_type))

                for md_label, paths in self.transport_events_global[event_type].items():
                    out_stream.write("from {}: ".format(md_label))
                    for path_id, traced_event in paths:
                        out_stream.write("{}, ({}), {}->{}; ".format(path_id, traced_event[0], traced_event[1][0],
                                                                     traced_event[1][1]))
                    out_stream.write("\n")

    def report_summary_line(self, widths: List[int], md_label: str = "overall") -> str:
        """
        Prepares information on transport events flagged as outliers for generation of summary of superclusters
        :param widths: column widths to match the format of the rest of the summary file
        :param md_label: summary of which simulations to report; by default report 'overall' stats
        :return: info on outlier events
        """

        return "{:{width1}s}{:{width2}d}, {:{width3}d}, " \
               "{:{width4}d}\n".format("Total number of unassigned events:", *self.count_events(md_label),
                                       width1=widths[0], width2=widths[1], width3=widths[2], width4=widths[3])

    def prepare_visualization(self, md_label: str = "overall") -> List[str]:
        """
        Generates lines for Pymol visualization script
        :param md_label: visualization of which simulations to prepare; by default 'overall' visualization
        :return: lines to load visualization of this SC into Pymol
        """

        root_folder = self.parameters["visualization_folder"]
        if "overall" not in md_label:
            root_folder = os.path.join(root_folder, "comparative_analysis", md_label)

        vis_folder = os.path.relpath(self.parameters["layered_aquaduct_vis_path"],
                                     root_folder)

        comparative_groups_definition = {}
        if self.parameters["perform_comparative_analysis"] \
                and self.parameters["comparative_groups_definition"] is not None:
            comparative_groups_definition = self.parameters["comparative_groups_definition"]

        plines = list()
        if "overall" not in md_label:
            events2process = self.transport_events
        else:
            events2process = self.transport_events_global

        for event_type in sorted(events2process.keys()):
            event_filenames = list()
            for _md_label, path_id in subsample_events(events2process[event_type],
                                                       self.parameters["random_seed"],
                                                       self.parameters["max_events_per_cluster4visualization"],
                                                       md_label, comparative_groups_definition):
                filename = os.path.join(vis_folder, _md_label, "paths",
                                        "wat_{}_{}_pathset.dump.gz".format(path_id, event_type))
                event_filenames.append("{}".format(utils.path_loader_string(filename)))

            if event_filenames:
                plines.append("events = [{}]\n".format(",\n".join(event_filenames)))
                plines.append("for event in events:\n")
                plines.append("    with gzip.open(event, 'rb') as in_stream:\n")
                plines.append("        pathset = pickle.load(in_stream)\n")
                plines.append("        for path in pathset:\n")
                plines.append("            path[3:6] = {}\n".format(utils.get_caver_color(None)))
                plines.append("            cmd.load_cgo(path, '{}_outlier')\n".format(event_type))
                plines.append("cmd.set('cgo_line_width', {}, '{}_outlier')\n\n".format(2, event_type))

        return plines


class TransportProcesses:
    def __init__(self, config: AnalysisConfig):
        """
        Class for analysis of transport processes and tunnels, contains 'global' work-flows using methods
        from SuperClusters (SC)
        :param config: configuration object with job parameters
        """

        self.parameters = config.get_parameters()
        self.caver_input_folders, self.traj_input_folders, self.aquaduct_input_folders = config.get_input_folders()
        self.reference_pdb_file = config.get_reference_pdb_file()
        self.transformation_folder = self.parameters["transformation_folder"]
        self._super_clusters: Dict[int, SuperCluster] = dict()
        self._prioritized_clusters: Dict[int, int] = dict()  # maps original and prioritized IDs of superclusters
        self._outlier_transport_events = OutlierTransportEvents(self.parameters)
        self._active_filters: Dict[str, Union[Tuple[float, float], float, int]] = define_filters()  # none active
        self._events_assigned = False
        self.vis_flag = 0
        self.filter_flag = 0
        self._aquaduct_single_event_inputs = False
        if self.parameters["start_from_stage"] == 1:
            logger.debug(str(config))  # log initial configuration

    def update_configuration(self, new_config: AnalysisConfig):
        """
        Updates parameters based on new job parameters
        :param new_config: object with job parameters
        """

        new_config.report_updates(self.parameters)
        self.parameters = new_config.get_parameters()
        self.caver_input_folders, self.traj_input_folders, self.aquaduct_input_folders = new_config.get_input_folders()
        self.reference_pdb_file = new_config.get_reference_pdb_file()
        self.transformation_folder = self.parameters["transformation_folder"]

        # update inside components
        self._outlier_transport_events.parameters.update(self.parameters)
        for supercluster in self._super_clusters.values():
            supercluster.parameters.update(self.parameters)

    @staticmethod
    def _calc_avg_cluster_distance(cls1: int, cls2: int, path_set1: LayeredPathSet, path_set2: LayeredPathSet,
                                   precision: int, cutoff: float) -> (int, int, float):
        """
        Computes average distance between the paths representing the two clusters
        :param cls1: order ID of evaluated cluster1 to enable its identification during distance matrix creation
        :param cls2: order ID of evaluated cluster2 to enable its identification during distance matrix creation
        :param path_set1: set of paths representing cluster1
        :param path_set2: set of paths representing cluster2
        :param precision: ith how many decimals are the calculated distances reported
        :param cutoff: clustering cutoff
        :return: cls1, cls2 and average distance between the two evaluated clusters
        """

        return cls1, cls2, np.around(path_set1.avg_distance2path_set(path_set2, cutoff), precision)

    def _compute_intercluster_distances(self, cluster_specifications: List[Tuple[str, int]],
                                        path_sets: Dict[Tuple[str, int], LayeredPathSet],
                                        precision: int = 4) -> np.array:
        """
        Computes inter-cluster distance matrix
        :param cluster_specifications: definition of clusters
        :param path_sets: sets of representative paths for all clusters
        :param precision: with how many decimals are the calculated distances reported
        :return: distance matrix for clustering
        """

        num_clusters = len(cluster_specifications)
        distance_matrix = np.full((num_clusters, num_clusters), np.inf)

        # parallel processing and progress monitoring related variables
        processing = list()
        n_jobs = int((num_clusters ** 2 - num_clusters) / 2)
        current_batch_size = 0
        done_calcs = 0
        jobs4batch = min(self.parameters["num_cpus"] * self.parameters["n_jobs_per_cpu_batch"], int(n_jobs / 10))
        progressbar(done_calcs, n_jobs)

        with Pool(processes=self.parameters["num_cpus"]) as pool:
            for cls1 in range(num_clusters):
                num_dists2compute = num_clusters - cls1 - 1

                # to avoid large memory allocation, keep processing queue moderately filled
                if jobs4batch <= current_batch_size + num_dists2compute and current_batch_size > 0:

                    # perform accumulated calculation jobs in this batch
                    for p in processing:
                        id1, id2, distance = p.get()
                        distance_matrix[id1, id2] = distance

                    done_calcs += len(processing)
                    progressbar(done_calcs, n_jobs)
                    processing = list()
                    current_batch_size = 0

                for cls2 in range(cls1 + 1, num_clusters):
                    cluster1_specification = cluster_specifications[cls1]
                    cluster2_specification = cluster_specifications[cls2]
                    processing.append(pool.apply_async(self._calc_avg_cluster_distance,
                                                       args=(cls1, cls2, path_sets[cluster1_specification],
                                                             path_sets[cluster2_specification], precision,
                                                             self.parameters["clustering_cutoff"])))
                current_batch_size += num_dists2compute

            # perform remaining calculation jobs
            for p in processing:
                id1, id2, distance = p.get()
                distance_matrix[id1, id2] = distance

            done_calcs += len(processing)
            progressbar(done_calcs, n_jobs)

        # complete the matrix
        for cls1 in range(num_clusters):
            distance_matrix[cls1, cls1] = 0
            for cls2 in range(cls1 + 1, num_clusters):
                distance_matrix[cls2, cls1] = distance_matrix[cls1, cls2]

        return distance_matrix

    def _does_super_cluster_exist(self, sc_id: int) -> bool:
        """
        Test if supercluster (SC) with given ID exists
        :param sc_id: ID of tested SC
        """

        return sc_id in self._super_clusters.keys()

    @staticmethod
    def _pre_process_single_tunnel_network(md_label: str, parameters: dict):
        """
        Transformation and visualization of original tunnel network for a single MD simulation
        :param md_label: name of folder with the source MD simulation data
        :param parameters: job configuration parameters
        """

        logger.debug("Processing a tunnel network from {}.".format(md_label))
        tunnel_network = TunnelNetwork(parameters, md_label)
        tunnel_network.read_tunnels_data()
        if parameters["visualize_transformed_tunnels"]:
            tunnel_network.save_orig_network_visualization()
        tunnel_network.save_orig_network()

    def _prioritize_super_clusters(self):
        """
        Sort superclusters (SCs) by their priority; stored in prioritized_sc_id of individual SC;
        and mapping of original sc_id to prioritized is stored in self._prioritized_clusters
        """

        with TimeProcess("Prioritization"):
            self._prioritized_clusters = dict()
            logger.debug("Sorting superclusters by their priority")

            sc_order = list()
            for sc_id, super_cluster in self._super_clusters.items():
                if super_cluster.has_passed_filter(consider_transport_events=self._events_assigned,
                                                   active_filters=self._active_filters):
                    sc_order.append((super_cluster.properties["overall"]["priority"], sc_id))

            sc_order = [sc_id[1] for sc_id in sorted(sc_order, reverse=True)]

            for new_id, old_id in enumerate(sc_order):
                self._super_clusters[old_id].prioritized_sc_id = new_id + 1
                self._prioritized_clusters[new_id + 1] = old_id

    def _report_super_cluster_details(self, filename: str):
        """
        Save file with information about tunnel clusters forming superclusters (SCs) and assigned transport events
        :param filename: output filename
        """

        details_file = os.path.join(self.parameters["super_cluster_details_folder"], filename)
        os.makedirs(os.path.dirname(details_file), exist_ok=True)
        with open(details_file, "w") as out_stream:
            out_stream.write("Tunnel clusters and transport events assignment to superclusters "
                             "(ordered by their priority):\n")

            for prio_sc in sorted(self._prioritized_clusters.keys()):
                out_stream.write("-" * 120 + "\n")
                sc_id = self._prioritized_clusters[prio_sc]
                out_stream.write(self._super_clusters[sc_id].report_details(self._events_assigned))

            out_stream.write("-" * 120 + "\n")

    def _report_filters(self) -> str:
        """
        Report currently active filters and pre-filters used during selection of tunnels for layering
        :return: string containing info on active filters and pre-filters
        """

        pr_min_length, pr_max_length = self._active_filters["length"]
        pr_min_radius, pr_max_radius = self._active_filters["radius"]
        pr_min_curvature, pr_max_curvature = self._active_filters["curvature"]
        min_sims_num = self._active_filters["min_sims_num"]
        min_snapshots_num = self._active_filters["min_snapshots_num"]
        min_avg_snapshots_num = self._active_filters["min_avg_snapshots_num"]
        pr_min_transport_events = self._active_filters["min_transport_events"]
        pr_min_entry_events = self._active_filters["min_entry_events"]
        pr_min_release_events = self._active_filters["min_release_events"]

        msg = "\nParameters used for pre-selection of input tunnels for clustering:\n"
        msg += "length >= {:.2f} A\n".format(self.parameters["min_tunnel_length4clustering"])
        msg += "radius >= {:.2f} A\n".format(self.parameters["min_tunnel_radius4clustering"])
        msg += "curvature <= {:.2f}\n\n".format(self.parameters["max_tunnel_curvature4clustering"])

        msg += "Active tunnel filters:\n"
        msg += "length = ({:.2f}, {:.2f}) A\n".format(pr_min_length, pr_max_length)
        msg += "radius = ({:.2f}, {:.2f}) A\n".format(pr_min_radius, pr_max_radius)
        msg += "curvature = ({:.2f}, {:.2f})\n".format(pr_min_curvature, pr_max_curvature)
        msg += "occurred in at least {:d} simulations\n".format(min_sims_num)
        msg += "occurred in at least {:d} snapshots\n".format(min_snapshots_num)
        msg += "occurred in at least {:.3f} snapshots on average per simulation\n".format(min_avg_snapshots_num)

        if self._events_assigned:  # transport event filters active
            msg += "has at least {:d} transport events\n".format(pr_min_transport_events)
            msg += "has at least {:d} entry events\n".format(pr_min_entry_events)
            msg += "has at least {:d} release events\n".format(pr_min_release_events)

        return msg

    def _precompute_cumulative_super_cluster_data(self):
        """
        Pre-compute superclusters (SCs) data - creating a single PathSet per SC and compute its overall direction
        """

        with Pool(processes=self.parameters["num_cpus"]) as pool:
            processing = list()
            for super_cluster in self._super_clusters.values():
                if super_cluster.avg_direction is None:
                    processing.append(pool.apply_async(super_cluster.compute_space_descriptors))

            items2process = len(processing)
            if items2process:
                logger.info("Aggregating supercluster data for {:d} superclusters "
                            "using {:d} {}:".format(items2process, self.parameters["num_cpus"],
                                                    process_count(self.parameters["num_cpus"])))

                progressbar(0, items2process)
                for i, p in enumerate(processing):
                    sc_id, avg_direction = p.get()
                    self._super_clusters[sc_id].avg_direction = avg_direction
                    del self._super_clusters[sc_id].tunnel_clusters  # remove extensive data
                    progressbar(i + 1, items2process)

    def assign_transport_events(self, md_labels: Optional[List[str]] = None):
        """
        Finds superclusters (SCs) through evaluated transport event happened, and assigns remaining events as outliers;
        this changes  self._events_assigned = True, enabling consideration of transport events during filtering
        :param md_labels: list to restrict assignment to particular Networks (simulations) only
        """

        self._events_assigned = False

        with TimeProcess("Assignment"):
            # assemble transport events
            path_sets = dict()
            if md_labels is None:
                folders2process = self.aquaduct_input_folders
            else:
                folders2process = [*set(md_labels).intersection(self.aquaduct_input_folders)]

            for md_label in folders2process:
                aquanet = AquaductNetwork(self.parameters, md_label, load_only=True)
                aquanet.load_layered_network()
                for event_label, layered_path_set in aquanet.layered_entities.items():
                    event_specification = (md_label, event_label, layered_path_set.traced_event)
                    path_sets[event_specification] = layered_path_set

            items2process = len(path_sets.keys())

            if items2process:
                # assign transport events
                logger.info("Assigning {:d} transport events to {:d} superclusters "
                            "using {:d} {}:".format(items2process, self.enumerate_valid_super_clusters(),
                                                    self.parameters["num_cpus"],
                                                    process_count(self.parameters["num_cpus"])))
                logger.debug(self._report_filters())

                self._outlier_transport_events = OutlierTransportEvents(self.parameters)

                with Pool(processes=self.parameters["num_cpus"]) as pool:
                    processing = list()
                    progressbar(0, items2process)
                    for event_specification, event_path_set in path_sets.items():
                        event_assigner = EventAssigner(self.parameters, event_specification, event_path_set,
                                                       self._super_clusters, self._active_filters)

                        processing.append(pool.apply_async(event_assigner.perform_assignment))

                    for i, p in enumerate(processing):
                        event_specification, assigned_sc_ids, max_buriedness, max_depth = p.get()
                        md_label = event_specification[0]
                        event_path_id, event_type = event_specification[1].split("_")
                        traced_event = event_specification[2]

                        if assigned_sc_ids is None:
                            logger.debug("Assigned transport event '{}' is not buried inside any supercluster"
                                         " (the highest buried_ratio was {:.2f})".format(event_specification,
                                                                                         max_buriedness))
                            self._outlier_transport_events.add_transport_event(md_label, event_path_id, event_type,
                                                                               traced_event)
                        else:

                            for assigned_sc_id in assigned_sc_ids:
                                logger.debug("Assigned transport event '{}' is buried inside the supercluster '{:d}' "
                                             "(buried_ratio = {:.2f}, "
                                             "penetration depth = {:.2f})".format(event_specification, assigned_sc_id,
                                                                                  max_buriedness, max_depth))

                                md_label_assigned = self._super_clusters[assigned_sc_id].add_transport_event(md_label,
                                                                                                             event_path_id,
                                                                                                             event_type,
                                                                                                             traced_event)
                                if not md_label_assigned:
                                    self._outlier_transport_events.add_transport_event(md_label, event_path_id,
                                                                                       event_type, traced_event,
                                                                                       globally_unassigned=False)

                        progressbar(i + 1, items2process)

                self._events_assigned = True

                if self._outlier_transport_events.exist():
                    logger.info("{:d} transport events were not assigned to superclusters.".format(
                                self._outlier_transport_events.count_events()[0]))
                else:
                    logger.info("All transport events were assigned to some supercluster.")

            else:
                logger.info("There are no transport events to assign.")

            self._report_super_cluster_details("initial_super_cluster_events_details.txt")
            if self._outlier_transport_events.exist():
                self._outlier_transport_events.report_events_details("outlier_transport_events_details.txt")

    def clear_results(self, overwrite: bool = False):
        """
        Removes output folder
        :param overwrite: if to perform the cleaning of non empty folder
        """

        from shutil import rmtree
        output_folder_keys = ["internal_folder", "data_folder", "visualization_folder", "statistics_folder"]

        for folder_key in output_folder_keys:
            output_path = self.parameters[folder_key]
            logger.debug("Cleaning content of results folder '{}'".format(output_path))
            if not overwrite and os.path.exists(output_path) and os.listdir(output_path):
                raise RuntimeError("Error output folder '{}' exists and is not empty. Specify different name in "
                                   "'output_path' parameter or enable overwrite option by using '--overwrite' option "
                                   "or setting 'overwrite' parameter to True".format(output_path))
            else:
                rmtree(output_path, True)

    def _save_distance_matrix(self, distance_matrix: np.array, cluster_specifications: List[Tuple[str, int]],
                              cluster_characteristics: Dict[Tuple[str, int], Tuple[float, int]]):
        """
        Save pair-wise cluster distance matrix and cluster specifications to files
        :param distance_matrix: distance matrix
        :param cluster_specifications: definition of clusters
        :param cluster_characteristics: data on cluster throughputs and number of tunnels
        """

        os.makedirs(self.parameters["clustering_folder"], exist_ok=True)
        np.save(os.path.join(self.parameters["clustering_folder"], "caver_clusters_clustering_matrix.npy"),
                distance_matrix)

        with open(os.path.join(self.parameters["clustering_folder"], "cluster_specifications.dump"), "wb") as out:
            pickle.dump((cluster_specifications, cluster_characteristics), out)

        if self.parameters["save_distance_matrix_csv"]:  # safe CSV formatted matrix

            # get natural order of labels
            natural_order_map = dict()
            for matrix_pos, cls_spec in enumerate(cluster_specifications):
                natural_order_map[cls_spec] = matrix_pos

            reordered_cluster_specifications = sorted(natural_order_map.keys())

            # define maximal label lengths for CSV formatting
            label_lengths1 = [1]  # initialize with the minimum label length
            label_lengths2 = [1]  # initialize with the minimum label length
            for cls_spec in reordered_cluster_specifications:
                label_lengths1.append(len(cls_spec[0]))
                label_lengths2.append(len(str(cls_spec[1])))
            label_length1 = max(label_lengths1)
            label_length2 = max(label_lengths2) + 4

            # save matrix
            os.makedirs(os.path.dirname(self.parameters["distance_matrix_csv_file"]), exist_ok=True)
            with open(self.parameters["distance_matrix_csv_file"], "w") as out_stream:
                line = " " * (label_length1+label_length2) + ", "
                for cls_spec in reordered_cluster_specifications[:-1]:
                    label1 = "{:>{label_length}s}:".format(cls_spec[0], label_length=label_length1)
                    label2 = "cls{}".format(cls_spec[1])
                    line += "{}{:>{label_length}s}, ".format(label1, label2, label_length=label_length2)

                label1 = "{:>{label_length}s}:".format(reordered_cluster_specifications[-1][0],
                                                       label_length=label_length1)
                label2 = "cls{}".format(reordered_cluster_specifications[-1][1])
                line += "{}{:>{label_length}s}\n".format(label1, label2, label_length=label_length2)
                out_stream.write(line)

                for cls_spec1 in reordered_cluster_specifications:
                    label1 = "{:>{label_length}s}:".format(cls_spec1[0], label_length=label_length1)
                    label2 = "cls{}".format(cls_spec1[1])
                    line = "{}{:>{label_length}s}, ".format(label1, label2, label_length=label_length2)

                    matrix_pos1 = natural_order_map[cls_spec1]
                    for cls_spec2 in reordered_cluster_specifications[:-1]:
                        matrix_pos2 = natural_order_map[cls_spec2]
                        line += "{:>{label_length}.3f}, ".format(distance_matrix[matrix_pos1, matrix_pos2],
                                                                 label_length=(label_length1 + label_length2))

                    matrix_pos2 = natural_order_map[reordered_cluster_specifications[-1]]
                    line += "{:>{label_length}.3f}\n".format(distance_matrix[matrix_pos1, matrix_pos2],
                                                             label_length=(label_length1 + label_length2))
                    out_stream.write(line)

    def _load_distance_matrix(self) -> Tuple[np.array, List[Tuple[str, int]], Dict[Tuple[str, int], Tuple[float, int]]]:
        """
        Loads distance matrix and cluster_specifications from files
        :return: distance matrix, cluster_specifications, cluster_characteristics
        """

        with open(os.path.join(self.parameters["clustering_folder"], "cluster_specifications.dump"), "rb") as in_stream:
            cluster_specifications, cluster_characteristics = pickle.load(in_stream)

        matrix = np.load(os.path.join(self.parameters["clustering_folder"], "caver_clusters_clustering_matrix.npy"))

        return matrix, cluster_specifications, cluster_characteristics

    def compute_tunnel_clusters_distances(self):
        """
        Compute pairwise cluster-cluster distances, and save their matrix
        """

        with TimeProcess("Cluster-cluster distances calculation"):
            # collect clusters from all analyzed MD simulations & order clusters according to their importance,
            # e.g., cluster ID, number of tunnels, and their mean throughput

            path_sets = dict()
            ordered_clusters = dict()
            cluster_characteristics = dict()

            for md_label in self.caver_input_folders:
                tunnel_network = TunnelNetwork(self.parameters, md_label)
                tunnel_network.load_layered_network()

                for cls_id, layered_path_set in tunnel_network.layered_entities.items():
                    cluster_specification = (md_label, cls_id)
                    avg_throughput, num_tunnels = layered_path_set.characteristics
                    cluster_characteristics[cluster_specification] = layered_path_set.characteristics
                    importance_key = (cls_id, 1 / avg_throughput, 1 / num_tunnels)
                    path_sets[cluster_specification] = layered_path_set
                    ordered_clusters[importance_key] = cluster_specification

            cluster_specifications = [ordered_clusters[x] for x in sorted(ordered_clusters.keys())]
            if len(cluster_specifications) <= 1:
                raise RuntimeError("Not enough tunnel clusters are available to perform calculate their distances")

            logger.info("Computing distances for {:d} tunnel clusters "
                        "using {:d} {}:".format(len(cluster_specifications), self.parameters["num_cpus"],
                                                process_count(self.parameters["num_cpus"])))

            distance_matrix = self._compute_intercluster_distances(cluster_specifications, path_sets)
            self._save_distance_matrix(distance_matrix, cluster_specifications, cluster_characteristics)

    def merge_tunnel_clusters2super_clusters(self):
        """
        Performs clustering of tunnel clusters and creates of their superclusters (SCs)
        """

        total_num_md_sims = len(self.caver_input_folders)
        with TimeProcess("Clustering"):
            logger.info("Clustering tunnel clusters into superclusters using {}-linkage agglomerative clustering "
                        "with distance cutoff {:.2f} A:".format(self.parameters["clustering_linkage"],
                                                                self.parameters["clustering_cutoff"]))

            # load and transform distance_matrix to condensed matrix of pairwise dissimilarities
            distance_matrix, cluster_specifications, cluster_characteristics = self._load_distance_matrix()
            condensed_matrix = distance_matrix[np.triu_indices(distance_matrix.shape[0], 1)]

            # cluster the tunnel clusters :)
            linkage_matrix = fastcluster.linkage(condensed_matrix, method=self.parameters["clustering_linkage"])
            cluster_labels = fcluster(linkage_matrix, t=self.parameters["clustering_cutoff"], criterion='distance')
            cluster_specs2label = dict(zip(cluster_specifications, cluster_labels))
            logger.info("Identified {:d} superclusters.".format(np.unique(cluster_labels).size))

            label2cluster_specs = dict()
            for cls_spec, cls_label in cluster_specs2label.items():
                if cls_label not in label2cluster_specs.keys():
                    label2cluster_specs[cls_label] = list()
                label2cluster_specs[cls_label].append(cls_spec)

            # order the labels according to the number of tunnels and their throughput to mimic later prioritization
            super_cluster_importance = dict()
            for label, cluster_specs in label2cluster_specs.items():
                throughput, num_tunnels = 0, 0
                md_sims = set()
                for cluster_spec in cluster_specs:
                    throughput += cluster_characteristics[cluster_spec][0]
                    num_tunnels += cluster_characteristics[cluster_spec][1]
                    md_sims.add(cluster_spec[0])
                super_cluster_importance[label] = (num_tunnels / self.parameters["snapshots_per_simulation"]) * \
                                                  (throughput / len(cluster_specs)) * (len(md_sims) / total_num_md_sims)

            label_order = {x[0]: i + 1 for i, x in enumerate(sorted(super_cluster_importance.items(),
                                                                    key=lambda cls_order: (cls_order[1]),
                                                                    reverse=True))}

            # reset status
            self._super_clusters: Dict[int, SuperCluster] = dict()
            self._outlier_transport_events = OutlierTransportEvents(self.parameters)
            self._events_assigned = False

            # create superclusters
            for md_label in self.caver_input_folders:
                tunnel_network = TunnelNetwork(self.parameters, md_label)
                tunnel_network.load_layered_network()
                for cls_id, layered_path_set in tunnel_network.layered_entities.items():
                    cluster_specification = (md_label, cls_id)
                    sc_id = label_order[cluster_specs2label[cluster_specification]]
                    if self._does_super_cluster_exist(sc_id):
                        self._super_clusters[sc_id].add_caver_cluster(md_label, int(cls_id), layered_path_set)
                    else:
                        self._super_clusters[sc_id] = SuperCluster(sc_id, self.parameters, total_num_md_sims)
                        self._super_clusters[sc_id].add_caver_cluster(md_label, int(cls_id), layered_path_set)

            self._precompute_cumulative_super_cluster_data()

    def compute_transformations(self):
        """
        Compute transformation matrices that are used to align all data
        """

        with TimeProcess("Computation"):
            os.makedirs(os.path.join(self.transformation_folder, self.parameters["caver_foldername"]), exist_ok=True)
            os.makedirs(os.path.join(self.transformation_folder, self.parameters["aquaduct_foldername"]), exist_ok=True)

            logger.info("Computing transformation to unify the simulations coordinate systems.")
            reference_tunnel_transform_mat = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

            logger.info("Aligning to reference file {}.".format(self.reference_pdb_file))

            progress_counter = 0
            num_folders2process = 2 * len(self.caver_input_folders) + len(self.aquaduct_input_folders)
            progressbar(progress_counter, num_folders2process)

            # compute transformation matrices from caver PDB files
            num_raw_paths = 0
            with Pool(processes=self.parameters["num_cpus"]) as pool:
                processing = list()
                tunnel_transform_mat = dict()
                for md_label in self.caver_input_folders:
                    md_folder = os.path.join(self.parameters["caver_results_path"], md_label)
                    in_pdb_file = utils.get_filepath(md_folder, self.parameters["caver_relative_pdb_file"])
                    processing.append(pool.apply_async(get_transform_matrix,
                                                       args=(in_pdb_file, self.reference_pdb_file, md_label)))

                for p in processing:
                    matrix, md_label = p.get()
                    tunnel_transform_mat[md_label] = matrix
                    progress_counter += 1
                    progressbar(progress_counter, num_folders2process)

                # compute overall average starting point across all MD simulations
                processing = list()
                transformed_average_sp_x = list()
                transformed_average_sp_y = list()
                transformed_average_sp_z = list()
                for md_label in self.caver_input_folders:
                    md_folder = os.path.join(self.parameters["caver_results_path"], md_label)
                    in_origin_file = utils.get_filepath(md_folder, self.parameters["caver_relative_origin_file"])
                    processing.append(pool.apply_async(average_starting_point, args=(in_origin_file, md_label)))

                for i, p in enumerate(processing):
                    md_original_sp, md_label = p.get()
                    transformed_sp = tunnel_transform_mat[md_label].dot(md_original_sp)
                    transformed_average_sp_x.append(transformed_sp[0])
                    transformed_average_sp_y.append(transformed_sp[1])
                    transformed_average_sp_z.append(transformed_sp[2])
                    progress_counter += 1
                    progressbar(progress_counter, num_folders2process)

                overall_starting_point = (np.average(transformed_average_sp_x), np.average(transformed_average_sp_y),
                                          np.average(transformed_average_sp_z))

                del transformed_average_sp_x
                del transformed_average_sp_y
                del transformed_average_sp_z
                logger.debug("Average starting point in network: {}".format(overall_starting_point))
                transform_mat2starting_point = np.array([
                    [0, 0, 0, -overall_starting_point[0]],
                    [0, 0, 0, -overall_starting_point[1]],
                    [0, 0, 0, -overall_starting_point[2]],
                    [0, 0, 0, 0]])

                # compute transformation matrices from AquaDuct PDB files
                processing = list()
                aquaduct_transform_mat = dict()
                for md_label in self.aquaduct_input_folders:
                    md_folder = os.path.join(self.parameters["aquaduct_results_path"], md_label)
                    tar_file = utils.get_filepath(md_folder, self.parameters["aquaduct_results_relative_tarfile"])
                    processing.append(pool.apply_async(transform_aquaduct,
                                                       args=(md_label, tar_file,
                                                             self.parameters["aquaduct_results_pdb_filename"],
                                                             self.reference_pdb_file)))

                for i, p in enumerate(processing):
                    matrix, md_label, _num_raw_paths = p.get()
                    num_raw_paths += _num_raw_paths
                    aquaduct_transform_mat[md_label] = matrix
                    progress_counter += 1
                    progressbar(progress_counter, num_folders2process)

            if self.aquaduct_input_folders and (num_raw_paths / len(self.aquaduct_input_folders)) \
                    <= self.parameters["num_cpus"]:
                self._aquaduct_single_event_inputs = True
            # compute general transformation matrix to have unified orientation of MD simulations less dependent on
            # the selection of reference PDB file
            trans_ref_pdb_file = os.path.join(self.transformation_folder, "ref.pdb")
            transform_pdb_file(self.reference_pdb_file, trans_ref_pdb_file,
                               reference_tunnel_transform_mat + transform_mat2starting_point)
            general_transform_mat = get_general_rot_mat_from_2_ca_atoms(trans_ref_pdb_file)
            os.remove(trans_ref_pdb_file)

            # save transformed reference pdb file
            transform_pdb_file(self.reference_pdb_file, os.path.join(self.transformation_folder, "ref_transformed.pdb"),
                               general_transform_mat.dot(reference_tunnel_transform_mat + transform_mat2starting_point))

            # save transformation matrices
            for md_label in self.caver_input_folders:
                tunnel_full_trans_mat = general_transform_mat.dot(tunnel_transform_mat[md_label] +
                                                                  transform_mat2starting_point)
                with open(os.path.join(self.transformation_folder, self.parameters["caver_foldername"],
                                       md_label + "-transform_mat.dump"), "wb") as out:
                    pickle.dump(tunnel_full_trans_mat, out)

            for md_label in self.aquaduct_input_folders:
                aquaduct_full_trans_mat = general_transform_mat.dot(aquaduct_transform_mat[md_label] +
                                                                    transform_mat2starting_point)
                with open(os.path.join(self.transformation_folder, self.parameters["aquaduct_foldername"],
                                       md_label + "-transform_mat.dump"), "wb") as out:
                    pickle.dump(aquaduct_full_trans_mat, out)

    def process_tunnel_networks(self):
        """
        Process tunnel networks for all MD simulations
        """

        logger.info("Processing {:d} tunnel networks "
                    "using {:d} {}:".format(len(self.caver_input_folders), self.parameters["num_cpus"],
                                            process_count(self.parameters["num_cpus"])))
        logger.debug("The networks are read from sub-folders in '{}' folder that match the following pattern "
                     "'{}'.".format(self.parameters["caver_results_path"],
                                    self.parameters["caver_results_folder_pattern"]))
        logger.debug("Using the following reference file to align caver clusters: '{}'".format(self.reference_pdb_file))

        with TimeProcess("Processing"):
            with Pool(processes=self.parameters["num_cpus"]) as pool:
                processing = list()
                for md_label in self.caver_input_folders:
                    processing.append(pool.apply_async(self._pre_process_single_tunnel_network,
                                                       args=(md_label, self.parameters)))

                items2process = len(processing)
                progressbar(0, items2process)
                for i, p in enumerate(processing):
                    p.get()
                    progressbar(i + 1, items2process)

    def create_layered_description4tunnel_networks(self):
        """
        Creates layered representation of tunnel networks for all MD simulations
        """

        with TimeProcess("Layering"):
            with Pool(processes=self.parameters["num_cpus"]) as pool:
                processing = list()
                cls_ids2process4md_label = dict()
                tunnel_networks = dict()
                for md_label in self.caver_input_folders:
                    tunnel_networks[md_label] = TunnelNetwork(self.parameters, md_label)

                # assemble clusters for layering
                for md_label in self.caver_input_folders:
                    tunnel_network = TunnelNetwork(self.parameters, md_label)
                    tunnel_network.load_orig_network()
                    cls_ids2process4md_label[md_label] = list()
                    for cluster in tunnel_network.get_clusters4layering():
                        processing.append(pool.apply_async(cluster.create_layered_cluster))
                        cls_ids2process4md_label[md_label].append(cluster.cluster_id)

                items2process = len(processing)
                if not items2process > 0:
                    raise RuntimeError("Not enough tunnel clusters are available to perform their layering")

                logger.info("Computing layered representation for {:d} tunnel clusters "
                            "using {:d} {}:".format(items2process, self.parameters["num_cpus"],
                                                    process_count(self.parameters["num_cpus"])))

                progressbar(0, items2process)
                for i, p in enumerate(processing):
                    cls_id, md_label, layered_path_set = p.get()
                    if layered_path_set.is_empty():
                        logger.warning("Cluster {} of {} cannot be layered".format(cls_id, md_label))
                        cls_ids2process4md_label[md_label].remove(cls_id)  # do not include in completeness testing
                    else:
                        tunnel_networks[md_label].add_layered_entity(cls_id, layered_path_set)

                    # save complete layered network
                    if tunnel_networks[md_label].is_layering_complete(cls_ids2process4md_label[md_label]):
                        logger.debug("Finished layering of network for '{}'.".format(md_label))
                        if self.parameters["visualize_layered_clusters"]:
                            tunnel_networks[md_label].save_layered_visualization(save_pdb_files=True)
                        tunnel_networks[md_label].save_layered_network()
                        del tunnel_networks[md_label]

                    progressbar(i + 1, items2process)

    @staticmethod
    def _pre_process_single_aquaduct_network(md_label: str, parameters: dict, parallel_processing: bool = True):
        """
        Transformation and visualization of original AQUA-DUCT network for a single MD simulation
        :param md_label: name of folder with the source MD simulation data
        :param parameters: job configuration parameters
        :param parallel_processing: if we process the raw_paths in parallel
        """

        logger.debug("Processing an AQUA-DUCT network from {}.".format(md_label))
        with AquaductNetwork(parameters, md_label) as aquanet:
            aquanet.read_raw_paths_data(parallel_processing)
            if parameters["visualize_transformed_transport_events"]:
                aquanet.save_orig_network_visualization()
            aquanet.save_orig_network()

    def process_aquaduct_networks(self):
        """
        Process AQUA-DUCT networks for all MD simulations
        """

        if not self.aquaduct_input_folders:
            raise RuntimeError("No data with AQUA-DUCT results were loaded to enable their analyses")

        items2process = len(self.aquaduct_input_folders)
        logger.info("Processing {:d} AQUA-DUCT networks "
                    "using {:d} {}:".format(items2process, self.parameters["num_cpus"],
                                            process_count(self.parameters["num_cpus"])))
        logger.debug("The networks are read from sub-folders in '{}' folder that match the following pattern "
                     "'{}'.".format(self.parameters["aquaduct_results_path"],
                                    self.parameters["aquaduct_results_folder_pattern"]))
        logger.debug("Using the following reference file to "
                     "align AQUA-DUCT networks: '{}'".format(self.reference_pdb_file))

        if not items2process > 0:
            raise RuntimeError("Not enough AQUA-DUCT networks are available to perform their layering")

        with TimeProcess("Processing"):
            progressbar(0, items2process)
            if self._aquaduct_single_event_inputs:
                with Pool(processes=self.parameters["num_cpus"]) as pool:
                    processing = list()
                    for md_label in self.aquaduct_input_folders:
                        processing.append(pool.apply_async(self._pre_process_single_aquaduct_network,
                                                           args=(md_label, self.parameters, False)))

                    for i, p in enumerate(processing):
                        p.get()
                        progressbar(i + 1, items2process)
            else:
                for i, md_label in enumerate(self.aquaduct_input_folders):
                    self._pre_process_single_aquaduct_network(md_label, self.parameters, True)
                    progressbar(i + 1, items2process)

    def create_layered_description4aquaduct_networks(self):
        """
        Creates layered representation of AquaDuct networks for all MD simulations
        """

        with TimeProcess("Layering"):
            with Pool(processes=self.parameters["num_cpus"]) as pool:
                processing = list()
                event_ids2process4md_label = dict()
                aqua_networks = dict()
                for md_label in self.aquaduct_input_folders:
                    aqua_networks[md_label] = AquaductNetwork(self.parameters, md_label, load_only=True)

                # assemble events for layering
                for md_label in self.aquaduct_input_folders:
                    aquanet = AquaductNetwork(self.parameters, md_label, load_only=True)
                    aquanet.load_orig_network()
                    event_ids2process4md_label[md_label] = list()
                    for event in aquanet.get_events4layering():
                        processing.append(pool.apply_async(event.create_layered_event))
                        event_ids2process4md_label[md_label].append(event.entity_label)

                items2process = len(processing)
                logger.info("Computing layered representation for {:d} transport events "
                            "using {:d} {}:".format(items2process, self.parameters["num_cpus"],
                                                    process_count(self.parameters["num_cpus"])))

                if not items2process > 0:
                    raise RuntimeError("Not enough transport events are available to perform their layering")

                progressbar(0, items2process)
                for i, p in enumerate(processing):
                    event_id, md_label, layered_path_set = p.get()
                    if layered_path_set.is_empty():
                        logger.warning("Event {} of {} cannot be layered".format(event_id, md_label))
                        event_ids2process4md_label[md_label].remove(event_id)  # do not include in completeness testing
                    else:
                        aqua_networks[md_label].add_layered_entity(event_id, layered_path_set)

                    # save complete layered network
                    if aqua_networks[md_label].is_layering_complete(event_ids2process4md_label[md_label]):
                        logger.debug("Finished layering of network for '{}'.".format(md_label))
                        # always saving layered visualizations to enable visualization of assigned events later
                        aqua_networks[md_label].get_pdb_file()
                        aqua_networks[md_label].save_layered_visualization(self.parameters["visualize_layered_events"])
                        aqua_networks[md_label].save_layered_network()
                        aqua_networks[md_label].clean_tempfile()
                        del aqua_networks[md_label]

                    progressbar(i + 1, items2process)

    def create_super_cluster_profiles(self):
        """
        Parallel merging of caver tunnel profiles into new ones for superclusters (SCs), and saving SC details
        """

        with TimeProcess("Creating"):
            os.makedirs(os.path.join(self.parameters["super_cluster_profiles_folder"], "initial"), exist_ok=True)
            with Pool(processes=self.parameters["num_cpus"]) as pool:
                logger.info("Creating {:d} supercluster tunnel profiles "
                            "using {:d} {}:".format(len(self._super_clusters.keys()), self.parameters["num_cpus"],
                                                    process_count(self.parameters["num_cpus"])))
                logger.debug(self._report_filters())

                processing = list()
                for super_cluster in self._super_clusters.values():
                    processing.append(pool.apply_async(super_cluster.process_cluster_profile))

                if not len(processing) > 0:
                    raise RuntimeError("Not enough superclusters are available to create their profiles")

                progressbar(0, len(processing))
                for i, p in enumerate(processing):
                    sc_id, sc_properties, sc_residues_freq, retained_tunnel_clusters = p.get()
                    # assign initial computed properties of SC to this SC
                    self._super_clusters[sc_id].set_properties(sc_properties)
                    self._super_clusters[sc_id].set_bottleneck_residue_freq(sc_residues_freq)
                    self._super_clusters[sc_id].update_caver_clusters_validity(retained_tunnel_clusters)
                    progressbar(i + 1, len(processing))

            self._prioritize_super_clusters()
            self._report_super_cluster_details("initial_super_cluster_details.txt")

    def enumerate_valid_super_clusters(self, consider_transport_events: bool = False) -> int:
        """
        Counts how many superclusters (SCs) are valid
        :param consider_transport_events: if related filters related to transport events should be considered
        :return: number of valid SCs
        """

        num_valid_super_clusters = 0

        for super_cluster in self._super_clusters.values():
            if super_cluster.has_passed_filter(consider_transport_events, active_filters=self._active_filters):
                num_valid_super_clusters += 1
        return num_valid_super_clusters

    def filter_super_cluster_profiles(self, min_length: float = -1, max_length: float = -1,
                                      min_bottleneck_radius: float = -1, max_bottleneck_radius: float = -1,
                                      min_curvature: float = -1, max_curvature: float = -1,
                                      min_sims_num: int = 1, min_snapshots_num: int = 1,
                                      min_avg_snapshots_num: float = -1, min_total_events: int = -1,
                                      min_entry_events: int = -1, min_release_events: int = -1):
        """
        Performs parallel filtering of superclusters (SC) profiles using defined filters, and saves SC details
        NOTE: -1 => filter is not active
        :param min_length: minimum tunnel length
        :param max_length: maximum tunnel length
        :param min_bottleneck_radius: minimum tunnel bottleneck radius
        :param max_bottleneck_radius: maximum tunnel bottleneck radius
        :param min_curvature: minimum tunnel curvature
        :param max_curvature: maximum tunnel curvature
        :param min_sims_num: present in minimum number of MD simulations
        :param min_snapshots_num: present in minimum number of snapshots
        :param min_avg_snapshots_num: present in minimum number of snapshots on average
        :param min_total_events: having minimum transport events
        :param min_entry_events: having minimum entry events
        :param min_release_events: having minimum release events
        """

        self.filter_flag += 1
        with TimeProcess("Filtering"):
            os.makedirs(os.path.join(self.parameters["super_cluster_profiles_folder"],
                                     "filtered{:02d}".format(self.filter_flag)), exist_ok=True)
            with Pool(processes=self.parameters["num_cpus"]) as pool:
                logger.info("Filtering {:d} supercluster profiles "
                            "using {:d} {}:".format(self.enumerate_valid_super_clusters(), self.parameters["num_cpus"],
                                                    process_count(self.parameters["num_cpus"])))
                self._active_filters = define_filters(min_length, max_length, min_bottleneck_radius,
                                                      max_bottleneck_radius, min_curvature, max_curvature, min_sims_num,
                                                      min_snapshots_num, min_avg_snapshots_num, min_total_events,
                                                      min_entry_events, min_release_events)
                logger.debug(self._report_filters())

                processing = list()

                for super_cluster in self._super_clusters.values():
                    super_cluster.properties = dict()
                    processing.append(pool.apply_async(super_cluster.filter_super_cluster,
                                                       args=(self._events_assigned, self._active_filters,
                                                             self.filter_flag)))
                num_retained_sc = 0
                if not len(processing) > 0:
                    raise RuntimeError("Not enough superclusters are available to filter their profiles")

                progressbar(0, len(processing))
                for i, p in enumerate(processing):
                    # SC properties cannot be directly assigned within the object since SC are copied during processing
                    sc_id, sc_properties, sc_residues_freq, retained_tunnel_clusters = p.get()
                    self._super_clusters[sc_id].set_properties(sc_properties)  # update with new properties
                    self._super_clusters[sc_id].set_bottleneck_residue_freq(sc_residues_freq)
                    self._super_clusters[sc_id].update_caver_clusters_validity(retained_tunnel_clusters)
                    if self._super_clusters[sc_id].has_passed_filter(consider_transport_events=self._events_assigned,
                                                                     active_filters=self._active_filters):
                        num_retained_sc += 1

                    progressbar(i + 1, len(processing))

            logger.info("{:d} superclusters kept after filtering.".format(num_retained_sc))
            self._prioritize_super_clusters()
            self._report_super_cluster_details("filtered_super_cluster_details{}.txt".format(self.filter_flag))

    def generate_super_cluster_summary(self, out_filename: str = "super_cluster_statistics.csv"):
        """
        Generates summary file with superclusters statistics.
        :param out_filename: name of file to save the summary to
        """

        with TimeProcess("Statistics generation"):
            logger.info("Generating statistics for superclusters.")
            no_md_snapshots = self.parameters["snapshots_per_simulation"]
            filename_parts = os.path.basename(out_filename).split(".")
            out_filename2 = filename_parts[0] + "_bottleneck_residues." + filename_parts[1]

            labels2process = set()
            labels2process.add("overall")
            num_sim = {
                "overall": len(self.caver_input_folders)
            }

            if self.parameters["perform_comparative_analysis"]:
                labels2process.update(self.caver_input_folders)
                if self.parameters["perform_comparative_analysis"] \
                        and self.parameters["comparative_groups_definition"] is not None:
                    for group, md_labels in self.parameters["comparative_groups_definition"].items():
                        labels2process.add(group)
                        labels2process -= set(md_labels)
                        num_sim[group] = len(md_labels)

            if self.parameters["perform_comparative_analysis"]:
                os.makedirs(os.path.join(self.parameters["statistics_folder"], "comparative_analysis"), exist_ok=True)
                cat_stream1 = open(os.path.join(self.parameters["statistics_folder"], "comparative_analysis",
                                                out_filename), "w")
                if self.parameters["process_bottleneck_residues"]:
                    cat_stream2 = open(os.path.join(self.parameters["statistics_folder"], "comparative_analysis",
                                                    out_filename2), "w")

            for md_label in sorted(labels2process):
                out_folder = self.parameters["statistics_folder"]
                if "overall" not in md_label:
                    out_folder = os.path.join(out_folder, "comparative_analysis", md_label)
                os.makedirs(out_folder, exist_ok=True)
                if md_label not in num_sim.keys():
                    num_sim[md_label] = 1

                if self.parameters["perform_comparative_analysis"] \
                        and self.parameters["comparative_groups_definition"] is not None\
                        and md_label in self.parameters["comparative_groups_definition"].keys():
                    md_labels = sorted(self.parameters["comparative_groups_definition"][md_label])
                    group_label = "{} {}".format(md_label, md_labels)
                else:
                    group_label = md_label

                with open(os.path.join(out_folder, out_filename), "w") as out_stream:
                    header_list = ["SC_ID", "No_Sims", "Total_No_Frames", "Avg_No_Frames", "Avg_BR", "StDev", "Max_BR",
                                   "Avg_Len", "StDev", "Avg_Cur", "StDev", "Avg_throug", "StDev", "Priority"]

                    if self._events_assigned:
                        header_list.extend(["Num_Events", "Num_entries", "Num_releases"])

                    dataset = [header_list]
                    for prio_sc in sorted(self._prioritized_clusters.keys()):  #
                        # we use prioritized IDs as those are also respecting active filters
                        sc_id = self._prioritized_clusters[prio_sc]
                        dataset.append(self._super_clusters[sc_id].get_summary_line_data(self._events_assigned,
                                                                                         md_label))

                    # find appropriate width of columns
                    widths = dict()
                    for i, column in enumerate(dataset[0]):
                        widths[i] = len(column)

                    for row in dataset[1:]:
                        for i, column in enumerate(row):
                            widths[i] = max(widths[i], len(column))

                    output = "Statistics for: {}\n".format(group_label)
                    output += "Total simulations = {}\n".format(num_sim[md_label])
                    output += "No snapshots per simulation = {}\n".format(no_md_snapshots)
                    output += self._report_filters()
                    output += "\n"

                    for row in dataset:
                        line = ""
                        for i, column in enumerate(row[:-1]):
                            line += "{:>{column_width}s}, ".format(column, column_width=widths[i])
                        line += "{:>{column_width}s}\n".format(row[-1], column_width=widths[len(row) - 1])  # last item

                        output += line

                    if self._outlier_transport_events.exist():
                        outlier_text_start = 0

                        for i in range(header_list.index("Priority") + 1):
                            outlier_text_start += widths[i] + 2

                        out_widths = [
                            outlier_text_start,
                            widths[header_list.index("Num_Events")],
                            widths[header_list.index("Num_entries")],
                            widths[header_list.index("Num_releases")]
                        ]
                        output += self._outlier_transport_events.report_summary_line(out_widths, md_label)

                    out_stream.write(output)
                    if "overall" not in md_label:
                        cat_stream1.write(output + "-" * 160 + "\n\n\n")

                if self.parameters["process_bottleneck_residues"]:
                    with open(os.path.join(out_folder, out_filename2), "w") as out_stream2:
                        output = "Bottleneck residues frequency for: {}\n".format(group_label)
                        output += "Total simulations = {}\n".format(num_sim[md_label])
                        output += "No snapshots per simulation = {}\n".format(no_md_snapshots)
                        output += self._report_filters()
                        output += "\n"
                        output += "SC_ID, Total_No_Frames, Bottleneck residues: frequencies\n"

                        for i, prio_sc in enumerate(sorted(self._prioritized_clusters.keys())):  #
                            # we use prioritized IDs as those are also respecting active filters
                            sc_id = self._prioritized_clusters[prio_sc]
                            super_cluster = self._super_clusters[sc_id]
                            output += "{:>5d}, {:>15s}, ".format(sc_id, dataset[i + 1][2])
                            if md_label not in super_cluster.bottleneck_residue_freq \
                                    or not super_cluster.bottleneck_residue_freq[md_label]:
                                output += "---\n"
                                continue
                            for residue, freq in sorted(super_cluster.bottleneck_residue_freq[md_label].items(),
                                                        key=lambda kv: (kv[1], kv[0]), reverse=True):
                                output += "{}:{:.3f}, ".format(residue, freq)
                            output += "\n"

                        out_stream2.write(output)
                        if "overall" not in md_label:
                            cat_stream2.write(output + "-" * 160 + "\n\n\n")

            if self.parameters["perform_comparative_analysis"]:
                cat_stream1.close()
                if self.parameters["process_bottleneck_residues"]:
                    cat_stream2.close()

    def save_super_clusters_visualization(self, script_name: str = "view_super_clusters.py"):
        """
        Save visualization of superclusters (SCs) for Pymol
        :param script_name: filename of the generated visualization script
        """

        from shutil import copyfile

        self.vis_flag += 1
        with TimeProcess("Visualization generation"):
            # self._precompute_cumulative_super_cluster_data()
            logger.info("Saving visualization of superclusters.")
            os.makedirs(self.parameters["super_cluster_vis_path"], exist_ok=True)
            copyfile(os.path.join(self.parameters["transformation_folder"], "ref_transformed.pdb"),
                     os.path.join(self.parameters["super_cluster_vis_path"], "ref_transformed.pdb"))

            labels2process = set()
            labels2process.add("overall")
            if self.parameters["perform_comparative_analysis"]:
                labels2process.update(self.caver_input_folders)
                md_labels_in_groups = list()
                if self.parameters["comparative_groups_definition"] is not None:
                    for md_labels in self.parameters["comparative_groups_definition"].values():
                        md_labels_in_groups.extend(md_labels)
                    labels2process -= set(md_labels_in_groups)
                    labels2process.update(self.parameters["comparative_groups_definition"].keys())

            for md_label in labels2process:
                viz_folder = self.parameters["visualization_folder"]

                if "overall" not in md_label:
                    viz_folder = os.path.join(viz_folder, "comparative_analysis", md_label)
                    os.makedirs(viz_folder, exist_ok=True)

                viz_pdb_file = os.path.relpath(os.path.join(self.parameters["super_cluster_vis_path"],
                                                            "ref_transformed.pdb"), viz_folder)

                with open(os.path.join(viz_folder, script_name), "w") as out_stream:
                    out_stream.write("import pickle, gzip, os\n\n")
                    # visualize protein
                    out_stream.write("cmd.load({}, "
                                     "'protein_structure')\n".format(utils.path_loader_string(viz_pdb_file)))
                    out_stream.write("cmd.show_as('cartoon', 'protein_structure')\n")
                    out_stream.write("cmd.color('gray', 'protein_structure')\n\n")

                    # visualize SCs and assigned transport events
                    data4vis = list()
                    for prio_sc_id in sorted(self._prioritized_clusters.keys()):
                        prio_super_cluster = self._super_clusters[self._prioritized_clusters[prio_sc_id]]
                        prio_super_cluster.load_path_sets()
                        script_lines, vis_data = prio_super_cluster.prepare_visualization(md_label, str(self.vis_flag))
                        if vis_data is None:
                            continue
                        out_stream.writelines(script_lines)
                        data4vis.append(vis_data)

                    # visualize unassigned transport events
                    if self._outlier_transport_events.exist():
                        out_stream.writelines(self._outlier_transport_events.prepare_visualization(md_label))

                    out_stream.write("cmd.do('set all_states, 1')\n")
                    out_stream.write("cmd.show('cgo')\n")
                    out_stream.write("cmd.disable('release_*')\n")
                    out_stream.write("cmd.disable('entry_*')\n")
                    out_stream.write("cmd.zoom()\n")

                surface_cgo = False
                if self.parameters["visualize_super_cluster_volumes"] and \
                        ("overall" in md_label or self.parameters["visualize_comparative_super_cluster_volumes"]):
                    surface_cgo = True

                with Pool(processes=self.parameters["num_cpus"]) as pool:
                    processing = list()
                    # parallel generation of SC visualization
                    for vis_data in data4vis:
                        path_set, params = vis_data
                        processing.append(pool.apply_async(path_set.visualize_cgo, args=(params[0], params[1],
                                                                                         params[2], params[3],
                                                                                         params[4], surface_cgo)))
                    items2process = len(processing)
                    progressbar(0, items2process)
                    for i, p in enumerate(processing):
                        p.get()
                        progressbar(i + 1, items2process)

    def get_property_time_evolution_data(self, property_name: str, active_filters: dict,
                                         sc_id: Optional[int] = None,
                                         missing_value_default: float = 0) -> Dict[int, Dict[str, np.array]]:
        """
        For each MD simulation in specified supercluster, return array containing values of given tunnel property
        for each simulation frame
        :param property_name: name of property to extract
        :param active_filters: filters to be applied (created by define_filters() function)
        :param sc_id: supercluster ID after prioritization if we want to focus on particular tunnel only
        :param missing_value_default: value to be used for frames where tunnels are missing or invalid in given cluster
        :return: for each supercluster ID we have mapping of tunnel property values per MD simulation
        """

        data = dict()
        if sc_id is None:
            for sc_id, supercluster in self._super_clusters.items():
                data[sc_id] = supercluster.get_property_time_evolution_data(property_name, active_filters,
                                                                            missing_value_default)
        else:
            data[sc_id] = self._super_clusters[sc_id].get_property_time_evolution_data(property_name, active_filters,
                                                                                       missing_value_default)

        return data

    def show_tunnels_passing_filter(self, sc_id: int, active_filters: dict, out_folder_path: str,
                                    md_labels: Optional[List[str]] = None,  start_snapshot: Optional[int] = None,
                                    end_snapshot: Optional[int] = None, trajectory: bool = False):
        """
        Visualize tunnels from particular supercluster that fulfill active_filters, possibly showing only particular
         snapshots, selected MD simulations and providing pdb file with corresponding protein ensemble
        :param sc_id: ID of source supercluster to visualize its tunnels
        :param active_filters: filters to be applied (created by define_filters() function)
        :param out_folder_path: path to folder into which we save the visualization
        :param md_labels: list to restrict visualization to tunnel networks from particular MD simulations only
        :param start_snapshot: start snapshot for tunnel visualization
        :param end_snapshot: end snapshot for tunnel visualization
        :param trajectory: if we should visualize tunnels per MD simulation trajectory with protein ensembles
        """

        def _save_pymol_script(_vis_inputs: List[Tuple[str, str, int]], _md_label: Optional[str] = None):
            """
            Prepare Pymol visualization script for given clusters, possibly limited to single MD trajectory
            :param _vis_inputs: data on clusters to visualize (filename, Pymol object label, CAVER color id)
            :param _md_label:  if only this particular simulations should be visualized dynamically
            """
            if _md_label is None:
                script_file = os.path.join(out_folder_path, "show_tunnels_{:03d}_{}.py".format(sc_id, "all"))
                _viz_pdb_file = viz_pdb_file
            else:
                script_file = os.path.join(out_folder_path, "show_tunnels_{:03d}_{}.py".format(sc_id, _md_label))
                _viz_pdb_file = "{}_structure.pdb.gz".format(_md_label)

            with open(script_file, "w") as out_stream:
                for _cls_pdbfilename, _cls_vis_label, _caver_color_id in _vis_inputs:
                    out_stream.write("cmd.load('{}', '{}')\n".format(_cls_pdbfilename, _cls_vis_label))
                    out_stream.write("cmd.set_color('caver{}', {})\n".format(_caver_color_id,
                                                                             utils.get_caver_color(_caver_color_id)))
                    out_stream.write("cmd.color('caver{}', \"{}\")\n".format(_caver_color_id, _cls_vis_label))

                out_stream.write("cmd.alter('Cls_*', 'vdw=b')\n")
                out_stream.write("cmd.show_as('spheres', 'Cls_*')\n")
                out_stream.write("cmd.load('{}', 'structure')\n".format(_viz_pdb_file))
                out_stream.write("cmd.show_as('cartoon', 'structure')\n")
                out_stream.write("cmd.show('lines', 'structure')\n")

        os.makedirs(out_folder_path, exist_ok=True)
        if start_snapshot is not None and end_snapshot is not None:
            snap_ids = [*range(start_snapshot, end_snapshot + 1)]
            viz_snap_ids = snap_ids
        else:
            snap_ids = None
            viz_snap_ids = [*range(1, self.parameters["snapshots_per_simulation"] + 1)]

        super_cluster = self._super_clusters[sc_id]
        viz_pdb_file = os.path.relpath(os.path.join(self.parameters["transformation_folder"],
                                                    "ref_transformed.pdb"), out_folder_path)
        logger.info("Preparing visualization of tunnels from supercluster {:d} present "
                    "between snapshots {} and {}.".format(sc_id, viz_snap_ids[0], viz_snap_ids[-1]))
        logger.info("Prepared visualization will be available in {}.".format(out_folder_path))

        vis_inputs = list()
        for md_label, clusters in super_cluster.get_caver_clusters().items():
            if md_labels is not None and md_label not in md_labels:
                continue

            for cluster in clusters:
                filtered_cluster = cluster.get_subcluster(snap_ids=snap_ids, active_filters=active_filters)
                num_filtered_tunnels = filtered_cluster.count_tunnels()
                if num_filtered_tunnels > 0:
                    logger.info("Saving visualization for {} tunnel(s) "
                                "from {} {}.".format(num_filtered_tunnels, md_label, filtered_cluster.entity_label))
                    caver_color_id = filtered_cluster.cluster_id - 1
                    cls_pdbfilename = "{}_{}.pdb.gz".format(md_label, filtered_cluster.entity_label)
                    cls_vis_label = "Cls_{}_{}".format(filtered_cluster.cluster_id, md_label)
                    filtered_cluster.save_pdb_files(viz_snap_ids, os.path.join(out_folder_path, cls_pdbfilename))
                    vis_inputs.append((cls_pdbfilename, cls_vis_label, caver_color_id))

            if trajectory:
                # we must save per trajectory
                _save_pymol_script(vis_inputs, md_label)
                vis_inputs = list()
                start_frame = start_snapshot - self.parameters["caver_traj_offset"]
                end_frame = end_snapshot - self.parameters["caver_traj_offset"]
                out_pdbfile = os.path.join(out_folder_path, "{}_structure.pdb.gz".format(md_label))

                if self.parameters["trajectory_engine"] == "mdtraj":
                    selector = "protein"  # to keep
                elif self.parameters["trajectory_engine"] == "pytraj":
                    selector = ":WAT,Cl-,Na+"  # to remove
                else:
                    selector = None
                TrajectoryFactory(self.parameters, md_label).write_frames(start_frame, end_frame, out_pdbfile, selector)

        if not trajectory:
            _save_pymol_script(vis_inputs)


class EventAssigner:
    def __init__(self, parameters: dict, event_specification: Tuple[str, str, Tuple[str, Tuple[int, int]]],
                 event: LayeredPathSet, superclusters: Dict[int, SuperCluster],  active_filters: dict):
        """
        Class to perform assignment of single transport event to matching superclusters
        :param parameters: job configuration parameters
        :param event_specification: folder name of source MD simulation, event label and tuple containing resname&resid
        of ligand responsible for this path, supplemented with info on beginning and last frames for the transport event
        :param event: pathset containing representative path for evaluated event
        :param superclusters: dictionary with SuperCluster objects to assign to
        """

        self.parameters = parameters
        self.super_clusters = superclusters
        self.event_specification = event_specification
        self.event = event
        self.active_filters = active_filters

    def _find_directionally_aligned_scs(self) -> List[int]:
        """
        Find superclusters (SCs) located in corresponding sector with transport event by comparing direction in which
        the terminal nodes of SC and event lies
        :return: list of unique supercluster IDs in the sector
        """

        directionally_fitting_super_clusters = set()
        # compute direction of transport_event based on its terminal node
        event_direction = np.ravel(self.event.nodes_data[self.event.nodes_data[:, 4] == 1][0, :3])

        for super_cluster in self.super_clusters.values():
            if super_cluster.has_passed_filter(consider_transport_events=False, active_filters=self.active_filters):
                if super_cluster.is_directionally_aligned(event_direction):
                    directionally_fitting_super_clusters.add(super_cluster.sc_id)

        return list(directionally_fitting_super_clusters)

    def perform_assignment(self) -> (Tuple[str, str, Tuple[str, Tuple[int, int]]], np.array, float, float):
        """
        Identifies most likely supercluster (SC) through which a single evaluated transport event occurred
        :return: event specification, array with IDs of SC to which the event is assigned, maximal buriedness and
        penetration depth of these SCs
        """

        # find which SC could be utilized by this event
        directionally_fitting_super_cluster_ids = self._find_directionally_aligned_scs()

        inside_ratios = list()
        max_depths = list()
        max_buriedness = -999
        max_depth = -999

        if not directionally_fitting_super_cluster_ids:
            logger.debug("Transport event '{}' is not directionally aligned to any valid "
                         "supercluster.".format(self.event_specification))
            return self.event_specification, None, max_buriedness, max_depth

        # evaluated suitable SC for event buriedness and penetration depth
        for sc_id in directionally_fitting_super_cluster_ids:
            super_cluster = self.super_clusters[sc_id]
            inside_ratio, depth = super_cluster.compute_distance2transport_event(self.event)
            inside_ratios.append(inside_ratio)
            max_depths.append(depth)

        # convert to numpy array for better processing
        directionally_fitting_super_clusters = np.array(directionally_fitting_super_cluster_ids)
        inside_ratios = np.array(inside_ratios)
        max_depths = np.array(max_depths)
        max_buriedness = np.max(inside_ratios)

        if max_buriedness < self.parameters["event_assignment_cutoff"]:  # not buried enough in the best SC
            logger.debug("Transport event '{}' could not be assigned to any valid supercluster "
                         "(max_buriedness = {:.2f})".format(self.event_specification, max_buriedness))
            return self.event_specification, None, max_buriedness, max_depth

        # find SCs in which the event is buried more than the cutoff, + we use here a 0.05 tolerance to account
        # for cases where much better fitting SC is just a tiny bit less buried
        max_buriedness_ids = np.nonzero(inside_ratios >= max_buriedness - 0.05)[0]
        buried_sc_ids = directionally_fitting_super_clusters[max_buriedness_ids]
        max_depths = max_depths[max_buriedness_ids]
        max_depth = np.max(max_depths)
        buriedness = None

        if self.parameters["perform_exact_matching_analysis"]:
            buriedness = self._exact_event_tunnel_matching(buried_sc_ids)

        # event is buried in more than one SC and we do want to perform assignment based on penetration depth
        if buried_sc_ids.size > 1 and self.parameters["ambiguous_event_assignment_resolution"] == "penetration_depth":

            msg = "Using penetration depth to identify the best supercluster for"
            msg += "transport event '{:s}' buried inside {:d} superclusters " \
                   "(buriedness = {:.2f}), ".format(str(self.event_specification), buried_sc_ids.size, max_buriedness)

            for sc_id, depth in zip(buried_sc_ids, max_depths):
                msg += "\n sc{:d} - penetration depth = {:.2f}".format(sc_id, depth)

            logger.debug(msg)

            # find the SC in which the event reaches max_depth
            buried_sc_ids = buried_sc_ids[max_depths == max_depth]

        # event is buried in more than one SC and we do want to perform assignment based on exact matching to tunnels
        if buried_sc_ids.size > 1 and self.parameters["ambiguous_event_assignment_resolution"] == "exact_matching":
            msg = "Using Exact matching to identify the best supercluster for transport event '{:s}' buried inside " \
                  "{:d} superclusters (buriedness = {:.2f})\n".format(str(self.event_specification),
                                                                      buried_sc_ids.size, max_buriedness)

            if not self.parameters["perform_exact_matching_analysis"]:  # not to run this twice
                buriedness = self._exact_event_tunnel_matching(buried_sc_ids)

            for sc_id in buried_sc_ids:
                msg += "Exact matching of transport event '{:s}' to " \
                      "tunnels from SC {:d}:\n".format(str(self.event_specification), sc_id)
                if sc_id in buriedness["4all_frames"].keys():
                    msg += "fraction of frames in which ligand is " \
                           "inside SC tunnels {:.2f}\n".format(buriedness["4all_frames"][sc_id])

                    msg += "fraction of frames with identified SC tunnels " \
                           "in which ligand is inside {:.2f}\n".format(buriedness["4existing_tunnels"][sc_id])
                else:
                    msg += "Cannot match - no tunnels from this SC exist in any snapshot/frame corresponding to this " \
                           "event!\n"
            logger.debug(msg)

            # use exact matching results to verify and filter event assignment to multiple clusters
            buried_sc_ids = np.array([*buriedness["4all_frames"].keys()])

            if buried_sc_ids.size:
                # some tunnels matched for some of evaluated SCs
                exact_buriedness = np.array([*buriedness["4all_frames"].values()])
                max_exact_buriedness = np.max(exact_buriedness)
                buried_sc_ids = buried_sc_ids[exact_buriedness == max_exact_buriedness]
            else:
                # no tunnels matched -> cannot assign event to any SC
                buried_sc_ids = None

        return self.event_specification, buried_sc_ids, max_buriedness, max_depth

    def _exact_event_tunnel_matching(self, considered_sc_ids: np.array) -> Dict[str, Dict[int, float]]:
        """
        Performs exact analyses of event with respect to tunnel clusters in frames during the event occurrence
        originating directly from the simulation sampling the event
        :param considered_sc_ids: superclusters considered for the matching with investigated event
        :return: details on buriedness of event in the actual tunnels of considered superclusters
        """

        # parse event info
        md_label = self.event_specification[0]
        residue = int(self.event_specification[2][0].split(":")[1])
        start_frame, end_frame = self.event_specification[2][1]
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        frames = range(start_frame, end_frame + 1)
        snap_ids = [x + self.parameters["caver_traj_offset"] for x in frames]
        if self.parameters["perform_exact_matching_analysis"]:
            details_path = os.path.join(self.parameters["exact_matching_details_folder"], md_label)
            os.makedirs(details_path, exist_ok=True)

        # analyze distances of events to actual tunnels in SCs
        buriedness = {
            "4all_frames": {},
            "4existing_tunnels": {}
        }

        # get coordinates of ligand
        trajectory = TrajectoryFactory(self.parameters, md_label)
        if not trajectory.inputs_exists():
            return buriedness

        if self.parameters["trajectory_engine"] == "mdtraj":
            selector = "resid {}".format(residue)  # to keep
        elif self.parameters["trajectory_engine"] == "pytraj":
            selector = "!:{}".format(residue + 1)  # to remove
        else:
            selector = None
        event_coords = trajectory.get_coords(start_frame, end_frame, selector)

        for sc_id in considered_sc_ids:
            super_cluster = self.super_clusters[sc_id]
            closest_tunnels_data = None
            # get subclusters of tunnel clusters for relevant frames from SCs
            subclusters = super_cluster.get_caver_clusters(md_labels=[md_label], snap_ids=snap_ids)
            if md_label in subclusters.keys():
                clusters2proc = subclusters[md_label]
            else:
                clusters2proc = []

            for frame_coords, frame, snap_id in zip(event_coords, frames, snap_ids):
                # find a tunnel sphere from caver clusters in this SC that is closest to any ligand atom
                min_dist2sphere = 9999999
                closest_sphere = None
                cluster_id = None
                closest_xyz = None

                for cluster in clusters2proc:
                    for xyz in frame_coords:
                        tmp_dist2sphere, tmp_sphere = cluster.get_closest_tunnel_sphere_in_frame2coords(xyz, snap_id)
                        if tmp_dist2sphere is not None and (tmp_dist2sphere <= min_dist2sphere):
                            min_dist2sphere = tmp_dist2sphere
                            closest_sphere = tmp_sphere
                            cluster_id = cluster.cluster_id
                            closest_xyz = xyz
                if closest_sphere is not None:
                    if closest_tunnels_data is None:
                        new_data = np.append(np.array([frame, min_dist2sphere, cluster_id]), closest_sphere)
                        new_data = np.insert(new_data, 1, closest_xyz)
                        closest_tunnels_data = new_data.reshape(1, 12)
                    else:
                        new_data = np.append(np.array([frame, min_dist2sphere, cluster_id]), closest_sphere)
                        new_data = np.insert(new_data, 1, closest_xyz)
                        closest_tunnels_data = np.concatenate((closest_tunnels_data, new_data.reshape(1, 12)))

            if closest_tunnels_data is None:
                # No tunnels from this SC exist in any snapshot/frame corresponding to the event
                continue

            # compute buriedness descriptors for event in the closest tunnels from the given SC
            surface_distances = closest_tunnels_data[:, 4]
            buried_dist_cutoff = 0 + self.parameters["aqauduct_ligand_effective_radius"]
            buried_nodes_data = closest_tunnels_data[surface_distances <= buried_dist_cutoff]
            num_buried_nodes = buried_nodes_data.shape[0]
            buriedness["4all_frames"][sc_id] = num_buried_nodes / (end_frame - start_frame + 1)
            buriedness["4existing_tunnels"][sc_id] = num_buried_nodes / surface_distances.shape[0]

            if self.parameters["perform_exact_matching_analysis"]:
                # save info on matching to files
                details_file = os.path.join(details_path, "{}_sc{}.txt".format(self.event_specification[1], sc_id))
                with open(details_file, "w") as out_stream:
                    out_stream.write("Exact matching of transport event '{:s}' to tunnels from "
                                     "Supercluster {:d} \n".format(str(self.event_specification), sc_id))
                    out_stream.write("fraction of frames in which ligand is inside "
                                     "SC tunnels {:.2f}\n".format(buriedness["4all_frames"][sc_id]))

                    out_stream.write("fraction of frames in which SC tunnels exist and the ligand is "
                                     "inside of them {:.2f}\n".format(buriedness["4existing_tunnels"][sc_id]))
                    out_stream.write("Data on the tunnel spheres closest to the ligand:\n")
                    out_stream.write("-------------------------------------------------\n")
                    out_stream.write("{:>10s},{:>10s},{:>10s},{:>10s},"
                                     "{:>10s},{:>10s},{:>10s},{:>10s},"
                                     "{:>10s},{:>10s},{:>10s},{:>10s},"
                                     "\n".format("Frame", "X-coordLig", "Y-coordLig", "Z-coordLig", "Dist2lig",
                                                 "CaverClsID", "X-coordSph", "Y-coordSph", "Z-coordSph", "Dist2SP",
                                                 "Radius", "TunLength"))
                    np.savetxt(out_stream, closest_tunnels_data, delimiter=',',
                               fmt=["%10d", "%10.3f", "%10.3f", "%10.3f", "%10.3f", "%10d", "%10.3f", "%10.3f",
                                    "%10.3f", "%10.3f", "%10.3f", "%10.3f"])

                # prepare full visualization of matched events
                if self.parameters["visualize_exact_matching_outcomes"]:
                    folder_path = os.path.join(self.parameters["exact_matching_vis_path"], md_label,
                                               "{}_sc{}".format(self.event_specification[1], sc_id))

                    if self.parameters["trajectory_engine"] == "mdtraj":
                        resid = [residue]
                    elif self.parameters["trajectory_engine"] == "pytraj":
                        resid = [residue + 1]
                    else:
                        resid = None
                    visualize_transport_details(folder_path, trajectory, start_frame, end_frame,
                                                self.parameters["caver_traj_offset"], clusters2proc, resids=resid)

        return buriedness


def visualize_transport_details(out_folder_path: str, trajectory: TrajectoryTT, start_frame: int,
                                end_frame: int, caver_traj_offset: int,
                                caver_clusters: Optional[List[TunnelCluster]] = None,
                                start_snapshot: Optional[int] = None, end_snapshot: Optional[int] = None,
                                resids: Optional[List[int]] = None):
    """
    Visualize dynamics of tunnels and/or events in given set of frames together with biomolecule
    :param out_folder_path: output folder path
    :param trajectory: MD simulation trajectory to process
    :param start_frame: start frame for visualization
    :param end_frame: end frame for visualization
    :param caver_traj_offset: difference in IDs of MD frames (from 0) and caver snapshots (often from 1)
    :param caver_clusters: list of tunnel clusters for visualization
    :param start_snapshot: start snapshot for visualization
    :param end_snapshot: end snapshot for visualization
    :param resids: residue ID(s) to show as events
    """

    if start_snapshot is None and end_snapshot is None:
        snap_ids = [x + caver_traj_offset for x in range(start_frame, end_frame + 1)]
    elif start_snapshot is None:
        start_snapshot = start_frame + caver_traj_offset
        snap_ids = [*range(start_snapshot, end_snapshot + 1)]
    elif end_snapshot is None:
        end_snapshot = end_frame + caver_traj_offset
        snap_ids = [*range(start_snapshot, end_snapshot + 1)]
    else:
        snap_ids = [*range(start_snapshot, end_snapshot + 1)]

    os.makedirs(out_folder_path, exist_ok=True)
    protein_filename = os.path.join(out_folder_path, "structure.pdb.gz")

    if trajectory.parameters["trajectory_engine"] == "mdtraj":
        selector = "protein"  # to keep
    elif trajectory.parameters["trajectory_engine"] == "pytraj":
        selector = ":WAT,Cl-,Na+"  # for removal
    else:
        selector = None
    trajectory.write_frames(start_frame, end_frame, protein_filename, selector)

    cluster_labels = list()
    if caver_clusters:
        for cluster in caver_clusters:
            if cluster.save_pdb_files(snap_ids, os.path.join(out_folder_path, cluster.entity_label + ".pdb.gz")):
                cluster_labels.append((cluster.entity_label, cluster.cluster_id - 1))

    resid_labels = list()
    if resids:
        for resid in resids:
            resid_label = "event_{}".format(resid)
            resid_labels.append(resid_label)
            event_filename = os.path.join(out_folder_path, resid_label + ".pdb.gz")
            if trajectory.parameters["trajectory_engine"] == "mdtraj":
                selector = "resid {}".format(resid)  # to keep
            elif trajectory.parameters["trajectory_engine"] == "pytraj":
                selector = "!:{}".format(resid + 1)  # to remove
            else:
                selector = None
            trajectory.write_frames(start_frame, end_frame, event_filename, selector)

    with open(os.path.join(out_folder_path, "show_matched_event.py"), "w") as out_stream:
        for cluster_label, color in cluster_labels:
            out_stream.write("cmd.load('{}.pdb.gz', '{}')\n".format(cluster_label, cluster_label))
            out_stream.write("cmd.set_color('caver{}', {})\n".format(color, utils.get_caver_color(color)))
            out_stream.write("cmd.color('caver{}', \"{}\")\n".format(color, cluster_label))

        if cluster_labels:
            out_stream.write("cmd.alter('Cluster_*', 'vdw=b')\n")
            out_stream.write("cmd.show_as('spheres', 'Cluster_*')\n")
            out_stream.write("cmd.set('sphere_transparency', 0.6, 'Cluster_*')\n")

        for resid_label in resid_labels:
            out_stream.write("cmd.load('{}.pdb.gz', '{}')\n".format(resid_label, resid_label))

        if resid_labels:
            out_stream.write("cmd.show_as('spheres', 'event_*')\n")

        out_stream.write("cmd.load('structure.pdb.gz', 'structure')\n")
        out_stream.write("cmd.show_as('cartoon', 'structure')\n")
        out_stream.write("cmd.show('lines', 'structure')\n")


def save_checkpoint(object_to_save: TransportProcesses, filename: str = "checkpoints/transport_processes.dump",
                    overwrite: bool = False):
    """
    Save TransportProcess object for later used
    :param object_to_save: TransportProcess object to save
    :param filename: filename where to save the object
    :param overwrite: if replace the content of the file in case the file with a given filename exists
    """

    if os.path.exists(filename) and not overwrite:
        raise RuntimeError("Error checkpoint file '{}' already exists. Specify different name or enable overwrite by"
                           " using '--overwrite' option or setting 'overwrite' parameter to True".format(filename))

    logger.info("CHECKPOINT saved to: {}".format(filename))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as out_stream:
        pickle.dump(object_to_save, out_stream)


def load_checkpoint(filename: str = "checkpoints/transport_processes.dump",
                    update_config: Optional[AnalysisConfig] = None) -> TransportProcesses:
    """
    Loads previously saved TransportProcess object, possibly updating new parameters from provided update_config object
    :param filename: input file with saved TransportProcess object
    :param update_config: configuration object with job parameters
    :return: loaded TransportProcess object
    """

    utils.test_file(filename)
    logger.info("CHECKPOINT loaded from: {}".format(filename))
    with open(filename, "rb") as in_stream:
        obj: TransportProcesses = pickle.load(in_stream)

    if update_config is not None:
        # refresh parameters from the configuration
        obj.update_configuration(update_config)
    return obj
