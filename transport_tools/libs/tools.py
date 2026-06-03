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
import pickle
import numpy as np
import fastcluster
from typing import List, Dict, Tuple, Any, Optional, cast
from scipy.cluster.hierarchy import fcluster
from multiprocessing import Pool, get_context, parent_process
from transport_tools.libs.config import AnalysisConfig
from logging import getLogger
from transport_tools.libs.ui import progressbar, TimeProcess, process_count
from transport_tools.libs import utils
from transport_tools.libs.networks import TunnelNetwork, AquaductNetwork, SuperCluster, define_filters, TunnelCluster, \
    TransportEvent, subsample_events, get_md_membership4groups
from transport_tools.libs.geometry import LayeredPathSet, average_starting_point, read_starting_points, \
    init_distance_worker, calc_distance_chunk, iter_pair_chunks, count_pairs_for_shard, \
    iter_shard_pair_chunks, condensed_pair_index, subcutoff_connected_components, gather_dense_submatrix, \
    gather_condensed_subvector
from transport_tools.libs.protein_files import TrajectoryTT, TrajectoryFactory, get_transform_matrix, \
    transform_pdb_file, get_general_rot_mat_from_2_ca_atoms, transform_aquaduct, save_caver_starting_points


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
        self.num_events_by_residue: Dict[str, Dict[str, Dict[str, int]]] = {"overall": {}}

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

    def count_events(self, md_label: str = "overall") -> Tuple[int, int, int]:
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
        resname = traced_event[0].split(":")[0]

        stat_md_label = md_label
        if self.parameters["perform_comparative_analysis"] \
                and self.parameters["comparative_groups_definition"] is not None:
            membership = get_md_membership4groups(self.parameters["comparative_groups_definition"])
            if md_label in membership.keys():
                stat_md_label = membership[md_label]

        if stat_md_label not in self.num_events.keys():
            self.num_events[stat_md_label] = {"entry": 0, "release": 0}
        self.num_events[stat_md_label][event_type] += 1
        if stat_md_label not in self.num_events_by_residue:
            self.num_events_by_residue[stat_md_label] = {}
        if resname not in self.num_events_by_residue[stat_md_label]:
            self.num_events_by_residue[stat_md_label][resname] = {"entry": 0, "release": 0}
        self.num_events_by_residue[stat_md_label][resname][event_type] += 1

        if globally_unassigned:
            self.num_events["overall"][event_type] += 1
            if resname not in self.num_events_by_residue["overall"]:
                self.num_events_by_residue["overall"][resname] = {"entry": 0, "release": 0}
            self.num_events_by_residue["overall"][resname][event_type] += 1
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
            if self.num_events_by_residue["overall"]:
                out_stream.write("Per-residue breakdown:\n")
                for resname in sorted(self.num_events_by_residue["overall"].keys()):
                    out_stream.write("  {}: entry={:d}, release={:d}\n".format(
                        resname, self.num_events_by_residue["overall"][resname]["entry"],
                        self.num_events_by_residue["overall"][resname]["release"]))

            for event_type in sorted(self.transport_events_global.keys()):
                out_stream.write("{}: (from Simulation: AQUA-DUCT ID, (Resname:Residue), "
                                 "start_frame->end_frame; ... )\n".format(event_type))

                for md_label, paths in self.transport_events_global[event_type].items():
                    out_stream.write("from {}: ".format(md_label))
                    for path_id, traced_event in paths:
                        out_stream.write("{}, ({}), {}->{}; ".format(path_id, traced_event[0], traced_event[1][0],
                                                                     traced_event[1][1]))
                    out_stream.write("\n")

    def report_summary_line(self, widths: List[int], md_label: str = "overall",
                            residue_names: List[str] | None = None) -> str:
        """
        Prepares information on transport events flagged as outliers for generation of summary of superclusters
        :param widths: column widths; first 4 are [text_start, Num_Events, Num_entries, Num_releases],
                       followed by pairs [RES_E_width, RES_R_width] for each residue in residue_names
        :param md_label: summary of which simulations to report; by default report 'overall' stats
        :param residue_names: sorted list of residue names matching the per-residue columns; None = skip
        :return: info on outlier events
        """

        line = "{:{width1}s}{:{width2}d}, {:{width3}d}, " \
               "{:{width4}d}".format("Total number of unassigned events:", *self.count_events(md_label),
                                     width1=widths[0], width2=widths[1], width3=widths[2], width4=widths[3])
        if residue_names:
            res_counts = self.num_events_by_residue.get(md_label, {})
            for i, resname in enumerate(residue_names):
                w_e, w_r = widths[4 + i * 2], widths[4 + i * 2 + 1]
                if resname in res_counts:
                    line += ", {:{we}d}, {:{wr}d}".format(
                        res_counts[resname]["entry"], res_counts[resname]["release"], we=w_e, wr=w_r)
                else:
                    line += ", {:{we}s}, {:{wr}s}".format("-", "-", we=w_e, wr=w_r)
        return line + "\n"

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

        # collect sorted residue names for deterministic color assignment
        all_residues: List[str] = sorted({
            path[1][0].split(":")[0]
            for events in events2process.values()
            for paths in events.values()
            for path in paths
        })

        for event_type in sorted(events2process.keys()):
            events_by_residue: Dict[str, List[str]] = {}
            for _md_label, path_id, resname in subsample_events(events2process[event_type],
                                                               self.parameters["random_seed"],
                                                               self.parameters["max_events_per_cluster4visualization"],
                                                               md_label, comparative_groups_definition):
                filename = os.path.join(vis_folder, _md_label, "paths",
                                        "{}_{}_pathset.dump.gz".format(resname.lower() + "_" + path_id, event_type))
                if resname not in events_by_residue:
                    events_by_residue[resname] = []
                events_by_residue[resname].append("{}".format(utils.path_loader_string(filename)))

            for resname in sorted(events_by_residue.keys()):
                color = utils.get_residue_color(all_residues.index(resname))
                obj_name = "{}_outlier".format(resname.lower() + "_" + event_type)
                plines.append("events = [{}]\n".format(",\n".join(events_by_residue[resname])))
                plines.append("for event in events:\n")
                plines.append("    with gzip.open(event, 'rb') as in_stream:\n")
                plines.append("        pathset = pickle.load(in_stream)\n")
                plines.append("        for path in pathset:\n")
                plines.append("            path[3:6] = {}\n".format(color))
                plines.append("            cmd.load_cgo(path, '{}')\n".format(obj_name))
                plines.append("cmd.set('cgo_line_width', {}, '{}')\n\n".format(2, obj_name))

        return plines


class TransportProcesses:
    def __init__(self, config: AnalysisConfig):
        """
        Class for analysis of transport processes and tunnels, contains 'global' work-flows using methods
        from SuperClusters (SC)
        :param config: configuration object with job parameters
        """

        self.parameters = config.get_parameters()
        self.config_file = config.source_file  # INI file the analysis was launched from (needed by SLURM shards)
        self.caver_input_folders, self.traj_input_folders, aquaduct_folders = config.get_input_folders()
        self.aquaduct_input_folders = sorted({md for mds in aquaduct_folders.values() for md in mds})
        self.aquaduct_md_to_roots: Dict[str, List[str]] = {
            md: [root for root, mds in aquaduct_folders.items() if md in mds]
            for md in self.aquaduct_input_folders
        }
        self.reference_pdb_file = config.get_reference_pdb_file()
        self.transformation_folder = self.parameters["transformation_folder"]
        self._super_clusters: Dict[int, SuperCluster] = dict()
        self._prioritized_clusters: Dict[int, int] = dict()  # maps original and prioritized IDs of superclusters
        self._outlier_transport_events = OutlierTransportEvents(self.parameters)
        self._active_filters: Dict[str, Tuple[float, float] | float | int] = define_filters()  # none active
        self._events_assigned = False
        self.vis_flag = 0
        self.filter_flag = 0
        self._aquaduct_single_event_inputs = False
        if self.parameters["start_from_stage"] == 1:
            from transport_tools.libs.ui import _LOG_HANDLER, set_logging_level
            if _LOG_HANDLER is not None:  # skip when logging has not been initialized (e.g. tests)
                set_logging_level("debug", _LOG_HANDLER)
                logger.debug(str(config))  # log initial configuration
                set_logging_level(self.parameters["log_level"], _LOG_HANDLER)

    def update_configuration(self, new_config: AnalysisConfig):
        """
        Updates parameters based on new job parameters
        :param new_config: object with job parameters
        """

        new_config.report_updates(self.parameters)
        self.parameters = new_config.get_parameters()
        self.config_file = new_config.source_file
        self.caver_input_folders, self.traj_input_folders, aquaduct_folders = new_config.get_input_folders()
        self.aquaduct_input_folders = sorted({md for mds in aquaduct_folders.values() for md in mds})
        self.aquaduct_md_to_roots = {
            md: [root for root, mds in aquaduct_folders.items() if md in mds]
            for md in self.aquaduct_input_folders
        }
        self.reference_pdb_file = new_config.get_reference_pdb_file()
        self.transformation_folder = self.parameters["transformation_folder"]

        # update inside components
        self._outlier_transport_events.parameters.update(self.parameters)
        for supercluster in self._super_clusters.values():
            supercluster.parameters.update(self.parameters)

    def _resolve_stage_backend(self, stage_knob: str) -> str:
        """
        Return the effective execution backend ('local' or 'slurm') for a SLURM-capable stage.
        Mirrors AnalysisConfig.resolve_stage_backend but operates on the cached parameters dict
        so TransportProcesses callers don't need to retain a reference to the original config.
        Per-stage override (e.g. parameters['stage04_backend']) wins over the global
        compute_backend default; both are validated at config load time.
        """

        return str(self.parameters.get(stage_knob) or self.parameters["compute_backend"]).lower()

    def _condensed_distances_path(self) -> str:
        """
        :return: path of the condensed pairwise-distance .npy file in the clustering folder
        """

        return os.path.join(self.parameters["clustering_folder"],
                            "caver_clusters_clustering_distances.npy")

    def _allocate_condensed_distances(self, num_pairs: int) -> np.ndarray:
        """
        Allocate the condensed pairwise-distance vector as a disk-backed memmap. The .npy file is
        created in the clustering folder so that the workers/shards scatter their results straight
        onto disk - the full N(N-1)/2 vector is never held fully resident, and stage 5 can load it
        via mmap.

        The memmap is left lazily zero-filled (a freshly created mode='w+' file is sparse and reads
        as zeros) rather than pre-filled with a +inf sentinel: for N in the 10**5 range the vector
        is ~1 TB and an explicit fill would write all of it through the memmap page by page before
        anything else could proceed (on a SLURM run, before the array is even submitted). Completeness
        is instead guaranteed by the callers without inspecting values - the local backend enumerates
        every pair exactly once, and the SLURM assembler verifies each shard contributed its expected
        strided pair count - which is also more robust than the old sentinel scan (0.0 is itself a
        valid distance, so no value could serve as a reliable "unfilled" marker).
        :param num_pairs: number of upper-triangle pairs, i.e. N(N-1)/2 for N clusters
        :return: memmap-backed condensed distance vector of shape (num_pairs,)
        """

        from numpy.lib.format import open_memmap

        os.makedirs(self.parameters["clustering_folder"], exist_ok=True)
        # numpy stubs declare open_memmap's mode/dtype/shape defaults as EllipsisType, so the
        # type checker rejects the real argument types here even though they are correct at
        # runtime; suppress the false positive.
        condensed_distances = open_memmap(self._condensed_distances_path(), mode='w+',  # type: ignore[arg-type]
                                          dtype=np.float64, shape=(num_pairs,))  # type: ignore[arg-type]
        return condensed_distances

    def _compute_intercluster_distances(self, cluster_specifications: List[Tuple[str, int]],
                                        path_sets: Dict[Tuple[str, int], LayeredPathSet],
                                        precision: int = 4) -> np.ndarray:
        """
        Computes the pairwise inter-cluster distances
        :param cluster_specifications: definition of clusters
        :param path_sets: sets of representative paths for all clusters
        :param precision: with how many decimals are the calculated distances reported
        :return: condensed (upper-triangle) vector of pairwise cluster distances for clustering
        """

        num_clusters = len(cluster_specifications)
        condensed_distances = self._allocate_condensed_distances(num_clusters * (num_clusters - 1) // 2)

        if num_clusters > 1:
            num_cpus = self.parameters["num_cpus"]
            n_jobs = (num_clusters ** 2 - num_clusters) // 2

            # split the pairwise jobs into balanced chunks; the path-sets are shared with the
            # workers only once via the Pool initializer, so each job carries merely the
            # lightweight cluster index pairs - keeping many chunks per CPU aids load balancing
            # (pair costs vary widely) and frequent progress updates
            target_num_chunks = max(num_cpus * 64, 1)
            chunk_size = max(1, -(-n_jobs // target_num_chunks))

            done_calcs = 0
            progressbar(done_calcs, n_jobs, self.parameters["log_level"])

            # use the 'spawn' start method: the workers must not be forked from this (typically
            # multi-threaded, e.g. via numpy/BLAS) process, as forking while another thread holds
            # an internal lock deadlocks the child - this happens intermittently, in particular
            # for the SLURM shard processes (see compute_distance_shard)
            with get_context("spawn").Pool(processes=num_cpus, initializer=init_distance_worker,
                                           initargs=(path_sets, cluster_specifications, precision,
                                                     self.parameters["clustering_cutoff"],
                                                     num_cpus, num_cpus)) as pool:
                timeout = self.parameters["worker_task_timeout_s"]
                imap_iter = pool.imap_unordered(calc_distance_chunk,
                                                iter_pair_chunks(num_clusters, chunk_size))
                for chunk_results in utils.iter_imap_results(imap_iter, timeout=timeout, pool=pool,
                                                             label="distance-chunk (local)"):
                    # scatter the chunk's pairs straight into the condensed vector - the dense
                    # N x N matrix is never materialized
                    chunk = np.asarray(chunk_results, dtype=np.float64)
                    rows = chunk[:, 0].astype(np.intp)
                    cols = chunk[:, 1].astype(np.intp)
                    condensed_distances[condensed_pair_index(rows, cols, num_clusters)] = chunk[:, 2]
                    done_calcs += chunk.shape[0]
                    progressbar(done_calcs, n_jobs, self.parameters["log_level"])

        return condensed_distances

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

        length_filter = self._active_filters["length"]
        assert isinstance(length_filter, tuple), f"Expected tuple for 'length' filter but got {type(length_filter).__name__}"
        pr_min_length, pr_max_length = length_filter

        radius_filter = self._active_filters["radius"]
        assert isinstance(radius_filter, tuple), f"Expected tuple for 'radius' filter but got {type(radius_filter).__name__}"
        pr_min_radius, pr_max_radius = radius_filter

        curvature_filter = self._active_filters["curvature"]
        assert isinstance(curvature_filter, tuple), f"Expected tuple for 'curvature' filter but got {type(curvature_filter).__name__}"
        pr_min_curvature, pr_max_curvature = curvature_filter
        min_sims_num = self._active_filters["min_sims_num"]
        min_snapshots_num = self._active_filters["min_snapshots_num"]
        min_avg_snapshots_num = self._active_filters["min_avg_snapshots_num"]
        pr_min_transport_events = self._active_filters["min_transport_events"]
        pr_min_entry_events = self._active_filters["min_entry_events"]
        pr_min_release_events = self._active_filters["min_release_events"]

        msg = "\nParameters used for pre-selection of input tunnels for clustering:\n"
        msg += "length >= {:.2f} A\n".format(self.parameters["relevant_tunnel_min_length"])
        msg += "radius >= {:.2f} A\n".format(self.parameters["relevant_tunnel_min_radius"])
        msg += "curvature <= {:.2f}\n\n".format(self.parameters["relevant_tunnel_max_curvature"])

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

        with get_context("spawn").Pool(processes=self.parameters["num_cpus"]) as pool:
            processing = list()
            for super_cluster in self._super_clusters.values():
                if super_cluster.avg_direction is None:
                    processing.append(("sc_id={}".format(super_cluster.sc_id),
                                       pool.apply_async(super_cluster.compute_space_descriptors)))

            items2process = len(processing)
            if items2process:
                logger.info("Aggregating supercluster data for {:d} superclusters "
                            "using {:d} {}:".format(items2process, self.parameters["num_cpus"],
                                                    process_count(self.parameters["num_cpus"])))

                progressbar(0, items2process, self.parameters["log_level"])
                timeout = self.parameters["worker_task_timeout_s"]
                for i, (sc_id, avg_direction) in enumerate(utils.iter_pool_results(processing,
                                                                                   timeout=timeout,
                                                                                   pool=pool)):
                    self._super_clusters[sc_id].avg_direction = avg_direction
                    del self._super_clusters[sc_id].tunnel_clusters  # remove extensive data
                    progressbar(i + 1, items2process, self.parameters["log_level"])

    def _validate_event_resids_in_trajectory_topologies(
            self, path_sets: Dict[Tuple[str, str, Tuple[str, Tuple[int, int]]],
                                  "LayeredPathSet | None"]):
        """
        Pre-flight check for exact-matching: verifies that every event's traced residue is
        actually present in its md_label's trajectory topology before any worker tries to
        strip it. Without this guard, a stripped trajectory (e.g. only some waters retained)
        produces an empty-topology error inside pytraj/mdtraj that surfaces as a cryptic
        worker exception ("need to have non-empty Topology") and kills the whole assignment
        pool partway through. Topologies are loaded once per md_label and reused across all
        of that md_label's events to keep the check cheap. All offending events are
        collected and reported in a single RuntimeError so the user can fix or filter them
        in one pass instead of one-per-rerun.
        :param path_sets: same mapping that assign_transport_events feeds into the Pool;
                          keys are (md_label, event_label, (resname:resid, frames))
        """

        # Validation is only meaningful when exact matching will actually run for some event.
        if not (self.parameters.get("perform_exact_matching_analysis")
                or self.parameters.get("ambiguous_event_assignment_resolution") == "exact_matching"):
            return

        if not path_sets:
            return

        engine = self.parameters.get("trajectory_engine")
        if engine not in ("pytraj", "mdtraj"):
            # Other engines don't enter the strip path in _exact_event_tunnel_matching.
            return

        # Group events by md_label so we open each topology only once.
        events_by_md: Dict[str, List[Tuple[str, str, int]]] = dict()
        for (md_label, event_label, traced_event) in path_sets.keys():
            try:
                resname, resid_str = traced_event[0].split(":")
                resid = int(resid_str)
            except (ValueError, AttributeError):
                # Unparseable traced_event[0] - record it so the user sees it instead of
                # the worker crashing on it later.
                events_by_md.setdefault(md_label, []).append((event_label, str(traced_event[0]), -1))
                continue
            events_by_md.setdefault(md_label, []).append((event_label, resname, resid))

        missing: List[Tuple[str, str, str, int, str]] = []  # (md_label, event_label, resname, resid, reason)
        for md_label, events in events_by_md.items():
            trajectory = TrajectoryFactory(self.parameters, md_label)
            if not trajectory.inputs_exists():
                # No trajectory configured for this md_label - exact matching would silently
                # return empty buriedness, which is the existing behaviour; nothing to flag.
                continue

            if engine == "pytraj":
                try:
                    import pytraj as pt  # type: ignore[import-untyped]
                except ModuleNotFoundError:
                    raise RuntimeError("Requested to use 'pytraj' as 'trajectory_engine' but pytraj "
                                       "package cannot be imported.")
                topology = pt.load_topology(trajectory.top)
                n_residues = topology.n_residues
                for event_label, resname, resid in events:
                    if resid < 0:
                        missing.append((md_label, event_label, resname, resid,
                                        "unparseable traced_event[0]"))
                        continue
                    pytraj_resid = resid + 1  # AquaDuct -> PyTraj 1-based offset
                    mask = ":{}".format(pytraj_resid)
                    if topology.select(mask).size == 0:
                        reason = ("residue index {} not in topology (n_residues={})".format(
                                  pytraj_resid, n_residues))
                        missing.append((md_label, event_label, resname, resid, reason))
            else:  # mdtraj
                try:
                    import mdtraj  # type: ignore[import-untyped]
                except ModuleNotFoundError:
                    raise RuntimeError("Requested to use 'mdtraj' as 'trajectory_engine' but mdtraj "
                                       "package cannot be imported.")
                topology = mdtraj.load_topology(trajectory.top)
                assert topology is not None, \
                    "mdtraj.load_topology returned None for {}".format(trajectory.top)
                n_residues = topology.n_residues
                for event_label, resname, resid in events:
                    if resid < 0:
                        missing.append((md_label, event_label, resname, resid,
                                        "unparseable traced_event[0]"))
                        continue
                    mask = "resid {}".format(resid)
                    if len(topology.select(mask)) == 0:
                        reason = ("resid {} not in topology (n_residues={})".format(resid, n_residues))
                        missing.append((md_label, event_label, resname, resid, reason))

        if not missing:
            return

        # Sort for stable, scannable output.
        missing.sort(key=lambda t: (t[0], t[1]))
        lines = ["{:d} transport event(s) reference residues that are NOT present in their "
                 "trajectory topology - exact matching cannot run for them and would otherwise "
                 "crash the assignment pool with 'need to have non-empty Topology'. Offending "
                 "events (md_label / event_label / resname:resid / reason):".format(len(missing))]
        for md_label, event_label, resname, resid, reason in missing:
            lines.append("  {} / {} / {}:{} - {}".format(md_label, event_label, resname, resid, reason))
        lines.append("This typically happens when 'trajectory_relative_file' / "
                     "'topology_relative_file' point to a trajectory that has been stripped of "
                     "atoms (e.g. only some waters retained) while AQUA-DUCT was run on the full "
                     "system. Provide the unstripped trajectory/topology for exact matching, or "
                     "filter the affected residues out of the AQUA-DUCT input.")
        raise RuntimeError("\n".join(lines))

    def assign_transport_events(self, md_labels: List[str] | None = None):
        """
        Finds superclusters (SCs) through evaluated transport event happened, and assigns remaining events as outliers;
        this changes  self._events_assigned = True, enabling consideration of transport events during filtering
        :param md_labels: list to restrict assignment to particular Networks (simulations) only
        """

        self._events_assigned = False

        with TimeProcess("Assignment"):
            # assemble transport events
            if md_labels is None:
                folders2process = self.aquaduct_input_folders
            else:
                folders2process = [*set(md_labels).intersection(self.aquaduct_input_folders)]

            # Pre-resolve backend: the SLURM launcher and the validator both only consume
            # `path_sets.keys()`, so for the SLURM path we can build a key-only mapping from
            # the lightweight per-MD .traced_event sidecar instead of pickle-loading every
            # full LayeredPathSet. Runs with millions of events would otherwise hang the
            # driver for hours and balloon RSS to 100s of GB before the array submits.
            backend = self._resolve_stage_backend("stage09_backend") if folders2process else None
            path_sets: Dict[Tuple[str, str, Tuple[str, Tuple[int, int]]], "LayeredPathSet" | None] = dict()

            for md_label in folders2process:
                aquanet = AquaductNetwork(self.parameters, md_label, load_only=True)
                if backend == "slurm":
                    for event_label, traced_event in aquanet.load_layered_summary().items():
                        event_specification = (md_label, str(event_label), traced_event)
                        path_sets[event_specification] = None
                else:
                    aquanet.load_layered_network()
                    for event_label, layered_path_set in aquanet.layered_entities.items():
                        event_specification = (md_label, event_label, layered_path_set.traced_event)
                        path_sets[event_specification] = layered_path_set

            # Per-event log dump so a stale/inconsistent layered_paths.dump (e.g. residues not in
            # the trajectory loaded for exact matching) can be diagnosed without rerunning the
            # whole stage; emitted only at DEBUG so production runs aren't spammed.
            if path_sets and logger.isEnabledFor(10):  # logging.DEBUG == 10
                logger.debug("Assembled {:d} transport event(s) for assignment "
                             "(md_label / event_label / traced_residue):".format(len(path_sets)))
                for (md_label, event_label, traced_event) in path_sets.keys():
                    logger.debug("  {} / {} / {}".format(md_label, event_label, traced_event[0]))

            # When exact matching is in play, every event's traced residue must be present in
            # its md_label's trajectory topology; otherwise pytraj/mdtraj will silently strip
            # to an empty topology in _exact_event_tunnel_matching and crash the whole pool
            # with a cryptic "need to have non-empty Topology" deep inside a worker. Fail loudly
            # up front with the full list of bad (md_label, event, residue) instead.
            self._validate_event_resids_in_trajectory_topologies(path_sets)

            items2process = len(path_sets.keys())

            if items2process:
                if backend == "slurm":
                    self._assign_transport_events_slurm(folders2process, path_sets)
                else:
                    self._assign_transport_events_local(path_sets)

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

    def _assign_transport_events_local(self, path_sets: Dict[Tuple[str, str, Tuple[str, Tuple[int, int]]],
                                                             "LayeredPathSet"]):
        """
        Local-backend body of `assign_transport_events`. Submits one task per event to a spawn
        multiprocessing pool, then applies the assignments to `self._super_clusters` and the
        outlier-event store in original submission order via `_apply_event_assignments`. Split
        out from the public entry point so that the SLURM-backed counterpart can share the
        post-streaming assembly logic.
        """

        items2process = len(path_sets)
        logger.info("Assigning {:d} transport events to {:d} superclusters "
                    "using {:d} {}:".format(items2process, self.enumerate_valid_super_clusters(),
                                            self.parameters["num_cpus"],
                                            process_count(self.parameters["num_cpus"])))
        logger.debug(self._report_filters())

        self._outlier_transport_events = OutlierTransportEvents(self.parameters)

        num_cpus = self.parameters["num_cpus"]
        with get_context("spawn").Pool(
                processes=num_cpus,
                initializer=init_event_assigner_worker,
                initargs=(self.parameters, self._super_clusters, self._active_filters,
                          num_cpus, num_cpus)) as pool:
            progressbar(0, items2process, self.parameters["log_level"])
            tasks = [(event_specification, event_path_set)
                     for event_specification, event_path_set in path_sets.items()]
            timeout = self.parameters["worker_task_timeout_s"]
            imap_iter = pool.imap_unordered(assign_event_worker, tasks)
            # Workers return (event_specification, ...); first element identifies the task.
            results_iter = utils.iter_imap_results(
                imap_iter, timeout=timeout, pool=pool, label="event-assignment",
                extract_task_label=lambda r: "{}/{}".format(r[0][0], r[0][1]))
            # imap_unordered yields in completion order, but add_transport_event() appends
            # to lists inside each SuperCluster and OutlierTransportEvents - the insertion
            # order is observable downstream (e.g. in the visualization scripts). Buffer
            # results during streaming, then apply them in submission order (= the order
            # of path_sets.keys()) so the produced state is stable regardless of worker
            # scheduling and matches the integration-test reference data.
            buffered_assignments: Dict[Tuple[str, str, Tuple[str, Tuple[int, int]]],
                                       Tuple] = dict()
            for i, result in enumerate(results_iter):
                event_specification = result[0]
                buffered_assignments[event_specification] = result
                progressbar(i + 1, items2process, self.parameters["log_level"])

            self._apply_event_assignments(path_sets.keys(), buffered_assignments)

    def _apply_event_assignments(self, ordered_event_specifications,
                                 buffered_assignments: Dict[Tuple[str, str, Tuple[str, Tuple[int, int]]],
                                                            Tuple]):
        """
        Apply event-to-supercluster assignments to `self._super_clusters` and
        `self._outlier_transport_events` in the original submission order. Shared by the
        local and SLURM backends; both produce the same `buffered_assignments` dict (keyed
        by event_specification) - the only difference is where the per-event work was run.
        :param ordered_event_specifications: iterable of event_specifications in submission
                                             order (= the order path_sets was assembled in)
        :param buffered_assignments: map event_specification -> assigner result tuple
                                     `(event_specification, assigned_sc_ids, max_buriedness,
                                       max_depth)`
        """

        for event_specification in ordered_event_specifications:
            _, assigned_sc_ids, max_buriedness, max_depth = buffered_assignments[event_specification]
            md_label = event_specification[0]
            event_path_id, event_type = event_specification[1].split("_")[-2:]
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

    def _assign_transport_events_slurm(self, folders2process: List[str],
                                       path_sets: Dict[Tuple[str, str, Tuple[str, Tuple[int, int]]],
                                                       "LayeredPathSet | None"]):
        """
        SLURM-backed counterpart to `_assign_transport_events_local`. The launcher writes a
        single `state.pkl` (containing `self._super_clusters` and `self._active_filters`)
        alongside `items.json` (the `(md_label, event_label)` enumeration in submission order),
        runs the SLURM array via `run_stage_on_slurm()`, then loads the per-shard pickles back
        and applies the assignments via the shared `_apply_event_assignments` helper so the
        produced state is byte-identical to what the local backend yields.

        The shard process re-loads `state.pkl` once and `aqua_layered_network.dump` per
        md_label (lazy, only for the md_labels its strided assignment actually touches), so
        the launcher does NOT need to ship `path_sets` itself - the events are rebuilt from
        disk inside each shard.

        :param folders2process: list of MD-label folders to process (mirror of the local path)
        :param path_sets: full per-event LayeredPathSet dict; here used only for the
                          submission-order key list (shards rebuild the path-set data on disk)
        """

        from transport_tools.libs import slurm

        if self.config_file is None:
            raise RuntimeError("The 'slurm' stage-9 backend requires the analysis to be run "
                               "from a configuration file - the SLURM compute nodes need it "
                               "to rebuild the analysis state.")

        items2process = len(path_sets)
        logger.info("Assigning {:d} transport events to {:d} superclusters via SLURM array "
                    "job:".format(items2process, self.enumerate_valid_super_clusters()))
        logger.debug(self._report_filters())

        self._outlier_transport_events = OutlierTransportEvents(self.parameters)

        # 1) Enumerate (md_label, event_label) pairs in stable submission order (== path_sets
        #    insertion order, which mirrors the local path).
        items: List[Tuple[str, str]] = [(spec[0], spec[1]) for spec in path_sets.keys()]

        # 2) Hand items + (super_clusters, active_filters) to the stage; run_stage_on_slurm
        #    invokes `prepare_state()` (which writes items.json + state.pkl and computes the
        #    state_hash fingerprint key), checks the fingerprint, and submits only the
        #    missing shards.
        stage = slurm.EventAssignmentShardStage(items, self._super_clusters, self._active_filters,
                                                self.parameters["pickle_protocol"])
        slurm_folder = stage.get_slurm_folder(self.parameters)
        num_shards, still_missing = slurm.run_stage_on_slurm(stage, self.config_file,
                                                             self.parameters)
        if still_missing:
            raise RuntimeError("SLURM event-assignment shard(s) {} did not produce results; "
                               "inspect their logs in '{}'. {}/{} shard(s) completed - "
                               "re-run stage 9 to resume from them.".format(
                                   still_missing, slurm_folder,
                                   num_shards - len(still_missing), num_shards))

        # 3) Stream the per-shard pickles back. Each result is
        #    `(event_specification, assigned_sc_ids, max_buriedness, max_depth)`; collect by
        #    event_specification, then apply in original submission order so the produced
        #    state matches the local backend byte-for-byte.
        buffered_assignments: Dict[Tuple[str, str, Tuple[str, Tuple[int, int]]], Tuple] = dict()
        for shard_id in range(num_shards):
            with open(stage.shard_result_path(slurm_folder, shard_id), "rb") as in_stream:
                shard_results: List[Tuple] = pickle.load(in_stream)
            for result in shard_results:
                buffered_assignments[result[0]] = result

        if len(buffered_assignments) != items2process:
            raise RuntimeError("SLURM event-assignment shards returned {} unique result(s) but "
                               "the launcher enumerated {} event(s); some shards likely failed "
                               "silently. Inspect the shard logs in '{}'.".format(
                                   len(buffered_assignments), items2process, slurm_folder))

        self._apply_event_assignments(path_sets.keys(), buffered_assignments)

        # Assembly succeeded end-to-end (every event is assigned and applied to
        # `_super_clusters` / `_outlier_transport_events`); drop the per-shard cache unless
        # the user opted in to keep it. The state.pkl artifact is part of this folder, so
        # dropping the whole tree also clears the (potentially hundreds-of-MB) supercluster
        # pickle along with it.
        slurm.cleanup_stage_folder(stage, self.parameters)

    def clear_results(self, overwrite: bool = False, output_folders: List[str] | None = None):
        """
        Removes output folder
        :param overwrite: if to perform the cleaning of non empty folder
        :param output_folders: which folders/files to be cleaned out, if not specified, all are removed
        """

        from shutil import rmtree

        if output_folders is None:  # default behavior is to remove all output folders produced
            folders2remove = list()
            for folder_key in ["internal_folder", "data_folder", "visualization_folder", "statistics_folder"]:
                folders2remove.append(self.parameters[folder_key])
        else:
            folders2remove = output_folders

        for output_path in folders2remove:
            if not os.path.exists(output_path):
                continue

            logger.debug("Cleaning '{}'".format(output_path))
            if os.path.isfile(output_path):
                # Handle files
                os.remove(output_path)
            elif os.path.isdir(output_path):
                # Handle directories            
                if not overwrite and os.path.exists(output_path) and os.listdir(output_path):
                    raise RuntimeError("Error output folder '{}' exists and is not empty. Specify different name in "
                                    "'output_path' parameter or enable overwrite option by using '--overwrite' option "
                                    "or setting 'overwrite' parameter to True".format(output_path))
                else:
                    rmtree(output_path, True)

    def _save_distance_matrix(self, condensed_distances: np.ndarray,
                              cluster_specifications: List[Tuple[str, int]],
                              cluster_characteristics: Dict[Tuple[str, int], Tuple[float, int]]):
        """
        Save the pairwise cluster distances (in condensed upper-triangle form) and the cluster
        specifications to files
        :param condensed_distances: condensed vector of pairwise cluster distances
        :param cluster_specifications: definition of clusters
        :param cluster_characteristics: data on cluster throughputs and number of tunnels
        """

        os.makedirs(self.parameters["clustering_folder"], exist_ok=True)
        # the condensed .npy was already created and filled in-place by
        # _allocate_condensed_distances during distance computation - just flush to make sure
        # it is durable on disk. The else branch handles legacy callers that pass a plain ndarray.
        if isinstance(condensed_distances, np.memmap):
            condensed_distances.flush()
        else:
            np.save(self._condensed_distances_path(), condensed_distances)

        with open(os.path.join(self.parameters["clustering_folder"], "cluster_specifications.dump"), "wb") as out:
            pickle.dump((cluster_specifications, cluster_characteristics), out)

        if self.parameters["save_distance_matrix_csv"]:  # save CSV formatted matrix
            from scipy.spatial.distance import squareform
            # the CSV export needs the full symmetric matrix - rebuild it from the condensed form
            # on demand; it is no longer persisted on disk in this dense form
            distance_matrix = squareform(condensed_distances)

            # get natural order of labels
            natural_order_map = {cls_spec: pos for pos, cls_spec
                                 in enumerate(cluster_specifications)}
            reordered_cluster_specifications = sorted(natural_order_map.keys())
            reorder_idx = np.fromiter(
                (natural_order_map[cs] for cs in reordered_cluster_specifications),
                dtype=np.intp, count=len(reordered_cluster_specifications),
            )

            # define maximal label lengths for CSV formatting
            label_length1 = max([1] + [len(cs[0]) for cs in reordered_cluster_specifications])
            label_length2 = max([1] + [len(str(cs[1])) for cs in reordered_cluster_specifications]) + 4

            # pre-format axis labels - identical strings serve as the column header cells
            # and as the leading label of each data row
            axis_labels = [
                "{:>{l1}s}:{:>{l2}s}".format(cs[0], "cls{}".format(cs[1]),
                                             l1=label_length1, l2=label_length2)
                for cs in reordered_cluster_specifications
            ]

            # vectorize per-cell float formatting: one C-level call replaces the
            # N**2 Python-level str.format calls
            cell_width = label_length1 + label_length2
            reordered = np.ascontiguousarray(distance_matrix[np.ix_(reorder_idx, reorder_idx)])
            formatted_cells = np.char.mod("%{}.3f".format(cell_width), reordered)

            # save matrix
            os.makedirs(os.path.dirname(self.parameters["distance_matrix_csv_file"]), exist_ok=True)
            with open(self.parameters["distance_matrix_csv_file"], "w") as out_stream:
                out_stream.write(" " * cell_width + ", " + ", ".join(axis_labels) + "\n")
                for axis_label, row in zip(axis_labels, formatted_cells):
                    out_stream.write(axis_label + ", " + ", ".join(row.tolist()) + "\n")

    def _load_distance_matrix(self) -> Tuple[np.ndarray, List[Tuple[str, int]], Dict[Tuple[str, int], Tuple[float, int]]]:
        """
        Loads the condensed pairwise cluster distances and the cluster specifications from files
        :return: condensed distance vector, cluster_specifications, cluster_characteristics
        """

        clustering_folder = self.parameters["clustering_folder"]
        with open(os.path.join(clustering_folder, "cluster_specifications.dump"), "rb") as in_stream:
            cluster_specifications, cluster_characteristics = pickle.load(in_stream)

        condensed_file = os.path.join(clustering_folder, "caver_clusters_clustering_distances.npy")
        if os.path.exists(condensed_file):
            # mmap so fastcluster.linkage and the optional CSV export can stream pages on demand
            # instead of pulling the full N(N-1)/2 vector into resident memory
            condensed_distances = np.load(condensed_file, mmap_mode='r')
        else:
            # backward compatibility: stage 4 of an older run stored the full dense N x N matrix
            from scipy.spatial.distance import squareform
            legacy_matrix = np.load(os.path.join(clustering_folder, "caver_clusters_clustering_matrix.npy"))
            condensed_distances = squareform(legacy_matrix, checks=False)

        return condensed_distances, cluster_specifications, cluster_characteristics

    def _assemble_cluster_path_sets(self, load_path_sets: bool = True) -> Tuple[List[Tuple[str, int]],
                                                                                Dict[Tuple[str, int], LayeredPathSet],
                                                                                Dict[Tuple[str, int],
                                                                                     Tuple[float, int]]]:
        """
        Collects tunnel clusters from all analyzed MD simulations and orders them by importance
        (cluster ID, mean throughput, number of tunnels, finally md_label to avoid duplicated keys
        in case the exact same clusters appear more times, e.g., due to analyses of parts of the
        same simulation). The resulting cluster ordering is deterministic, so it is identical for
        the local and SLURM backends and across all SLURM shards.
        :param load_path_sets: if False, the layered path-sets are not retained (only their
                               characteristics); used by the SLURM launcher which does not need them
        :return: ordered list of cluster specifications, dict of path-sets keyed by specification
                 (empty when load_path_sets is False), and dict of cluster characteristics
        """

        path_sets: Dict[Tuple[str, int], LayeredPathSet] = dict()
        ordered_clusters = dict()
        cluster_characteristics: Dict[Tuple[str, int], Tuple[float, int]] = dict()

        for md_label in self.caver_input_folders:
            tunnel_network = TunnelNetwork(self.parameters, md_label)
            if load_path_sets:
                tunnel_network.load_layered_network()
                items = (
                    (cls_id, lps.characteristics, lps)
                    for cls_id, lps in tunnel_network.layered_entities.items()
                )
            else:
                # SLURM launcher only needs the (avg_throughput, num_tunnels) tuple to
                # order clusters by importance; pickle.load-ing every LayeredPathSet just
                # to peek at .characteristics costs hours of IO and 100s of GB of peak
                # memory on large studies (600k+ clusters).
                items = (
                    (cls_id, characteristics, None)
                    for cls_id, characteristics
                    in tunnel_network.load_layered_summary().items()
                )

            for cls_id, characteristics, layered_path_set in items:
                cluster_specification = (md_label, int(cls_id))  # tunnel cluster IDs are integers
                if characteristics is None:
                    raise RuntimeError(f"Variable 'characteristics' is not specified for tunnel cluster {cls_id} in {md_label}")
                avg_throughput, num_tunnels = characteristics
                cluster_characteristics[cluster_specification] = characteristics
                importance_key = (cls_id, 1 / avg_throughput, 1 / num_tunnels, md_label)
                if load_path_sets:
                    path_sets[cluster_specification] = layered_path_set
                ordered_clusters[importance_key] = cluster_specification

        cluster_specifications = [ordered_clusters[x] for x in sorted(ordered_clusters.keys())]

        return cluster_specifications, path_sets, cluster_characteristics

    def compute_tunnel_clusters_distances(self):
        """
        Compute pairwise cluster-cluster distances, and save their matrix
        """

        with TimeProcess("Cluster-cluster distances calculation"):
            backend = self._resolve_stage_backend("stage04_backend")
            # the SLURM launcher does not need the path-sets in this process - the compute nodes
            # rebuild them themselves - so we avoid loading them here to save memory
            cluster_specifications, path_sets, cluster_characteristics = \
                self._assemble_cluster_path_sets(load_path_sets=(backend != "slurm"))

            if len(cluster_specifications) <= 1:
                raise RuntimeError("Not enough tunnel clusters available to calculate their distances")

            if backend == "slurm":
                logger.info("Computing distances for {:d} tunnel clusters using SLURM array "
                            "jobs:".format(len(cluster_specifications)))
                condensed_distances = self._compute_intercluster_distances_slurm(cluster_specifications)
            else:
                logger.info("Computing distances for {:d} tunnel clusters "
                            "using {:d} {}:".format(len(cluster_specifications), self.parameters["num_cpus"],
                                                    process_count(self.parameters["num_cpus"])))
                condensed_distances = self._compute_intercluster_distances(cluster_specifications, path_sets)

            self._save_distance_matrix(condensed_distances, cluster_specifications, cluster_characteristics)

    def _compute_intercluster_distances_slurm(self, cluster_specifications: List[Tuple[str, int]]) -> np.ndarray:
        """
        Computes the inter-cluster distance matrix by distributing the pairwise calculations over
        SLURM array jobs (one task per shard) using submitit. The rounding precision is applied by
        the shards themselves (see compute_distance_shard). Each shard writes a compact (K, 3)
        float64 array of (cluster1_id, cluster2_id, distance) rows atomically; the launcher
        concatenates them and scatters the pairs into the condensed upper-triangle vector
        (the dense N x N matrix is never materialised).
        :param cluster_specifications: definition of clusters
        :return: condensed (upper-triangle) vector of pairwise cluster distances for clustering
        """

        from transport_tools.libs import slurm

        if self.config_file is None:
            raise RuntimeError("The 'slurm' distance backend requires the analysis to be run from a configuration "
                               "file - the SLURM compute nodes need it to rebuild the analysis state.")

        num_clusters = len(cluster_specifications)
        num_pairs = num_clusters * (num_clusters - 1) // 2
        if num_clusters <= 1:
            return self._allocate_condensed_distances(num_pairs)

        # 1) Submit and wait via the generic launcher; framework handles fingerprint check,
        #    completion-order polling, and per-shard runtime timeout.
        stage = slurm.DistanceShardStage(num_clusters)
        slurm_folder = stage.get_slurm_folder(self.parameters)
        num_shards, still_missing = slurm.run_stage_on_slurm(stage, self.config_file, self.parameters)
        if still_missing:
            raise RuntimeError("SLURM distance shard(s) {} did not produce results; inspect "
                               "their logs in '{}'. {}/{} shard(s) completed - re-run stage 4 "
                               "to resume from them.".format(
                                   still_missing, slurm_folder,
                                   num_shards - len(still_missing), num_shards))

        # 2) Allocate the condensed vector only now that the shards have completed. Deferring it
        #    past run_stage_on_slurm keeps the launcher from blocking for hours on the creation of
        #    the full ~N(N-1)/2 memmap before any shard is even submitted (it is disk-backed and
        #    lazily zero-filled; see _allocate_condensed_distances).
        condensed_distances = self._allocate_condensed_distances(num_pairs)

        # 3) Scatter each shard's compact (K, 3) float64 array straight into the condensed vector
        #    one shard at a time, freeing each before loading the next. The full result set
        #    (~24 B/pair, terabytes for large studies) is never held resident and the dense N x N
        #    matrix is never materialised. Each shard must return exactly its strided pair count;
        #    a short array means a silently truncated result file.
        total_rows = 0
        for sid in range(num_shards):
            shard_array = np.load(stage.shard_result_path(slurm_folder, sid))
            expected_rows = count_pairs_for_shard(num_clusters, num_shards, sid)
            if shard_array.shape[0] != expected_rows:
                raise RuntimeError("SLURM distance shard {} returned {} pair(s) but {} were "
                                   "expected; its result file in '{}' is incomplete - re-run "
                                   "stage 4 to recompute it.".format(
                                       sid, shard_array.shape[0], expected_rows, slurm_folder))
            if expected_rows:
                rows = shard_array[:, 0].astype(np.intp)
                cols = shard_array[:, 1].astype(np.intp)
                condensed_distances[condensed_pair_index(rows, cols, num_clusters)] = shard_array[:, 2]
                total_rows += expected_rows
            del shard_array
        logger.info("Assembled distance matrix from {:d} SLURM shards.".format(num_shards))

        # 4) Completeness check without inspecting values: the strided assignment partitions the
        #    upper triangle exactly, so the shards' pair counts must sum to N(N-1)/2. This replaces
        #    the old +inf-sentinel scan, which required pre-filling the whole ~1 TB memmap.
        if total_rows != num_pairs:
            raise RuntimeError("SLURM distance shards produced {} pair(s) but {} were expected; "
                               "some shard results are incomplete - inspect the shard logs in "
                               "'{}'.".format(total_rows, num_pairs, slurm_folder))

        # 5) Assembly succeeded end-to-end; drop the per-shard cache unless the user opted in
        #    to keep it via slurm_keep_shard_results. The condensed_distances memmap above is
        #    already on disk under the clustering folder, outside the SLURM tree.
        slurm.cleanup_stage_folder(stage, self.parameters)
        return condensed_distances

    def compute_distance_shard(self, num_shards: int, shard_id: int,
                               precision: int = 4) -> List[Tuple[int, int, float]]:
        """
        Computes a single shard of the inter-cluster distance matrix - the strided subset of
        upper-triangle cluster pairs assigned to shard_id. Used by the SLURM distance backend;
        executed on a SLURM compute node.
        :param num_shards: total number of shards the calculation is split into
        :param shard_id: 0-based ID of the shard to compute
        :param precision: with how many decimals are the calculated distances reported
        :return: list of (cluster1_id, cluster2_id, distance) tuples for this shard's pairs
        """

        cluster_specifications, path_sets, _ = self._assemble_cluster_path_sets(load_path_sets=True)
        num_clusters = len(cluster_specifications)
        if num_clusters <= 1:
            return list()

        num_cpus = self.parameters["slurm_cpus_per_task"]
        cutoff = self.parameters["clustering_cutoff"]
        n_jobs = count_pairs_for_shard(num_clusters, num_shards, shard_id)
        if n_jobs == 0:
            return list()

        logger.info("SLURM shard %d/%d: computing %d cluster-cluster distances using %d %s.",
                    shard_id + 1, num_shards, n_jobs, num_cpus, process_count(num_cpus))

        chunk_size = max(1, -(-n_jobs // max(num_cpus * 64, 1)))
        results: List[Tuple[int, int, float]] = list()
        done_calcs = 0
        progressbar(done_calcs, n_jobs, self.parameters["log_level"])

        # 'spawn' start method - this shard process is multi-threaded (numpy/BLAS), so forking
        # the worker processes here would intermittently deadlock the children on inherited locks.
        # NB: the parent's path_sets + cluster_specifications cannot be freed while the pool runs -
        # multiprocessing.Pool retains `initargs` (to re-spawn dead workers) for the pool's whole
        # lifetime, so they stay resident until this `with` block exits no matter what is del'd here.
        # Each worker also unpickles its own private copy, so the shard holds ~(1 + num_cpus) copies
        # of the corpus; cutting that to ~1 needs the workers to share the corpus rather than copy it
        # (shared memory / fork-COW), which is tracked separately.
        with get_context("spawn").Pool(processes=num_cpus, initializer=init_distance_worker,
                                       initargs=(path_sets, cluster_specifications,
                                                 precision, cutoff,
                                                 num_cpus, num_cpus)) as pool:
            timeout = self.parameters["worker_task_timeout_s"]
            imap_iter = pool.imap_unordered(calc_distance_chunk,
                                            iter_shard_pair_chunks(num_clusters, num_shards,
                                                                   shard_id, chunk_size))
            for chunk_results in utils.iter_imap_results(imap_iter, timeout=timeout, pool=pool,
                                                         label="distance-chunk (shard {})".format(shard_id)):
                results.extend(chunk_results)
                done_calcs += len(chunk_results)
                progressbar(done_calcs, n_jobs, self.parameters["log_level"])

        return results

    def compute_tunnel_layering_shard(self, num_shards: int, shard_id: int) -> int:
        """
        Compute one shard of the stage-3 tunnel-layering work - the strided subset of the
        (md_label, cluster_id) enumeration assigned to shard_id. Used by the SLURM stage-3
        backend; executed on a SLURM compute node. The enumeration itself was written by the
        launcher into <slurm_folder>/items.json; we read it from there instead of re-deriving
        it from orig_network files (which would multiply per-shard I/O by num_shards). For
        each (md_label, cls_id) pair we own, we load the cluster's TunnelCluster object from
        the corresponding orig_network.dump (orig_networks are loaded once per md_label even
        when the shard touches many clusters from the same md_label), run the existing
        `create_layered_cluster_worker` over an inner spawn-Pool sized to slurm_cpus_per_task,
        and write the per-shard results as a single pickle. The launcher concatenates the
        per-shard pickles and applies the same submission-order commit as the local backend.
        :param num_shards: total number of shards the layering is split into
        :param shard_id: 0-based ID of the shard to compute
        :return: number of (md_label, cluster_id) pairs this shard layered
        """

        from transport_tools.libs.slurm import TunnelLayeringShardStage

        slurm_folder = TunnelLayeringShardStage.get_slurm_folder(self.parameters)
        all_items = TunnelLayeringShardStage.load_items(slurm_folder)

        # strided assignment: shard i gets every Nth item from the enumeration. The
        # enumeration was built in stable submission order by the launcher, so this slicing
        # is deterministic across launcher + shard processes
        my_items: List[List[Any]] = all_items[shard_id::num_shards]
        if not my_items:
            logger.info("SLURM tunnel-layering shard %d/%d: nothing to do.", shard_id + 1, num_shards)
            self._write_tunnel_layering_shard_result(slurm_folder, shard_id, [])
            return 0

        # group by md_label so we load each orig_network.dump at most once even when the
        # shard touches many clusters from the same MD simulation
        clusters_by_md: Dict[str, set] = dict()
        for md_label, cls_id in my_items:
            clusters_by_md.setdefault(md_label, set()).add(int(cls_id))

        clusters_to_layer: list = list()
        for md_label, wanted_ids in clusters_by_md.items():
            tunnel_network = TunnelNetwork(self.parameters, md_label)
            tunnel_network.load_orig_network()
            for cluster in tunnel_network.get_clusters4layering():
                if cluster.cluster_id in wanted_ids:
                    clusters_to_layer.append(cluster)

        n_jobs = len(clusters_to_layer)
        num_cpus = self.parameters["slurm_cpus_per_task"]
        logger.info("SLURM tunnel-layering shard %d/%d: layering %d cluster(s) using %d %s.",
                    shard_id + 1, num_shards, n_jobs, num_cpus, process_count(num_cpus))

        results: List[Tuple[int, str, LayeredPathSet]] = list()
        done = 0
        progressbar(done, n_jobs, self.parameters["log_level"])

        # 'spawn' inner pool with the BLAS thread cap - identical pattern to compute_distance_shard
        with get_context("spawn").Pool(processes=num_cpus,
                                       initializer=utils.cap_blas_threads_for_worker,
                                       initargs=(num_cpus, num_cpus)) as pool:
            timeout = self.parameters["worker_task_timeout_s"]
            imap_iter = pool.imap_unordered(create_layered_cluster_worker, clusters_to_layer)
            for result in utils.iter_imap_results(
                    imap_iter, timeout=timeout, pool=pool,
                    label="layer-tunnel-cluster (shard {})".format(shard_id),
                    extract_task_label=lambda r: "{}/{}".format(r[1], r[0])):
                results.append(result)
                done += 1
                progressbar(done, n_jobs, self.parameters["log_level"])

        # persist the side-effects backup (per-cluster .py scripts + per-layer node PDBs) so
        # a future resume hit with a wiped visualisation folder can restore them without
        # re-running the per-cluster sklearn stack. Backup is written BEFORE the result
        # pickle so the existence of the pickle implies the manifest + archive are present.
        from transport_tools.libs.slurm import TunnelLayeringShardStage
        if self.parameters["visualize_layered_clusters"]:
            side_effects = TunnelLayeringShardStage.derive_side_effect_paths(
                self.parameters, results)
            TunnelLayeringShardStage.write_side_effects_archive(
                slurm_folder, shard_id, side_effects, self.parameters["output_path"])
            TunnelLayeringShardStage.write_shard_manifest(slurm_folder, shard_id, side_effects)
        else:
            # consistency: write an empty manifest so the resume check has a definitive
            # answer ("no side-effects expected") rather than relying on a missing file
            TunnelLayeringShardStage.write_shard_manifest(slurm_folder, shard_id, [])

        self._write_tunnel_layering_shard_result(slurm_folder, shard_id, results)
        return len(results)

    def _write_tunnel_layering_shard_result(self, slurm_folder: str, shard_id: int,
                                            results: List[Tuple[int, str, "LayeredPathSet"]]):
        """
        Atomically persist a tunnel-layering shard's results to its per-shard pickle file
        (write to `.tmp.pkl`, then os.replace). Used by `compute_tunnel_layering_shard()`;
        kept separate so the empty-shard fast path can reuse it without duplicating the
        atomic-write idiom.
        """

        from transport_tools.libs.slurm import TunnelLayeringShardStage

        out_file = TunnelLayeringShardStage.shard_result_path(slurm_folder, shard_id)
        tmp_file = out_file + ".tmp.pkl"
        with open(tmp_file, "wb") as out_stream:
            pickle.dump(results, out_stream, protocol=self.parameters["pickle_protocol"])
        os.replace(tmp_file, out_file)

    def compute_aquaduct_layering_shard(self, num_shards: int, shard_id: int) -> int:
        """
        Compute one shard of the stage-8 aquaduct-layering work - the strided subset of the
        (md_label, event_label) enumeration assigned to shard_id. Used by the SLURM stage-8
        backend; executed on a SLURM compute node. Mirror of `compute_tunnel_layering_shard`
        but for AQUA-DUCT transport events instead of tunnel clusters. The enumeration itself
        was written by the launcher into <slurm_folder>/items.json; we read it from there
        instead of re-deriving it from aqua_orig_network files (which would multiply per-shard
        I/O by num_shards). For each (md_label, event_label) pair we own, we load the event
        object from the corresponding aqua_orig_network.dump (aqua_orig_networks are loaded
        once per md_label even when the shard touches many events from the same md_label),
        run the existing `create_layered_event_worker` over an inner spawn-Pool sized to
        slurm_cpus_per_task, and write the per-shard results as a single pickle. The launcher
        concatenates the per-shard pickles and applies the same submission-order commit as
        the local backend.
        :param num_shards: total number of shards the layering is split into
        :param shard_id: 0-based ID of the shard to compute
        :return: number of (md_label, event_label) pairs this shard layered
        """

        from transport_tools.libs.slurm import AquaductLayeringShardStage

        slurm_folder = AquaductLayeringShardStage.get_slurm_folder(self.parameters)
        all_items = AquaductLayeringShardStage.load_items(slurm_folder)

        # strided assignment: shard i gets every Nth item from the enumeration. The
        # enumeration was built in stable submission order by the launcher, so this slicing
        # is deterministic across launcher + shard processes
        my_items: List[List[Any]] = all_items[shard_id::num_shards]
        if not my_items:
            logger.info("SLURM aquaduct-layering shard %d/%d: nothing to do.",
                        shard_id + 1, num_shards)
            self._write_aquaduct_layering_shard_result(slurm_folder, shard_id, [])
            return 0

        # group by md_label so we load each aqua_orig_network.dump at most once even when the
        # shard touches many events from the same MD simulation
        events_by_md: Dict[str, set] = dict()
        for md_label, event_label in my_items:
            events_by_md.setdefault(md_label, set()).add(str(event_label))

        events_to_layer: list = list()
        for md_label, wanted_labels in events_by_md.items():
            aquanet = AquaductNetwork(self.parameters, md_label, load_only=True)
            aquanet.load_orig_network()
            for event in aquanet.get_events4layering():
                if event.entity_label in wanted_labels:
                    events_to_layer.append(event)

        n_jobs = len(events_to_layer)
        num_cpus = self.parameters["slurm_cpus_per_task"]
        logger.info("SLURM aquaduct-layering shard %d/%d: layering %d event(s) using %d %s.",
                    shard_id + 1, num_shards, n_jobs, num_cpus, process_count(num_cpus))

        results: List[Tuple[str, str, LayeredPathSet]] = list()
        done = 0
        progressbar(done, n_jobs, self.parameters["log_level"])

        # 'spawn' inner pool with the BLAS thread cap - identical pattern to
        # compute_tunnel_layering_shard / compute_distance_shard
        with get_context("spawn").Pool(processes=num_cpus,
                                       initializer=utils.cap_blas_threads_for_worker,
                                       initargs=(num_cpus, num_cpus)) as pool:
            timeout = self.parameters["worker_task_timeout_s"]
            imap_iter = pool.imap_unordered(create_layered_event_worker, events_to_layer)
            for result in utils.iter_imap_results(
                    imap_iter, timeout=timeout, pool=pool,
                    label="layer-aquaduct-event (shard {})".format(shard_id),
                    extract_task_label=lambda r: "{}/{}".format(r[1], r[0])):
                results.append(result)
                done += 1
                progressbar(done, n_jobs, self.parameters["log_level"])

        # persist the side-effects backup (per-event .py scripts + per-layer node PDBs) so a
        # future resume hit with a wiped visualisation folder can restore them without
        # re-running the per-event sklearn stack. `visualize_layered_events` gates whether
        # the worker called `prep_visualization()` - when False, no side-effects were
        # produced and we write only an empty manifest (no archive needed) so the resume
        # check has a definitive "nothing to validate" answer.
        from transport_tools.libs.slurm import AquaductLayeringShardStage
        if self.parameters["visualize_layered_events"]:
            side_effects = AquaductLayeringShardStage.derive_side_effect_paths(
                self.parameters, results)
            AquaductLayeringShardStage.write_side_effects_archive(
                slurm_folder, shard_id, side_effects, self.parameters["output_path"])
            AquaductLayeringShardStage.write_shard_manifest(slurm_folder, shard_id, side_effects)
        else:
            AquaductLayeringShardStage.write_shard_manifest(slurm_folder, shard_id, [])

        self._write_aquaduct_layering_shard_result(slurm_folder, shard_id, results)
        return len(results)

    def _write_aquaduct_layering_shard_result(self, slurm_folder: str, shard_id: int,
                                              results: List[Tuple[str, str, "LayeredPathSet"]]):
        """
        Atomically persist an aquaduct-layering shard's results to its per-shard pickle file
        (write to `.tmp.pkl`, then os.replace). Used by `compute_aquaduct_layering_shard()`;
        kept separate so the empty-shard fast path can reuse it without duplicating the
        atomic-write idiom.
        """

        from transport_tools.libs.slurm import AquaductLayeringShardStage

        out_file = AquaductLayeringShardStage.shard_result_path(slurm_folder, shard_id)
        tmp_file = out_file + ".tmp.pkl"
        with open(tmp_file, "wb") as out_stream:
            pickle.dump(results, out_stream, protocol=self.parameters["pickle_protocol"])
        os.replace(tmp_file, out_file)

    def compute_event_assignment_shard(self, num_shards: int, shard_id: int) -> int:
        """
        Compute one shard of the stage-9 event-assignment work - the strided subset of the
        (md_label, event_label) enumeration assigned to shard_id. Used by the SLURM stage-9
        backend; executed on a SLURM compute node. The enumeration was written by the launcher
        into <slurm_folder>/items.json; the supercluster dict + active filter set were written
        into <slurm_folder>/state.pkl (so each shard does not have to re-derive them by
        replaying stages 5/6). We load both at startup, inject the state into our
        TransportProcesses instance, group our strided items by md_label (so each
        aqua_layered_network.dump is loaded at most once per shard regardless of how many of
        its events this shard owns), build per-event (event_specification, LayeredPathSet)
        tasks for the events we own, run the existing `init_event_assigner_worker` /
        `assign_event_worker` over an inner spawn-Pool sized to slurm_cpus_per_task, and write
        the per-shard results as a single pickle. The launcher concatenates the per-shard
        pickles and applies the assignments to its `_super_clusters` / OutlierTransportEvents
        in original submission order via the shared `_apply_event_assignments` helper.
        :param num_shards: total number of shards the assignment is split into
        :param shard_id: 0-based ID of the shard to compute
        :return: number of events this shard assigned
        """

        from transport_tools.libs.slurm import EventAssignmentShardStage

        slurm_folder = EventAssignmentShardStage.get_slurm_folder(self.parameters)
        all_items = EventAssignmentShardStage.load_items(slurm_folder)
        super_clusters, active_filters = EventAssignmentShardStage.load_state(slurm_folder)

        # inject the launcher-prepared state so the local code-paths (assigner workers, the
        # _validate_event_resids check) see the same superclusters and filters as the launcher
        self._super_clusters = super_clusters
        self._active_filters = active_filters

        # strided assignment: shard i gets every Nth item from the enumeration. The
        # enumeration was built in stable submission order by the launcher, so this slicing
        # is deterministic across launcher + shard processes
        my_items: List[List[Any]] = all_items[shard_id::num_shards]
        if not my_items:
            logger.info("SLURM event-assignment shard %d/%d: nothing to do.",
                        shard_id + 1, num_shards)
            self._write_event_assignment_shard_result(slurm_folder, shard_id, [])
            return 0

        # group by md_label so we load each aqua_layered_network.dump at most once even when
        # the shard touches many events from the same MD simulation
        events_by_md: Dict[str, set] = dict()
        for md_label, event_label in my_items:
            events_by_md.setdefault(md_label, set()).add(str(event_label))

        # AquaductNetwork.layered_entities is shared with TunnelNetwork and typed as
        # Dict[str | int, LayeredPathSet] with traced_event Optional; for events the keys are
        # always str and traced_event is always set, so narrow via cast to match the strict
        # signature of assign_event_worker (mirrors what the local-backend path implicitly does
        # via _assign_transport_events_local's typed path_sets parameter).
        tasks: List[Tuple[Tuple[str, str, Tuple[str, Tuple[int, int]]], "LayeredPathSet"]] = list()
        for md_label, wanted_events in events_by_md.items():
            aquanet = AquaductNetwork(self.parameters, md_label, load_only=True)
            aquanet.load_layered_network()
            for event_label, layered_path_set in aquanet.layered_entities.items():
                if event_label in wanted_events:
                    event_specification = cast(Tuple[str, str, Tuple[str, Tuple[int, int]]],
                                               (md_label, event_label, layered_path_set.traced_event))
                    tasks.append((event_specification, layered_path_set))

        n_jobs = len(tasks)
        num_cpus = self.parameters["slurm_cpus_per_task"]
        logger.info("SLURM event-assignment shard %d/%d: assigning %d event(s) using %d %s.",
                    shard_id + 1, num_shards, n_jobs, num_cpus, process_count(num_cpus))

        results: List[Tuple] = list()
        done = 0
        progressbar(done, n_jobs, self.parameters["log_level"])

        # 'spawn' inner pool with the shared event-assigner initializer - identical pattern to
        # the local backend, with super_clusters + active_filters routed via the initializer
        with get_context("spawn").Pool(
                processes=num_cpus,
                initializer=init_event_assigner_worker,
                initargs=(self.parameters, self._super_clusters, self._active_filters,
                          num_cpus, num_cpus)) as pool:
            timeout = self.parameters["worker_task_timeout_s"]
            imap_iter = pool.imap_unordered(assign_event_worker, tasks)
            for result in utils.iter_imap_results(
                    imap_iter, timeout=timeout, pool=pool,
                    label="event-assignment (shard {})".format(shard_id),
                    extract_task_label=lambda r: "{}/{}".format(r[0][0], r[0][1])):
                results.append(result)
                done += 1
                progressbar(done, n_jobs, self.parameters["log_level"])

        self._write_event_assignment_shard_result(slurm_folder, shard_id, results)
        return len(results)

    def _write_event_assignment_shard_result(self, slurm_folder: str, shard_id: int,
                                             results: List[Tuple]):
        """
        Atomically persist an event-assignment shard's results to its per-shard pickle file
        (write to `.tmp.pkl`, then os.replace). Used by `compute_event_assignment_shard()`;
        kept separate so the empty-shard fast path can reuse it without duplicating the
        atomic-write idiom.
        """

        from transport_tools.libs.slurm import EventAssignmentShardStage

        out_file = EventAssignmentShardStage.shard_result_path(slurm_folder, shard_id)
        tmp_file = out_file + ".tmp.pkl"
        with open(tmp_file, "wb") as out_stream:
            pickle.dump(results, out_stream, protocol=self.parameters["pickle_protocol"])
        os.replace(tmp_file, out_file)

    def _assign_supercluster_labels(self, condensed_matrix: np.ndarray,
                                    cluster_specifications: List[Tuple[str, int]]
                                    ) -> Dict[Tuple[str, int], int]:
        """
        Assigns a supercluster label to every tunnel cluster from the condensed pairwise-distance
        vector. Dispatches by clustering_method:
          * 'agglomerative' (default): 'single'/'complete'/'average' run through the
            connected-components partition driver (transparent, exact, and far cheaper than the
            full-matrix linkage), while 'ward' uses the historical full-matrix fastcluster.linkage
            (its cross-component validity under the partition is unresolved, so it is excluded).
          * 'hdbscan': density-based clustering run per connected component via the same driver,
            with clustering_cutoff acting as a hard merge floor; noise within the cutoff of a cluster
            is absorbed into the nearest one, only noise beyond the cutoff becomes a singleton.
        When consolidate_orphan_superclusters is enabled, a final method-agnostic pass
        (_consolidate_orphans_to_cores) reassigns every sub-core-size supercluster to the nearest
        core, stitching periphery the partition isolated back into the major branches.
        Only the grouping of clusters matters downstream - the raw label integers are remapped to
        1..K by supercluster importance in the caller.
        :param condensed_matrix: condensed upper-triangle pairwise-distance vector (may be a memmap)
        :param cluster_specifications: tunnel-cluster specs; index i matches row/column i of the matrix
        :return: mapping of each cluster specification to its integer supercluster label
        """

        method = self.parameters["clustering_method"]
        cutoff = self.parameters["clustering_cutoff"]
        num_points = len(cluster_specifications)

        if method == "hdbscan":
            # the partition scale is decoupled from the HDBSCAN selection/merge scales; a generous
            # partition keeps each branch and its periphery in one component (see _cluster_component_hdbscan)
            partition_cutoff = self.parameters["clustering_partition_cutoff"]
            if partition_cutoff is None:
                partition_cutoff = cutoff
            cluster_labels = self._cluster_labels_partitioned(condensed_matrix, num_points, cutoff, "hdbscan",
                                                              partition_cutoff)
        else:  # agglomerative - partitions at the flat cut (clustering_partition_cutoff does not apply here, as a
               # sub-flat-cut partition would break the exact equivalence with the full-matrix linkage)
            linkage = self.parameters["clustering_linkage"]
            if linkage == "ward":
                linkage_matrix = fastcluster.linkage(condensed_matrix, method=linkage)
                cluster_labels = fcluster(linkage_matrix, t=cutoff, criterion='distance')
            else:
                cluster_labels = self._cluster_labels_partitioned(condensed_matrix, num_points, cutoff, linkage)

        if self.parameters["consolidate_orphan_superclusters"]:
            cluster_labels = self._consolidate_orphans_to_cores(
                cluster_labels, condensed_matrix, num_points,
                self.parameters["supercluster_core_min_size"],
                self.parameters["orphan_assignment_cutoff"])

        return dict(zip(cluster_specifications, cluster_labels))

    @staticmethod
    def _consolidate_orphans_to_cores(cluster_labels: np.ndarray, condensed_matrix: np.ndarray,
                                      num_points: int, core_min_size: int,
                                      assignment_cutoff: Optional[float]) -> np.ndarray:
        """
        Global, method-agnostic post-clustering consolidation: every 'orphan' tunnel cluster - one
        whose supercluster has fewer than core_min_size members - is reassigned to the nearest
        'core' supercluster (>= core_min_size members) by the minimum distance to any of that core's
        members. This is the inter-component counterpart to the per-component HDBSCAN noise
        absorption: the connected-components partition can isolate a peripheral cluster in its own
        component (or linkage/HDBSCAN can leave it a small fragment) even though it spatially belongs
        to a major branch; this pass stitches such fragments back into the nearest major supercluster.

        Only orphans move, and only onto the ORIGINAL cores in one simultaneous pass, so an orphan
        never attaches to another orphan and the assignment cannot transitively chain distant
        clusters together. An orphan farther than assignment_cutoff from every core stays on its own
        (a true outlier); assignment_cutoff=None means uncapped (every orphan joins its nearest
        core). Relabel-only: every tunnel cluster still maps to exactly one supercluster.
        :param cluster_labels: per-point supercluster labels (a relabeled copy is returned)
        :param condensed_matrix: condensed upper-triangle distance vector of the full matrix (may be a memmap)
        :param num_points: number of tunnel clusters (side length of the implied dense matrix)
        :param core_min_size: minimum membership for a supercluster to count as a core / assignment target
        :param assignment_cutoff: max orphan-to-core distance allowed for absorption; None => uncapped
        :return: consolidated label array
        """

        cluster_labels = np.asarray(cluster_labels).copy()
        if num_points == 0:
            return cluster_labels

        _, inverse, counts = np.unique(cluster_labels, return_inverse=True, return_counts=True)
        point_in_core = (counts >= core_min_size)[inverse.ravel()]
        core_points = np.flatnonzero(point_in_core)
        orphan_points = np.flatnonzero(~point_in_core)
        if core_points.size == 0 or orphan_points.size == 0:
            return cluster_labels  # no cores to attach to, or nothing orphaned

        cap = np.inf if assignment_cutoff is None else float(assignment_cutoff)
        for orphan in orphan_points:
            # nearest core member to this orphan, read straight from the condensed vector
            lows = np.minimum(orphan, core_points)
            highs = np.maximum(orphan, core_points)
            dists = np.asarray(condensed_matrix[condensed_pair_index(lows, highs, num_points)])
            nearest = int(np.argmin(dists))
            if dists[nearest] <= cap:
                cluster_labels[orphan] = cluster_labels[core_points[nearest]]
        return cluster_labels

    def _cluster_labels_partitioned(self, condensed_matrix: np.ndarray, num_points: int, cutoff: float,
                                    backend: str, partition_cutoff: Optional[float] = None) -> np.ndarray:
        """
        Clusters tunnel clusters by partitioning them into the connected components of the
        sub-partition_cutoff graph (pairs with distance <= partition_cutoff) and clustering each
        component independently with the requested backend, then stitching the per-component labels
        into one global labelling. Peak cost is O(max |component|^2) instead of O(N^2). Components
        are processed in ascending smallest-member-index order, so the labelling is deterministic
        across runs and backends.

        partition_cutoff (the graph/component scale) is decoupled from cutoff (the per-component
        clustering scale) so a single pass can be multi-scale: a generous partition keeps a branch
        and its periphery in one component for context, while the backend separates them at a tighter
        cutoff. For agglomerative linkages the caller passes partition_cutoff == cutoff (the flat
        cut), which keeps the partition exact; for HDBSCAN partition_cutoff may be larger than the
        cluster-selection scale. Defaults to cutoff when not given (the legacy single-scale behavior).
        :param condensed_matrix: condensed upper-triangle pairwise-distance vector (may be a memmap)
        :param num_points: number of tunnel clusters (side length of the implied dense matrix)
        :param cutoff: per-component clustering scale (flat cut for agglomerative; passed to the
                       HDBSCAN backend as its selection/merge fallback)
        :param backend: clustering backend - an agglomerative linkage ('single'/'complete'/'average')
                        or 'hdbscan'
        :param partition_cutoff: sub-cutoff graph threshold defining the components; None => cutoff
        :return: integer supercluster-label array of length num_points
        """

        if partition_cutoff is None:
            partition_cutoff = cutoff
        n_components, component_labels = subcutoff_connected_components(condensed_matrix, num_points,
                                                                        partition_cutoff)
        components = self._components_by_min_index(component_labels)
        self._log_component_size_histogram(components, num_points)

        cluster_labels = np.empty(num_points, dtype=np.intp)
        next_label = 1  # 1-based to mirror scipy.fcluster (purely cosmetic; labels are remapped later)
        for members in components:
            if members.size == 1:
                cluster_labels[members[0]] = next_label
                next_label += 1
                continue
            local_labels = self._cluster_component(condensed_matrix, num_points, members, cutoff, backend)
            next_label = self._stitch_component_labels(cluster_labels, members, local_labels, next_label)

        return cluster_labels

    @staticmethod
    def _components_by_min_index(component_labels: np.ndarray) -> List[np.ndarray]:
        """
        Groups point indices by their connected-component id and returns the per-component index
        arrays ordered by their smallest member index; within each array the indices are ascending.
        This fixed order makes the stitched labelling deterministic and independent of the order in
        which scipy happened to number the components.
        :param component_labels: per-point component id (as returned by connected_components)
        :return: list of per-component global-index arrays, ordered by smallest member index
        """

        if component_labels.size == 0:
            return list()
        sort_idx = np.argsort(component_labels, kind='stable')
        sorted_labels = component_labels[sort_idx]
        boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
        components = np.split(sort_idx, boundaries)
        components.sort(key=lambda member_indices: member_indices[0])
        return components

    def _cluster_component(self, condensed_matrix: np.ndarray, num_points: int, members: np.ndarray,
                           cutoff: float, backend: str) -> np.ndarray:
        """
        Clusters a single connected component (>= 2 members) and returns local labels for its
        members. For an agglomerative backend the component's dense distance block is condensed and
        fed to fastcluster.linkage + fcluster, reproducing exactly what the full-matrix linkage would
        yield for these methods (a flat cluster at the cutoff never spans two sub-cutoff components).
        For the 'hdbscan' backend the dense block is clustered by HDBSCAN (see
        _cluster_component_hdbscan).
        :param condensed_matrix: condensed upper-triangle pairwise-distance vector of the full matrix
        :param num_points: side length of the full dense matrix
        :param members: global indices of this component's members (ascending)
        :param cutoff: clustering distance cutoff
        :param backend: 'hdbscan' or an agglomerative linkage ('single'/'complete'/'average')
        :return: local integer label per member (in the order of 'members'; -1 marks HDBSCAN noise)
        """

        if backend == "hdbscan":
            # HDBSCAN(metric="precomputed") needs the full dense block
            dense_block = gather_dense_submatrix(condensed_matrix, num_points, members)
            return self._cluster_component_hdbscan(dense_block, cutoff)

        # Agglomerative: gather the component's distances directly in condensed form and feed them
        # to fastcluster, skipping the dense |members| x |members| block and its squareform copy
        # (the direct condensed vector holds ~4 |C|**2 bytes vs. ~12 |C|**2 for the dense round-trip).
        # The ordering matches squareform exactly, so the linkage is bit-identical to before.
        sub_condensed = gather_condensed_subvector(condensed_matrix, num_points, members)
        linkage_matrix = fastcluster.linkage(sub_condensed, method=backend)
        return fcluster(linkage_matrix, t=cutoff, criterion='distance')

    def _cluster_component_hdbscan(self, dense_block: np.ndarray, cutoff: float) -> np.ndarray:
        """
        Clusters a single connected component's dense precomputed distance block with HDBSCAN
        (density-based, opt-in alternative to agglomerative linkage). Two scales govern it, each
        decoupled from the partition and defaulting to clustering_cutoff (passed in as cutoff):
          * hdbscan_cluster_selection_epsilon - HDBSCAN's cluster_selection_epsilon, a hard merge
            floor: clusters HDBSCAN would split below it are kept merged (the branch-separation
            fineness).
          * hdbscan_noise_merge_cutoff - members HDBSCAN marks as noise (-1) but lying within this
            distance of an actual cluster are absorbed into the nearest such cluster by
            _reassign_hdbscan_noise_within_cutoff, so the merge floor also holds for noise: a tunnel
            cluster coincident with (or within the floor of) a real cluster never becomes its own
            spatially-overlapping singleton.
        Only noise farther than the noise-merge floor from every clustered point keeps label -1,
        which the stitching step turns into a singleton supercluster - preserving the 'every tunnel
        cluster maps to exactly one supercluster' invariant. A component smaller than
        min_cluster_size cannot form a cluster, so it is reported entirely as noise (-> singletons)
        without calling HDBSCAN.
        :param dense_block: dense |C|x|C| precomputed distance block of the component
        :param cutoff: clustering_cutoff, the fallback for both the selection epsilon and the noise
                       merge floor when their dedicated parameters are None
        :return: local integer label per member (-1 marks residual noise beyond the merge floor)
        """

        import hdbscan

        epsilon = self.parameters["hdbscan_cluster_selection_epsilon"]
        if epsilon is None:
            epsilon = cutoff
        noise_merge_cutoff = self.parameters["hdbscan_noise_merge_cutoff"]
        if noise_merge_cutoff is None:
            noise_merge_cutoff = cutoff

        min_cluster_size = self.parameters["hdbscan_min_cluster_size"]
        if dense_block.shape[0] < min_cluster_size:
            return np.full(dense_block.shape[0], -1, dtype=np.intp)

        clusterer = hdbscan.HDBSCAN(metric="precomputed",
                                    min_cluster_size=min_cluster_size,
                                    min_samples=self.parameters["hdbscan_min_samples"],
                                    cluster_selection_method=self.parameters["hdbscan_cluster_selection_method"],
                                    cluster_selection_epsilon=float(epsilon))
        local_labels = clusterer.fit_predict(np.ascontiguousarray(dense_block, dtype=np.float64))
        return self._reassign_hdbscan_noise_within_cutoff(local_labels, dense_block, noise_merge_cutoff)

    @staticmethod
    def _reassign_hdbscan_noise_within_cutoff(local_labels: np.ndarray, dense_block: np.ndarray,
                                              cutoff: float) -> np.ndarray:
        """
        Enforces clustering_cutoff as a hard merge floor for HDBSCAN noise: each point HDBSCAN
        labelled as noise (-1) but lying within cutoff of an actual cluster is absorbed into the
        nearest such cluster (single-link style - by the minimum distance to any clustered member).
        This stops a tunnel cluster that spatially overlaps a real cluster (down to a coincident
        0 A pair) from being emitted as its own singleton supercluster. Points farther than the
        cutoff from every clustered point keep label -1 and become singletons downstream.

        The reassignment is a single simultaneous pass keyed only on HDBSCAN's original clustered
        points: noise is never attached to other (reassigned) noise, so this cannot transitively
        chain distant points together the way single linkage would. Ties on the nearest clustered
        point are broken by lowest index, keeping the labelling deterministic.
        :param local_labels: per-member HDBSCAN labels (-1 marks noise)
        :param dense_block: dense |C|x|C| precomputed distance block these labels belong to
        :param cutoff: merge floor; noise within this distance of a cluster is absorbed into it
        :return: local labels with within-cutoff noise reassigned (a copy; input is not mutated)
        """

        local_labels = np.asarray(local_labels).copy()
        noise = np.flatnonzero(local_labels == -1)
        clustered = np.flatnonzero(local_labels != -1)
        if noise.size == 0 or clustered.size == 0:
            # nothing to absorb, or HDBSCAN found no cluster in this component to absorb into
            return local_labels

        # |noise| x |clustered| distances; nearest clustered point per noise point
        sub = dense_block[np.ix_(noise, clustered)]
        nearest = np.argmin(sub, axis=1)
        nearest_dist = sub[np.arange(noise.size), nearest]
        within = nearest_dist <= cutoff
        local_labels[noise[within]] = local_labels[clustered[nearest[within]]]
        return local_labels

    @staticmethod
    def _stitch_component_labels(cluster_labels: np.ndarray, members: np.ndarray,
                                 local_labels: np.ndarray, next_label: int) -> int:
        """
        Writes a component's local labels into the global label array under fresh global labels. Each
        distinct non-negative local label becomes one global label; each -1 (reserved for HDBSCAN
        noise) becomes its own singleton global label. The raw integers are cosmetic - only the
        grouping matters downstream - but they are assigned deterministically given the fixed member
        order.
        :param cluster_labels: global label array being filled (modified in place)
        :param members: global indices this component's labels belong to
        :param local_labels: per-member local labels (non-negative; -1 marks noise)
        :param next_label: next unused global label
        :return: advanced next_label
        """

        local_labels = np.asarray(local_labels)
        out = np.empty(local_labels.size, dtype=np.intp)

        noise_mask = local_labels == -1
        n_noise = int(np.count_nonzero(noise_mask))
        if n_noise:
            out[noise_mask] = next_label + np.arange(n_noise)
            next_label += n_noise

        clustered = local_labels[~noise_mask]
        if clustered.size:
            unique_labels, inverse = np.unique(clustered, return_inverse=True)
            out[~noise_mask] = next_label + inverse.ravel()
            next_label += unique_labels.size

        cluster_labels[members] = out
        return next_label

    @staticmethod
    def _log_component_size_histogram(components: List[np.ndarray], num_points: int):
        """
        Logs a one-line summary of the sub-cutoff component-size distribution at INFO. It explains
        both the runtime and the O(max |component|^2) peak memory of the partition driver, and is
        cheap to compute.
        :param components: per-component index arrays
        :param num_points: total number of tunnel clusters
        """

        if not components:
            return
        sizes = np.fromiter((member_indices.size for member_indices in components),
                            dtype=np.intp, count=len(components))
        logger.info("Sub-cutoff partition: %d tunnel clusters -> %d connected components "
                    "(largest %d, median %d, %d singletons).",
                    num_points, len(components), int(sizes.max()), int(np.median(sizes)),
                    int(np.count_nonzero(sizes == 1)))

    def merge_tunnel_clusters2super_clusters(self):
        """
        Performs clustering of tunnel clusters and creates of their superclusters (SCs)
        """

        # clean data produced by previous merging, if any are present
        folders2remove = list()
        folders2remove.append(self.parameters["statistics_folder"])
        folders2remove.append(os.path.join(self.parameters["internal_folder"], "super_cluster_pathsets"))
        folders2remove.append(os.path.join(self.parameters["internal_folder"], "super_cluster_profiles"))

        # Only remove specific visualization outputs, preserve sources generated before merging stage, except for superclusters CGOs
        if os.path.exists(self.parameters["visualization_folder"]):
            for item in os.listdir(self.parameters["visualization_folder"]):
                if item != "sources":  # Preserve the sources folder
                    folders2remove.append(os.path.join(self.parameters["visualization_folder"], item))
        folders2remove.append(os.path.join(self.parameters["visualization_folder"], "sources",
                                           "super_cluster_CGOs"))
                    
        if os.path.exists(self.parameters["data_folder"]):
            for folder in os.listdir(self.parameters["data_folder"]):
                if folder != "clustering":  # since this contains data from previous stage
                    folders2remove.append(os.path.join(self.parameters["data_folder"], folder))
        self.clear_results(self.parameters["overwrite"], folders2remove)

        total_num_md_sims = len(self.caver_input_folders)
        with TimeProcess("Clustering"):
            if self.parameters["clustering_method"] == "hdbscan":
                logger.info("Clustering tunnel clusters into superclusters using HDBSCAN density-based clustering "
                            "(min_cluster_size {:d}, min_samples {}, selection method '{}') with distance cutoff "
                            "{:.2f} A as a hard merge floor:".format(
                                self.parameters["hdbscan_min_cluster_size"],
                                self.parameters["hdbscan_min_samples"],
                                self.parameters["hdbscan_cluster_selection_method"],
                                self.parameters["clustering_cutoff"]))
            else:
                logger.info("Clustering tunnel clusters into superclusters using {}-linkage agglomerative clustering "
                            "with distance cutoff {:.2f} A:".format(self.parameters["clustering_linkage"],
                                                                    self.parameters["clustering_cutoff"]))

            # load the condensed pairwise dissimilarities (already in the upper-triangle form
            # fastcluster expects - no dense matrix is materialized)
            condensed_matrix, cluster_specifications, cluster_characteristics = self._load_distance_matrix()

            # cluster the tunnel clusters :)
            cluster_specs2label = self._assign_supercluster_labels(condensed_matrix, cluster_specifications)
            logger.info("Identified {:d} superclusters.".format(len(set(cluster_specs2label.values()))))

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
                    assert isinstance(cls_id, int), f"Expected int but got {type(cls_id).__name__}"
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
            progressbar(progress_counter, num_folders2process, self.parameters["log_level"])

            # compute transformation matrices from caver PDB files
            num_raw_paths = 0
            timeout = self.parameters["worker_task_timeout_s"]
            with get_context("spawn").Pool(processes=self.parameters["num_cpus"]) as pool:
                processing = list()
                tunnel_transform_mat = dict()
                for md_label in self.caver_input_folders:
                    md_folder = os.path.join(self.parameters["caver_results_path"], md_label)
                    in_pdb_file = utils.get_filepath(md_folder, self.parameters["caver_relative_pdb_file"])
                    processing.append(("caver-transform[{}]".format(md_label),
                                       pool.apply_async(get_transform_matrix,
                                                        args=(in_pdb_file, self.reference_pdb_file, md_label))))

                for matrix, md_label in utils.iter_pool_results(processing, timeout=timeout, pool=pool):
                    tunnel_transform_mat[md_label] = matrix
                    progress_counter += 1
                    progressbar(progress_counter, num_folders2process, self.parameters["log_level"])

                # Batch 2 (avg-starting-point) and batch 3 (aquaduct-transform) are independent of
                # each other - both only depend on batch 1's per-md_label transform matrices, which
                # are now in tunnel_transform_mat. Submit both up front so the pool's workers can
                # process them concurrently: batch 2 is small/fast (one tiny PDB read per md_label),
                # batch 3 is slow (tar.gz extraction + matrix computation). Draining batch 2 first
                # lets the launcher compute overall_starting_point and transform_mat2starting_point
                # while batch 3 is still running on the pool's other workers.
                batch2_processing = list()
                transformed_average_sp_x = list()
                transformed_average_sp_y = list()
                transformed_average_sp_z = list()
                for md_label in self.caver_input_folders:
                    md_folder = os.path.join(self.parameters["caver_results_path"], md_label)
                    in_origin_file = utils.get_filepath(md_folder, self.parameters["caver_relative_origin_file"])
                    batch2_processing.append(("avg-starting-point[{}]".format(md_label),
                                              pool.apply_async(average_starting_point,
                                                               args=(in_origin_file, md_label))))

                batch3_processing = list()
                aquaduct_transform_mat = dict()
                for md_label in self.aquaduct_input_folders:
                    md_folder = os.path.join(self.aquaduct_md_to_roots[md_label][0], md_label)
                    tar_file = utils.get_filepath(md_folder, self.parameters["aquaduct_results_relative_tarfile"])
                    batch3_processing.append(("aquaduct-transform[{}]".format(md_label),
                                              pool.apply_async(transform_aquaduct,
                                                               args=(md_label, tar_file,
                                                                     self.parameters["aquaduct_results_pdb_filename"],
                                                                     self.reference_pdb_file))))

                for md_original_sp, md_label in utils.iter_pool_results(batch2_processing,
                                                                        timeout=timeout, pool=pool):
                    transformed_sp = tunnel_transform_mat[md_label].dot(md_original_sp)
                    transformed_average_sp_x.append(transformed_sp[0])
                    transformed_average_sp_y.append(transformed_sp[1])
                    transformed_average_sp_z.append(transformed_sp[2])
                    progress_counter += 1
                    progressbar(progress_counter, num_folders2process, self.parameters["log_level"])

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

                # batch 3 has been running on pool workers in parallel with batch 2; drain it now
                for matrix, md_label, _num_raw_paths in utils.iter_pool_results(batch3_processing,
                                                                                timeout=timeout, pool=pool):
                    num_raw_paths += _num_raw_paths
                    aquaduct_transform_mat[md_label] = matrix
                    progress_counter += 1
                    progressbar(progress_counter, num_folders2process, self.parameters["log_level"])

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

                # save transformed origin pdb file
                in_origin_file = utils.get_filepath(os.path.join(self.parameters["caver_results_path"], md_label),
                                                    self.parameters["caver_relative_origin_file"])
                out_origin_file = os.path.join(self.transformation_folder, self.parameters["caver_foldername"],
                                               "v_origin-" + md_label + "-transformed.pdb")
                sp_coords = read_starting_points(in_origin_file)
                save_caver_starting_points(out_origin_file, sp_coords, tunnel_full_trans_mat)

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

        backend = self._resolve_stage_backend("stage02_backend")
        if backend == "slurm":
            self._process_tunnel_networks_slurm()
            return

        logger.info("Processing {:d} tunnel networks "
                    "using {:d} {}:".format(len(self.caver_input_folders), self.parameters["num_cpus"],
                                            process_count(self.parameters["num_cpus"])))
        logger.debug("The networks are read from sub-folders in '{}' folder that match the following pattern "
                     "'{}'.".format(self.parameters["caver_results_path"],
                                    self.parameters["caver_results_folder_pattern"]))
        logger.debug("Using the following reference file to align caver clusters: '{}'".format(self.reference_pdb_file))

        with TimeProcess("Processing"):
            with get_context("spawn").Pool(processes=self.parameters["num_cpus"]) as pool:
                processing = list()
                for md_label in self.caver_input_folders:
                    processing.append(("tunnel-network[{}]".format(md_label),
                                       pool.apply_async(self._pre_process_single_tunnel_network,
                                                        args=(md_label, self.parameters))))

                items2process = len(processing)
                progressbar(0, items2process, self.parameters["log_level"])
                timeout = self.parameters["worker_task_timeout_s"]
                for i, _ in enumerate(utils.iter_pool_results(processing, timeout=timeout, pool=pool)):
                    progressbar(i + 1, items2process, self.parameters["log_level"])

    def _process_tunnel_networks_slurm(self):
        """
        SLURM-backed counterpart to `process_tunnel_networks`. The launcher writes the
        canonical md_label enumeration to `<slurm_folder>/items.json` via the stage's
        `prepare_state()` hook, runs the SLURM array via `run_stage_on_slurm()`, then asserts
        every md_label was processed. The per-shard pickle stored under the slurm folder is a
        tiny JSON-shaped marker (the list of md_labels the shard handled); the real outputs -
        the `<md_label>_caver.dump` files in `orig_caver_network_data_path` and, when
        `visualize_transformed_tunnels` is True, the per-md visualisation folder - live in
        the regular user-facing paths and are tracked as side-effects backed up to a per-shard
        tar.gz so the resume path can restore a wiped output folder without re-reading CAVER.
        """

        from transport_tools.libs import slurm

        if self.config_file is None:
            raise RuntimeError("The 'slurm' stage-2 backend requires the analysis to be run "
                               "from a configuration file - the SLURM compute nodes need it "
                               "to rebuild the analysis state.")

        logger.info("Processing {:d} tunnel networks via SLURM array job:".format(
            len(self.caver_input_folders)))
        logger.debug("The networks are read from sub-folders in '{}' folder that match the "
                     "following pattern '{}'.".format(self.parameters["caver_results_path"],
                                                      self.parameters["caver_results_folder_pattern"]))

        with TimeProcess("Processing"):
            items: List[str] = list(self.caver_input_folders)
            stage = slurm.TunnelNetworksShardStage(items)
            slurm_folder = stage.get_slurm_folder(self.parameters)
            num_shards, still_missing = slurm.run_stage_on_slurm(stage, self.config_file,
                                                                 self.parameters)
            if still_missing:
                raise RuntimeError("SLURM tunnel-networks shard(s) {} did not produce "
                                   "results; inspect their logs in '{}'. {}/{} shard(s) "
                                   "completed - re-run stage 2 to resume from them.".format(
                                       still_missing, slurm_folder,
                                       num_shards - len(still_missing), num_shards))

            # Verify completeness: collect every md_label reported by the shards and assert
            # the set matches the original enumeration. Catches the failure mode where a
            # shard succeeded but silently processed a subset (e.g. a CAVER folder that
            # vanished from the shared FS between submission and run); the dump-file
            # side-effect manifest would still validate, so the explicit set check is what
            # makes the launcher fail loudly here.
            processed: List[str] = list()
            for shard_id in range(num_shards):
                with open(stage.shard_result_path(slurm_folder, shard_id), "rb") as in_stream:
                    processed.extend(pickle.load(in_stream))
            expected = set(items)
            actual = set(processed)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise RuntimeError("SLURM tunnel-networks shards processed an unexpected set "
                                   "of md_labels. Missing: {}. Unexpected: {}. Inspect the "
                                   "shard logs in '{}'.".format(missing, extra, slurm_folder))

            # Assembly succeeded end-to-end (every md_label is accounted for and its dump +
            # optional visualisation files are on disk in the user-facing folders); drop the
            # per-shard cache unless the user opted in to keep it.
            slurm.cleanup_stage_folder(stage, self.parameters)

    def compute_tunnel_networks_shard(self, num_shards: int, shard_id: int) -> int:
        """
        Compute one shard of the stage-2 tunnel-networks work - the strided subset of the
        flat md_label enumeration assigned to shard_id. Used by the SLURM stage-2 backend;
        executed on a SLURM compute node. The enumeration was written by the launcher into
        `<slurm_folder>/items.json` (a list of md_label strings in original submission order);
        we read it from there instead of re-deriving it via the config so every shard agrees
        on which md_labels are owned by which shard. For each md_label we own, we run the
        existing `_pre_process_single_tunnel_network` over an inner spawn-Pool sized to
        `slurm_cpus_per_task` (CAVER parsing + numpy transformation per MD is moderate, so an
        inner pool inside the shard amortises shard-startup costs across many md_labels), then
        write the per-shard side-effects archive (`<md_label>_caver.dump` + optional viz
        files) + manifest + result pickle (a tiny JSON-shaped list of processed md_labels).
        :param num_shards: total number of shards the work is split into
        :param shard_id: 0-based ID of the shard to compute
        :return: number of md_labels this shard processed
        """

        from transport_tools.libs.slurm import TunnelNetworksShardStage

        slurm_folder = TunnelNetworksShardStage.get_slurm_folder(self.parameters)
        all_items = TunnelNetworksShardStage.load_items(slurm_folder)

        # strided assignment: shard i gets every Nth item from the enumeration. Identical
        # pattern to the tunnel-layering / aquaduct-layering shards
        my_items: List[str] = all_items[shard_id::num_shards]
        if not my_items:
            logger.info("SLURM tunnel-networks shard %d/%d: nothing to do.",
                        shard_id + 1, num_shards)
            TunnelNetworksShardStage.write_shard_manifest(slurm_folder, shard_id, [])
            self._write_tunnel_networks_shard_result(slurm_folder, shard_id, [])
            return 0

        n_jobs = len(my_items)
        num_cpus = self.parameters["slurm_cpus_per_task"]
        logger.info("SLURM tunnel-networks shard %d/%d: processing %d md_label(s) using %d %s.",
                    shard_id + 1, num_shards, n_jobs, num_cpus, process_count(num_cpus))

        progressbar(0, n_jobs, self.parameters["log_level"])
        done = 0
        processed_md_labels: List[str] = list()

        # 'spawn' inner pool - identical pattern to compute_tunnel_layering_shard. The worker
        # is a @staticmethod (already module-importable via the class) and takes only
        # (md_label, parameters), so no special initializer is needed beyond the shared BLAS
        # thread cap. iter_pool_results preserves submission order; the worker returns None,
        # so we walk my_items in parallel with the iterator to record which md_label each
        # finished result corresponds to.
        with get_context("spawn").Pool(processes=num_cpus,
                                       initializer=utils.cap_blas_threads_for_worker,
                                       initargs=(num_cpus, num_cpus)) as pool:
            timeout = self.parameters["worker_task_timeout_s"]
            processing = list()
            for md_label in my_items:
                processing.append(("tunnel-network[{}]".format(md_label),
                                   pool.apply_async(self._pre_process_single_tunnel_network,
                                                    args=(md_label, self.parameters))))
            for md_label, _result in zip(my_items,
                                          utils.iter_pool_results(processing,
                                                                   timeout=timeout, pool=pool)):
                processed_md_labels.append(md_label)
                done += 1
                progressbar(done, n_jobs, self.parameters["log_level"])

        # Persist the side-effects backup (per-md dump + optional viz folder) BEFORE the
        # result pickle so the existence of the pickle implies the manifest + archive are
        # present. Mirrors the tunnel-layering shard's ordering. The dump files are always
        # produced regardless of `visualize_transformed_tunnels`; the visualisation files are
        # gated on the knob. derive_side_effect_paths() reflects both branches.
        side_effects = TunnelNetworksShardStage.derive_side_effect_paths(
            self.parameters, processed_md_labels)
        TunnelNetworksShardStage.write_side_effects_archive(
            slurm_folder, shard_id, side_effects, self.parameters["output_path"])
        TunnelNetworksShardStage.write_shard_manifest(slurm_folder, shard_id, side_effects)

        self._write_tunnel_networks_shard_result(slurm_folder, shard_id, processed_md_labels)
        return len(processed_md_labels)

    def _write_tunnel_networks_shard_result(self, slurm_folder: str, shard_id: int,
                                            processed_md_labels: List[str]):
        """
        Atomically persist a tunnel-networks shard's results to its per-shard pickle file
        (write to `.tmp.pkl`, then os.replace). The "result" here is just a list of md_labels
        the shard processed - the real outputs (orig_network dumps + optional visualisations)
        live under the user-facing internal/visualisation folders and are tracked as
        side-effects. Kept separate so the empty-shard fast path can reuse it without
        duplicating the atomic-write idiom.
        """

        from transport_tools.libs.slurm import TunnelNetworksShardStage

        out_file = TunnelNetworksShardStage.shard_result_path(slurm_folder, shard_id)
        tmp_file = out_file + ".tmp.pkl"
        with open(tmp_file, "wb") as out_stream:
            pickle.dump(list(processed_md_labels), out_stream,
                        protocol=self.parameters["pickle_protocol"])
        os.replace(tmp_file, out_file)

    def create_layered_description4tunnel_networks(self):
        """
        Creates layered representation of tunnel networks for all MD simulations
        """

        backend = self._resolve_stage_backend("stage03_backend")
        if backend == "slurm":
            self._create_layered_description4tunnel_networks_slurm()
            return

        with TimeProcess("Layering"):
            num_cpus = self.parameters["num_cpus"]
            with get_context("spawn").Pool(processes=num_cpus,
                                           initializer=utils.cap_blas_threads_for_worker,
                                           initargs=(num_cpus, num_cpus)) as pool:
                clusters_to_process: list = list()
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
                        clusters_to_process.append(cluster)
                        cls_ids2process4md_label[md_label].append(cluster.cluster_id)

                items2process = len(clusters_to_process)
                if not items2process > 0:
                    raise RuntimeError("Not enough tunnel clusters are available to perform their layering")

                logger.info("Computing layered representation for {:d} tunnel clusters "
                            "using {:d} {}:".format(items2process, num_cpus, process_count(num_cpus)))

                progressbar(0, items2process, self.parameters["log_level"])
                timeout = self.parameters["worker_task_timeout_s"]
                imap_iter = pool.imap_unordered(create_layered_cluster_worker, clusters_to_process)
                # On timeout/failure, log the last few completed identities to point at the region
                # of the workload where the stall happened (workers return (cls_id, md_label, ...)).
                results_iter = utils.iter_imap_results(
                    imap_iter, timeout=timeout, pool=pool, label="layer-tunnel-cluster",
                    extract_task_label=lambda r: "{}/{}".format(r[1], r[0]))
                # imap_unordered yields in completion order, but the saved layered_entities dict
                # must be inserted in submission order so its pickled byte layout (and therefore
                # the integration-test reference data) is stable regardless of worker scheduling.
                # Buffer per-md_label results as they stream in and commit each md_label's entities
                # in the original cls_id order as soon as the last cluster for that md_label
                # arrives - this preserves the per-md_label save+free memory profile of the prior
                # streaming implementation.
                buffered_layered_paths: Dict[str, Dict[int, LayeredPathSet]] = {
                    md: {} for md in self.caver_input_folders}
                expected_count: Dict[str, int] = {
                    md: len(cls_ids2process4md_label[md]) for md in self.caver_input_folders}
                seen_count: Dict[str, int] = {md: 0 for md in self.caver_input_folders}
                for i, result in enumerate(results_iter):
                    cls_id, md_label, layered_path_set = result
                    seen_count[md_label] += 1
                    if layered_path_set.is_empty():
                        logger.warning("Cluster {} of {} cannot be layered".format(cls_id, md_label))
                        cls_ids2process4md_label[md_label].remove(cls_id)
                    else:
                        buffered_layered_paths[md_label][cls_id] = layered_path_set

                    if seen_count[md_label] == expected_count[md_label]:
                        # All workers for this md_label have reported; commit in submission order.
                        for c_id in cls_ids2process4md_label[md_label]:
                            tunnel_networks[md_label].add_layered_entity(
                                c_id, buffered_layered_paths[md_label][c_id])
                        logger.debug("Finished layering of network for '{}'.".format(md_label))
                        if self.parameters["visualize_layered_clusters"]:
                            tunnel_networks[md_label].save_layered_visualization(save_pdb_files=True)
                        tunnel_networks[md_label].save_layered_network()
                        del tunnel_networks[md_label]
                        del buffered_layered_paths[md_label]

                    progressbar(i + 1, items2process, self.parameters["log_level"])

    def _create_layered_description4tunnel_networks_slurm(self):
        """
        SLURM-backed counterpart to `create_layered_description4tunnel_networks`. The launcher
        enumerates the (md_label, cluster_id) pairs once (in the same submission order the
        local backend uses), persists them to `<slurm_folder>/items.json` so every shard reads
        a single artifact instead of re-deriving the enumeration from the orig_network.dump
        files, runs the SLURM array via `run_stage_on_slurm()`, then assembles the per-shard
        pickles into the same per-md_label `layered_network.dump` artifacts the local backend
        produces (and the integration test compares byte-for-byte).
        """

        from transport_tools.libs import slurm

        if self.config_file is None:
            raise RuntimeError("The 'slurm' stage-3 backend requires the analysis to be run "
                               "from a configuration file - the SLURM compute nodes need it "
                               "to rebuild the analysis state.")

        with TimeProcess("Layering"):
            # 1) Enumerate (md_label, cluster_id) pairs in stable submission order.
            #    Mirrors the local path's enumeration so produced artifacts are byte-identical.
            items: List[Tuple[str, int]] = list()
            cls_ids2process4md_label: Dict[str, List[int]] = dict()
            for md_label in self.caver_input_folders:
                tunnel_network = TunnelNetwork(self.parameters, md_label)
                # enumerate via the size sidecar to avoid loading every orig_network in full on
                # this single launcher node (falls back to a full load for legacy/stale sidecars)
                cls_ids2process4md_label[md_label] = list()
                for cls_id in tunnel_network.get_clusters4layering_ids():
                    items.append((md_label, int(cls_id)))
                    cls_ids2process4md_label[md_label].append(int(cls_id))

            items2process = len(items)
            if not items2process > 0:
                raise RuntimeError("Not enough tunnel clusters are available to perform their layering")

            logger.info("Computing layered representation for {:d} tunnel clusters via SLURM "
                        "array job:".format(items2process))

            # 2) Hand the enumeration to the stage; run_stage_on_slurm will write items.json
            #    (prepare_state hook), check the fingerprint, submit only missing shards,
            #    and poll in completion order.
            stage = slurm.TunnelLayeringShardStage(items)
            slurm_folder = stage.get_slurm_folder(self.parameters)
            num_shards, still_missing = slurm.run_stage_on_slurm(stage, self.config_file,
                                                                 self.parameters)
            if still_missing:
                raise RuntimeError("SLURM tunnel-layering shard(s) {} did not produce results; "
                                   "inspect their logs in '{}'. {}/{} shard(s) completed - "
                                   "re-run stage 3 to resume from them.".format(
                                       still_missing, slurm_folder,
                                       num_shards - len(still_missing), num_shards))

            # 3) Stream the per-shard pickles back. Each result is (cls_id, md_label,
            #    LayeredPathSet); buffer per md_label and commit in original submission order
            #    so the saved layered_network.dump matches the local backend byte-for-byte.
            buffered_layered_paths: Dict[str, Dict[int, LayeredPathSet]] = {
                md: {} for md in self.caver_input_folders}
            empty_clusters: List[Tuple[str, int]] = list()
            total_received = 0
            for shard_id in range(num_shards):
                with open(stage.shard_result_path(slurm_folder, shard_id), "rb") as in_stream:
                    shard_results: List[Tuple[int, str, LayeredPathSet]] = pickle.load(in_stream)
                for cls_id, md_label, layered_path_set in shard_results:
                    total_received += 1
                    if layered_path_set.is_empty():
                        empty_clusters.append((md_label, cls_id))
                    else:
                        buffered_layered_paths[md_label][cls_id] = layered_path_set

            if total_received != items2process:
                raise RuntimeError("SLURM tunnel-layering shards returned {} result(s) but the "
                                   "launcher enumerated {} (md_label, cluster_id) pair(s); some "
                                   "shards likely failed silently. Inspect the shard logs in "
                                   "'{}'.".format(total_received, items2process, slurm_folder))

            # report and drop empty clusters from the per-md_label order so the local-path
            # save_layered_network() path sees the same list it would have seen locally
            for md_label, cls_id in empty_clusters:
                logger.warning("Cluster {} of {} cannot be layered".format(cls_id, md_label))
                cls_ids2process4md_label[md_label].remove(cls_id)

            # 4) Commit per md_label in submission order, save layered networks/visualisations.
            #    Progress is reported per md_label (not per cluster) - clusters are already
            #    finished at this point, and the heavy work that remains is the
            #    save_layered_network() pickle dump which has md_label granularity.
            num_md = len(self.caver_input_folders)
            progressbar(0, num_md, self.parameters["log_level"])
            for i, md_label in enumerate(self.caver_input_folders):
                tunnel_network = TunnelNetwork(self.parameters, md_label)
                for cls_id in cls_ids2process4md_label[md_label]:
                    tunnel_network.add_layered_entity(cls_id, buffered_layered_paths[md_label][cls_id])
                logger.debug("Finished layering of network for '{}'.".format(md_label))
                if self.parameters["visualize_layered_clusters"]:
                    tunnel_network.save_layered_visualization(save_pdb_files=True)
                tunnel_network.save_layered_network()
                progressbar(i + 1, num_md, self.parameters["log_level"])

            # Assembly succeeded end-to-end (every md_label's layered_network.dump and
            # optional per-cluster visualisations are on disk); drop the per-shard cache
            # unless the user opted in to keep it.
            slurm.cleanup_stage_folder(stage, self.parameters)

    @staticmethod
    def _pre_process_single_aquaduct_network(md_label: str, parameters: dict, root_paths: List[str],
                                             parallel_processing: bool = True):
        """
        Transformation and visualization of original AQUA-DUCT network for a single MD simulation.
        Merges paths from all root paths that contain this md_label.
        :param md_label: name of folder with the source MD simulation data
        :param parameters: job configuration parameters
        :param root_paths: pre-filtered list of aquaduct_results_path entries that actually contain this
                           md_label with the required tar/summary files (already accounts for
                           aquaduct_allow_empty_folders)
        :param parallel_processing: if we process the raw_paths in parallel
        """

        # Guard against nested-pool deadlocks: if this function is invoked from inside a
        # multiprocessing worker (parent_process() is non-None), the inner per-raw-path Pool
        # would create children that the worker process cannot schedule. The caller is
        # responsible for passing parallel_processing=False in that case; fail loudly otherwise.
        if parallel_processing and parent_process() is not None:
            raise RuntimeError("_pre_process_single_aquaduct_network was called from a Pool "
                               "worker with parallel_processing=True; pass parallel_processing="
                               "False when invoking from inside a Pool to avoid a nested pool.")

        logger.debug("Processing AQUA-DUCT network from {} ({} source(s)).".format(md_label, len(root_paths)))
        with AquaductNetwork(parameters, md_label, root_path=root_paths[0]) as aquanet:
            aquanet.read_raw_paths_data(parallel_processing)
            for extra_root in root_paths[1:]:
                extra_net = AquaductNetwork(parameters, md_label, load_only=True, root_path=extra_root)
                extra_net.read_raw_paths_data(parallel_processing)
                aquanet.orig_entities.extend(extra_net.orig_entities)
            if parameters["visualize_transformed_transport_events"]:
                aquanet.save_orig_network_visualization()
            aquanet.save_orig_network()

    def process_aquaduct_networks(self):
        """
        Process AQUA-DUCT networks for all MD simulations
        """

        if not self.aquaduct_input_folders:
            raise RuntimeError("No data with AQUA-DUCT results were loaded to enable their analyses")

        backend = self._resolve_stage_backend("stage07_backend")
        if backend == "slurm":
            self._process_aquaduct_networks_slurm()
            return

        items2process = len(self.aquaduct_input_folders)
        logger.info("Processing {:d} AQUA-DUCT networks "
                    "using {:d} {}:".format(items2process, self.parameters["num_cpus"],
                                            process_count(self.parameters["num_cpus"])))
        logger.debug("The networks are read from sub-folders matching pattern '{}' in: {}.".format(
            self.parameters["aquaduct_results_folder_pattern"],
            ", ".join(self.parameters["aquaduct_results_path"])))
        logger.debug("Using the following reference file to "
                     "align AQUA-DUCT networks: '{}'".format(self.reference_pdb_file))

        if not items2process > 0:
            raise RuntimeError("Not enough AQUA-DUCT networks are available to perform their layering")

        with TimeProcess("Processing"):
            progressbar(0, items2process, self.parameters["log_level"])
            if self._aquaduct_single_event_inputs:
                num_cpus = self.parameters["num_cpus"]
                with get_context("spawn").Pool(processes=num_cpus,
                                               initializer=utils.cap_blas_threads_for_worker,
                                               initargs=(num_cpus, num_cpus)) as pool:
                    tasks = [(md_label, self.parameters, self.aquaduct_md_to_roots[md_label], False)
                             for md_label in self.aquaduct_input_folders]
                    # Submitted md_labels in submission order, so the i-th task corresponds to
                    # aquaduct_input_folders[i] for the timeout extract_task_label callback.
                    submitted = [md_label for md_label in self.aquaduct_input_folders]
                    timeout = self.parameters["worker_task_timeout_s"]
                    imap_iter = pool.imap_unordered(process_aquaduct_network_worker, tasks)
                    # The worker returns None (it just writes results to disk), so identity has to
                    # come from completion count - we report it via "N task(s) completed" suffix.
                    results_iter = utils.iter_imap_results(
                        imap_iter, timeout=timeout, pool=pool,
                        label="aquaduct-network[{} MD label(s)]".format(len(submitted)))
                    for i, _ in enumerate(results_iter):
                        progressbar(i + 1, items2process, self.parameters["log_level"])
            else:
                for i, md_label in enumerate(self.aquaduct_input_folders):
                    self._pre_process_single_aquaduct_network(md_label, self.parameters,
                                                              self.aquaduct_md_to_roots[md_label], True)
                    progressbar(i + 1, items2process, self.parameters["log_level"])

    def _process_aquaduct_networks_slurm(self):
        """
        SLURM-backed counterpart to `process_aquaduct_networks`. Mirror of
        `_process_tunnel_networks_slurm` for the AQUA-DUCT side. The launcher writes the
        canonical md_label enumeration to `<slurm_folder>/items.json` via the stage's
        `prepare_state()` hook, runs the SLURM array via `run_stage_on_slurm()`, then asserts
        every md_label was processed. The per-shard pickle stored under the slurm folder is a
        tiny JSON-shaped marker (the list of md_labels the shard handled); the real outputs -
        the `<md_label>_aqua.dump` files in `orig_aquaduct_network_data_path` and, when
        `visualize_transformed_transport_events` is True, the per-md visualisation folder -
        live in the regular user-facing paths and are tracked as side-effects backed up to a
        per-shard tar.gz so the resume path can restore a wiped output folder without
        re-extracting the AQUA-DUCT tarballs.
        """

        from transport_tools.libs import slurm

        if self.config_file is None:
            raise RuntimeError("The 'slurm' stage-7 backend requires the analysis to be run "
                               "from a configuration file - the SLURM compute nodes need it "
                               "to rebuild the analysis state.")

        logger.info("Processing {:d} AQUA-DUCT networks via SLURM array job:".format(
            len(self.aquaduct_input_folders)))
        logger.debug("The networks are read from sub-folders matching pattern '{}' in: {}.".format(
            self.parameters["aquaduct_results_folder_pattern"],
            ", ".join(self.parameters["aquaduct_results_path"])))

        with TimeProcess("Processing"):
            items: List[str] = list(self.aquaduct_input_folders)
            stage = slurm.AquaductNetworksShardStage(items)
            slurm_folder = stage.get_slurm_folder(self.parameters)
            num_shards, still_missing = slurm.run_stage_on_slurm(stage, self.config_file,
                                                                 self.parameters)
            if still_missing:
                raise RuntimeError("SLURM aquaduct-networks shard(s) {} did not produce "
                                   "results; inspect their logs in '{}'. {}/{} shard(s) "
                                   "completed - re-run stage 7 to resume from them.".format(
                                       still_missing, slurm_folder,
                                       num_shards - len(still_missing), num_shards))

            # Verify completeness: collect every md_label reported by the shards and assert
            # the set matches the original enumeration. Catches the failure mode where a
            # shard succeeded but silently processed a subset (e.g. an AQUA-DUCT folder that
            # vanished from the shared FS between submission and run); the dump-file
            # side-effect manifest would still validate, so the explicit set check is what
            # makes the launcher fail loudly here.
            processed: List[str] = list()
            for shard_id in range(num_shards):
                with open(stage.shard_result_path(slurm_folder, shard_id), "rb") as in_stream:
                    processed.extend(pickle.load(in_stream))
            expected = set(items)
            actual = set(processed)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise RuntimeError("SLURM aquaduct-networks shards processed an unexpected "
                                   "set of md_labels. Missing: {}. Unexpected: {}. Inspect "
                                   "the shard logs in '{}'.".format(missing, extra,
                                                                     slurm_folder))

            # Assembly succeeded end-to-end (every md_label's aqua.dump and optional
            # visualisation files are on disk in the user-facing folders); drop the per-shard
            # cache unless the user opted in to keep it.
            slurm.cleanup_stage_folder(stage, self.parameters)

    def compute_aquaduct_networks_shard(self, num_shards: int, shard_id: int) -> int:
        """
        Compute one shard of the stage-7 aquaduct-networks work - the strided subset of the
        flat md_label enumeration assigned to shard_id. Used by the SLURM stage-7 backend;
        executed on a SLURM compute node. The enumeration was written by the launcher into
        `<slurm_folder>/items.json` (a list of md_label strings in original submission order);
        we read it from there instead of re-deriving it via the config so every shard agrees
        on which md_labels are owned by which shard. For each md_label we own, we run the
        existing `_pre_process_single_aquaduct_network` over an inner spawn-Pool sized to
        `slurm_cpus_per_task` with `parallel_processing=False` so the inner raw-paths pool is
        never spawned from a worker (the nested-pool guard at the top of
        `_pre_process_single_aquaduct_network` would otherwise raise). Per-MD work
        (tar.gz extraction + numpy transformation of every traced raw path) is moderate, so an
        inner pool inside the shard amortises shard-startup costs across many md_labels. After
        all workers complete, we write the per-shard side-effects archive (`<md_label>_aqua.dump`
        + optional viz files) + manifest + result pickle (a tiny JSON-shaped list of processed
        md_labels).
        :param num_shards: total number of shards the work is split into
        :param shard_id: 0-based ID of the shard to compute
        :return: number of md_labels this shard processed
        """

        from transport_tools.libs.slurm import AquaductNetworksShardStage

        slurm_folder = AquaductNetworksShardStage.get_slurm_folder(self.parameters)
        all_items = AquaductNetworksShardStage.load_items(slurm_folder)

        # strided assignment: shard i gets every Nth item from the enumeration. Identical
        # pattern to the tunnel-networks / layering shards.
        my_items: List[str] = all_items[shard_id::num_shards]
        if not my_items:
            logger.info("SLURM aquaduct-networks shard %d/%d: nothing to do.",
                        shard_id + 1, num_shards)
            AquaductNetworksShardStage.write_shard_manifest(slurm_folder, shard_id, [])
            AquaductNetworksShardStage.write_side_effects_archive(
                slurm_folder, shard_id, [], self.parameters["output_path"])
            self._write_aquaduct_networks_shard_result(slurm_folder, shard_id, [])
            return 0

        n_jobs = len(my_items)
        num_cpus = self.parameters["slurm_cpus_per_task"]
        logger.info("SLURM aquaduct-networks shard %d/%d: processing %d md_label(s) using "
                    "%d %s.", shard_id + 1, num_shards, n_jobs, num_cpus,
                    process_count(num_cpus))

        progressbar(0, n_jobs, self.parameters["log_level"])
        done = 0
        processed_md_labels: List[str] = list()

        # 'spawn' inner pool - identical pattern to compute_tunnel_networks_shard. We pass
        # parallel_processing=False so the inner raw-paths pool inside
        # _pre_process_single_aquaduct_network is NOT spawned from these worker processes -
        # the nested-pool guard at the top of that function would otherwise raise. iter_pool_results
        # preserves submission order; the worker returns None, so we walk my_items in parallel
        # with the iterator to record which md_label each finished result corresponds to.
        with get_context("spawn").Pool(processes=num_cpus,
                                       initializer=utils.cap_blas_threads_for_worker,
                                       initargs=(num_cpus, num_cpus)) as pool:
            timeout = self.parameters["worker_task_timeout_s"]
            tasks = [(md_label, self.parameters, self.aquaduct_md_to_roots[md_label], False)
                     for md_label in my_items]
            processing = list()
            for md_label, task in zip(my_items, tasks):
                processing.append(("aquaduct-network[{}]".format(md_label),
                                   pool.apply_async(process_aquaduct_network_worker,
                                                    args=(task,))))
            for md_label, _result in zip(my_items,
                                          utils.iter_pool_results(processing,
                                                                   timeout=timeout, pool=pool)):
                processed_md_labels.append(md_label)
                done += 1
                progressbar(done, n_jobs, self.parameters["log_level"])

        # Persist the side-effects backup (per-md dump + optional viz folder) BEFORE the
        # result pickle so the existence of the pickle implies the manifest + archive are
        # present. Mirrors the tunnel-networks shard's ordering. The dump files are always
        # produced regardless of `visualize_transformed_transport_events`; the visualisation
        # files are gated on the knob. derive_side_effect_paths() reflects both branches.
        side_effects = AquaductNetworksShardStage.derive_side_effect_paths(
            self.parameters, processed_md_labels)
        AquaductNetworksShardStage.write_side_effects_archive(
            slurm_folder, shard_id, side_effects, self.parameters["output_path"])
        AquaductNetworksShardStage.write_shard_manifest(slurm_folder, shard_id, side_effects)

        self._write_aquaduct_networks_shard_result(slurm_folder, shard_id, processed_md_labels)
        return len(processed_md_labels)

    def _write_aquaduct_networks_shard_result(self, slurm_folder: str, shard_id: int,
                                              processed_md_labels: List[str]):
        """
        Atomically persist an aquaduct-networks shard's results to its per-shard pickle file
        (write to `.tmp.pkl`, then os.replace). The "result" here is just a list of md_labels
        the shard processed - the real outputs (orig_network dumps + optional visualisations)
        live under the user-facing internal/visualisation folders and are tracked as
        side-effects. Kept separate so the empty-shard fast path can reuse it without
        duplicating the atomic-write idiom.
        """

        from transport_tools.libs.slurm import AquaductNetworksShardStage

        out_file = AquaductNetworksShardStage.shard_result_path(slurm_folder, shard_id)
        tmp_file = out_file + ".tmp.pkl"
        with open(tmp_file, "wb") as out_stream:
            pickle.dump(list(processed_md_labels), out_stream,
                        protocol=self.parameters["pickle_protocol"])
        os.replace(tmp_file, out_file)

    def create_layered_description4aquaduct_networks(self):
        """
        Creates layered representation of AquaDuct networks for all MD simulations
        """

        backend = self._resolve_stage_backend("stage08_backend")
        if backend == "slurm":
            self._create_layered_description4aquaduct_networks_slurm()
            return

        with TimeProcess("Layering"):
            num_cpus = self.parameters["num_cpus"]
            with get_context("spawn").Pool(processes=num_cpus,
                                           initializer=utils.cap_blas_threads_for_worker,
                                           initargs=(num_cpus, num_cpus)) as pool:
                events_to_process: list = list()
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
                        events_to_process.append(event)
                        event_ids2process4md_label[md_label].append(event.entity_label)

                items2process = len(events_to_process)
                logger.info("Computing layered representation for {:d} transport events "
                            "using {:d} {}:".format(items2process, num_cpus, process_count(num_cpus)))

                if not items2process > 0:
                    raise RuntimeError("Not enough transport events are available to perform their layering")

                progressbar(0, items2process, self.parameters["log_level"])
                timeout = self.parameters["worker_task_timeout_s"]
                imap_iter = pool.imap_unordered(create_layered_event_worker, events_to_process)
                # Workers return (event_id, md_label, ...); use that for last-completed labels.
                results_iter = utils.iter_imap_results(
                    imap_iter, timeout=timeout, pool=pool, label="layer-aquaduct-event",
                    extract_task_label=lambda r: "{}/{}".format(r[1], r[0]))
                # imap_unordered yields in completion order, but the saved layered_entities dict
                # must be inserted in submission order so its pickled byte layout (and the
                # integration-test reference data) is stable regardless of worker scheduling.
                # Buffer per-md_label results and commit each md_label's entries in the original
                # event order once its last event arrives - same pattern as the tunnel-layering
                # site above; preserves the per-md_label save+free memory profile.
                buffered_layered_paths: Dict[str, Dict[str, LayeredPathSet]] = {
                    md: {} for md in self.aquaduct_input_folders}
                expected_count: Dict[str, int] = {
                    md: len(event_ids2process4md_label[md]) for md in self.aquaduct_input_folders}
                seen_count: Dict[str, int] = {md: 0 for md in self.aquaduct_input_folders}
                for i, result in enumerate(results_iter):
                    event_id, md_label, layered_path_set = result
                    seen_count[md_label] += 1
                    if layered_path_set.is_empty():
                        logger.warning("Event {} of {} cannot be layered".format(event_id, md_label))
                        event_ids2process4md_label[md_label].remove(event_id)
                    else:
                        buffered_layered_paths[md_label][event_id] = layered_path_set

                    if seen_count[md_label] == expected_count[md_label]:
                        # All workers for this md_label have reported; commit in submission order.
                        for e_id in event_ids2process4md_label[md_label]:
                            aqua_networks[md_label].add_layered_entity(
                                e_id, buffered_layered_paths[md_label][e_id])
                        logger.debug("Finished layering of network for '{}'.".format(md_label))
                        # always saving layered visualizations to enable visualization of assigned events later
                        aqua_networks[md_label].get_pdb_file()
                        aqua_networks[md_label].save_layered_visualization(self.parameters["visualize_layered_events"])
                        aqua_networks[md_label].save_layered_network()
                        aqua_networks[md_label].clean_tempfile()
                        del aqua_networks[md_label]
                        del buffered_layered_paths[md_label]

                    progressbar(i + 1, items2process, self.parameters["log_level"])

    def _create_layered_description4aquaduct_networks_slurm(self):
        """
        SLURM-backed counterpart to `create_layered_description4aquaduct_networks`. Mirror of
        `_create_layered_description4tunnel_networks_slurm` for transport events. The launcher
        enumerates the (md_label, event_label) pairs once (in the same submission order the
        local backend uses), persists them to `<slurm_folder>/items.json` so every shard reads
        a single artifact instead of re-deriving the enumeration from the
        aqua_orig_network.dump files, runs the SLURM array via `run_stage_on_slurm()`, then
        assembles the per-shard pickles into the same per-md_label `aqua_layered_network.dump`
        artifacts the local backend produces (and the integration test compares byte-for-byte).
        """

        from transport_tools.libs import slurm

        if self.config_file is None:
            raise RuntimeError("The 'slurm' stage-8 backend requires the analysis to be run "
                               "from a configuration file - the SLURM compute nodes need it "
                               "to rebuild the analysis state.")

        with TimeProcess("Layering"):
            # 1) Enumerate (md_label, event_label) pairs in stable submission order.
            #    Mirrors the local path's enumeration so produced artifacts are byte-identical.
            #    Enumerate via the per-md .event_ids sidecar (get_events4layering_ids) to avoid
            #    loading every orig_network in full on this single launcher node (falls back to
            #    a full load for legacy/missing sidecars); the per-md AquaductNetwork objects
            #    are constructed lazily in the assembly loop below rather than held resident
            #    across the whole array wait.
            items: List[Tuple[str, str]] = list()
            event_ids2process4md_label: Dict[str, List[str]] = dict()
            for md_label in self.aquaduct_input_folders:
                aquanet = AquaductNetwork(self.parameters, md_label, load_only=True)
                event_ids2process4md_label[md_label] = list()
                for event_id in aquanet.get_events4layering_ids():
                    items.append((md_label, str(event_id)))
                    event_ids2process4md_label[md_label].append(str(event_id))

            items2process = len(items)
            if not items2process > 0:
                raise RuntimeError("Not enough transport events are available to perform their layering")

            logger.info("Computing layered representation for {:d} transport events via SLURM "
                        "array job:".format(items2process))

            # 2) Hand the enumeration to the stage; run_stage_on_slurm will write items.json
            #    (prepare_state hook), check the fingerprint, submit only missing shards,
            #    and poll in completion order.
            stage = slurm.AquaductLayeringShardStage(items)
            slurm_folder = stage.get_slurm_folder(self.parameters)
            num_shards, still_missing = slurm.run_stage_on_slurm(stage, self.config_file,
                                                                 self.parameters)
            if still_missing:
                raise RuntimeError("SLURM aquaduct-layering shard(s) {} did not produce results; "
                                   "inspect their logs in '{}'. {}/{} shard(s) completed - "
                                   "re-run stage 8 to resume from them.".format(
                                       still_missing, slurm_folder,
                                       num_shards - len(still_missing), num_shards))

            # 3) Stream the per-shard pickles back. Each result is (event_label, md_label,
            #    LayeredPathSet); buffer per md_label and commit in original submission order
            #    so the saved aqua_layered_network.dump matches the local backend byte-for-byte.
            buffered_layered_paths: Dict[str, Dict[str, LayeredPathSet]] = {
                md: {} for md in self.aquaduct_input_folders}
            empty_events: List[Tuple[str, str]] = list()
            total_received = 0
            for shard_id in range(num_shards):
                with open(stage.shard_result_path(slurm_folder, shard_id), "rb") as in_stream:
                    shard_results: List[Tuple[str, str, LayeredPathSet]] = pickle.load(in_stream)
                for event_id, md_label, layered_path_set in shard_results:
                    total_received += 1
                    if layered_path_set.is_empty():
                        empty_events.append((md_label, event_id))
                    else:
                        buffered_layered_paths[md_label][event_id] = layered_path_set

            if total_received != items2process:
                raise RuntimeError("SLURM aquaduct-layering shards returned {} result(s) but the "
                                   "launcher enumerated {} (md_label, event_label) pair(s); some "
                                   "shards likely failed silently. Inspect the shard logs in "
                                   "'{}'.".format(total_received, items2process, slurm_folder))

            # report and drop empty events from the per-md_label order so the local-path
            # save_layered_network() path sees the same list it would have seen locally
            for md_label, event_id in empty_events:
                logger.warning("Event {} of {} cannot be layered".format(event_id, md_label))
                event_ids2process4md_label[md_label].remove(event_id)

            # 4) Commit per md_label in submission order, save layered networks/visualisations.
            #    Progress is reported per md_label (not per event) - events are already finished
            #    at this point, and the heavy work that remains is the save_layered_network()
            #    pickle dump + visualisation prep which have md_label granularity.
            num_md = len(self.aquaduct_input_folders)
            progressbar(0, num_md, self.parameters["log_level"])
            for i, md_label in enumerate(self.aquaduct_input_folders):
                aquanet = AquaductNetwork(self.parameters, md_label, load_only=True)
                for event_id in event_ids2process4md_label[md_label]:
                    aquanet.add_layered_entity(
                        event_id, buffered_layered_paths[md_label][event_id])
                logger.debug("Finished layering of network for '{}'.".format(md_label))
                # always saving layered visualizations to enable visualization of assigned events later
                aquanet.get_pdb_file()
                aquanet.save_layered_visualization(self.parameters["visualize_layered_events"])
                aquanet.save_layered_network()
                aquanet.clean_tempfile()
                progressbar(i + 1, num_md, self.parameters["log_level"])

            # Assembly succeeded end-to-end (every md_label's aqua_layered_network.dump and
            # visualisations are on disk); drop the per-shard cache unless the user opted in
            # to keep it.
            slurm.cleanup_stage_folder(stage, self.parameters)

    def create_super_cluster_profiles(self):
        """
        Parallel merging of caver tunnel profiles into new ones for superclusters (SCs), and saving SC details
        """

        with TimeProcess("Creating"):
            os.makedirs(os.path.join(self.parameters["super_cluster_profiles_folder"], "initial"), exist_ok=True)
            # Use the spawn context with cap_blas_threads_for_worker as the initializer so each
            # worker pins its BLAS thread budget to the parent's CPU allocation. Submitting
            # process_supercluster_profile_worker (a top-level function) instead of
            # super_cluster.process_cluster_profile (a bound method) keeps the per-task pickle
            # to the SuperCluster only.
            num_cpus = self.parameters["num_cpus"]
            with get_context("spawn").Pool(processes=num_cpus,
                                           initializer=utils.cap_blas_threads_for_worker,
                                           initargs=(num_cpus, num_cpus)) as pool:
                logger.info("Creating {:d} supercluster tunnel profiles "
                            "using {:d} {}:".format(len(self._super_clusters.keys()), num_cpus,
                                                    process_count(num_cpus)))
                logger.debug(self._report_filters())

                superclusters_to_process = list(self._super_clusters.values())
                items2process = len(superclusters_to_process)
                if items2process == 0:
                    raise RuntimeError("Not enough superclusters are available to create their profiles")

                progressbar(0, items2process, self.parameters["log_level"])
                timeout = self.parameters["worker_task_timeout_s"]
                imap_iter = pool.imap_unordered(process_supercluster_profile_worker, superclusters_to_process)
                # Workers return (sc_id, ...); identify completions by sc_id in error logs.
                results_iter = utils.iter_imap_results(
                    imap_iter, timeout=timeout, pool=pool, label="sc-profile",
                    extract_task_label=lambda r: "sc_id={}".format(r[0]))
                for i, result in enumerate(results_iter):
                    sc_id, sc_properties, sc_residues_freq, retained_tunnel_clusters = result
                    # assign initial computed properties of SC to this SC
                    self._super_clusters[sc_id].set_properties(sc_properties)
                    self._super_clusters[sc_id].set_bottleneck_residue_freq(sc_residues_freq)
                    self._super_clusters[sc_id].update_caver_clusters_validity(retained_tunnel_clusters)
                    progressbar(i + 1, items2process, self.parameters["log_level"])

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
            # Use the spawn context with cap_blas_threads_for_worker as the initializer so each
            # worker pins its BLAS thread budget to the parent's CPU allocation. Submitting
            # filter_supercluster_worker (a top-level function) instead of
            # super_cluster.filter_super_cluster (a bound method) keeps the per-task pickle to the
            # SuperCluster plus the small filter args, making the worker SLURM-ready.
            num_cpus = self.parameters["num_cpus"]
            with get_context("spawn").Pool(processes=num_cpus,
                                           initializer=utils.cap_blas_threads_for_worker,
                                           initargs=(num_cpus, num_cpus)) as pool:
                logger.info("Filtering {:d} supercluster profiles "
                            "using {:d} {}:".format(self.enumerate_valid_super_clusters(), num_cpus,
                                                    process_count(num_cpus)))
                self._active_filters = define_filters(min_length, max_length, min_bottleneck_radius,
                                                      max_bottleneck_radius, min_curvature, max_curvature, min_sims_num,
                                                      min_snapshots_num, min_avg_snapshots_num, min_total_events,
                                                      min_entry_events, min_release_events)
                logger.debug(self._report_filters())

                processing = list()

                for super_cluster in self._super_clusters.values():
                    super_cluster.properties = dict()
                    processing.append(("sc-filter[sc_id={},flag={}]".format(super_cluster.sc_id, self.filter_flag),
                                       pool.apply_async(filter_supercluster_worker,
                                                        args=(super_cluster, self._events_assigned,
                                                              self._active_filters, self.filter_flag))))
                num_retained_sc = 0
                if not len(processing) > 0:
                    raise RuntimeError("Not enough superclusters are available to filter their profiles")

                progressbar(0, len(processing), self.parameters["log_level"])
                timeout = self.parameters["worker_task_timeout_s"]
                for i, result in enumerate(utils.iter_pool_results(processing, timeout=timeout, pool=pool)):
                    # SC properties cannot be directly assigned within the object since SC are copied during processing
                    sc_id, sc_properties, sc_residues_freq, retained_tunnel_clusters = result
                    self._super_clusters[sc_id].set_properties(sc_properties)  # update with new properties
                    self._super_clusters[sc_id].set_bottleneck_residue_freq(sc_residues_freq)
                    self._super_clusters[sc_id].update_caver_clusters_validity(retained_tunnel_clusters)
                    if self._super_clusters[sc_id].has_passed_filter(consider_transport_events=self._events_assigned,
                                                                     active_filters=self._active_filters):
                        num_retained_sc += 1

                    progressbar(i + 1, len(processing), self.parameters["log_level"])

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

            cat_stream1 = None
            cat_stream2 = None
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

                    residue_names: List[str] = []
                    if self._events_assigned:
                        header_list.extend(["Num_Events", "Num_entries", "Num_releases"])
                        residue_set: set = set()
                        for sc in self._super_clusters.values():
                            residue_set.update(sc.num_events_by_residue.get(md_label, {}).keys())
                        if len(residue_set) > 1:
                            residue_names = sorted(residue_set)
                            for res in residue_names:
                                header_list.extend(["{}_E".format(res), "{}_R".format(res)])

                    dataset = [header_list]
                    for prio_sc in sorted(self._prioritized_clusters.keys()):  #
                        # we use prioritized IDs as those are also respecting active filters
                        sc_id = self._prioritized_clusters[prio_sc]
                        dataset.append(self._super_clusters[sc_id].get_summary_line_data(
                            self._events_assigned, md_label, residue_names or None))

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
                        for res in residue_names:
                            out_widths.append(widths[header_list.index("{}_E".format(res))])
                            out_widths.append(widths[header_list.index("{}_R".format(res))])
                        output += self._outlier_transport_events.report_summary_line(
                            out_widths, md_label, residue_names or None)

                    out_stream.write(output)
                    if "overall" not in md_label and cat_stream1 is not None:
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
                        if "overall" not in md_label and cat_stream2 is not None:
                            cat_stream2.write(output + "-" * 160 + "\n\n\n")

            if cat_stream1 is not None:
                cat_stream1.close()
            if cat_stream2 is not None:
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
                        data4vis.append((prio_sc_id, vis_data))

                    # visualize unassigned transport events
                    if self._outlier_transport_events.exist():
                        out_stream.writelines(self._outlier_transport_events.prepare_visualization(md_label))

                    out_stream.write("cmd.do('set all_states, 1')\n")
                    out_stream.write("cmd.show('cgo')\n")
                    out_stream.write("cmd.disable('*_release*')\n")
                    out_stream.write("cmd.disable('*_entry*')\n")
                    out_stream.write("cmd.zoom()\n")

                surface_cgo = False
                if self.parameters["visualize_super_cluster_volumes"] and \
                        ("overall" in md_label or self.parameters["visualize_comparative_super_cluster_volumes"]):
                    surface_cgo = True

                with get_context("spawn").Pool(processes=self.parameters["num_cpus"]) as pool:
                    processing = list()
                    # parallel generation of SC visualization
                    for prio_sc_id, vis_data in data4vis:
                        path_set, params = vis_data
                        processing.append(("sc-visualize[prio_sc_id={},md={}]".format(prio_sc_id, md_label),
                                           pool.apply_async(path_set.visualize_cgo,
                                                            args=(params[0], params[1], params[2], params[3],
                                                                  params[4], surface_cgo))))
                    if len(processing) > 0:
                        items2process = len(processing)
                        progressbar(0, items2process, self.parameters["log_level"])
                        timeout = self.parameters["worker_task_timeout_s"]
                        for i, _ in enumerate(utils.iter_pool_results(processing, timeout=timeout, pool=pool)):
                            progressbar(i + 1, items2process, self.parameters["log_level"])

    def get_property_time_evolution_data(self, property_name: str, active_filters: dict, sc_id: int | None = None,
                                         missing_value_default: float = 0) -> Dict[int, Dict[str, np.ndarray]]:
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

    def show_tunnels_passing_filter(self, sc_id: int, active_filters: dict, out_folder_path: str, md_labels: List[str] | None = None,  
                                    start_snapshot: int | None = None, end_snapshot: int | None = None, trajectory: bool = False):
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

        def _save_pymol_script(_vis_inputs: List[Tuple[str, str, int]], _md_label: str | None = None):
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
        :param active_filters: mapping of active supercluster filters applied during assignment
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
        try:
            event_direction = np.ravel(self.event.nodes_data[self.event.nodes_data[:, 4] == 1][0, :3])  # type: ignore[index]
        except (IndexError, TypeError):
            logger.debug("Event {} could not be assigned\n".format(self.event.entity_label))
            return list()

        for super_cluster in self.super_clusters.values():
            if super_cluster.has_passed_filter(consider_transport_events=False, active_filters=self.active_filters):
                if super_cluster.is_directionally_aligned(event_direction):
                    directionally_fitting_super_clusters.add(super_cluster.sc_id)

        return list(directionally_fitting_super_clusters)

    def perform_assignment(self) -> Tuple[Tuple[str, str, Tuple[str, Tuple[int, int]]], np.ndarray | None, float, float]:
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

            if buriedness is None:
                raise RuntimeError("Variable 'buriedness' should be set by exact matching analysis")

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

    def _exact_event_tunnel_matching(self, considered_sc_ids: np.ndarray) -> Dict[str, Dict[int, float]]:
        """
        Performs exact analyses of event with respect to tunnel clusters in frames during the event occurrence
        originating directly from the simulation sampling the event
        :param considered_sc_ids: superclusters considered for the matching with investigated event
        :return: details on buriedness of event in the actual tunnels of considered superclusters
        """
        from pathlib import Path
        # parse event info
        md_label = self.event_specification[0]
        residue = int(self.event_specification[2][0].split(":")[1])
        start_frame, end_frame = self.event_specification[2][1]
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        frames = range(start_frame, end_frame + 1)
        snap_ids = [x + self.parameters["caver_traj_offset"] for x in frames]
        perform_exact_matching_analysis = self.parameters["perform_exact_matching_analysis"] and\
                                          md_label in [a.name for a in Path(self.parameters["trajectory_path"]).glob(self.parameters["folder_pattern4exact_matching_analysis"]) if a.is_dir()]

        if perform_exact_matching_analysis:
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
                    assert closest_xyz is not None, "closest_xyz should be set when closest_sphere is set"
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

            if perform_exact_matching_analysis:
                # save info on matching to files
                details_file = os.path.join(details_path, "{}_sc{}.txt".format(self.event_specification[1], sc_id))  # type: ignore[arg-type]
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


# Shared state for the event-assigner worker processes; populated once per worker by
# init_event_assigner_worker() so that individual tasks need to carry only the per-event data
# instead of re-pickling the (potentially hundreds of MB) supercluster dict for every event.
_EVENT_WORKER_STATE: dict = {}


def init_event_assigner_worker(parameters: dict,
                               superclusters: Dict[int, SuperCluster],
                               active_filters: dict,
                               allocated_cpus: int, pool_size: int):
    """
    Initializer for the event-assigner multiprocessing Pool; stores read-only state shared by all
    tasks (analysis parameters, the supercluster dict, the active filter set) exactly once per
    worker, so individual tasks need to carry only (event_specification, event_path_set) instead
    of re-pickling the supercluster dict for every event. Also caps this worker's BLAS thread
    budget so pool_size * threads_per_worker stays within the parent's CPU allocation.
    :param parameters: job configuration parameters
    :param superclusters: dictionary of SuperCluster objects to assign events to (read-only)
    :param active_filters: filter set currently active when the assignment was requested
    :param allocated_cpus: total CPU budget the parent has (num_cpus on local, slurm_cpus_per_task
                           inside a SLURM shard); used by cap_blas_threads_for_worker()
    :param pool_size: number of worker processes in this pool, used by cap_blas_threads_for_worker()
    """

    utils.cap_blas_threads_for_worker(allocated_cpus, pool_size)
    _EVENT_WORKER_STATE["parameters"] = parameters
    _EVENT_WORKER_STATE["superclusters"] = superclusters
    _EVENT_WORKER_STATE["active_filters"] = active_filters


def assign_event_worker(task: Tuple[Tuple[str, str, Tuple[str, Tuple[int, int]]], LayeredPathSet]):
    """
    Top-level worker for transport-event-to-supercluster assignment. Reconstructs an EventAssigner
    from the shared _EVENT_WORKER_STATE (populated once by init_event_assigner_worker()) and the
    per-task event data, then delegates to EventAssigner.perform_assignment().
    Lives at module scope (not a bound method) so the multiprocessing Pool can submit it under
    the 'spawn' start method without pickling the surrounding TransportProcesses instance.
    :param task: (event_specification, event_path_set) tuple
    :return: same as EventAssigner.perform_assignment()
    """

    event_specification, event_path_set = task
    assigner = EventAssigner(_EVENT_WORKER_STATE["parameters"], event_specification, event_path_set,
                             _EVENT_WORKER_STATE["superclusters"], _EVENT_WORKER_STATE["active_filters"])
    return assigner.perform_assignment()


def process_supercluster_profile_worker(super_cluster: SuperCluster) -> Tuple[int, Dict[str, Dict[str, float | int]],
                                                                              Dict[str, Dict[str, float]],
                                                                              Dict[str, List[int]]]:
    """
    Top-level worker that delegates to SuperCluster.process_cluster_profile(). Lives at module
    scope (not as a bound method) so the Pool can submit it under 'spawn' without dragging the
    surrounding TransportProcesses instance into the per-task pickle.
    :param super_cluster: the SuperCluster whose profile is being built; carries only SC metadata
                          at this stage (tunnel_clusters has been freed in stage 5 precompute)
    :return: same as SuperCluster.process_cluster_profile()
    """

    return super_cluster.process_cluster_profile()


def filter_supercluster_worker(super_cluster: SuperCluster, consider_transport_events: bool,
                               active_filters: dict, flag: int) -> Tuple[int, Dict[str, Dict[str, float | int]],
                                                                         Dict[str, Dict[str, float]],
                                                                         Dict[str, List[int]]]:
    """
    Top-level worker that delegates to SuperCluster.filter_super_cluster(). Lives at module scope
    (not as a bound method) so the Pool can submit it under 'spawn' without dragging the
    surrounding TransportProcesses instance into the per-task pickle.
    :param super_cluster: the SuperCluster being filtered
    :param consider_transport_events: forwarded to filter_super_cluster()
    :param active_filters: forwarded to filter_super_cluster()
    :param flag: filtering-pass ID, forwarded to filter_super_cluster()
    :return: same as SuperCluster.filter_super_cluster()
    """

    return super_cluster.filter_super_cluster(consider_transport_events, active_filters, flag)


def create_layered_cluster_worker(cluster: TunnelCluster) -> Tuple[int, str, LayeredPathSet]:
    """
    Top-level worker that delegates to TunnelCluster.create_layered_cluster(). Lives at module
    scope (not as a bound method) so the Pool can submit it under 'spawn' without dragging the
    surrounding TransportProcesses instance, and so that pool.imap_unordered can consume an
    iterable of clusters directly.
    :param cluster: the TunnelCluster being layered
    :return: same as TunnelCluster.create_layered_cluster()
    """

    return cluster.create_layered_cluster()


def create_layered_event_worker(event: TransportEvent) -> Tuple[str, str, LayeredPathSet]:
    """
    Top-level worker that delegates to TransportEvent.create_layered_event(). Lives at module
    scope (not as a bound method) so the Pool can submit it under 'spawn' without dragging the
    surrounding TransportProcesses instance, and so that pool.imap_unordered can consume an
    iterable of events directly.
    :param event: the TransportEvent being layered
    :return: same as TransportEvent.create_layered_event()
    """

    return event.create_layered_event()


def process_aquaduct_network_worker(task: Tuple[str, dict, List[str], bool]) -> None:
    """
    Top-level worker that unpacks a (md_label, parameters, root_paths, parallel_processing) tuple
    and delegates to TransportProcesses._pre_process_single_aquaduct_network(). Used with
    pool.imap_unordered() (which only supports single-argument workers); the staticmethod itself
    cannot be passed directly because it takes multiple positional arguments.
    :param task: (md_label, parameters, root_paths, parallel_processing) tuple
    """

    md_label, parameters, root_paths, parallel_processing = task
    TransportProcesses._pre_process_single_aquaduct_network(md_label, parameters, root_paths,
                                                            parallel_processing)


def visualize_transport_details(out_folder_path: str, trajectory: TrajectoryTT, start_frame: int, end_frame: int, caver_traj_offset: int,
                                caver_clusters: List[TunnelCluster] | None = None, start_snapshot: int | None = None, end_snapshot: int | None = None,
                                resids: List[int] | None = None):
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
    elif start_snapshot is None and end_snapshot is not None:
        start_snapshot = start_frame + caver_traj_offset
        snap_ids = [*range(start_snapshot, end_snapshot + 1)]
    elif end_snapshot is None and start_snapshot is not None:
        end_snapshot = end_frame + caver_traj_offset
        snap_ids = [*range(start_snapshot, end_snapshot + 1)]
    else:
        snap_ids = [*range(start_snapshot, end_snapshot + 1)]  # type: ignore[arg-type, operator]

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
                selector = "!:{}".format(resid)  # to remove, here we already have resid so no need to alter
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
                    update_config: AnalysisConfig | None = None) -> TransportProcesses:
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
