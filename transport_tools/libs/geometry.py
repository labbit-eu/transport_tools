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

from __future__ import annotations

__version__ = '0.9.8'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import numpy as np
import os
import hdbscan
import gzip
import pickle
import warnings
from joblib import parallel_backend
from itertools import groupby
from logging import getLogger
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from transport_tools.libs.protein_files import VizAtom
from transport_tools.libs.utils import node_labels_split, convert_coords2cgo, get_caver_color
from typing import Dict, List, Mapping, Set, Tuple, TYPE_CHECKING, Iterable
if TYPE_CHECKING:  # to enable type_checking without cyclic imports
    from transport_tools.libs.networks import Tunnel, TransportEvent

logger = getLogger(__name__)


class Point:
    def __init__(self, xyz: List[float], distance: float = -1, radius: float = 0):
        """
        Class for point storing and manipulation
        :param xyz: coordinates
        :param distance: distance to arbitrary point
        :param radius: radius of allocated space
        """

        self.data = np.append(xyz, [distance, radius]).astype(float).reshape(1, 5)

    def __str__(self):
        return "(x, y, z, dist, radius) = ({:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}, {:4.2f})".format(self.data[0, 0],
                                                                                                self.data[0, 1],
                                                                                                self.data[0, 2],
                                                                                                self.data[0, 3],
                                                                                                self.data[0, 4])

    def distance2point(self, other: Point) -> float:
        """
        Calculate distance to another point
        :param other: point to which we calculated the distance
        :return: distance between the points
        """

        return einsum_dist(self.data[:, 0:3], other.data[:, 0:3])[0]

    def convert2viz_atom(self, atom_id: int, res_id: int, resname: str = "UNK") -> VizAtom:
        """
        Convert points to VizAtom object for saving PDB lines
        :param atom_id: ID of atom in PDB record
        :param res_id: ID of residue in PDB record
        :param resname: name of residue in PDB record
        :return: VizAtom object
        """

        return VizAtom([atom_id, "UNK", resname, res_id, self.data[0, 0], self.data[0, 1], self.data[0, 2]])

    def save_point(self, filename):
        """
        Save point as PDB line to file
        :param filename: file to which we save the point
        """

        with open(filename, "w") as out_stream:
            out_stream.write(str(self.convert2viz_atom(1, 1)))


class PointMatrix:
    def __init__(self, points_mat: np.ndarray):
        """
        Class to handle points data
        :param points_mat: input points data
        """

        self.points_mat = points_mat.copy()
        self.dist_column = 3
        self.radii_column = 4
        self.points_ids_column = 5
        self.tunnel_ids_column = 6

    def __str__(self):
        return str(self.points_mat)

    def __add__(self, other: PointMatrix):
        return PointMatrix(np.concatenate((self.points_mat, other.points_mat)))

    def is_empty(self) -> bool:
        """
        Test if this matrix is empty
        """

        return self.points_mat.size == 0

    def get_num_columns(self) -> int:
        """
        Compute number of columns this matrix has
        """

        if self.is_empty():
            raise IndexError("Cannot return number of columns of empty matrix")

        return self.points_mat.shape[1]

    def get_num_points(self) -> int:
        """
        Compute number of points this matrix contain
        """

        return self.points_mat.shape[0]

    def get_whole_matrix(self) -> np.ndarray:
        """
        Return numpy representation of this matrix
        """

        return self.points_mat

    def alter_coords(self, new_xyz: np.ndarray) -> None:
        """
        Updates current coordinates of points with new set of coordinates
        :param new_xyz: new coordinates use
        """

        self.points_mat[:, :3] = new_xyz

    def get_coords(self) -> np.ndarray:
        """
        Return coordinates of all points in Matrix
        """

        return self.points_mat[:, :3]

    def get_radii(self) -> np.ndarray:
        """
        Return radii of all points in Matrix
        """

        return self.points_mat[:, self.radii_column]

    def get_start_points_indexing(self) -> np.ndarray:
        """
        Return indices of points representing start of tunnels or events
        """

        return self.points_mat[:, self.points_ids_column] == 0

    def get_end_points_indexing(self) -> np.ndarray:
        """
        Return indices of points representing end of tunnels or events
        """

        return self.points_mat[:, self.points_ids_column] < 0

    def get_tunnels_ids(self) -> np.ndarray:
        """
        Return IDs of all tunnels(events) in Matrix
        """

        return self.points_mat[:, self.tunnel_ids_column].astype(int)

    def get_points_ids4tunnel(self, tunnel_id: int) -> np.ndarray:
        """
        Return IDs of points for particular tunnel(event)
        :param tunnel_id: ID of query tunnel
        """

        tunnel_selector = self.get_tunnels_ids() == tunnel_id
        return np.absolute(self.points_mat[:, self.points_ids_column][tunnel_selector].astype(int))


class ClusterInLayer:
    def __init__(self, points_mat: np.ndarray, thickness: float, quantile: float, end_point: bool = False,
                 cls_id: int = -1, layer_id: int = -1):
        """
        Clusters representing original points in layers
        :param cls_id: ID of this cluster in a given layer
        :param layer_id: ID of layer to which this cluster belongs
        :param points_mat: data of included points
        :param thickness: layer thickness
        :param quantile: to use for calculation of representative radius of this cluster
        :param end_point: does cluster represent end point of original tunnel or transport event
        """

        self.cls_id = cls_id
        self.layer_id = layer_id
        self.matrix = PointMatrix(points_mat)
        self.average: np.ndarray | None = None
        self.rmsf: float | None = None
        self.radius: float | None = None
        self.end_point = end_point
        self.start_point = False
        self.num_end_points = 0
        self.num_points = self.matrix.get_num_points()
        self.thickness = thickness
        self.quantile = quantile
        self.tunnel_ids = set(self.matrix.get_tunnels_ids())
        if self.num_points > 0:
            self.compute_averages()

    def __str__(self) -> str:
        return "L{}".format(self.get_node_label())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ClusterInLayer):
           return False # differnt object
        return self.layer_id == other.layer_id and self.cls_id == other.cls_id

    def get_node_label(self) -> str:
        """
        Create label node representing this cluster from its layer and cls IDs
        :return: node label
        """

        return "{}_{}".format(self.layer_id, self.cls_id)

    def resolve_avg_failures(self, random_seed: int, num_new_clusters: int = 0,
                             max_new_clusters: int | None = None, tolerance: float = 0) -> List[ClusterInLayer]:
        """
        Keep splitting this cluster to two, until all created clusters represents well their points or the maximum
        number of new clusters is created
        :param random_seed: value to initiate the random number generator
        :param num_new_clusters: number of newly created clusters so far during the resolution process
        :param max_new_clusters: limit on the maximum of newly created clusters
        :param tolerance: additional tolerance for testing the representativeness
        :return: list of representative clusters
        """

        new_clusters = list()
        # split cluster data to two new clusters
        process_data = self.matrix.get_whole_matrix().copy()
        with parallel_backend('loky', n_jobs=1):
            clustering_method = KMeans(n_clusters=2, random_state=random_seed, n_init=10, algorithm='elkan')
            clustering = clustering_method.fit_predict(process_data)

        for cls_id in np.unique(clustering):
            if cls_id == -1:
                continue
            cluster_data = process_data[clustering == cls_id, :]
            new_cluster = ClusterInLayer(cluster_data, self.thickness, self.quantile, self.end_point)
            if not new_cluster.is_representative(tolerance=tolerance):
                tmp_clusters = new_cluster.resolve_avg_failures(random_seed, num_new_clusters, max_new_clusters,
                                                                tolerance)
                num_new_clusters += len(tmp_clusters)
                if max_new_clusters is not None and num_new_clusters > max_new_clusters:
                    new_clusters.append(new_cluster)
                    return new_clusters

                new_clusters.extend(tmp_clusters)
            else:
                num_new_clusters += 1
                new_clusters.append(new_cluster)
                if max_new_clusters is not None and num_new_clusters > max_new_clusters:
                    return new_clusters

        return new_clusters

    def is_representative(self, tolerance: float = 0) -> bool:
        """
        Test if cluster average represents well the original tunnel points
        :param tolerance: additional tolerance
        """

        if self.average is None or self.radius is None or self.rmsf is None:
            raise ValueError(f"Cluster {self} properties must be computed before checking representativeness of data ")

        distances = einsum_dist(self.get_coords(), self.average)
        dist2closest_point = np.min(distances)
        dist2furthest_point = np.max(distances)

        if self.end_point:  # stricter cutoffs for outlier content in terminal nodes
            outlier_content_cutoff = max([2 * self.rmsf, self.radius, self.thickness])
        else:
            outlier_content_cutoff = max([3 * self.rmsf, 2 * self.radius, 2 * self.thickness])

        if dist2closest_point > 1 + tolerance or dist2furthest_point > outlier_content_cutoff + tolerance:
            # average point is more than 1 A from the points it should represent, or contains outlying points
            return False
        else:
            return True

    def merge_with_cluster(self, other_cluster: ClusterInLayer):
        """
        Join other cluster to this cluster
        :param other_cluster: Cluster to add
        """

        self.num_points += other_cluster.num_points
        self.tunnel_ids |= other_cluster.tunnel_ids
        self.matrix += other_cluster.matrix
        self.end_point = self.end_point or other_cluster.end_point

    def get_coords(self) -> np.ndarray:
        """
        Return coordinates of all points in this cluster
        """

        return self.matrix.get_coords()

    def compute_averages(self):
        """
        Compute average coordinates, rmsf, radius for this cluster, determine if it contains starting or end points
        """

        self.average = np.average(self.get_coords(), axis=0)
        squares = np.power(einsum_dist(self.get_coords(), self.average), 2)  # type: ignore[arg-type]
        self.rmsf = np.sqrt(np.average(squares))

        # to represent occupied void, we want to have quite large radius from the distribution but avoiding extremes
        self.radius = float(np.quantile(self.matrix.get_radii(), self.quantile))
        self.num_end_points = self.matrix.get_coords()[self.matrix.get_end_points_indexing()].shape[0]
        self.start_point = self.matrix.get_start_points_indexing().any()  # contain starting points of tunnels


class LayeredPathSet:
    def __init__(self, entity_label: str, md_label: str, parameters: dict, starting_point_coords: np.ndarray | None = None):
        """
        Class for manipulation and analyses of simplified set of paths designed to represent original tunnel
         clusters or transport events
        :param entity_label: name of the entity, e.g., Cluster_1, Path_2 ...
        :param md_label: name of folder with the source MD simulation data
        :param parameters: job configuration parameters
        :param starting_point_coords: coordinates of average starting point for this simulation
        """

        self.entity_label = entity_label
        self.md_label = md_label
        self.node_paths: List[np.ndarray] = list()
        self.starting_point_coords = starting_point_coords
        self.parameters = parameters
        self.traced_event: Tuple[str, Tuple[int, int]] | None = None
        self.visualization_prefix: str | None = None
        self.node_depths: Dict[str, float] = dict()
        self.characteristics: Tuple[float, int] | None = None

        # nodes_data = np.array of cluster.average, cluster.layer_id, cluster.end_point, cluster.radius, cluster.rmsf
        if starting_point_coords is not None:  # include overall SP as the first node
            self.node_labels: List[str] = ["SP"]
            self.nodes_data = np.append(starting_point_coords,
                                        np.array([-1, 0, self.parameters["sp_radius"], 0.5])).reshape(1, -1)
        else:
            self.node_labels: List[str] = list()
            self.nodes_data: np.ndarray | None = None

    def is_empty(self) -> bool:
        """
        If this pathset contain no valid node_paths
        """

        return len(self.node_paths) == 0

    def transform_coordinates(self, transform_mat: np.ndarray):
        """
        Transform nodes coordinates forming this pathset according to the transformation matrix
        :param transform_mat: transformation matrix to be applied on the nodes coordinates
        """

        if self.nodes_data is None: #should not be called from empty pahtset but to be safe
            return  # Nothing to transform for empty pathset

        data4transform = np.append(self.nodes_data[:, 0:3],
                                   np.full((self.nodes_data[:, 0:3].shape[0], 1), 1.), axis=1).T
        self.nodes_data[:, 0:3] = transform_mat.dot(data4transform).T[:, 0:3]

        # recompute layer memberships based on distances to global SP [0,0,0] to be comparable among different systems
        # despite the fact that this results to altered layer only super rarely
        self.nodes_data[:, 3] = assign_layer_from_distances(einsum_dist(self.nodes_data[:, 0:3],
                                                                        np.array([0., 0., 0.])),
                                                            self.parameters["layer_thickness"])[1]

        if "SP" in self.node_labels:  # global SP xyz must remain [0,0,0] and be in layer -1
            self.nodes_data[0, 0:3] = np.array([0., 0., 0.])
            self.nodes_data[0, 3] = -1.

    def is_same(self, other: LayeredPathSet, precision: float=0.1) -> bool:
        """
        Test if the pathset is same as the other pathset
        :param other: other pathset to compare with
        :precision: tolerated deviation when comparing data
        :return: are two pathset same?
        """

        # Handle empty pathsets
        if self.nodes_data is None or other.nodes_data is None:
            return self.nodes_data is None and other.nodes_data is None

        # compare basic structure
        if self.nodes_data.shape != other.nodes_data.shape:
            print("shape not matching", str(self), str(other))
            return False

        # Create a mapping of labels based on layer_id (ignore sub-cluster id)
        # Labels format: "layer_subcluster" or "SP" for starting point
        def get_layer_id(label: str) -> str:
            """Extract layer ID from label, ignoring sub-cluster ID"""
            if label == "SP":
                return "SP"
            return label.split("_")[0]  # Get only the layer part

        # Group labels by layer
        self_layers = [get_layer_id(label) for label in self.node_labels]
        other_layers = [get_layer_id(label) for label in other.node_labels]

        # Check if both have the same layer structure
        if sorted(set(self_layers)) != sorted(set(other_layers)):
            print("layer structure not matching", str(self), str(other))
            return False

        # Create label mapping based on spatial coordinates and layer_id in data
        # Two nodes are considered "the same" if they have the same layer_id (geometric layer) and similar coordinates
        # We ignore the label itself since cluster IDs may have changed due to non-deterministic clustering order
        label_mapping = {}

        for self_idx, self_label in enumerate(self.node_labels):
            self_coords = self.nodes_data[self_idx, 0:3]
            self_layer_id = self.nodes_data[self_idx, 3]

            # Find matching node in other pathset
            best_match = None
            best_dist = float('inf')

            for other_idx, other_label in enumerate(other.node_labels):
                # Must have same layer_id in data (geometric layer assignment)
                if not np.isclose(self_layer_id, other.nodes_data[other_idx, 3]):
                    continue

                # Check spatial distance
                other_coords = other.nodes_data[other_idx, 0:3]
                dist = np.linalg.norm(self_coords - other_coords)

                if dist < best_dist:
                    best_dist = dist
                    best_match = other_label

            # Tolerance of 0.2 Angstroms to account for:
            # - Numerical precision in averaging cluster coordinates
            # - Different cluster ID assignment order leading to slightly different cluster memberships
            # - Small variations in pathfinding that select nearby but not identical clusters
            if best_match is None or best_dist > precision:  # No close match found
                print(f"no matching node found for {self_label} in other pathset")
                print(f"  coords: {self_coords}, layer_id: {self_layer_id}, best_dist: {best_dist}")
                return False

            label_mapping[self_label] = best_match

        # Now compare node paths using the mapping
        for path, other_path in zip(self.node_paths, other.node_paths):
            # Map self path labels to expected other path labels
            mapped_path = np.array([label_mapping.get(label, label) for label in path])
            if not np.all(mapped_path == other_path):
                print("node paths not matching", str(self), str(other))
                return False

        # Compare node data (coordinates, radius, rmsf) for mapped nodes
        for self_idx, self_label in enumerate(self.node_labels):
            other_label = label_mapping[self_label]
            # Find the index of other_label in other.node_labels
            other_idx = other.node_labels.index(other_label)

            # Compare node data with relaxed tolerances to handle small variations from cluster reordering
            if not np.allclose(self.nodes_data[self_idx], other.nodes_data[other_idx], rtol=precision, atol=precision):
                print(f"node data not matching for {self_label} vs {other_label}")
                print(f"  self data: {self.nodes_data[self_idx]}")
                print(f"  other data: {other.nodes_data[other_idx]}")
                return False

        return True

    def set_traced_event(self, traced_residue: Tuple[str, int, Tuple[int, int], Tuple[int, int]]):
        """
        Sets information on residue and frames traced by AQUA-DUCT that is source for this layered event
        :param traced_residue: tuple containing resname & resid of ligand responsible for this path,
                                and beginning and last frames for entry and release events
        """

        if self.starting_point_coords is not None:
            raise RuntimeError("This functionality is meant for events only")

        event_type = self.entity_label.split("_")[-1]
        if event_type == "entry":  # which frame ranges to report
            frame_range = traced_residue[2]
        else:
            frame_range = traced_residue[3]

        self.traced_event = ("{}:{}".format(traced_residue[0], traced_residue[1]), frame_range)
        self.visualization_prefix = self.entity_label

    def _get_direction(self) -> np.ndarray:
        """
        Return average direction of path end-points in this set
        """

        if self.nodes_data is None:
            return np.array([0., 0., 0.])  # Return zero vector for empty pathset
        return np.average(self.nodes_data[self.nodes_data[:, 4] == 1, :3], axis=0)

    def _get_extended_labels4paths(self) -> List[List[str]]:
        """
        Returns list of node paths from this pathset in which each node label is enhanced by information on source
        MD simulation foldername and the name of entity it represents (cluster_1, event_2 etc)
        """

        if "data merged" in self.entity_label:  # to avoid extending names of already extended names
            return [path.tolist() if isinstance(path, np.ndarray) else path for path in self.node_paths]

        node_paths = list()
        for path in self.node_paths:
            node_path = list(map(lambda node: "{}-{}-{}".format(self.md_label, self.entity_label, node), path))
            node_paths.append(node_path)

        return node_paths

    def _get_extended_labels4nodes(self) -> List[str]:
        """
        Returns list of node labels from this pathset that are enhanced by information on source MD simulation
        foldername and the name of entity it represents (cluster_1, event_2 etc)
        """
        if "data merged" in self.entity_label:  # to avoid extending names of already extended names
            return self.node_labels

        return list(map(lambda node: "{}-{}-{}".format(self.md_label, self.entity_label, node), self.node_labels))

    def _get_most_complete_shortest_path_from_ids(self, path_ids: List[int]) -> int:
        """
        Finds node path that 1) starts from the lowest layer, and 2) is the shortest
        :param path_ids: list of IDs of node_paths to analyze
        :return: ID of the selected path
        """

        path_sort = list()
        for path_id in path_ids:
            start_layer = node_labels_split(self.node_paths[path_id][1])[0]
            path_sort.append((start_layer, self.node_paths[path_id].size, path_id))

        return min(path_sort)[-1]

    def remove_unnecessary_paths(self):
        """
        Remove paths that do lead to already explored terminal nodes, and do not include any additional unexplored node
        """

        if self.starting_point_coords is None:
            raise RuntimeError("This functionality is meant for tunnels only")

        necessary_nodes = set(self.node_labels[1:])
        paths2keep = list()
        paths2terminal = dict()

        # find paths leading to the same terminal nodes
        for terminal in self._get_terminal_node_labels():
            paths2terminal[terminal] = list()

        for path_id, path in enumerate(self.node_paths):
            terminal = path[-1]
            paths2terminal[terminal].append(path_id)

        # for each terminal node, find the optimal path
        for terminal in paths2terminal:
            if len(paths2terminal[terminal]) >= 1:  # find shortest path that starts in a layer closest to the SP
                optimal_path_id = self._get_most_complete_shortest_path_from_ids(paths2terminal[terminal])
                paths2keep.append(optimal_path_id)

        # find missing nodes not covered by the optimal paths to terminal nodes
        current_nodes = set()
        for path_id in paths2keep:
            path = self.node_paths[path_id]
            current_nodes = current_nodes.union(set(path[1:]))

        missing_nodes = necessary_nodes - current_nodes

        # add smallest set of optimal paths containing all missing nodes
        while missing_nodes:
            missing_scores = dict()

            # calculate how many missing nodes are visited by each path
            for path_id, path in enumerate(self.node_paths):
                missing_scores[path_id] = 0
                for missing_node in missing_nodes:
                    if missing_node in path:
                        missing_scores[path_id] += 1

            # find paths covering the most missing nodes
            max_score = max(missing_scores.values())
            optimal_path_ids = list()
            for path_id, score in missing_scores.items():
                if score == max_score:
                    optimal_path_ids.append(path_id)

            # add optimal path covering the most missing nodes
            optimal_path_id = self._get_most_complete_shortest_path_from_ids(optimal_path_ids)
            opt_path = self.node_paths[optimal_path_id]
            paths2keep.append(optimal_path_id)
            current_nodes = current_nodes.union(set(opt_path[1:]))
            missing_nodes = necessary_nodes - current_nodes

        # purge unnecessary paths
        all_paths = self.node_paths.copy()
        self.node_paths = list()
        for path_id, path in enumerate(all_paths):
            if path_id in paths2keep:
                self.node_paths.append(path)

    def compute_node_depths(self):
        """
        Assign average depth along the paths to each node
        """

        if "-unique" not in self.entity_label:
            raise RuntimeError("This functionality is meant for merged pathset after duplicate removal")

        self.node_depths: Dict[str, float] = dict()

        for path in self.node_paths:
            for i, label in enumerate(path):
                if label not in self.node_depths.keys():
                    self.node_depths[label] = 1 / (i + 1)  # inversion to have maximum depth close to the starting point
                else:
                    self.node_depths[label] = max(1 / (i + 1), self.node_depths[label])

    def remove_duplicates(self):
        """
        Remove duplicate nodes and paths from a merged pathset to speed-up the assignment of other pathsets
        """

        if "data merged" not in self.entity_label:
            raise RuntimeError("This functionality is meant for merged pathset only")

        if self.nodes_data is None:
            return  # Nothing to deduplicate for empty pathset

        self.entity_label += "-unique"
        ids_for_removal = np.array([])
        labels = np.array(self.node_labels).astype(np.str_)
        labels2add = list()
        combined_label_id = dict()  # info on ID of new nodes created in a given layer
        with parallel_backend('loky', n_jobs=1):
            cluster_method = DBSCAN(min_samples=1, n_jobs=1, eps=0.75, metric='euclidean')
            clustering = cluster_method.fit_predict(self.nodes_data[:, :3])
        unique, counts = np.unique(clustering, return_counts=True)

        # parse similar nodes
        for dupe_id in unique[counts > 1]:
            dupes_indices = np.nonzero(clustering == dupe_id)[0]
            dupes_data = self.nodes_data[dupes_indices]
            dupes_labels = labels[dupes_indices]

            # make sure that only duplicates in the same layers are merged
            layers, l_counts = np.unique(dupes_data[:, 3], return_counts=True)
            for layer in layers[l_counts > 1]:
                if layer not in combined_label_id:
                    combined_label_id[layer] = 0  # first merged node in this layer
                combined_label_id[layer] += 1  # there is already some merged node in this layer, use next ID

                # get duplicates in this layer
                layer_selector = dupes_data[:, 3] == layer
                layer_dupes_data = dupes_data[layer_selector]
                layer_dupes_labels = dupes_labels[layer_selector]
                layer_dupes_indices = dupes_indices[layer_selector]
                ids_for_removal = np.append(ids_for_removal, layer_dupes_indices).astype(int)

                # merge data -  avg coordinates,  maintain layer, and for rmsf and radii consider their maximum
                combined_data = np.average(layer_dupes_data[:, :5], axis=0)
                combined_data = np.append(combined_data, np.max(layer_dupes_data[:, 5:7], axis=0))
                if np.any(layer_dupes_data[:, 4] == 1):  # new merged node is still terminal node
                    combined_data[4] = 1
                combined_data = combined_data.reshape(1, -1)

                # add new data to end of data matrix
                self.nodes_data = np.concatenate((self.nodes_data, combined_data))

                # merge labels
                if layer < 0:
                    combined_layer = "SP"
                else:
                    combined_layer = int(layer)

                combined_label = "merged-{}_{}".format(combined_layer, combined_label_id[layer])
                labels2add.append(combined_label)

                # update labels in paths
                for i, path in enumerate(self.node_paths):
                    self.node_paths[i] = np.array([combined_label if label in layer_dupes_labels else label for label in path]).astype(np.str_)

        # removing duplicate data and labels
        self.nodes_data = np.delete(self.nodes_data, ids_for_removal, 0)
        self.node_labels = list(np.delete(labels, ids_for_removal, 0).tolist())
        self.node_labels.extend(labels2add)

        # remove redundant paths created due to merging of duplicate clusters
        redundant_path_ids = get_redundant_path_ids({i: path.tolist() if isinstance(path, np.ndarray) else path for i, path in enumerate(self.node_paths)})

        unique_paths = list()
        for path_id, path in enumerate(self.node_paths):
            if path_id not in redundant_path_ids:
                unique_paths.append(path)

        self.node_paths = unique_paths.copy()

    def __add__(self, other_set: LayeredPathSet) -> LayeredPathSet:
        if self.starting_point_coords is None and other_set.starting_point_coords is not None \
                or self.starting_point_coords is not None and other_set.starting_point_coords is None:
            raise RuntimeError("Joining LayeredPathSet for tunnels and events is not possible")

        #TODO unclear if we should not be able to merge one correct with one empty pathset despite passing tests
        if self.nodes_data is None or other_set.nodes_data is None:
            raise RuntimeError("Cannot merge pathsets with empty nodes_data")

        new_set = LayeredPathSet("data merged", "various cls", self.parameters, self.starting_point_coords)
        new_set.nodes_data = np.concatenate((self.nodes_data, other_set.nodes_data))
        new_set.node_labels = self._get_extended_labels4nodes() + other_set._get_extended_labels4nodes()
        new_set.node_paths = [np.array(path).astype(np.str_) for path in
                              self._get_extended_labels4paths() + other_set._get_extended_labels4paths()]

        return new_set

    def visualize_cgo(self, output_folder: str, entity_label: str, color_id: int = 0, merged: bool = False,
                      flag: str = "", surface_cgo: bool = False):
        """
        Save this pathset as Pymol compiled graphics object(CGO) for visualization
        :param output_folder: folder to which CGO will be saved
        :param entity_label: name of the layered entity (tunnel cluster or transport event) to visualize
        :param color_id: Pymol ID of color to use for this pathset
        :param merged: if all paths should be in single CGO
        :param surface_cgo: if to generate also surface visualization
        :param flag: additional description enabling differentiation of cgo files among various results after filtering
        """

        if self.nodes_data is None:
            raise RuntimeError("No data for visualization of PathSet from {} of {}".format(entity_label, output_folder))

        node_data = dict(zip(self.node_labels, self.nodes_data))
        filename1 = os.path.join(output_folder, "{}_pathset{}.dump.gz".format(entity_label, flag))
        os.makedirs(os.path.dirname(filename1), exist_ok=True)
        pathset_cgos = list()
        if merged:
            for path_id, path in enumerate(self.node_paths):
                xyz = node_data[path[0]][:3].reshape(1, 3)
                for node_label in path[1:]:
                    xyz = np.concatenate((xyz, node_data[node_label][:3].reshape(1, 3)), axis=0)
                pathset_cgos.extend(convert_coords2cgo(xyz, color_id=color_id))

            if surface_cgo:
                from transport_tools.libs.utils import convert_spheres2cgo_surface
                from transport_tools.libs.msms import filter_spheres

                spheres = list()
                for path_id, path in enumerate(self.node_paths):
                    for node_label in path:
                        xyz = node_data[node_label][:3]
                        radius = node_data[node_label][5]
                        spheres.append((xyz, radius))

                filtered_spheres = filter_spheres(spheres)

                filename2 = os.path.join(output_folder, "{}_volume{}.dump.gz".format(entity_label, flag))
                with gzip.open(filename2, "wb") as out_stream:
                    if self.parameters["msms"] is None:
                        pickle.dump(convert_spheres2cgo_surface(filtered_spheres, color_id=color_id,),
                                    out_stream, self.parameters["pickle_protocol"])
                    else:
                        from transport_tools.libs.msms import msms_surface
                        try:
                            pickle.dump(msms_surface(self.parameters["msms"], filtered_spheres, color_id=color_id, ),
                                        out_stream, self.parameters["pickle_protocol"])
                        except RuntimeError:
                            logger.warning("The program 'msms' was not found or is not correctly configured.\n")
                            logger.warning("Default calculation of surfaces will be employed.\n")
                            pickle.dump(convert_spheres2cgo_surface(filtered_spheres, color_id=color_id, ),
                                        out_stream, self.parameters["pickle_protocol"])

        else:
            for path_id, path in enumerate(self.node_paths):
                xyz = node_data[path[0]][:3].reshape(1, 3)
                for node_label in path[1:]:
                    xyz = np.concatenate((xyz, node_data[node_label][:3].reshape(1, 3)), axis=0)
                pathset_cgos.append(convert_coords2cgo(xyz, color_id=color_id))

        with gzip.open(filename1, "wb") as out_stream:
            pickle.dump(pathset_cgos, out_stream, self.parameters["pickle_protocol"])

    def __str__(self):
        msg = "LayeredPathSet: {} of {}\nPaths: \n".format(self.entity_label, self.md_label)
        for i, path in enumerate(self.node_paths):
            msg += "{:2d}: {}\n".format(i, path)
        if self.nodes_data is not None:
            msg += "num nodes = {}\nLabels {}:\n{}\nData:\n{}".format(self.nodes_data.shape[0], len(self.node_labels),
                                                                      self.node_labels, self.nodes_data)
        else:
            msg += "num nodes = 0\nLabels {}:\n{}\nData:\nNone".format(len(self.node_labels), self.node_labels)
        return msg

    def add_node_path(self, node_path: List[str], layers: Mapping[int, Layer]):
        """
        Add path of node labels representing the original entity (tunnel cluster or transport event) and the data for
        clusters visited along the path to this pathset
        :param node_path: path formed by node labels to add
        :param layers: layers containing processed clusters
        """

        def _generate_node_data(in_node_label) -> np.ndarray:
            """
            Generates data for the input node
            :param in_node_label: label of the node to process
            :return: data for the processed node
            """

            in_layer_id, in_cls_id = node_labels_split(in_node_label)
            in_cluster = layers[in_layer_id].clusters[in_cls_id]
            if in_cluster.average is None or in_cluster.radius is None or in_cluster.rmsf is None:
                raise ValueError(f"Cluster properties must be computed before generating node data for node {in_node_label}")

            return np.append(in_cluster.average, np.array([in_cluster.layer_id, in_cluster.end_point, in_cluster.radius,
                                                          in_cluster.rmsf]).astype(float), axis=0).reshape(1, -1)

        if self.nodes_data is None:  # Events - we add just a single event path only
            # first process the first node, to initialize nodes_data
            self.node_labels.append(node_path[0])

            layer_id, cls_id = node_labels_split(node_path[0])
            cluster = layers[layer_id].clusters[cls_id]
            if cluster.average is None or cluster.radius is None or cluster.rmsf is None:
                raise ValueError(f"Cluster properties must be computed before generating node data for node {node_path[0]}")
            self.nodes_data = np.append(cluster.average,
                                        np.array([cluster.layer_id, cluster.end_point, cluster.radius,
                                                  cluster.rmsf]).astype(float), axis=0).reshape(1, -1)
            # add remaining nodes
            for node_label in node_path[1:]:
                self.node_labels.append(node_label)
                self.nodes_data = np.concatenate((self.nodes_data, _generate_node_data(node_label)))

        else:  # TunnelClusters - multiple paths can be added in a single Cluster
            # nodes_data already initialized with SP
            for node_label in node_path:
                if node_label in self.node_labels:  # we do not duplicate same nodes involved in multiple paths
                    continue
                self.node_labels.append(node_label)
                self.nodes_data = np.concatenate((self.nodes_data, _generate_node_data(node_label)))

            node_path.insert(0, "SP")  # include SP as the start node of each tunnel

        self.node_paths.append(np.array(node_path).astype(np.str_))

    def _get_adjacent_nodes_data(self, query_node_data: np.ndarray, query_last_layer_id: float,
                                 query_first_terminal_layer: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get information on the nodes from this PathSet located in the same and surrounding layers as the query node
        :param query_node_data: data of the query node
        :param query_last_layer_id: ID of the last layer in query pathset
        :param query_first_terminal_layer: ID of the first layer with terminal node in the query pathset
        :return: array with labels and data of adjacent nodes
        """

        if self.nodes_data is None:
            raise ValueError(f"Empty pathset {self} should not be processed")

        query_layer_id = query_node_data[3]
        is_last_layer = query_last_layer_id == query_layer_id
        last_layer_id = np.max(self.nodes_data[:, 3])
        nodes = np.array(self.node_labels).astype(np.str_)

        selector = np.logical_or.reduce((self.nodes_data[:, 3] == query_layer_id + 1,
                                         self.nodes_data[:, 3] == query_layer_id,
                                         self.nodes_data[:, 3] == query_layer_id - 1))

        # include also all nodes after first terminal one as distance to this one could be evaluated too
        selector = np.logical_or(selector, self.nodes_data[:, 3] >= query_first_terminal_layer)

        if is_last_layer:  # include also all nodes in layers beyond, as query node is likely the closest one for them
            selector = np.logical_or(selector, self.nodes_data[:, 3] > query_layer_id)

        if query_layer_id > last_layer_id:  # include also last two layers
            selector = np.logical_or(selector, np.logical_or.reduce((self.nodes_data[:, 3] == last_layer_id,
                                                                     self.nodes_data[:, 3] == last_layer_id - 1)))

        # include also SP as possible adjacent node for misaligned clusters
        selector = np.logical_or(selector, self.nodes_data[:, 3] < 0)  # SP has layer ID == -1
        adjacent_nodes = nodes[np.nonzero(selector)[0]]
        adjacent_data = self.nodes_data[np.nonzero(selector)[0], :]

        return adjacent_nodes, adjacent_data

    @staticmethod
    def _compute_distances(node_data: np.ndarray, last_layer: float, first_terminal_layer: float,
                           eval_pathset: LayeredPathSet,
                           consider_rmsf: bool = False) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Computes three types of distances between considerd node and nodes in adjacent layers from evalulated pathset:
        Types of distances:
        1) center to center distances
        2) distances of their surfaces (considering their radii)
        3) distances of their surfaces decreased by mean RMSF in position of the evaluated nodes
        :param node_data: considered node
        :param last_layer: ID of the last layer in pathset from which considered node originate
        :param first_terminal_layer: ID of the first layer with terminal node in the pathset from which considered node
         originate
        :param eval_pathset: evaluated pathset
        :param consider_rmsf: if the RMSF correction should be computed
        :return: zip object with adjacent node labels, and three types of computed distances
        """

        adjacent_nodes, adjacent_data = eval_pathset._get_adjacent_nodes_data(node_data, last_layer,
                                                                              first_terminal_layer)
        distances = einsum_dist(node_data[0:3], adjacent_data[:, 0:3])
        radii_sum = adjacent_data[:, 5] + node_data[5]  # sum of radii of adjacent nodes and the evaluated node
        surface_dists = distances - radii_sum
        if consider_rmsf:
            avg_rmsfs = (adjacent_data[:, 6] + node_data[6]) / 2
            max_rmsf = 3
            avg_rmsfs = np.where(avg_rmsfs > max_rmsf, max_rmsf, avg_rmsfs)
            rmsf_dists = surface_dists - avg_rmsfs
        else:
            rmsf_dists = surface_dists

        return zip(adjacent_nodes, distances, surface_dists, rmsf_dists)

    def _update_missing_distances(self, missing_node: str, other: LayeredPathSet,
                                  consider_rmsf: bool = False) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
        """
        Computes distances between missing node and adjacent nodes form other set.
        :param missing_node: label of missing node
        :param other: other evaluated pathset
        :param consider_rmsf: if the RMSF correction should be computed
        :return: missing node distance matrix
        """

        update_dist_mat = {
            missing_node: dict()
        }

        if other.nodes_data is None:
            raise ValueError(f"Empty pathset {other} should not be processed")

        missing_node_data = dict(zip(other.node_labels, other.nodes_data))[missing_node]
        last_layer = np.max(other.nodes_data[:, 3])

        for other_node_label, dist, surf_dist, rmsf_dist in self._compute_distances(missing_node_data, last_layer,
                                                                                    other._get_first_terminal_layer(),
                                                                                    self, consider_rmsf):
            update_dist_mat[missing_node][other_node_label] = (dist, surf_dist, rmsf_dist)

        return update_dist_mat

    def _get_first_terminal_layer(self) -> float:
        """
        Find layer of terminal node closest to the starting point
        :return: id of the layer
        """
        if self.nodes_data is None:
            raise ValueError(f"Empty pathset {self} should not be processed")

        selector = (self.nodes_data[:, 4] == 1)
        terminal_nodes = self.nodes_data[np.nonzero(selector)[0], :]

        return np.min(terminal_nodes[:, 3])

    def _pre_compute_distance_matrices(self, other_set: LayeredPathSet, consider_rmsf: bool = False) \
            -> Tuple[Dict[str, Dict[str, Tuple[float, float, float]]], Dict[str, Dict[str, Tuple[float, float, float]]]]:
        """
        For each pair of nodes (clusters) in adjacent layers of this and other pathsets, three distances are precomputed
        :param other_set: other evaluated pathset
        :param consider_rmsf: if the RMSF correction should be computed
        :return: node2node distance matrix and its inverted form
        """
        if self.nodes_data is None:
            raise ValueError(f"Empty pathset {self} should not be processed")

        dist_mat = dict()
        last_layer = np.max(self.nodes_data[:, 3])

        for node_label, node_data in zip(self.node_labels, self.nodes_data):
            if node_label not in dist_mat.keys():
                dist_mat[node_label] = dict()

            for other_node_label, dist, surf_dist, rmsf_dist in \
                    self._compute_distances(node_data, last_layer, self._get_first_terminal_layer(), other_set,
                                            consider_rmsf):
                dist_mat[node_label][other_node_label] = (dist, surf_dist, rmsf_dist)

        inverted_dist_mat = dict()
        for node1 in dist_mat.keys():
            for node2 in dist_mat[node1].keys():
                if node2 not in inverted_dist_mat.keys():
                    inverted_dist_mat[node2] = dict()
                inverted_dist_mat[node2][node1] = dist_mat[node1][node2]

        return dist_mat, inverted_dist_mat

    def _get_terminal_node_labels(self) -> np.ndarray:
        """
        Return array with labels of terminal nodes in this pathset
        """
        if self.nodes_data is None:
            raise ValueError(f"Empty pathset {self} should not be processed")

        nodes = np.array(self.node_labels).astype(np.str_)
        return nodes[np.nonzero(self.nodes_data[:, 4] == 1)]

    @staticmethod
    def _get_dist2closest_node(node_label: str, dist_mat: Dict[str, Dict[str, Tuple[float, float, float]]],
                               dist_type: int = 1, nodes_subset: Set[str] | None = None) -> Tuple[float, str]:
        """
        Finds surface distance from query node to the closest node in the distance matrix
        :param node_label: label of query node
        :param dist_mat: precomputed node2node distance matrix
        :param dist_type: 0 - center distances, 1 - surface distances, 2 - surface and rmsf distances
        :param nodes_subset: node labels from among which the closest node must be search for
        :return: minimal distance of query node from node from other pathset used to compute matrix, closest node label
        """

        distances = list()
        if nodes_subset is not None:
            nodes2process = list(nodes_subset.intersection(dist_mat[node_label].keys()))
        else:
            nodes2process = [*dist_mat[node_label].keys()]

        if not nodes2process:
            raise KeyError

        for node2 in nodes2process:
            distances.append(dist_mat[node_label][node2][dist_type])
        min_id = int(np.argmin(distances))
        min_dist = max(distances[min_id], 0)  # to correct for possibly negative distances in type 1 and 2
        min_label = nodes2process[min_id]

        return min_dist, min_label

    def _get_path_fragments(self, path: np.ndarray, dividing_nodes: set) -> List[List[str]]:
        """
        Returns fragment of two paths from starting point to each of their terminal nodes
        :param path: evaluated path
        :param dividing_nodes: terminal nodes of this path to use for fragmentation
        :return: path fragments
        """

        num_div_nodes = len(dividing_nodes)
        if num_div_nodes == 1:  # no fragments encoded here
            return [path.tolist()]

        # create list of dividing nodes for processing
        nodes2process = list()
        for node in dividing_nodes:
            node_location = np.nonzero(path == node)[0][0] + 1
            nodes2process.append((node_location, node))
        nodes2process.sort()

        if 0 < self.parameters["clustering_max_num_rep_frag"] < num_div_nodes:
            # we will always add last terminal node to create the longest path
            worknodes = []
            if self.parameters["clustering_max_num_rep_frag"] >= 2:
                # evaluate also the shortest path
                worknodes.append(nodes2process[0])
            if self.parameters["clustering_max_num_rep_frag"] > 2:
                # add remaining num of fragments
                from random import seed, sample
                seed(self.parameters["random_seed"])  # to have consistent results
                nodes2add = sample(nodes2process[1:-1], self.parameters["clustering_max_num_rep_frag"] - 2)
                worknodes.extend(nodes2add)

            worknodes.append(nodes2process[-1])
        else:
            # no fragment selection, keeping all
            worknodes = nodes2process
        worknodes.sort()
        fragmented_path = list()
        for old_location, node in worknodes:
            node_location = np.nonzero(path == node)[0][0] + 1
            fragmented_path.append(path[:node_location].tolist())
            path = path[node_location:]

        return fragmented_path

    def how_much_is_inside(self, other_set: LayeredPathSet) -> Tuple[float, float]:
        """
        Computes the fraction of nodes from this set buried inside the nodes of the other set, and maximal depth
        (counted towards starting point (SP) along shortest path)
        :param other_set: other set in which the buriedness is calculated
        :return: buriedness, and maximal depth towards SP
        """
        if self.nodes_data is None:
            raise ValueError(f"Empty pathset {self} should not be processed")

        dist_mat, inverted_dist_mat = self._pre_compute_distance_matrices(other_set,
                                                                          self.parameters["use_cluster_spread"])

        max_depth = -99999999
        surface_distances = list()

        for node in self.node_labels:
            if self.parameters["use_cluster_spread"]:
                # have surface with RMSF correction
                surface_dist, surface_node = self._get_dist2closest_node(node, dist_mat, dist_type=2)
            else:
                # only surface
                surface_dist, surface_node = self._get_dist2closest_node(node, dist_mat, dist_type=1)
            surface_distances.append(surface_dist)
            max_depth = max(max_depth, other_set.node_depths[surface_node])

        num_buried_nodes = np.count_nonzero(np.array(surface_distances) == 0)
        buriedness = num_buried_nodes / self.nodes_data.shape[0]

        return buriedness, max_depth

    def get_fragmented_paths(self) -> Tuple[int, List[List[List[str]]]]:
        """
        Compile path fragment's for all paths forming this pathset
        :return: number of fragmented paths, list of fragmented paths
        """

        fragmented_paths = list()
        num_paths = 0
        terminal_nodes = set(self._get_terminal_node_labels())
        for path in self.node_paths:
            term_nodes4path = terminal_nodes & set(path)
            fragments = self._get_path_fragments(path, term_nodes4path)
            num_paths += len(fragments)
            fragmented_paths.append(fragments)

        return num_paths, fragmented_paths

    def _pre_compute_dense_distance_matrix(self, other_set: LayeredPathSet, dist_type: int,
                                           consider_rmsf: bool = False) -> np.ndarray:
        """
        Build a dense node-to-node distance matrix between this and the other pathset for the requested distance type.
        Equivalent to the forward matrix of _pre_compute_distance_matrices, but stored as a 2D NumPy array with NaN
        marking node pairs that are not adjacent (and hence were not evaluated).
        :param other_set: other evaluated pathset
        :param dist_type: 0 - center distances, 1 - surface distances, 2 - surface and rmsf distances
        :param consider_rmsf: if the RMSF correction should be computed
        :return: dense distance matrix of shape (#self nodes, #other nodes)
        """

        if self.nodes_data is None:
            raise ValueError(f"Empty pathset {self} should not be processed")

        other_index = {label: j for j, label in enumerate(other_set.node_labels)}
        dist_mat = np.full((len(self.node_labels), len(other_set.node_labels)), np.nan)
        last_layer = np.max(self.nodes_data[:, 3])
        first_terminal_layer = self._get_first_terminal_layer()

        for i, node_data in enumerate(self.nodes_data):
            for other_label, dist, surf_dist, rmsf_dist in \
                    self._compute_distances(node_data, last_layer, first_terminal_layer, other_set, consider_rmsf):
                dist_mat[i, other_index[str(other_label)]] = (dist, surf_dist, rmsf_dist)[dist_type]

        return dist_mat

    def _build_inverted_dense_distance_matrix(self, fwd_mat: np.ndarray, other_set: LayeredPathSet,
                                              expanded_paths: List[List[List[str]]], dist_type: int) -> np.ndarray:
        """
        Build the inverted (other-to-self) dense distance matrix from the forward one. Nodes of the other pathset
        that are not adjacent to any node of this pathset - and hence absent from the forward matrix - have their
        distances computed on demand via _update_missing_distances, mirroring the missing-node handling of the
        original _get_dist2closest_node based kernel.
        :param fwd_mat: forward dense distance matrix from _pre_compute_dense_distance_matrix
        :param other_set: other evaluated pathset
        :param expanded_paths: fragmented paths of the other pathset (limits the missing-node search to used nodes)
        :param dist_type: 0 - center distances, 1 - surface distances, 2 - surface and rmsf distances
        :return: dense distance matrix of shape (#other nodes, #self nodes)
        """

        inv_mat = fwd_mat.T.copy()
        self_index = {label: i for i, label in enumerate(self.node_labels)}
        other_index = {label: j for j, label in enumerate(other_set.node_labels)}
        used_labels = {label for fragmented_path in expanded_paths
                       for fragment in fragmented_path for label in fragment}

        for label in used_labels:
            j = other_index[label]
            if np.all(np.isnan(inv_mat[j])):
                # node never adjacent to any node of this pathset - compute its missing distances
                missing = self._update_missing_distances(label, other_set)
                for self_label, distances in missing[label].items():
                    inv_mat[j, self_index[self_label]] = distances[dist_type]

        return inv_mat

    def avg_distance2path_set(self, other_set: LayeredPathSet, distance_cutoff: float = 999,
                              dist_type: int = 1) -> float:
        """
        Compute mean closest surface-to-surface distance of all paths from two pathsets
        Note that if during calculation the mean distance is projected to surpass distance cutoff, the two patsets are
        deemed faraway with 999 distance. The same distance is also assumed for directionally misaligned patset pairs.

        Vectorized re-implementation of the closest-node distance kernel: the per-node Python loops of the original
        algorithm are replaced by NumPy operations over dense node-to-node distance matrices. Pathset pairs that
        require the original kernel's stateful missing-node handling (detected as a non-finite running sum) are
        delegated to the pure-Python _avg_distance2path_set_reference, which also serves as its regression reference.
        :param other_set: other set to which the distance is calculated
        :param distance_cutoff: cutoff on accurate distance calculation, anything beyond this value is far (999)
        :param dist_type: 0 - center distances, 1 - surface distances, 2 - surface and rmsf distances
        :return: mean distance (surface-to-surface by default) between pathsets
        """

        exact = self.parameters["calculate_exact_path_distances"]
        angle = vector_angle(self._get_direction(), other_set._get_direction())
        if not exact and self.parameters["directional_cutoff"] <= angle <= \
                (2 * np.pi - self.parameters["directional_cutoff"]):
            # directionally misaligned
            return 999

        # expand shorter paths
        num_paths1, expanded_paths1 = self.get_fragmented_paths()
        num_paths2, expanded_paths2 = other_set.get_fragmented_paths()

        total_num_dists = num_paths1 * num_paths2
        if num_paths1 == 0:
            raise RuntimeError("Empty LayeredPathSet:\n{}".format(str(self)))
        if num_paths2 == 0:
            raise RuntimeError("Empty LayeredPathSet:\n{}".format(str(other_set)))

        too_distant = 3 * distance_cutoff * total_num_dists

        # dense node-to-node distance matrices (NaN marks non-adjacent, i.e. non-evaluated, node pairs)
        fwd_mat = self._pre_compute_dense_distance_matrix(other_set, dist_type)
        inv_mat = self._build_inverted_dense_distance_matrix(fwd_mat, other_set, expanded_paths2, dist_type)

        self_index = {label: i for i, label in enumerate(self.node_labels)}
        other_index = {label: j for j, label in enumerate(other_set.node_labels)}
        fragmented1 = [_index_fragmented_path(fp, self_index) for fp in expanded_paths1]
        fragmented2 = [_index_fragmented_path(fp, other_index) for fp in expanded_paths2]

        sum_distances = 0.0
        for node_ids1, prefix_lengths1, is_sp1 in fragmented1:
            for node_ids2, prefix_lengths2, is_sp2 in fragmented2:
                # forward direction: surface distance to the closest path2 node for each path1 node
                fwd = fwd_mat[np.ix_(node_ids1, node_ids2)]
                fwd = np.where(np.isnan(fwd), np.inf, fwd)
                # running minimum over the growing path2 prefix; negative (overlap) distances clamped to 0
                fwd = np.maximum(np.minimum.accumulate(fwd, axis=1), 0.0)
                fwd[is_sp1, :] = 0.0  # SP nodes are excluded from the path1 sum
                fwd_cumsum = np.cumsum(fwd, axis=0)
                dist4path1 = fwd_cumsum[np.ix_(prefix_lengths1 - 1, prefix_lengths2 - 1)]

                # inverted direction: surface distance to the closest path1 node for each path2 node
                inv = inv_mat[np.ix_(node_ids2, node_ids1)]
                inv = np.where(np.isnan(inv), np.inf, inv)
                inv = np.maximum(np.minimum.accumulate(inv, axis=1), 0.0)
                inv[is_sp2, :] = 0.0  # SP nodes are excluded from the path2 sum
                inv_cumsum = np.cumsum(inv, axis=0)
                cum_frag_dist = inv_cumsum[np.ix_(prefix_lengths2 - 1, prefix_lengths1 - 1)].T

                # mean closest distance per (path1 prefix, path2 prefix); SP excluded from both path lengths
                dists2evaluate = (prefix_lengths1[:, None] - 1) + (prefix_lengths2[None, :] - 1)
                sum_distances += float(np.sum((dist4path1 + cum_frag_dist) / dists2evaluate))

        if not np.isfinite(sum_distances):
            # a path2 node had no adjacent node of this pathset for some path1 prefix; the original
            # kernel resolves this by a stateful per-node switch to self-adjacency distances that
            # cannot be reproduced by the static dense matrices - fall back to the reference kernel
            return self._avg_distance2path_set_reference(other_set, distance_cutoff, dist_type)

        if not exact and sum_distances > too_distant:
            # avg of two path_sets that are too distant will still be beyond the cutoff,
            # not messing the clustering much
            return 999

        return sum_distances / total_num_dists

    def _avg_distance2path_set_reference(self, other_set: LayeredPathSet, distance_cutoff: float = 999,
                                         dist_type: int = 1) -> float:
        """
        Original pure-Python implementation of avg_distance2path_set, retained as a reference for regression
        testing of the vectorized avg_distance2path_set. Not used in production code.
        :param other_set: other set to which the distance is calculated
        :param distance_cutoff: cutoff on accurate distance calculation, anything beyond this value is far (999)
        :param dist_type: 0 - center distances, 1 - surface distances, 2 - surface and rmsf distances
        :return: mean distance (surface-to-surface by default) between pathsets
        """

        angle = vector_angle(self._get_direction(), other_set._get_direction())
        if not self.parameters["calculate_exact_path_distances"] and self.parameters["directional_cutoff"] <= angle <= \
                (2 * np.pi - self.parameters["directional_cutoff"]):
            # directionally misaligned
            return 999

        # expand shorter paths
        num_paths1, expanded_paths1 = self.get_fragmented_paths()
        num_paths2, expanded_paths2 = other_set.get_fragmented_paths()

        total_num_dists = num_paths1 * num_paths2
        if num_paths1 == 0:
            raise RuntimeError("Empty LayeredPathSet:\n{}".format(str(self)))
        if num_paths2 == 0:
            raise RuntimeError("Empty LayeredPathSet:\n{}".format(str(other_set)))

        too_distant = 3 * distance_cutoff * total_num_dists
        dist_mat, inverted_dist_mat = self._pre_compute_distance_matrices(other_set)

        sum_distances = 0
        for path_fragments1 in expanded_paths1:
            path1 = list()

            for path_id1 in range(len(path_fragments1)):
                path1.extend(path_fragments1[path_id1])
                for path_fragments2 in expanded_paths2:
                    path2 = list()
                    overlapping_nodes_path1 = set()
                    cum_frag_dist = 0
                    cum_frag_len = -1  # to exclude SP from first fragment of path2

                    for path_id2 in range(len(path_fragments2)):
                        path2.extend(path_fragments2[path_id2])
                        path_frag = path_fragments2[path_id2]
                        dist4path1 = 0
                        dist4frag = 0
                        cum_frag_len += len(path_frag)
                        dists2evaluate = (len(path1) - 1) + cum_frag_len  # to exclude SP from path1
                        overlapping_nodes_path2 = set()
                        for node in path1:
                            if node in overlapping_nodes_path1 or "SP" in node:
                                # skip search for closest nodes for those that are overlapping and cannot get better
                                continue
                            node_dist, node2 = self._get_dist2closest_node(node, dist_mat, dist_type, set(path2))
                            dist4path1 += node_dist

                            if node_dist == 0:
                                overlapping_nodes_path1.add(node)
                                overlapping_nodes_path2.add(node2)

                        for node in path_frag:
                            if node in overlapping_nodes_path2 or "SP" in node:
                                # skip search for overlapping nodes as those cannot get better and hence will
                                # contribute 0 to the fragment distance
                                continue
                            try:
                                node_dist, node2 = self._get_dist2closest_node(node, inverted_dist_mat, dist_type,
                                                                               set(path1))
                            except KeyError:
                                # This indicates some missing nodes, frequently in layers close to SP, due to large
                                # fluctuations of SP -> compute missing data
                                inverted_dist_mat.update(self._update_missing_distances(node, other_set))
                                node_dist, node2 = self._get_dist2closest_node(node, inverted_dist_mat, dist_type,
                                                                               set(path1))

                            dist4frag += node_dist

                        cum_frag_dist += dist4frag

                        # project the mean value to test for far away pathsets
                        sum_distances += (dist4path1 + cum_frag_dist) / dists2evaluate
                        if not self.parameters["calculate_exact_path_distances"] and sum_distances > too_distant:
                            # avg of two path_sets that are too distant will still be beyond the cutoff,
                            # not messing the clustering much
                            return 999

        return sum_distances / total_num_dists


class Layer:
    def __init__(self, layer_id: int, layer_thickness: float, parameters: dict, entity_label: str, md_label: str):
        """
        Class storing layered points; enable clustering of these points to Clusters
        :param layer_id: ID of this layer
        :param layer_thickness: layer thickness
        :param parameters: job configuration parameters
        :param entity_label: name of the entity (tunnel cluster or transport event) to be layered
        """

        self.id = layer_id
        self.parameters = parameters
        self.clusters: Dict[int, ClusterInLayer] = dict()
        self.thickness = layer_thickness
        self.num_points = 0
        self.entity_label = entity_label
        self.md_label = md_label

    def save_points(self, out_folder: str, save_points_prefix: str, transform_mat: np.ndarray) -> List[str]:
        """
        Save clusters (nodes) and their constituent points for visualization
        :param out_folder: folder to which PDB files with nodes will be saved
        :param save_points_prefix: prefix for names of PDB file defining nature of layered entity
        :param transform_mat: transformation matrix to transform output points
        :return: list of names of created PDB files
        """

        def _subsample_points(in_cluster: ClusterInLayer, random_seed: int, max_points: int) -> np.ndarray:
            """
            Randomly selects coordinates of limited number of points from a cluster
            :param in_cluster: evaluated cluster
            :param random_seed: value to initiate the random number generator
            :param max_points: maximum points to keep
            :return: coordinates of retained points from evaluated cluster
            """
            import random
            coords = in_cluster.get_coords()
            if len(coords) > max_points:
                random_point = coords.copy()
                random.seed(random_seed)  # to have consistent results
                random.shuffle(random_point)  # type: ignore[arg-type]
                return random_point[:max_points]
            else:
                return coords

        filenames = list()
        for cls_id, cluster in self.clusters.items():

            filename = "{}-{}-{}.pdb".format(save_points_prefix, self.id, cls_id)
            filenames.append(filename)
            with open(os.path.join(out_folder, filename), "w") as out:
                last = 0
                for i, xyz in enumerate(_subsample_points(cluster, self.parameters["random_seed"],
                                                          self.parameters["max_layered_points4visualization"])):
                    data4transform = np.append(xyz, np.array([1.]))
                    new_xyz = transform_mat.dot(data4transform)[0:3]
                    out.write(str(Point(new_xyz).convert2viz_atom(atom_id=i, res_id=i, resname="L{}".format(self.id))))
                    out.write("TER\n")
                    last = i
                last += 1

                if cluster.average is None:
                    raise ValueError(f"Cluster average must be computed before transformation")
                data4transform = np.append(cluster.average, np.array([1.]))
                new_xyz = transform_mat.dot(data4transform)[0:3]
                out.write(str(Point(new_xyz).convert2viz_atom(atom_id=last, res_id=last,
                                                              resname="A{}".format(self.id))))
                out.write("END\n")

        return filenames

    def _out_filtering(self, points_coords: np.ndarray) -> np.ndarray:
        """
        Filters out outlying points from this layer based on their coordinates
        :param points_coords: cartesian coordinates of points for outlier filtering
        :return: clustering labels, with outliers annotated as -1
        """

        with parallel_backend('loky', n_jobs=1):
            filter_method = IsolationForest(random_state=self.parameters["random_seed"], contamination="auto",
                                            n_estimators=50, n_jobs=1)
            clustering = filter_method.fit_predict(points_coords)

        unique_, counts_ = np.unique(clustering, return_counts=True)
        clusters_counts_ = dict(zip(unique_, counts_))
        if -1 in clusters_counts_.keys() and clusters_counts_[-1] > points_coords.shape[0] * 0.5:
            # too many outliers => keep all points
            return np.full(points_coords.shape[0], 0)
        else:
            return clustering

    def _cluster_data(self, points_coords: np.ndarray) -> np.ndarray:
        """
        Perform clustering/outlier filtering of points from this layer based on their coordinates
        :param points_coords: cartesian coordinates of points for data clustering
        :return: clustering labels, with outliers annotated as -1
        """

        num_points = points_coords.shape[0]
        if 1 < num_points < 50:
            try:
                cluster_method = AgglomerativeClustering(n_clusters=None, metric="euclidean", linkage="average",
                                                        distance_threshold=2)
            except TypeError:
                cluster_method = AgglomerativeClustering(n_clusters=None, affinity="euclidean", linkage="average",  # type: ignore[call-arg]
                                                        distance_threshold=2)

        elif num_points == 1:
            return np.array([0])
        else:
            # HDBSCAN with adaptive clustering thresholds
            fraction_cutoff = 0.01
            min_cls_size = max(int(num_points * fraction_cutoff), 5)
            min_samples = max(5, int(min_cls_size / 2))

            cluster_method = hdbscan.HDBSCAN(min_cluster_size=min_cls_size, metric="euclidean",
                                             allow_single_cluster=False, approx_min_span_tree=False,
                                             core_dist_n_jobs=1, min_samples=min_samples,
                                             match_reference_implementation=True)

        # try clustering to find significant nodes of alternative paths
        with warnings.catch_warnings():
            # hdbscan (>=this site only) still passes the deprecated force_all_finite to
            # sklearn.check_array (renamed to ensure_all_finite in sklearn 1.6, removed in 1.8).
            # Silence only that external warning here; FutureWarnings from our own code remain visible.
            warnings.filterwarnings("ignore", category=FutureWarning,
                                    message="'force_all_finite' was renamed to 'ensure_all_finite'")
            clustering = cluster_method.fit_predict(points_coords)
        unique, counts = np.unique(clustering, return_counts=True)
        clusters_counts = dict(zip(unique, counts))

        # if too many outliers assigned during the clustering, try just IF outlier filtering which is often milder
        if -1 in clusters_counts.keys() and clusters_counts[-1] > num_points * 0.5:
            return self._out_filtering(points_coords)
        else:
            # order the labels based on the cluster sizes
            cls_id = -10000
            for cls, count in sorted(clusters_counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
                if cls < 0:
                    continue
                cls_id -= 1
                clustering = np.where(clustering == cls, cls_id, clustering)

            clustering = np.where(clustering < -10000, (clustering * -1) - 10000, clustering)

            return clustering

    def pop_cluster(self, cls_id: int) -> ClusterInLayer:
        """
        Pops a required cluster with given id from this layer
        :param cls_id: id of the required cluster
        :return: the required cluster
        """

        return self.clusters.pop(cls_id)

    def add_cluster(self, cluster: ClusterInLayer):
        """
        Adds new cluster to this layer, making sure it adheres to the layer specification
        :param cluster: cluster to add
        :return: id of the new cluster
        """

        if self.clusters.keys():
            last_cluster_id = max(self.clusters.keys())
        else:
            last_cluster_id = -1

        new_cluster_id = last_cluster_id + 1
        cluster.cls_id = new_cluster_id
        cluster.layer_id = self.id
        self.clusters[new_cluster_id] = cluster

        return new_cluster_id

    def _generate_clusters_from_data(self, points_mat: np.ndarray, clustering: np.ndarray, end_point: bool = False):
        """
        Based on clustering of points, creates clusters of points in this layer.
        Clusters are added in spatially sorted order to ensure stable, deterministic labeling.
        :param points_mat: points data
        :param clustering: assignment of points to clusters
        :param end_point: if the cluster is of end point type
        """

        cluster_objects = []
        for cls_id in np.unique(clustering):
            if cls_id == -1:
                continue
            cluster_data = points_mat[clustering == cls_id, :]
            cluster = ClusterInLayer(cluster_data, self.thickness,
                                     self.parameters["tunnel_properties_quantile"],
                                     end_point=end_point)
            cluster_objects.append(cluster)
        
        #Sort clusters by their average coordinates (z, y, x for stability)
        cluster_objects.sort(key=lambda c: (c.average[2], c.average[1], c.average[0]))        
        
        for cluster in cluster_objects:
             self.add_cluster(cluster)

    def _make_layered_clusters(self, points_mat: np.ndarray):
        raise NotImplementedError("Provide implementation of this method.")

    def _assure_end_points(self, end_points_mat: np.ndarray):
        """
        Creates layered clusters from points that represent ends of layered entities (tunnel clusters or events)
        :param end_points_mat: data of end-points
        """

        # cluster the end points
        end_points_coords = end_points_mat[:, :3]
        with parallel_backend('loky', n_jobs=1):
            cluster_method = DBSCAN(min_samples=1, n_jobs=1, eps=1, metric='euclidean')
            clustering = cluster_method.fit_predict(end_points_coords)

        # create new terminal clusters from end points
        self._generate_clusters_from_data(end_points_mat, clustering, end_point=True)

        # if we have some path end points as outliers, we perform their re-clustering to keep all endpoint nodes
        if -1 in clustering:
            # clustering of outliers
            out_mat = end_points_mat[clustering == -1, :]
            out_coords = out_mat[:, :3]
            with parallel_backend('loky', n_jobs=1):
                try:
                    cluster_method = AgglomerativeClustering(n_clusters=None, metric="euclidean", linkage="average", distance_threshold=2)
                except TypeError:
                    cluster_method = AgglomerativeClustering(n_clusters=None, affinity="euclidean", linkage="average", distance_threshold=2)   # type: ignore[attr-defined]
                    
                clustering = cluster_method.fit_predict(out_coords)

            # create additional clusters
            self._generate_clusters_from_data(out_mat, clustering, end_point=True)

    def contains_data(self) -> bool:
        """
        Test if layer contain any data (has been assigned any point)
        """

        return self.num_points > 0

    def cluster_data(self, points_mat: np.ndarray):
        """
        Perform clustering of data of points assigned to this layer, separately for common points and end points,
        finally representativeness of formed clusters is enforced
        :param points_mat: points to cluster
        """

        self.num_points = int(points_mat.shape[0])
        if self.contains_data():
            # cluster common points
            common_points = points_mat[points_mat[:, 5] >= 0]
            if common_points.size > 0:
                self._make_layered_clusters(common_points)

            # cluster end points
            end_points = points_mat[points_mat[:, 5] < 0]
            if end_points.size > 0:
                self._assure_end_points(end_points)

            # make sure that all cluster centers are close to any of their point
            for cls_id, cluster in self.clusters.copy().items():
                if not cluster.is_representative():
                    split_clusters = self.pop_cluster(cls_id).resolve_avg_failures(self.parameters["random_seed"])
                    for new_cluster in split_clusters:
                        self.add_cluster(new_cluster)

            # update info on num points in layer after possible outlier removal
            self.num_points = 0
            for cls_id, cluster in self.clusters.items():
                self.num_points += cluster.num_points


class LayeredRepresentation:
    def __init__(self, parameters: dict, entity_label: str):
        """
        Class responsible for splitting of original data to layers and defining of layered (coarse-grained) paths
        representing the original entity (tunnel cluster or transport event)
        :param parameters: job configuration parameters
        :param entity_label: name of the entity to be layered
        """

        self.entity_label = entity_label
        self.parameters = parameters
        self.points_mat: np.ndarray | None = None
        self.layer_thickness = self.parameters["layer_thickness"]
        self.layers: Dict[int, Layer] | None = None
        self.save_points_folder: str | None = None
        self.md_label = self.parameters["md_label"]
        self.original_data_path: str | None = None
        self.visualization_prefix: str | None = None

    def _merge_duplicate_clusters(self):
        """
        Merge similar clusters of points in each layer separately
        """

        if self.layers is None:
            raise ValueError(f"Layers not initialized for {self.entity_label}")

        # this must be limited to single layer, otherwise we can loose inter-layer connections
        for layer_id in self.layers.keys():
            clusters = self.layers[layer_id].clusters

            if len(clusters) > 1:
                keys = list(clusters.keys())
                first_cluster = clusters[keys[0]]
                if first_cluster.average is None:
                    raise ValueError(f"Cluster {keys[0]} in layer {layer_id} has not computed averages")
                values = first_cluster.average.reshape(1, 3)  # initialize with first item
                ids = [keys[0]]

                for cls_id in keys[1:]:
                    cluster = clusters[cls_id]
                    if cluster.average is None:
                        raise ValueError(f"Cluster {cls_id} in layer {layer_id} has not computed averages")
                    ids.append(cls_id)
                    values = np.concatenate((values, cluster.average.reshape(1, 3)), axis=0)

                try:
                    clustering_method = AgglomerativeClustering(n_clusters=None, metric="euclidean", linkage="complete", distance_threshold=2)
                except TypeError:
                    clustering_method = AgglomerativeClustering(n_clusters=None, affinity="euclidean", linkage="complete", distance_threshold=2)  # type: ignore[attr-defined]

                clustering = clustering_method.fit_predict(values)
                ids = np.array(ids).astype(int)
                unique, counts = np.unique(clustering, return_counts=True)

                # merge duplicate clusters
                for un_id, count in zip(unique, counts):
                    if count > 1:
                        ids2merge = ids[clustering == un_id]
                        tmp_cluster = ClusterInLayer(clusters[ids2merge[0]].matrix.get_whole_matrix(),
                                                     self.parameters["layer_thickness"],
                                                     self.parameters["tunnel_properties_quantile"],
                                                     end_point=clusters[ids2merge[0]].end_point)
                        for merge_id in ids2merge[1:]:
                            tmp_cluster.merge_with_cluster(clusters[merge_id])
                            del self.layers[layer_id].clusters[merge_id]
                        tmp_cluster.compute_averages()

                        # new cluster is representative and is not substituted by > the number of merged duplicates
                        if not tmp_cluster.is_representative(tolerance=2):
                            clusters2add = tmp_cluster.resolve_avg_failures(self.parameters["random_seed"],
                                                                            max_new_clusters=len(ids2merge),
                                                                            tolerance=2)
                        else:
                            clusters2add = [tmp_cluster]

                        for new_cluster in clusters2add:
                            self.layers[layer_id].add_cluster(new_cluster)

    def _get_unique_pathset(self, putative_paths: Dict[int, List[str]],
                            starting_point_coords: np.ndarray) -> LayeredPathSet:
        """
        Create LayeredPathsSet object from node paths, storing only unique paths that are not subset of another path
        :param putative_paths: input node paths
        :param starting_point_coords: coordinates of average starting point for this simulation
        :return: path set with unique node paths
        """

        if self.layers is None:
            raise ValueError(f"Layers not initialized for {self.entity_label}")

        unique_path_ids = set(putative_paths.keys()) - get_redundant_path_ids(putative_paths)

        layered_path_set = LayeredPathSet(self.entity_label, self.md_label, self.parameters, starting_point_coords)

        for i, unique_path_id in enumerate(unique_path_ids):
            layered_path_set.add_node_path(putative_paths[unique_path_id], self.layers)

        return layered_path_set

    def load_points(self, source_entity: Tunnel | TransportEvent):
        """
        Load data of points from the original entity (tunnel cluster or transport event)
        :param source_entity: the original entity from which data is to be extracted
        """

        self.points_mat = source_entity.get_points_data()

    def prep_visualization(self, layered_paths: List[np.ndarray], transform_mat: np.ndarray, show_original_data: bool = False):
        """
        Prepare visualization of the layered representation - clusters (nodes) and representative paths
        :param layered_paths: analyzed node paths = list of arrays of node labels
        :param transform_mat: transformation matrix to transform output pathset
        :param show_original_data: should original data be visualized too
        """

        if self.save_points_folder is None:
            raise ValueError("Variable 'self.save_points_folder' is not specified")

        if self.original_data_path is None:
            raise ValueError("Variable 'self.original_data_path' is not specified")

        if self.visualization_prefix is None:
            raise ValueError("Variable 'self.visualization_prefix' is not specified")

        if self.layers is None:
            raise ValueError(f"Layers not initialized for {self.entity_label}")

        out_folder = os.path.join(self.save_points_folder, "nodes")
        script_path = os.path.join(self.save_points_folder, "{}.py".format(self.entity_label))
        os.makedirs(out_folder, exist_ok=True)
        relative_path2orig_data = os.path.relpath(self.original_data_path, self.save_points_folder)

        with open(script_path, "w") as view_out:
            view_out.write("import pickle, gzip\n\n")
            if show_original_data:  # visualize original data
                view_out.write("with gzip.open('{}', 'rb') as in_stream:\n".format(relative_path2orig_data))
                view_out.write("    orig_cluster = pickle.load(in_stream)\n")
                view_out.write("cmd.load_cgo(orig_cluster, 'org_data')\n")
                view_out.write("cmd.disable('org_data')\n\n")

            # visualize nodes
            for layer in self.layers.values():
                filenames = layer.save_points(out_folder, self.visualization_prefix, transform_mat)
                for filename in filenames:
                    view_out.write("cmd.load('{}')\n".format(os.path.join("nodes", filename)))
                    color = layer.id + 1
                    view_out.write("cmd.set_color('caver{}', {})\n".format(color, get_caver_color(color)))
                    view_out.write("cmd.color('{}', \"{}\")\n".format(color, filename.split(".")[0]))

            view_out.write("cmd.show_as('wire', 'all')\n")
            view_out.write("cmd.show_as('spheres', 'r. A*')\n\n")

            # visualize layered paths
            if layered_paths:
                filename = "{}_pathset.dump.gz".format(self.visualization_prefix)
                in_path = os.path.join("paths",  filename)
                view_out.write("with gzip.open('{}', 'rb') as in_stream:\n".format(in_path))
                view_out.write("    pathset = pickle.load(in_stream)\n")
                view_out.write("    for path in pathset:\n")
                view_out.write("        cmd.load_cgo(path, 'path_{}')\n".format(self.visualization_prefix))
                view_out.write("cmd.set('cgo_line_width', {}, 'path_{}')\n\n".format(10, self.visualization_prefix))
                view_out.write("cmd.load('origin.pdb', 'starting_point')\n")
                view_out.write("cmd.show_as('spheres', 'starting_point')\n")
                view_out.write("cmd.do('set all_states, 1')\n")

            view_out.write("cmd.set('sphere_scale', 0.25)\n")
            view_out.write("cmd.zoom('all')\n")
            view_out.write("cmd.show('cgo')\n")

    def _assign_entity_points2clusters(self) -> Dict[int, Dict[int, str]]:
        """
        Make inverse mapping of pointsID to Clusters (nodes) to which they belong for all original entities
        :return: map of points to node label per original entity (tunnel, event)
        """
        if self.layers is None:
            raise ValueError(f"Layers not initialized for {self.entity_label}")

        point2cluster_map = dict()
        for layer in self.layers.values():
            for cluster in layer.clusters.values():
                for tunnel_id in cluster.tunnel_ids:
                    if tunnel_id not in point2cluster_map.keys():
                        point2cluster_map[tunnel_id] = dict()
                    for point in cluster.matrix.get_points_ids4tunnel(tunnel_id):
                        point2cluster_map[tunnel_id][point] = cluster.get_node_label()

        return point2cluster_map

    def find_representative_paths(self, transform_mat: np.ndarray,  starting_point_coords: np.ndarray,
                                  visualize: bool = False) -> LayeredPathSet:
        raise NotImplementedError("Provide implementation of this method.")


class LayeredRepresentationOfTunnels(LayeredRepresentation):
    def __init__(self, parameters: dict, entity_label: str):
        """
        Class responsible for splitting of tunnel cluster data to layers and defining of their representative paths
        :param parameters: job configuration parameters
        :param entity_label: name of the entity (tunnel cluster) to be layered
        """

        LayeredRepresentation.__init__(self, parameters, entity_label)
        self.layers: Dict[int, Layer4Tunnels] = dict()  # type: ignore[assignment]
        self.save_points_folder = os.path.join(self.parameters["layered_caver_vis_path"], self.md_label)

        cls_id = int(entity_label.split("_")[1])
        self.visualization_prefix = "cls{:03d}".format(cls_id)
        self.original_data_path = os.path.join(self.parameters["orig_caver_vis_path"],
                                               self.md_label, "cls_{:03d}_cgo.dump.gz".format(cls_id))

    def __add__(self, other: LayeredRepresentationOfTunnels) -> LayeredRepresentationOfTunnels:
        if self.layer_thickness != other.layer_thickness:
            raise RuntimeError("Joining two layered representations with different layer thickness is not possible")

        new = LayeredRepresentationOfTunnels(self.parameters, self.entity_label)

        if self.points_mat is not None and other.points_mat is not None:
            new.points_mat = np.concatenate((self.points_mat, other.points_mat))
        elif self.points_mat is not None:
            new.points_mat = self.points_mat
        else:
            new.points_mat = other.points_mat

        return new

    def split_points_to_layers(self):
        """
        Splits points to Layers and execute their per layer clustering
        """

        if self.points_mat is None or self.points_mat.size == 0:
            raise RuntimeError("Layering of empty entity attempted! in {} of {}".format(self.entity_label,
                                                                                        self.md_label))
        num_points = self.points_mat.shape[0]

        # assign tunnel ids
        self.points_mat = np.append(self.points_mat, np.zeros((num_points, 1)), axis=1)
        tunnel_id = 0
        for point in self.points_mat:
            point[6] = tunnel_id
            if point[5] < 0:  # end point
                tunnel_id += 1

        layer_ids, layers = assign_layer_from_distances(self.points_mat[:, 3], self.layer_thickness)
        for layer_id in layer_ids:
            self.layers[layer_id] = Layer4Tunnels(layer_id, self.layer_thickness, self.parameters, self.entity_label,
                                                  self.md_label)
            self.layers[layer_id].cluster_data(self.points_mat[layers == layer_id, :])

        self._merge_duplicate_clusters()

    def _validate_path(self, coarse_grained_path: List[str]) -> List[str] | None:
        """
        Check if the node path is continuous through the layers and ends at node with actual tunnel endpoint
        :param coarse_grained_path: path consisting of node labels produced by get_coarse_grained_path()
        :return: validated path
        """

        # discard discontinuous tunnels in which there are no nodes in some layers
        for node_id in range(len(coarse_grained_path) - 1):
            layer1 = node_labels_split(coarse_grained_path[node_id])[0]
            layer2 = node_labels_split(coarse_grained_path[node_id + 1])[0]
            if abs(layer1 - layer2) > 1:
                return None

        # discard tunnels ending before reaching any original end point
        terminal = coarse_grained_path[-1]
        terminal_layer, terminal_cls = node_labels_split(terminal)
        if not self.layers[terminal_layer].clusters[terminal_cls].end_point:  # last node is not tunnel end point
            new_terminal = None
            for node_id in coarse_grained_path[:-1]:
                _node_layer, _node_cls = node_labels_split(node_id)
                if self.layers[_node_layer].clusters[_node_cls].end_point:
                    new_terminal = node_id
                    break

            if new_terminal is None:  # no end point present, discarding the tunnel
                return None
            else:  # getting new path truncated at the last existing end point
                new_terminal_index = np.where(np.array(coarse_grained_path) == new_terminal)[0][-1]  # last occurrence
                return coarse_grained_path[:new_terminal_index + 1]
        else:
            return coarse_grained_path

    def find_representative_paths(self, transform_mat: np.ndarray, starting_point_coords: np.ndarray,
                                  visualize: bool = False) -> LayeredPathSet:
        """
        Find representative paths leading from starting point to terminal clusters (nodes)
        :param transform_mat: transformation matrix to transform output pathset
        :param starting_point_coords: coordinates of average starting point for this simulation
        :param visualize: should the layered representation be prepared for visualization
        :return: set of representative paths
        """

        # adjust layering for global starting point placed at system origin
        clusters2reassign = list()
        for layer in self.layers.values():
            for cluster in layer.clusters.values():
                if cluster.average is None:
                    raise ValueError(f"Cluster {cluster.cls_id} in layer {cluster.layer_id} has not computed averages")
                distance2origin = einsum_dist(cluster.average, starting_point_coords)
                global_layer = int(assign_layer_from_distances(np.array([distance2origin]),
                                                                self.parameters["layer_thickness"])[1].item())
                if cluster.layer_id != global_layer and len(self.layers[cluster.layer_id].clusters.keys()) > 1:
                    # mismatch, need to move cluster to different layer, which we can do without emptying whole layer
                    clusters2reassign.append((cluster.get_node_label(), global_layer))

        for cluster_label, new_layer in clusters2reassign:
            layer_id, cls_id = node_labels_split(cluster_label)
            moved_cluster = self.layers[layer_id].pop_cluster(cls_id)
            if new_layer not in self.layers.keys():
                self.layers[new_layer] = Layer4Tunnels(new_layer, self.layer_thickness, self.parameters,
                                                       self.entity_label, self.md_label)
            self.layers[new_layer].add_cluster(moved_cluster)

        # make inverse mapping of pointsID to Clusters to which they belong
        point2cluster_map = self._assign_entity_points2clusters()
        putative_paths = dict()

        for tunnel_id in sorted(point2cluster_map.keys()):
            coarse_grained_path = get_coarse_grained_path(point2cluster_map, tunnel_id)
            validated_path = self._validate_path(coarse_grained_path)
            if validated_path is None:
                continue

            direct_path = remove_loops_from_path(validated_path)
            if direct_path is None:
                continue

            putative_paths[tunnel_id] = direct_path

        del point2cluster_map

        # create LayeredPathsSet object and filter paths that are part of other paths
        layered_path_set = self._get_unique_pathset(putative_paths, starting_point_coords)
        del putative_paths

        if not layered_path_set.is_empty():
            # get minimal set of paths that cover all nodes and lead to all terminal nodes
            layered_path_set.remove_unnecessary_paths()

            #  here we transform the pathset to fit the reference structure
            layered_path_set.transform_coordinates(transform_mat)

            if visualize:
                self.prep_visualization(layered_path_set.node_paths, transform_mat,
                                        self.parameters["visualize_transformed_tunnels"])

        return layered_path_set


class LayeredRepresentationOfEvents(LayeredRepresentation):
    def __init__(self, parameters: dict, entity_label: str, resname: str = "WAT"):
        """
        Class responsible for splitting of transport event data to layers and defining of their representative paths
        :param parameters: job configuration parameters
        :param entity_label: name of the entity (transport event) to be layered
        :param resname: name of the traced residue (e.g. "WAT", "O2")
        """

        LayeredRepresentation.__init__(self, parameters, entity_label)
        self.layers: Dict[int, Layer4Events] = dict()  # type: ignore[assignment]
        self.save_points_folder = os.path.join(self.parameters["layered_aquaduct_vis_path"], self.md_label)

        event_id = int(entity_label.split("_")[-2])
        self.original_data_path = os.path.join(self.parameters["orig_aquaduct_vis_path"], self.md_label,
                                               "raw_paths_{:d}_cgo.dump.gz".format(event_id))

        self.resname = resname
        self.visualization_prefix = entity_label

    def __add__(self, other: LayeredRepresentationOfEvents) -> LayeredRepresentationOfEvents:
        if self.layer_thickness != other.layer_thickness:
            raise RuntimeError("Joining two layered representations with different layer thickness is not possible")

        new = LayeredRepresentationOfEvents(self.parameters, self.entity_label, self.resname)

        if self.points_mat is not None and other.points_mat is not None:
            new.points_mat = np.concatenate((self.points_mat, other.points_mat))
        elif self.points_mat is not None:
            new.points_mat = self.points_mat
        else:
            new.points_mat = other.points_mat

        return new

    def split_points_to_layers(self):
        """
        Splits points to Layers and execute their per layer clustering
        """

        if self.points_mat is None or self.points_mat.size == 0:
            raise RuntimeError("Layering of empty entity attempted! in {} of {}".format(self.entity_label,
                                                                                        self.md_label))
        layer_ids, layers = assign_layer_from_distances(self.points_mat[:, 3], self.layer_thickness)

        for layer_id in layer_ids:
            self.layers[layer_id] = Layer4Events(layer_id, self.layer_thickness, self.parameters, self.entity_label,
                                                 self.md_label)
            self.layers[layer_id].cluster_data(self.points_mat[layers == layer_id, :])

        self._merge_duplicate_clusters()

    def find_representative_paths(self, transform_mat: np.ndarray,  starting_point_coords: np.ndarray | None = None,
                                  visualize: bool = False) -> LayeredPathSet:
        """
        Find representative paths leading from starting point to terminal clusters (nodes)
        :param transform_mat: transformation matrix to transform output pathset
        :param starting_point_coords: NOT USED HERE
        :param visualize: should the layered representation be prepared for visualization
        :return: set of representative paths
        """

        # make inverse mapping of pointsID to Clusters to which they belong
        point2cluster_map = self._assign_entity_points2clusters()
        coarse_grained_path = get_coarse_grained_path(point2cluster_map, 0)  # a single path exists for events => ID 0
        direct_path = remove_loops_from_path(coarse_grained_path)

        # convert to LayeredPathSet
        layered_path_set = LayeredPathSet(self.entity_label, self.md_label, self.parameters)
        if direct_path is not None:
            layered_path_set.add_node_path(direct_path, self.layers)

        if not layered_path_set.is_empty():
            #  here we transform the pathset to fit the reference structure
            layered_path_set.transform_coordinates(transform_mat)

            if visualize:
                self.prep_visualization(layered_path_set.node_paths, transform_mat,
                                        self.parameters["visualize_transformed_transport_events"])

        return layered_path_set


class Layer4Tunnels(Layer):
    def __init__(self, layer_id: int, layer_thickness: float, parameters: dict, entity_label: str, md_label: str):
        Layer.__init__(self, layer_id, layer_thickness, parameters, entity_label, md_label)

    def _make_layered_clusters(self, points_mat: np.ndarray):
        """
        Create clusters from points in this layer
        :param points_mat: points data
        """

        clustering = self._cluster_data(points_mat[:, :3])
        self._generate_clusters_from_data(points_mat, clustering)


class Layer4Events(Layer):
    def __init__(self, layer_id: int, layer_thickness: float, parameters: dict, entity_label: str, md_label: str):
        Layer.__init__(self, layer_id, layer_thickness, parameters, entity_label, md_label)

    def _make_layered_clusters(self, points_mat: np.ndarray):
        """
        Create clusters from points in this layer
        :param points_mat: points data
        """

        self.add_cluster(ClusterInLayer(points_mat, self.thickness, self.parameters["tunnel_properties_quantile"]))


def get_coarse_grained_path(point2cluster_map: Dict[int, Dict[int, str]], path_id: int) -> List[str]:
    """
    Node path formed by nodes labels along the original path
    :param point2cluster_map:
    :param path_id: ID of original path that is being coarse-grained
    :return: path consisting of node labels
    """

    path = list()

    for point in sorted(point2cluster_map[path_id].keys()):
        path.append(point2cluster_map[path_id][point])

    return [node[0] for node in groupby(path)]  # squeeze consecutive keys


def get_redundant_path_ids(all_paths: Dict[int, List[str]]) -> Set[int]:
    """
    Detects node paths that are subset of another node paths from all_paths
    :param all_paths: paths to analyze
    :return: set of IDs of redundant paths that are subsets of others
    """

    # sort paths based on their length
    length_paths = list()

    for path_id, path in all_paths.items():
        length_paths.append((len(path), path_id))
    sorted_paths = sorted(length_paths, reverse=True)

    # pairwise path comparison
    redundant_path_ids = set()
    for i, path_tuple in enumerate(sorted_paths[:-1]):
        path_len, path_id = path_tuple
        if path_id in redundant_path_ids:
            continue

        path = all_paths[path_id]
        for other_path_tuple in sorted_paths[i + 1:]:
            other_path_len, other_path_id = other_path_tuple
            other_path = all_paths[other_path_id]
            # since after sorting other_path_len <= path_len
            last_index = other_path_len
            if path[:last_index] == other_path[:last_index] or path[path_len - other_path_len:] == other_path:
                # is the shared part same from the beginning or from the end
                redundant_path_ids.add(other_path_id)

    return redundant_path_ids


def remove_loops_from_path(node_path: List[str]) -> List[str] | None:
    """
    Removes loops (repetitively visited nodes) from the node path while guaranteeing the largest span of layers
    :param node_path: analyzed node path = list of node labels
    :return: node path without loops
    """

    def _loop_prune(_path: List[str]) -> List[str]:
        """
        Removes loops from given path segment
        :param _path: path segment
        :return: direct path
        """

        _seen = set()
        _duplicates = dict()
        _reversed_path = _path[::-1]
        _direct_path = list()

        for _node in _path:  # detect nodes that have been repetitively explored along the path, excluding ends
            if _node in _seen:
                # store the last index of this node
                _duplicates[_node] = len(_path) - 1 - _reversed_path.index(_node)
            _seen.add(_node)

        _i = 0
        while _i < len(_path):
            _node = _path[_i]
            _direct_path.append(_node)
            if _node in _duplicates.keys() and _i != _duplicates[_node]:
                # skip to the item following the last occurrence of this node
                _i = _duplicates[_node] + 1
            else:
                _i += 1

        return _direct_path

    # get longest path between the furthest layers
    layers = np.array([node_labels_split(node)[0] for node in node_path])
    first_min_layer = np.min(np.where(layers == np.min(layers))[0])
    last_max_layer = np.max(np.where(layers == np.max(layers))[0])
    path_reminder = node_path[last_max_layer + 1:]
    min_max_path = node_path[first_min_layer:last_max_layer + 1]

    # remove loops to get direct paths
    try:
        direct_path = [min_max_path[0]]
    except IndexError:
        return None

    direct_path.extend(_loop_prune(min_max_path[1:]))

    if path_reminder:
        direct_path.extend(_loop_prune(path_reminder))

    return direct_path


def assign_layer_from_distances(distances: np.ndarray, layer_thickness: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate membership of points in layers based on point distance form starting point and the layer thickness
    :param distances: distance of points from starting point
    :param layer_thickness: layer thickness
    :return: unique IDs of layers and membership of points in layers
    """

    if layer_thickness <= 0:
        raise ValueError("Layer thickness has to be positive number")

    layers_membership = np.apply_along_axis(get_layer_id_from_distance, 0, distances, layer_thickness)
    layers_membership = np.where(layers_membership < 0, 0, layers_membership)

    return np.unique(layers_membership).astype(int), layers_membership


def get_layer_id_from_distance(distances: np.ndarray, layer_thickness: float) -> np.ndarray:
    """
    Calculate layers for points based on their distances form starting point and the layer thickness
    :param distances: distance of points from starting point
    :param layer_thickness: layer thickness
    :return: layerIDs
    """

    if layer_thickness <= 0:
        raise ValueError("Layer thickness has to be positive number")

    return np.ceil(np.ceil(distances) / layer_thickness) - 1


def read_starting_points(tunnel_origin_file: str) -> np.ndarray:
    """
    Extracts ensemble of coordinates of starting points from origin_file
    :param tunnel_origin_file: file with caver starting points
    :return: array with coordinates of all starting points extracted from the file
    """

    from transport_tools.libs.utils import test_file
    from transport_tools.libs.protein_files import AtomFromPDB

    starting_points = np.empty((1, 4))
    test_file(tunnel_origin_file)
    with open(tunnel_origin_file) as ORIGIN:
        for line in ORIGIN.readlines():
            if line.startswith("ATOM"):
                atom = AtomFromPDB(line.rstrip("\n"))
                starting_points = np.concatenate((starting_points, np.array([atom.x, atom.y, atom.z, 1]).reshape(1, 4)),
                                                 axis=0)

    return starting_points[1:]


def average_starting_point(tunnel_origin_file: str, md_label: str = "") -> Tuple[np.ndarray, str]:
    """
    Computes the average coordinates of starting points from origin_file
    :param tunnel_origin_file: file with caver starting points
    :param md_label: name of folder with the source MD simulation data
    :return: array containing the average starting point coordinates & md_label
    """

    starting_points = read_starting_points(tunnel_origin_file)
    return np.mean(starting_points, axis=0).reshape(4, 1), md_label
            

def cart2spherical(xyz: np.ndarray) -> np.ndarray:
    """
    Converts cartesian coordinates to spherical ones
    :param xyz: cartesian coords
    :return: spherical coords
    """

    xy = xyz[0] ** 2 + xyz[1] ** 2
    r = np.sqrt(xy + xyz[2] ** 2)
    theta = np.arctan2(np.sqrt(xy), xyz[2])
    phi = np.arctan2(xyz[1], xyz[0])

    return np.array([r, theta, phi])


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Returns angle between two vectors
    :param v1: input vector
    :param v2: input vector
    :return: angle
    """

    cosang = np.dot(v1, v2)
    sinang = np.linalg.norm(np.cross(v1, v2))

    return np.arctan2(sinang, cosang)


def einsum_dist(xyz1: np.ndarray, xyz2: np.ndarray):
    """
    Calculates distance between two points in computationally rather efficient manner
    :param xyz1: coordinates of the first point
    :param xyz2: coordinates of the second point
    :return:
    """

    z = (xyz1 - xyz2).T
    return np.sqrt(np.einsum('ij,ij->j', z, z))


def _index_fragmented_path(fragmented_path: List[List[str]],
                           node_index: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a fragmented path (list of node-label fragments) into the integer node indices along its concatenated
    nodes, the cumulative lengths of its fragment prefixes, and a boolean mask of its starting-point (SP) nodes.
    Used by LayeredPathSet.avg_distance2path_set to address the dense distance matrices.
    :param fragmented_path: path fragments, each a list of node labels
    :param node_index: mapping from node label to its row/column index in the dense distance matrix
    :return: node indices along the concatenated path, prefix lengths per fragment, SP-node mask
    """

    labels = [label for fragment in fragmented_path for label in fragment]
    node_ids = np.array([node_index[label] for label in labels], dtype=int)
    prefix_lengths = np.cumsum([len(fragment) for fragment in fragmented_path])
    is_sp = np.array(["SP" in label for label in labels], dtype=bool)

    return node_ids, prefix_lengths, is_sp


# Shared state for the cluster-distance worker processes; populated once per worker by
# init_distance_worker() so that individual jobs carry only lightweight cluster index pairs
# instead of re-pickling the (potentially large) path-sets for every pairwise comparison.
_DIST_WORKER_STATE: dict = {}


def init_distance_worker(path_sets: Dict[Tuple[str, int], LayeredPathSet],
                         cluster_specifications: List[Tuple[str, int]],
                         precision: int, cutoff: float,
                         allocated_cpus: int = 1, pool_size: int = 1):
    """
    Initializer for the cluster-distance multiprocessing Pool; stores data shared by all jobs
    once per worker process, so individual jobs need to carry only lightweight cluster index
    pairs instead of re-pickling the path-sets for every pairwise comparison.
    :param path_sets: sets of representative paths for all clusters
    :param cluster_specifications: definition of clusters, indexed by their order ID
    :param precision: number of decimals with which the calculated distances are reported
    :param cutoff: clustering cutoff
    :param allocated_cpus: total CPU budget the parent has (num_cpus on local, slurm_cpus_per_task
                           inside a SLURM shard); used to cap this worker's BLAS thread count so
                           pool_size * threads_per_worker stays within the parent's allocation
    :param pool_size: number of worker processes in this pool, used in the BLAS-thread cap
    """

    # Cap BLAS threads first, before any numpy operation in this worker triggers BLAS-pool init.
    from transport_tools.libs.utils import cap_blas_threads_for_worker
    cap_blas_threads_for_worker(allocated_cpus, pool_size)

    _DIST_WORKER_STATE["path_sets"] = path_sets
    _DIST_WORKER_STATE["cluster_specifications"] = cluster_specifications
    _DIST_WORKER_STATE["precision"] = precision
    _DIST_WORKER_STATE["cutoff"] = cutoff


def calc_distance_chunk(index_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, float]]:
    """
    Computes average cluster-cluster distances for a chunk of cluster index pairs, using the
    path-sets shared with the worker process via init_distance_worker().
    :param index_pairs: list of (cls1, cls2) order ID pairs to evaluate
    :return: list of (cls1, cls2, average distance) tuples
    """

    path_sets = _DIST_WORKER_STATE["path_sets"]
    cluster_specifications = _DIST_WORKER_STATE["cluster_specifications"]
    precision = _DIST_WORKER_STATE["precision"]
    cutoff = _DIST_WORKER_STATE["cutoff"]

    results = list()
    for cls1, cls2 in index_pairs:
        path_set1 = path_sets[cluster_specifications[cls1]]
        path_set2 = path_sets[cluster_specifications[cls2]]
        distance = np.around(path_set1.avg_distance2path_set(path_set2, cutoff), precision)
        results.append((cls1, cls2, distance))

    return results


def iter_pair_chunks(num_clusters: int, chunk_size: int) -> Iterable[List[Tuple[int, int]]]:
    """
    Lazily yields chunks of upper-triangle cluster index pairs to keep memory bounded even for
    large numbers of clusters.
    :param num_clusters: total number of clusters
    :param chunk_size: maximal number of index pairs per yielded chunk
    """

    chunk = list()
    for cls1 in range(num_clusters):
        for cls2 in range(cls1 + 1, num_clusters):
            chunk.append((cls1, cls2))
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = list()
    if chunk:
        yield chunk


def condensed_pair_index(i: np.ndarray, j: np.ndarray, num_clusters: int) -> np.ndarray:
    """
    Maps upper-triangle (i, j) index arrays (with i < j) to their positions in the condensed 1-D
    representation of a num_clusters x num_clusters symmetric distance matrix, using the same
    row-major ordering as scipy.spatial.distance.squareform and numpy.triu_indices(n, 1). This
    lets stage 4 assemble and store the pairwise distances directly in condensed form, avoiding
    the dense N x N matrix entirely.
    :param i: array of lower (row) indices
    :param j: array of higher (column) indices
    :param num_clusters: total number of clusters (side length of the dense matrix)
    :return: array of positions in the condensed distance vector
    """

    return i * num_clusters - (i * (i + 1)) // 2 + j - i - 1


def count_pairs_for_shard(num_clusters: int, num_shards: int, shard_id: int) -> int:
    """
    Returns the number of upper-triangle cluster index pairs assigned to a single shard under the
    strided assignment used by iter_shard_pair_chunks().
    :param num_clusters: total number of clusters
    :param num_shards: total number of shards the work is split into
    :param shard_id: 0-based ID of the shard
    """

    total_pairs = (num_clusters ** 2 - num_clusters) // 2
    return len(range(shard_id, total_pairs, num_shards))


def iter_shard_pair_chunks(num_clusters: int, num_shards: int, shard_id: int,
                           chunk_size: int) -> Iterable[List[Tuple[int, int]]]:
    """
    Lazily yields chunks of the upper-triangle cluster index pairs assigned to a single shard.
    Pairs are distributed across shards in a strided fashion (pair i goes to shard i % num_shards),
    which balances the load well because the per-pair cost varies along the matrix rows.

    The shard's pairs are exactly the condensed positions range(shard_id, N(N-1)/2, num_shards) -
    the upper triangle is enumerated row-major, so a pair's running index equals its condensed
    position (see condensed_pair_index). Rather than scanning all N(N-1)/2 positions in Python to
    pick out this shard's stride (an O(N**2) interpreter loop in *every* shard - tens of minutes of
    pure enumeration overhead for N in the 10**5 range), generate only this shard's positions and
    invert them to (i, j) vectorised: i = searchsorted(row_offsets, pos) - 1, j = pos -
    row_offsets[i] + i + 1, where row_offsets[i] is the condensed start of row i. This yields the
    identical set of pairs, in the identical (ascending-position) order, as the old scan.
    :param num_clusters: total number of clusters
    :param num_shards: total number of shards the work is split into
    :param shard_id: 0-based ID of the shard whose pairs are yielded
    :param chunk_size: maximal number of index pairs per yielded chunk
    """

    total_pairs = num_clusters * (num_clusters - 1) // 2
    if total_pairs == 0 or shard_id >= total_pairs:
        return

    # condensed start offset of each matrix row: row i holds columns i+1..num_clusters-1, so its
    # first pair (i, i+1) sits at row_offsets[i] in the condensed vector
    rows_axis = np.arange(num_clusters, dtype=np.int64)
    row_offsets = rows_axis * num_clusters - (rows_axis * (rows_axis + 1)) // 2

    # this shard owns every num_shards-th condensed position starting at shard_id; walk them in
    # blocks of chunk_size positions (a stride of num_shards * chunk_size between block starts)
    block_stride = num_shards * chunk_size
    for block_start in range(shard_id, total_pairs, block_stride):
        positions = np.arange(block_start, min(block_start + block_stride, total_pairs),
                              num_shards, dtype=np.int64)
        i = np.searchsorted(row_offsets, positions, side='right') - 1
        j = positions - row_offsets[i] + i + 1
        yield list(zip(i.tolist(), j.tolist()))


def subcutoff_connected_components(condensed: np.ndarray, num_points: int, cutoff: float,
                                   chunk_size: int = 1 << 22) -> Tuple[int, np.ndarray]:
    """
    Partition points into the connected components of the sub-cutoff graph G = {(i, j) : d(i, j) <=
    cutoff}, where d is the condensed (upper-triangle) pairwise-distance vector of a
    num_points x num_points symmetric matrix. This is the shared substrate of stage-5 clustering:
    partitioning by sub-cutoff connectivity lets each component be clustered independently, dropping
    linkage/HDBSCAN cost from O(N**2) to O(max |C_k|**2). It keys off the cutoff threshold only and
    is correct regardless of whether far pairs carry real distances or the approximate-mode 999
    saturation.

    The condensed vector (potentially terabytes for large N, so possibly a memmap) is thresholded in
    element-bounded chunks rather than all at once - each chunk's comparison and the recovery of its
    surviving (i, j) pairs are fully vectorised. The condensed position k of a survivor is mapped
    back to (i, j) exactly via a vectorised search over the precomputed row-start offsets:
    i = searchsorted(row_offsets, k) - 1 and j = k - row_offsets[i] + i + 1 (no fragile float-sqrt
    inversion, no per-row Python loop). The chunk_size is a pure performance/memory knob: the
    surviving edge set, and hence the partition, is identical for any chunk_size. The graph handed to
    scipy is a structural all-ones uint8 adjacency: only the connectivity matters, so storing 1 per
    edge keeps a legitimate zero-distance edge from being pruned (tocsr() does not call
    eliminate_zeros(), but relying on that is version-fragile) and shrinks the data array 8x versus
    float64.

    :param condensed: condensed upper-triangle distance vector of length num_points*(num_points-1)/2
                      (may be a numpy memmap; only chunk_size elements are materialised at a time)
    :param num_points: number of points (side length of the implied dense matrix)
    :param cutoff: distance threshold; pairs with d <= cutoff form the graph edges
    :param chunk_size: number of condensed elements thresholded per vectorised pass (perf/memory only)
    :return: (number of connected components, per-point component-label array of length num_points)
    """

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    num_points = int(num_points)
    if num_points <= 0:
        return 0, np.empty(0, dtype=np.intp)
    if num_points == 1:
        return 1, np.zeros(1, dtype=np.intp)

    # start offset of each matrix row within the condensed vector; row i holds columns i+1..num_points-1
    rows_axis = np.arange(num_points, dtype=np.int64)
    row_offsets = rows_axis * num_points - (rows_axis * (rows_axis + 1)) // 2

    total_pairs = num_points * (num_points - 1) // 2
    row_indices: List[np.ndarray] = list()
    col_indices: List[np.ndarray] = list()
    for start in range(0, total_pairs, chunk_size):
        block = np.asarray(condensed[start: start + chunk_size])
        local_hits = np.nonzero(block <= cutoff)[0]
        if not local_hits.size:
            continue
        # recover the surviving (i, j) pairs from their condensed positions, fully vectorised
        positions = local_hits + start
        i = np.searchsorted(row_offsets, positions, side='right') - 1
        j = positions - row_offsets[i] + i + 1
        row_indices.append(i.astype(np.int32, copy=False))
        col_indices.append(j.astype(np.int32, copy=False))

    if row_indices:
        rows = np.concatenate(row_indices)
        cols = np.concatenate(col_indices)
    else:
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)
    # the per-chunk edge arrays are now duplicated inside the concatenated rows/cols; drop them so
    # the edge set is held once rather than twice (matters when the sub-cutoff graph is dense, i.e.
    # many pairs <= cutoff, which is exactly the redundant-dataset regime that produces large E)
    del row_indices, col_indices

    data = np.ones(rows.size, dtype=np.uint8)
    graph = coo_matrix((data, (rows, cols)), shape=(num_points, num_points)).tocsr()
    # the CSR holds its own copies of the indices; release the COO inputs before the traversal so
    # the edge arrays are not pinned while connected_components allocates its own O(V + E) structures
    del rows, cols, data
    n_components, labels = connected_components(graph, directed=False, return_labels=True)

    return n_components, labels.astype(np.intp, copy=False)


def gather_dense_submatrix(condensed: np.ndarray, num_points: int, members: np.ndarray,
                           stripe_elements: int = 1 << 22) -> np.ndarray:
    """
    Build the dense |members| x |members| distance block of a subset of points from the condensed
    upper-triangle distance vector of the full num_points x num_points matrix. Used by the stage-5
    partition driver to hand one connected component's distances to the HDBSCAN backend (which needs
    a dense precomputed block; the agglomerative backend uses gather_condensed_subvector instead).
    The diagonal stays 0.

    The block is filled in row-stripes so the index/value temporaries stay bounded to
    O(stripe_rows * m) rather than the O(m**2) full-triangle arrays a single np.triu_indices(m, 1)
    gather would allocate (~24 m**2 transient bytes of int64/float64 - several times the block
    itself for m in the 10**5 range, which is the stage-5 peak-memory bottleneck on large studies).

    :param condensed: condensed upper-triangle distance vector of the full matrix (may be a memmap)
    :param num_points: side length of the full dense matrix the condensed vector represents
    :param members: global indices of the points in the block; order defines the block's row/column
                    order (the returned block is symmetric regardless of whether members is sorted)
    :param stripe_elements: target number of (row x m) grid elements materialised per stripe; pure
                            performance/memory knob, the returned block is identical for any value
    :return: dense (|members|, |members|) float64 distance block with a zero diagonal
    """

    members = np.asarray(members, dtype=np.intp)
    m = members.size
    block = np.zeros((m, m), dtype=np.float64)
    if m <= 1:
        return block

    cols = members[np.newaxis, :]                                   # (1, m)
    stripe_rows = max(1, stripe_elements // m)
    for a in range(0, m, stripe_rows):
        b = min(a + stripe_rows, m)
        rows = members[a:b, np.newaxis]                             # (h, 1)
        # min/max so condensed_pair_index always sees i < j even if members is not sorted ascending
        global_i = np.minimum(rows, cols)                           # (h, m)
        global_j = np.maximum(rows, cols)
        # members are distinct, so global_i == global_j marks exactly the block diagonal; its slots
        # are placeholders (condensed_pair_index assumes i < j) and are zeroed after the gather
        on_diagonal = global_i == global_j
        positions = condensed_pair_index(global_i.ravel(), global_j.ravel(), num_points)
        positions[on_diagonal.ravel()] = 0
        values = np.asarray(condensed[positions]).reshape(b - a, m)
        values[on_diagonal] = 0.0
        block[a:b, :] = values

    return block


def gather_condensed_subvector(condensed: np.ndarray, num_points: int, members: np.ndarray,
                               chunk_size: int = 1 << 22) -> np.ndarray:
    """
    Gather a subset's pairwise distances directly in condensed (upper-triangle) form, i.e. exactly
    what squareform(gather_dense_submatrix(...), checks=False) would return, but WITHOUT
    materialising the dense |members| x |members| block. The stage-5 agglomerative backend feeds
    this straight to fastcluster.linkage, so for a component of m members the path holds ~4 m**2
    bytes (this vector) instead of ~12 m**2 (the dense block plus its squareform copy).

    The output enumerates local upper-triangle pairs (i, j), i < j, in the same row-major order as
    scipy.spatial.distance.squareform / condensed_pair_index, so the linkage it produces is
    bit-identical to the previous dense-block path. It is gathered in chunks of the condensed index
    space, so the temporaries are bounded to O(chunk_size).

    :param condensed: condensed upper-triangle distance vector of the full matrix (may be a memmap)
    :param num_points: side length of the full dense matrix the condensed vector represents
    :param members: global indices of the points; order defines the local pair ordering
    :param chunk_size: number of condensed sub-vector positions gathered per vectorised pass
                       (performance/memory only; the returned vector is identical for any value)
    :return: condensed (m(m-1)/2,) float64 distance vector over the members, in squareform order
    """

    members = np.asarray(members, dtype=np.intp)
    m = members.size
    sub_pairs = m * (m - 1) // 2
    out = np.empty(sub_pairs, dtype=np.float64)
    if sub_pairs == 0:
        return out

    # condensed start offset of each local row i within the sub-vector (row i holds cols i+1..m-1)
    rows_axis = np.arange(m, dtype=np.int64)
    sub_row_offsets = rows_axis * m - (rows_axis * (rows_axis + 1)) // 2

    for start in range(0, sub_pairs, chunk_size):
        sub_positions = np.arange(start, min(start + chunk_size, sub_pairs), dtype=np.int64)
        # invert sub-vector position -> local (i, j), then map to global via members (min/max so
        # condensed_pair_index sees i < j even when members is unsorted)
        i = np.searchsorted(sub_row_offsets, sub_positions, side='right') - 1
        j = sub_positions - sub_row_offsets[i] + i + 1
        global_i = np.minimum(members[i], members[j])
        global_j = np.maximum(members[i], members[j])
        out[start: start + sub_positions.size] = condensed[condensed_pair_index(global_i, global_j,
                                                                                num_points)]

    return out
