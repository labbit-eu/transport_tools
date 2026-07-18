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

import os
import pickle
import tarfile
import gzip
import numpy as np
from collections import OrderedDict
from threadpoolctl import threadpool_limits
from logging import getLogger
from multiprocessing import Pool, get_context
from re import search
from typing import Dict, List, TextIO, Set, Tuple
from transport_tools.libs.geometry import Point, LayeredRepresentationOfTunnels, \
    LayeredRepresentationOfEvents, LayeredPathSet, vector_angle, assign_layer_from_distances, einsum_dist, \
    average_starting_point
from transport_tools.libs import utils
from transport_tools.libs.protein_files import VizAtom

logger = getLogger(__name__)


class Network:
    def __init__(self, parameters: dict, md_label: str):
        """
        Generic class for handling transport tunnels and paths
        :param parameters: job configuration parameters
        :param md_label: name of folder with the source MD simulation data
        """

        self.parameters: dict = parameters.copy()
        self.md_label: str = md_label
        self.transform_mat: np.ndarray | None = None
        self.layered_entities: Dict[str | int, LayeredPathSet] = dict()
        self.entity_pymol_abbreviation: str | None = None
        self.transformed_pdb_file_name: str | None = None
        self.orig_entities: List[TunnelCluster | AquaductPath] = list()
        self.starting_point_coords: np.ndarray | None = None
        # newly added parameter to dict that is passed down to LayeredRepresentation:
        self.parameters["md_label"] = md_label

        # input paths
        self.pdb_file: str | None = None

        # output paths
        self.transformation_folder: str | None = None
        self.orig_viz_path: str | None = None
        self.orig_dump_file: str | None = None
        self.layered_dump_file: str | None = None
        self.layered_viz_path: str | None = None

        # Sidecar written next to layered_dump_file by save_layered_network() containing only
        # the lightweight per-entity field listed in `_layered_summary_attr` (subclasses set
        # these). Stage 4 (caver) and stage 9 SLURM launcher (aquaduct) read this sidecar
        # via load_layered_summary() to avoid the heavy full-pickle load. None disables.
        self._layered_summary_attr: str | None = None
        self._layered_summary_suffix: str | None = None

        # Sidecar written next to orig_dump_file by save_orig_network() holding only the
        # per-entity summary needed to enumerate layering work (see _compute_orig_summary).
        # The stage-3 / stage-8 SLURM layering launchers read it (via get_clusters4layering_ids
        # / get_events4layering_ids) so they can build the (md_label, entity_id) enumeration
        # without the heavy full-pickle load of every orig_network on a single node. Subclasses
        # that set it must override _compute_orig_summary(); None disables emission.
        self._orig_summary_suffix: str | None = None

    def add_layered_entity(self, entity_id: int | str, layered_pathset: LayeredPathSet):
        """
        Add layered entity to the network
        :param entity_id: ID of layered entity
        :param layered_pathset:  layered entity to add
        """

        self.layered_entities[entity_id] = layered_pathset

    def is_layering_complete(self, entity_ids: List[int | str]) -> bool:
        """
        Tests is all original entities to verify existence of their layered counterparts
        :param entity_ids: list of IDs of original entities
        :return: if all original entities have been layered
        """

        return set(entity_ids) == set(self.layered_entities.keys())

    def _save_transformed_pdb_file(self, transformed_pdb_filename: str):
        """
        Transform the reference structure of this network and save it  PDB file
        :param transformed_pdb_filename: path to the output file
        """

        from transport_tools.libs.protein_files import transform_pdb_file
        if self.transformed_pdb_file_name is None:
            raise ValueError("Variable 'self.transformed_pdb_file_name' is not specified")

        if self.pdb_file is None:
            raise ValueError("Variable 'self.pdb_file' is not specified")

        if self.transform_mat is None:
            raise ValueError("Variable 'self.transform_mat' is not specified")

        transform_pdb_file(self.pdb_file, transformed_pdb_filename, self.transform_mat)

    def _load_transformations(self):
        """
        Load transformation matrix for this network from self.transformation_folder
        """

        if self.transformation_folder is None:
            raise ValueError("Variable 'self.transformation_folder' is not specified")

        mat_file = os.path.join(self.transformation_folder, self.md_label + "-transform_mat.dump")
        with open(mat_file, "rb") as in_stream:
            self.transform_mat = pickle.load(in_stream)

    def load_orig_network(self):
        """
        Load pre-computed original entities (AquaductPath, TunnelCluster)
        """

        if self.orig_dump_file is None:
            raise ValueError("Variable 'self.orig_dump_file' is not specified")

        utils.test_file(self.orig_dump_file)
        with open(self.orig_dump_file, "rb") as in_stream:
            self.orig_entities: List[TunnelCluster | AquaductPath] = pickle.load(in_stream)
        for entity in self.orig_entities:
            entity.parameters.update(self.parameters)

    def save_orig_network(self):
        """
        Dump original entities (AquaductPath, TunnelCluster) to files for later processing
        """

        if self.orig_dump_file is None:
            raise ValueError("Variable 'self.orig_dump_file' is not specified")

        os.makedirs(os.path.dirname(self.orig_dump_file), exist_ok=True)
        with open(self.orig_dump_file, "wb") as out_stream:
            pickle.dump(self.orig_entities, out_stream)

        self._write_orig_summary_sidecar()

    def _orig_summary_path(self) -> str | None:
        """
        Path to the lightweight summary sidecar written next to orig_dump_file, or None when
        the subclass disables sidecar emission (_orig_summary_suffix unset).
        """

        if self._orig_summary_suffix is None:
            return None
        if self.orig_dump_file is None:
            raise ValueError("Variable 'self.orig_dump_file' is not specified")
        return self.orig_dump_file + self._orig_summary_suffix

    def _compute_orig_summary(self) -> dict:
        """
        Build the lightweight per-entity summary persisted to the orig size sidecar. Subclasses
        that set `_orig_summary_suffix` must override this. The default raises so a misconfigured
        subclass fails loudly rather than writing an empty/meaningless sidecar.
        """

        raise NotImplementedError("Subclass {} set _orig_summary_suffix but did not override "
                                  "_compute_orig_summary().".format(type(self).__name__))

    def _write_orig_summary_sidecar(self):
        """
        Dump the lightweight per-entity size summary (see _compute_orig_summary) to a small
        sidecar next to orig_dump_file. The stage-3 SLURM tunnel-layering launcher reads it
        to enumerate layering work without the heavy full orig_network load on a single node.
        No-op when the subclass leaves _orig_summary_suffix unset.
        """

        sidecar_path = self._orig_summary_path()
        if sidecar_path is None:
            return

        summary = self._compute_orig_summary()
        sidecar_tmp = sidecar_path + ".tmp"
        with open(sidecar_tmp, "wb") as out_stream:
            pickle.dump(summary, out_stream, self.parameters.get("pickle_protocol", 4))
        os.replace(sidecar_tmp, sidecar_path)

    def load_layered_network(self):
        """
        Load pre-computed layered entities (LayeredPathSet representing transport events or tunnel clusters)
        """

        if self.layered_dump_file is None:
            raise ValueError("Variable 'self.layered_dump_file' is not specified")

        if os.path.exists(self.layered_dump_file):
            with open(self.layered_dump_file, "rb") as in_stream:
                self.layered_entities: Dict[str | int, LayeredPathSet] = pickle.load(in_stream)

            for entity in self.layered_entities.values():
                entity.parameters.update(self.parameters)

    def save_layered_network(self):
        """
        Dump layered entities (LayeredPathSet representing transport events or tunnel clusters) to files
        for later processing
        """

        if self.layered_dump_file is None:
            raise ValueError("Variable 'self.layered_dump_file' is not specified")

        os.makedirs(os.path.dirname(self.layered_dump_file), exist_ok=True)
        with open(self.layered_dump_file, "wb") as out_stream:
            pickle.dump(self.layered_entities, out_stream)

        self._write_layered_summary_sidecar()

    def _layered_summary_path(self) -> str | None:
        """
        Path to the lightweight summary sidecar written next to layered_dump_file, or None
        when the subclass disables sidecar emission (both class attrs unset).
        """

        if self._layered_summary_attr is None or self._layered_summary_suffix is None:
            return None
        if self.layered_dump_file is None:
            raise ValueError("Variable 'self.layered_dump_file' is not specified")
        return self.layered_dump_file + self._layered_summary_suffix

    def _write_layered_summary_sidecar(self):
        """
        Dump the per-entity summary field (e.g., (avg_throughput, num_tunnels) for tunnel
        clusters, traced_event for aquaduct events) to a small sidecar next to
        layered_dump_file. Stage 4's SLURM launcher and stage 9's SLURM launcher use the
        sidecar to skip the multi-100-GB / multi-hour full-pickle load on large studies.
        """

        sidecar_path = self._layered_summary_path()
        if sidecar_path is None:
            return

        attr_name = self._layered_summary_attr
        assert attr_name is not None  # for type-checkers; guarded by _layered_summary_path
        summary = {
            entity_id: getattr(entity, attr_name)
            for entity_id, entity in self.layered_entities.items()
        }
        sidecar_tmp = sidecar_path + ".tmp"
        with open(sidecar_tmp, "wb") as out_stream:
            pickle.dump(summary, out_stream, self.parameters.get("pickle_protocol", 4))
        os.replace(sidecar_tmp, sidecar_path)

    def load_layered_summary(self) -> Dict[str | int, object]:
        """
        Load the per-entity summary sidecar (the lightweight field declared by the subclass:
        characteristics for tunnel clusters, traced_event for aquaduct events). Falls back
        to a full load_layered_network() + extraction when the sidecar is missing (legacy
        checkpoints), and writes the sidecar so subsequent runs skip the heavy load.
        :return: dict mapping entity_id to the summary value (may be None per-entity)
        """

        sidecar_path = self._layered_summary_path()
        if sidecar_path is None:
            raise RuntimeError("Subclass {} did not configure _layered_summary_attr / "
                               "_layered_summary_suffix; load_layered_summary() is not "
                               "available.".format(type(self).__name__))

        if os.path.exists(sidecar_path):
            with open(sidecar_path, "rb") as in_stream:
                return pickle.load(in_stream)

        # Fallback: legacy checkpoint without sidecar. Load the heavy pickle, extract the
        # summary field, and write the sidecar so subsequent runs skip the full load.
        if self.layered_dump_file is None or not os.path.exists(self.layered_dump_file):
            return dict()
        self.load_layered_network()
        attr_name = self._layered_summary_attr
        assert attr_name is not None
        summary = {
            entity_id: getattr(entity, attr_name)
            for entity_id, entity in self.layered_entities.items()
        }
        try:
            sidecar_tmp = sidecar_path + ".tmp"
            with open(sidecar_tmp, "wb") as out_stream:
                pickle.dump(summary, out_stream, self.parameters.get("pickle_protocol", 4))
            os.replace(sidecar_tmp, sidecar_path)
        except OSError as exc:
            logger.debug("Could not write summary sidecar '%s': %s", sidecar_path, exc)
        # Drop the heavy dict so the caller doesn't keep it in memory.
        self.layered_entities = dict()
        return summary

    def save_orig_network_visualization(self):
        """
        Saves CGO files with original entities AquaDuctPath, TunnelCluster) and PDBs of transformed protein
        structure for visualization
        """

        if self.orig_viz_path is None:
            raise ValueError("Variable 'self.orig_viz_path' is not specified")

        if self.transformed_pdb_file_name is None:
            raise ValueError("Variable 'self.transformed_pdb_file_name' is not specified")

        os.makedirs(self.orig_viz_path, exist_ok=True)
        self._save_transformed_pdb_file(os.path.join(self.orig_viz_path, self.transformed_pdb_file_name))

        with open(os.path.join(self.orig_viz_path, "view_network.py"), "w") as view_out:
            view_out.write("import pickle, gzip\n\n")
            view_out.write("cmd.load('{}', 'protein_structure')\n".format(self.transformed_pdb_file_name))
            view_out.write("cmd.show_as('cartoon', 'protein_structure')\n")
            view_out.write("cmd.color('gray', 'protein_structure')\n\n")

            if self.orig_entities:
                for entity in self.orig_entities:
                    entity.save_cgo(self.orig_viz_path)
                    view_out.write("with gzip.open('{}', 'rb') as in_stream:\n".format(entity.vis_file))
                    view_out.write("    load_cluster = pickle.load(in_stream)\n")
                    view_out.write("cmd.load_cgo(load_cluster, '{}')\n".format(entity.entity_pymol_label))

                view_out.write("cmd.disable('{}*')\n\n".format(self.orig_entities[0].entity_pymol_label[0:3]))

    def save_layered_visualization(self, save_pdb_files: bool = False, save_cgo_files: bool = True):
        """
        Saves CGO files with layered entities (LayeredPathSet representing transport events or tunnel clusters)
        and optionally also PDBs of transformed protein structure and tunnel starting point
        :param save_pdb_files: if the PDB files are to be saved
        :param save_cgo_files: if a per-entity CGO file is to be written; disabled for transport events when
                               bundle_events_visualization is on, since stage 9 then builds bundled CGOs from
                               the layered_network.dump instead of one file per event
        """

        if self.layered_viz_path is None:
            raise ValueError("Output folder for layered_visualization not correctly specified in variable "
                             "'self.layered_viz_path'")

        if save_pdb_files:
            os.makedirs(self.layered_viz_path, exist_ok=True)
            if self.transformed_pdb_file_name is None:
                raise ValueError("Variable 'self.transformed_pdb_file_name' is not specified")
            Point([0, 0, 0]).save_point(os.path.join(self.layered_viz_path, "origin.pdb"))
            self._save_transformed_pdb_file(os.path.join(self.layered_viz_path, self.transformed_pdb_file_name))

        if not save_cgo_files:
            # Nothing per-entity to write (e.g. bundled event visualization); do not create empty viz folders.
            return

        if self.entity_pymol_abbreviation is None:
            raise ValueError("Variable 'self.entity_pymol_abbreviation' is not specified")

        os.makedirs(os.path.join(self.layered_viz_path, "paths"), exist_ok=True)

        for entity_id, layered_path_set in self.layered_entities.items():
            # Use per-entity visualization_prefix if available (e.g. for mixed-residue aqueduct networks)
            if layered_path_set.visualization_prefix is not None:
                entity_label = layered_path_set.visualization_prefix
            else:
                try:
                    entity_label = "{}{:03d}".format(self.entity_pymol_abbreviation, entity_id)
                except ValueError:
                    entity_label = "{}{:s}".format(self.entity_pymol_abbreviation, entity_id)

            layered_path_set.visualize_cgo(os.path.join(self.layered_viz_path, "paths"), entity_label)


# --- CAVER tunnels related classes
class Tunnel:
    def __init__(self, parameters: dict, transform_mat: np.ndarray):
        """
        Class for processing of tunnels from CAVER - an elemental unit carrying info on a tunnel
        :param parameters: job configuration parameters
        :param transform_mat: transformation matrix to be applied on the input coordinates
        """

        self.parameters = parameters
        self.transform_mat = transform_mat
        self.snapshot = ""
        self.caver_cluster_id = -1
        self.bottleneck_radius = -1.0
        self.curvature = -1.0
        self.length = -1.0
        self.throughput = -1.0
        self.cost = -1.0
        self.tunnel_id = -1
        self.filters_passed = False
        self.spheres_data: np.ndarray | None = None
        self.layer_membership: np.ndarray | None = None
        self.bottleneck_residues = list()
        self.bottleneck_xyz: np.ndarray | None = None
        self.weight = 1.0

    def __str__(self):
        output = ""
        if self.spheres_data is None:
            raise ValueError(f"Tunnel sphere data must be provided before useage for tunnel {self.tunnel_id} of cluster {self.caver_cluster_id}")

        for sphere in self.spheres_data:
            output = output + "(x, y, z, distance, radius, length) = " \
                              "({:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}, {:4.2f}, {:4.2f})-".format(sphere[0], sphere[1],
                                                                                               sphere[2],
                                                                                               sphere[3], sphere[4],
                                                                                               sphere[5])
        return output

    def get_center_line(self) -> np.ndarray:
        """
        Returns coordinates of spheres centers forming the tunnel
        """
        if self.spheres_data is None:
            raise ValueError(f"Tunnel sphere data must be provided before useage for tunnel {self.tunnel_id} of cluster {self.caver_cluster_id}")

        return self.spheres_data[:, :3]

    def get_snapshot_id(self, id_position: int = 1, delimiter: str = ".") -> int:
        """
        Returns numerical position of snapshot to which this tunnel belongs based on snapshot name from CAVER
        :param id_position: which field will contain snapshot ID after splitting with the delimiter
        :param delimiter: delimiter to find the snapshot ID at the specified position
        :return: snapshot ID
        """

        try:
            return int(self.snapshot.split(delimiter)[id_position])
        except (ValueError, IndexError):
            raise RuntimeError("snapshot ID not present at position {} after splitting with {}".format(id_position,
                                                                                                       delimiter))

    def get_points_data(self) -> np.ndarray:
        """
        Convert data of points forming this tunnel, adding info on order of points and identifying the tunnel end point
        :return: augmented data for this tunnel suitable for LayeredRepresentation.load_points() method
        """

        if self.spheres_data is None:
            raise ValueError(f"Tunnel sphere data must be provided before useage for tunnel {self.tunnel_id} of cluster {self.caver_cluster_id}")

        num_points = self.spheres_data.shape[0]
        # store info on points order in the sixth column (id = 5)
        order = np.arange(0, num_points).astype(float)
        order[-1] = -order[-1]  # end point id is inverted
        points_data = self.spheres_data.copy()
        points_data[:, 5] = order

        # to guarantee independence on analyzed set of simulations this needs to be performed on untransformed data
        data4transform = np.append(points_data[:, 0:3], np.full((points_data[:, 0:3].shape[0], 1), 1.), axis=1).T
        rev_trans_mat = np.linalg.inv(self.transform_mat)
        orig_dat = rev_trans_mat.dot(data4transform).T[:, :3]
        points_data[:, 0:3] = orig_dat

        return points_data

    def get_csv_lines(self, md_label: str) -> str:
        """
        Produces this tunnel's data in CVS format analogous to caver tunnel_profile files, adding info on its md_label
        :param md_label: name of folder with the source MD simulation data
        :return: seven lines in CVS format
        """

        temp = "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, , " \
               "".format(md_label, self.snapshot, self.caver_cluster_id, self.tunnel_id, self.throughput, self.cost,
                         self.bottleneck_radius, "-", "-", "-", self.curvature, self.length)
        line1 = temp + "X"
        line2 = temp + "Y"
        line3 = temp + "Z"
        line4 = temp + "distance"
        line5 = temp + "length"
        line6 = temp + "R"
        line7 = temp + "Upper limit of R overestimation"
        
        if self.spheres_data is None:
            raise ValueError(f"Tunnel sphere data must be provided before useage for tunnel {self.tunnel_id} of cluster {self.caver_cluster_id}")

        for sphere in self.spheres_data:
            line1 += ", {}".format(sphere[0])
            line2 += ", {}".format(sphere[1])
            line3 += ", {}".format(sphere[2])
            line4 += ", {}".format(sphere[3])
            line5 += ", {}".format(sphere[5])
            line6 += ", {}".format(sphere[4])
            line7 += ", -"

        line1 += "\n"
        line2 += "\n"
        line3 += "\n"
        line4 += "\n"
        line5 += "\n"
        line6 += "\n"
        line7 += "\n"

        return line1 + line2 + line3 + line4 + line5 + line6 + line7

    def get_bottleneck_line(self, md_label: str) -> str:
        """
        Produces this tunnel's data in CVS format analogous to CAVER bottlenecks.csv files, adding info on its md_label
        :param md_label: name of folder with the source MD simulation data
        :return: line in CVS format
        """

        if self.bottleneck_xyz is None or self.bottleneck_radius < 0 or not self.bottleneck_residues:
            raise RuntimeError("Data on bottlenecks has not been processed previously. Make sure that "
                               "'process_bottleneck_residues' parameter was set to 'True' during stage 2.")

        return "{},{},{},{},{},{},{},{},{},{}," \
               "{}\n".format(md_label, self.snapshot, self.caver_cluster_id, self.tunnel_id, self.throughput, self.cost,
                             self.bottleneck_xyz[0], self.bottleneck_xyz[1], self.bottleneck_xyz[2],
                             self.bottleneck_radius, ",".join(self.bottleneck_residues))

    def has_better_throughput(self, other_tunnel: Tunnel) -> bool:
        """
        Check if this tunnel has better throughput than the other one
        :param other_tunnel: other tunnel to compare with
        """

        if self.throughput > other_tunnel.throughput:
            return True
        return False

    def is_same(self, other: Tunnel) -> bool:
        """
        Test if the tunnel is same as the other tunnel
        :param other: other tunnel to compare with
        :return: are two tunnels same?
        """

        if self.snapshot != other.snapshot:
            return False

        if self.caver_cluster_id != other.caver_cluster_id:
            return False

        if self.spheres_data is None and other.spheres_data is None:
            return True

        if self.spheres_data is None or other.spheres_data is None:
            # one of them is None
            return False

        if self.spheres_data.shape != other.spheres_data.shape:
            return False

        return np.allclose(self.spheres_data, other.spheres_data, atol=1e-3)

    def get_parameters(self) -> Tuple[float, float, float, float]:
        """
        Get basic parameters of this tunnel
        :return: length, bottleneck_radius, curvature, and throughput of this tunnel
        """

        return self.length, self.bottleneck_radius, self.curvature, self.throughput

    def does_tunnel_pass_filters(self, active_filters: dict) -> bool:
        """
        Check if this tunnel adhere to the applied filters
        :param active_filters: filters to be applied (created by define_filters() function)
        """

        min_length, max_length = active_filters["length"]
        min_radius, max_radius = active_filters["radius"]
        min_curvature, max_curvature = active_filters["curvature"]

        if (min_length >= 0 and round(self.length, 6) < min_length) or \
                (min_radius >= 0 and round(self.bottleneck_radius, 6) < min_radius) or \
                (min_curvature >= 0 and round(self.curvature, 4) < min_curvature):
            return False
        if (0 <= max_length < round(self.length, 6)) or (0 <= max_radius < round(self.bottleneck_radius, 6)) \
                or (0 <= max_curvature < round(self.curvature, 6)):
            return False
        return True

    def fill_bottleneck_data(self, bottleneck_data: List[str], transform_mat: np.ndarray | None = None):
        """
        Process line from bottlenecks.csv produced by CAVER to fill in data about Tunnel bottleneck
        :param bottleneck_data: line of data from bottlenecks.csv corresponding to bottleneck residues of this tunnel
        :param transform_mat: transformation matrix to be applied on the input coordinates
        """

        self.bottleneck_xyz = np.array(bottleneck_data[0:3]).astype(float)
        if transform_mat is not None:
            self.bottleneck_xyz = np.append(self.bottleneck_xyz, np.array([1.0]))
            self.bottleneck_xyz = transform_mat.dot(self.bottleneck_xyz)[0:3]
        self.bottleneck_residues = [res.strip() for res in bottleneck_data[4:]]

    def fill_data(self, data_section: List[str]):
        """
        Processes seven lines of data from tunnel_profiles.csv produced by CAVER to create Tunnel object
        :param data_section:  seven lines of data from tunnel_profiles.csv corresponding to a single tunnel
        """

        dataset = dict()
        for line in data_section[:6]:  # do not process line with Upper limit of R overestimation
            line = line.rstrip("\n")
            array = line.split(",")
            property_name = array[12].lstrip()
            if property_name == "X":
                self.snapshot = str(array[0])
                self.caver_cluster_id = int(array[1])
                self.tunnel_id = int(array[2])
                self.throughput = float(array[3])
                self.cost = float(array[4])
                self.bottleneck_radius = float(array[5])
                self.curvature = float(array[9])
                self.length = float(array[10])
            dataset[property_name] = array[13:]

        if round(self.bottleneck_radius, 6) >= self.parameters["relevant_tunnel_min_radius"] and \
                round(self.length, 6) >= self.parameters["relevant_tunnel_min_length"] and \
                round(self.curvature, 6) <= self.parameters["relevant_tunnel_max_curvature"]:
            self.filters_passed = True

        #  here we transform the data to fit the reference structure
        try:
            xyz = np.array([dataset["X"], dataset["Y"], dataset["Z"], [1.0] * len(dataset["X"])]).astype(float)
        except ValueError:
            raise RuntimeError("\nThe input CAVER data in '{}' file from '{}' seems to be corrupted for "
                               "CAVER cluster '{}' and snapshot '{}'"
                               "\n".format(self.parameters["caver_relative_profile_file"], self.parameters["md_label"],
                                           self.caver_cluster_id, self.snapshot))

        if self.transform_mat is not None:
            xyz = self.transform_mat.dot(xyz)

        try:
            self.spheres_data = np.append(xyz[0:3, :], np.array([dataset["distance"], dataset["R"],
                                                                 dataset["length"]]).astype(float), axis=0).T
        except TypeError:
            raise RuntimeError("\nThe input CAVER data in '{}' file from '{}' seems to be corrupted for "
                               "CAVER cluster '{}' and snapshot '{}'"
                               "\n".format(self.parameters["caver_relative_profile_file"], self.parameters["md_label"],
                                           self.caver_cluster_id, self.snapshot))

        self.layer_membership = assign_layer_from_distances(einsum_dist(self.spheres_data[:, 0:3],
                                                                        np.array([0., 0., 0.])),
                                                            self.parameters["layer_thickness"])[1]

    def get_closest_sphere2coords(self, xyz: np.ndarray) ->  Tuple[float, np.ndarray] | Tuple[None, None]:
        """
        Identifies closest sphere from this tunnel to given coordinates, using spherical grid for efficiency.
        :param xyz: coordinates of point to which the distance is computed
        :return: distance to the closest sphere and the closest sphere data
        """

        if self.layer_membership is None:
            raise ValueError(f"Layer membership must be computed before usage for tunnel {self.tunnel_id} of cluster {self.caver_cluster_id}")

        distance2sp = np.linalg.norm(xyz, axis=0)
        xyz_layer = int(assign_layer_from_distances(np.array([distance2sp]), self.parameters["layer_thickness"])[1].item())
        last_layer_id = np.max(self.layer_membership)

        # now select layers with tunnel spheres adjacent to the point

        if xyz_layer > last_layer_id:
            # xyz is located beyond any layer of the tunnel -> consider the last two tunnel layers
            selector = np.logical_or.reduce((self.layer_membership == last_layer_id,
                                             self.layer_membership == last_layer_id - 1))
        else:
            if xyz_layer in self.layer_membership:
                # point share layer with some tunnel sphere, we also take layers around
                selector = np.logical_or.reduce((self.layer_membership == xyz_layer + 1,
                                                 self.layer_membership == xyz_layer,
                                                 self.layer_membership == xyz_layer - 1))
            else:
                # this is most likely error from tunnel calculation but we have to deal with it by choosing the closest
                # layers below and/or above

                warn_msg = "Could not find matching tunnel layer, this could indicate that some problems occurred " \
                           "during the tunnel computation provided as inputs for this analysis for \"tunnel {}\" from" \
                           " \"cluster {}\" in snapshot \"{}\" from simulation \"{}\"".format(self.tunnel_id,
                                                                                              self.caver_cluster_id,
                                                                                              self.snapshot,
                                                                                              self.parameters[
                                                                                                  "md_label"])
                logger.debug(warn_msg)

                closest_layers_below = self.layer_membership[self.layer_membership < xyz_layer]
                closest_layers_above = self.layer_membership[self.layer_membership > xyz_layer]

                if closest_layers_below.size > 0:
                    min_closest_layer = np.max(closest_layers_below)
                else:
                    min_closest_layer = -999  # no layers below, -999 will allow for none selected from this side

                if closest_layers_above.size > 0:
                    max_closest_layer = np.min(closest_layers_above)
                else:
                    max_closest_layer = -999  # no layers above, -999 will allow for none selected from this side

                selector = np.logical_or.reduce((self.layer_membership == min_closest_layer - 1,
                                                 self.layer_membership == min_closest_layer,
                                                 self.layer_membership == max_closest_layer,
                                                 self.layer_membership == max_closest_layer + 1))
        if self.spheres_data is None:
            raise ValueError(f"Tunnel sphere data must be provided before useage for tunnel {self.tunnel_id} of cluster {self.caver_cluster_id}")


        adjacent_spheres = self.spheres_data[np.nonzero(selector)[0]]

        if not adjacent_spheres.size > 0:
            error_msg = "This should not happen but no adjacent sphere was found for \"tunnel {}\" from " \
                        "\"cluster {}\" in snapshot \"{}\" from simulation \"{}\"".format(self.tunnel_id,
                                                                                          self.caver_cluster_id,
                                                                                          self.snapshot,
                                                                                          self.parameters["md_label"])
            logger.error(error_msg)
            return None, None

        # find the closest sphere surface to xyz among the adjacent ones
        distances = einsum_dist(adjacent_spheres[:, 0:3], xyz)
        distances2surface = distances - adjacent_spheres[:, 4]

        min_distance = np.min(distances2surface)
        closest_sphere = adjacent_spheres[distances2surface == min_distance][0]

        return min_distance, closest_sphere

    def create_layered_tunnel(self, entity_label: str = "") -> LayeredRepresentationOfTunnels:
        """
        Loads tunnel spheres(points) into LayeredRepresentation and forwards to the cluster layering
        :param entity_label: name of the entity (a cluster to which this tunnel belong) to be layered
        :return: raw, unprocessed LayeredRepresentation of the tunnel
        """

        tmp_repre = LayeredRepresentationOfTunnels(self.parameters, entity_label)
        tmp_repre.load_points(self)  # this passes spheres_data from this tunnel to the function

        return tmp_repre

    def get_visualization_cgo(self) -> List[float]:
        """
        Converts tunnel points to Pymol compiled graphics object(CGO) for visualization
        :return: CGO of tunnel points
        """
        if self.spheres_data is None:
            raise ValueError(f"Tunnel sphere data must be provided before useage for tunnel {self.tunnel_id} of cluster {self.caver_cluster_id}")

        return utils.convert_coords2cgo(self.spheres_data[:, 0:3], color_id=self.caver_cluster_id - 1)

    def get_pdb_file_format(self) -> List[str]:
        """
        Saves tunnel points as PDB file according to CAVER format
        :return
        """
        if self.spheres_data is None:
            raise ValueError(f"Tunnel sphere data must be provided before useage for tunnel {self.tunnel_id} of cluster {self.caver_cluster_id}")

        out_lines = list()
        sphere = self.spheres_data[0]
        out_lines.append(str(VizAtom([1, "H", "FIL", 1, sphere[0], sphere[1], sphere[2], sphere[4]], use_hetatm=False)))
        for i, sphere in enumerate(self.spheres_data[1:]):
            out_lines.append(str(VizAtom([i + 2, "H", "FIL", 1, sphere[0], sphere[1], sphere[2], sphere[4]],
                                         use_hetatm=False)))
            out_lines.append("CONECT{:5d}{:5d}\n".format(i + 1, i + 2))

        return out_lines


class TunnelCluster:
    def __init__(self, cluster_id: int, parameters: dict,  transform_mat: np.ndarray, starting_point_coords: np.ndarray):
        """
        Class for processing of tunnel clusters from CAVER, containing tunnels
        :param cluster_id: ID of the cluster from CAVER
        :param parameters: job configuration parameters
        :param transform_mat: transformation matrix to be applied on the nodes coordinates
        :param starting_point_coords: coordinates of the tunnel starting point
        """

        self.tunnels: Dict[int, Tunnel] = dict()
        self.cluster_id = int(cluster_id)
        self.parameters = parameters
        self.transform_mat = transform_mat
        self.starting_point_coords = starting_point_coords
        self.entity_label = "Cluster_{}".format(self.cluster_id)
        self.entity_pymol_label = "cls_{:03d}".format(self.cluster_id)
        self.vis_file = self.entity_pymol_label + "_cgo.dump.gz"

    def __str__(self):
        return "MD: {}, ID:{:4d}, num tunnels: {:d}".format(self.parameters["md_label"], self.cluster_id,
                                                            self.count_tunnels())

    def get_characteristics(self):
        """
        Get information on the average throughput and the number of tunnels in this cluster
        :return: the average throughput and the number of tunnels
        """
        throughput = 0
        for tunnel in self.tunnels.values():
            throughput += tunnel.throughput

        return throughput / self.count_tunnels(), self.count_tunnels()

    def is_same(self, other: TunnelCluster) -> bool:
        """
        Test if the cluser is same as the other cluster
        :param other: other cluster to compare with
        :return: are two clusters same?
        """

        if set(self.tunnels.keys()) != set(other.tunnels.keys()):
            return False

        for tunnel_id in self.tunnels.keys():
            if not self.tunnels[tunnel_id].is_same(other.tunnels[tunnel_id]):
                return False

        return True

    def count_tunnels(self) -> int:
        """
        Counts number of tunnels in this cluster
        :return: number of tunnels
        """

        return len(self.tunnels.keys())

    def get_closest_tunnel_sphere_in_frame2coords(self, xyz: np.ndarray, snap_id: int) -> Tuple[float, np.ndarray] | Tuple[None, None]:
        """
        Identifies closest sphere from a tunnel in the investigated snapshot to given coordinates
        :param xyz: coordinates of point to which the distance is computed
        :param snap_id: ID of investigated snapshot
        :return: distance to the closest sphere and the closest sphere data, or None if no tunnel exists in the snapshot
        """

        if snap_id in self.tunnels.keys():
            return self.tunnels[snap_id].get_closest_sphere2coords(xyz)
        else:
            return None, None

    def get_subcluster(self, snap_ids: List[int] | None = None, active_filters: dict | None = None) -> TunnelCluster:
        """
        Returns subcluster with tunnels from the frames and/or all tunnels in cluster fulfilling the active filters
        :param snap_ids: list of snapshot IDs to create the subcluster from
        :param active_filters: filters to be applied (created by define_filters() function)
        :return: subcluster with tunnels from given frames and/or fulfilling filters
        """

        if snap_ids is None and active_filters is None:
            raise RuntimeError("Either list of analyzed snapshot IDs or filters to apply have to be specified here.")

        subcluster = TunnelCluster(self.cluster_id, self.parameters, self.transform_mat, self.starting_point_coords)

        # when no explicit snapshot selection is given, take every tunnel actually present in the cluster
        # (sorted to keep the previous ascending iteration order); this is both correct under sparse/
        # strided CAVER snapshot IDs and avoids scanning a dense 1..snapshots_per_simulation range that
        # is mostly absent
        if snap_ids is None:
            snap_ids = sorted(self.tunnels.keys())

        for snap_id in snap_ids:
            if snap_id in self.tunnels.keys():
                subcluster.add_tunnel(self.tunnels[snap_id])

        if active_filters is not None:
            tunnels2remove = list()
            for snap_id, tunnel in subcluster.tunnels.items():
                if not tunnel.does_tunnel_pass_filters(active_filters):
                    tunnels2remove.append(snap_id)

            for snap_id in tunnels2remove:
                subcluster.remove_tunnel(snap_id)

        return subcluster

    def get_property(self, property_name: str, active_filters: dict) -> np.ndarray:
        """
        Returns array containing all values of given property for all tunnels in cluster that fulfill active filters
        :param property_name: name of property to extract
        :param active_filters: filters to be applied (created by define_filters() function)
        :return: array of tunnel property values adhering to filters
        """

        array = list()
        for tunnel in self.tunnels.values():
            if tunnel.does_tunnel_pass_filters(active_filters):
                array.append(getattr(tunnel, property_name))

        return np.array(array)

    def add_tunnel(self, tunnel: Tunnel):
        """
        Adds a tunnel to this cluster
        :param tunnel: Tunnel object to add
        """

        snap_id = tunnel.get_snapshot_id(self.parameters["snapshot_id_position"], self.parameters["snapshot_delimiter"])
        if snap_id in self.tunnels.keys():
            raise KeyError("Tunnel with this snapshot id {} already exists in this cluster {}!".format(snap_id,
                                                                                                       str(self)))
        self.tunnels[snap_id] = tunnel

    def remove_tunnel(self, snap_id: int):
        """
        Remove tunnel with a given snapshot ID from this cluster
        :param snap_id: snapshot ID
        """

        del self.tunnels[snap_id]

    def count_valid_tunnels(self) -> int:
        """
        Counts number of tunnels that passed the initial filters for creation of cluster representative
        :return: number of valid tunnels
        """

        num_tunnels = 0
        for tunnel in self.tunnels.values():
            if tunnel.filters_passed:
                num_tunnels += 1

        return num_tunnels

    def create_layered_cluster(self) -> Tuple[int, str, LayeredPathSet]:
        """
        Combines LayeredRepresentation of all valid tunnels within the cluster, executes the actual layering, and
        identifies the representative set of Layered paths
        :return: cluster ID, md_label and a set of layered paths representing this cluster
        """

        with threadpool_limits(limits=1, user_api=None):
            tmp_repre = LayeredRepresentationOfTunnels(self.parameters, self.entity_label)
            for tunnel in self.tunnels.values():
                if tunnel.filters_passed:
                    tmp_repre += tunnel.create_layered_tunnel(self.entity_label)

            tmp_repre.split_points_to_layers()
            layered_pathset = tmp_repre.find_representative_paths(self.transform_mat, self.starting_point_coords,
                                                                  self.parameters["visualize_layered_clusters"])
            layered_pathset.characteristics = self.get_characteristics()
        return self.cluster_id, self.parameters["md_label"], layered_pathset

    def save_cgo(self, out_path: str):
        """
        Dump all tunnel as single Pymol compiled graphics object(CGO) to gzipped files for visualization of this cluster
        :param out_path: folder where the CGO should be dumped to
        """

        cluster_cgos = list()
        for tunnel in self.tunnels.values():
            cluster_cgos.extend(tunnel.get_visualization_cgo())

        with gzip.open(os.path.join(out_path, self.vis_file), "wb") as out_stream:
            pickle.dump(cluster_cgos, out_stream, self.parameters["pickle_protocol"])

    def save_pdb_files(self, snap_ids: List[int], out_file: str) -> bool:
        """
        Writes tunnels with specified snapshot IDs to MULTIMODEL PDB file
        :param snap_ids: list of snapshot IDs to save
        :param out_file: file to save to
        :return: if saved successfully
        """

        if set(snap_ids).intersection(self.tunnels.keys()):
            with gzip.open(out_file, "wt") as out_stream:
                for snap_id in snap_ids:  # we loop over snap_ids to guarantee match between models
                    out_stream.write("MODEL {:5d}\n".format(snap_id))
                    if snap_id in self.tunnels.keys():
                        out_stream.writelines(self.tunnels[snap_id].get_pdb_file_format())
                    out_stream.write("ENDMDL\n")
            return True
        else:
            logger.debug("No tunnels available between snapshots " + str(snap_ids[0]) + " and " + str(snap_ids[-1])
                         + " of " + self.entity_label + " from " + self.parameters["md_label"]
                         + " to save their visualization")
            return False


class TunnelNetwork(Network):
    def __init__(self, parameters: dict, md_label: str):
        """
        Class for processing of CAVER results for tunnels, contains TunnelCluster objects
        :param parameters: job configuration parameters
        :param md_label: name of folder with the source MD simulation data
        """

        Network.__init__(self, parameters, md_label)
        self.entity_pymol_abbreviation = "cls"
        self.transformed_pdb_file_name = self.md_label + "_C_trans_rot.pdb"

        # Stage 4's SLURM launcher orders tunnel clusters by importance using
        # (avg_throughput, num_tunnels); the sidecar lets it skip the full LayeredPathSet load.
        self._layered_summary_attr = "characteristics"
        self._layered_summary_suffix = ".characteristics"

        # Stage 3's SLURM tunnel-layering launcher enumerates (md_label, cluster_id) pairs by
        # the per-cluster valid-tunnel count; this sidecar lets it skip the full orig_network
        # load (see get_clusters4layering_ids).
        self._orig_summary_suffix = ".cluster_sizes"

        # input paths - the raw CAVER files (PDB, tunnel profiles, bottlenecks, v_origins) are read
        # only when the network actually (re)parses tunnels (stage 2) or renders the structure (the
        # optional visualisations). They are resolved lazily on first access (see the pdb_file,
        # tunnel_profile_file, bottleneck_file and starting_point_coords properties below), so a
        # load-only rebuild from the layered/orig dump - clustering at stage 5, filtering, a resume
        # into a later stage - does not require the original CAVER folders to still be present.
        self.caver_root_folder = os.path.join(self.parameters["caver_results_path"], self.md_label)
        self._tunnel_profile_file: str | None = None
        self._bottleneck_file: str | None = None
        self._bottleneck_resolved: bool = False

        # output paths
        self.transformation_folder = os.path.join(self.parameters["transformation_folder"],
                                                  self.parameters["caver_foldername"])
        self.orig_viz_path = os.path.join(self.parameters["orig_caver_vis_path"], self.md_label)
        self.orig_dump_file = os.path.join(self.parameters["orig_caver_network_data_path"],
                                           self.md_label + "_caver.dump")

        self.layered_viz_path = os.path.join(self.parameters["layered_caver_vis_path"], self.md_label)
        self.layered_dump_file = os.path.join(self.parameters["layered_caver_network_data_path"],
                                              self.md_label + "_layered_paths.dump")

        # initiation methods
        self._load_transformations()

    @property
    def pdb_file(self) -> str | None:
        """Path to the CAVER protein PDB, resolved on first access (used by structure visualisation
        in stages 1-2 and the layered-cluster visualisation). None until then so a load-only rebuild
        never touches the raw CAVER folder."""
        if self._pdb_file is None:
            self._pdb_file = utils.get_filepath(self.caver_root_folder,
                                                self.parameters["caver_relative_pdb_file"])
        return self._pdb_file

    @pdb_file.setter
    def pdb_file(self, value: str | None):
        self._pdb_file = value

    @property
    def tunnel_profile_file(self) -> str:
        """Path to the CAVER tunnel_profiles.csv, resolved on first access (used by read_tunnels_data
        at stage 2)."""
        if self._tunnel_profile_file is None:
            self._tunnel_profile_file = utils.get_filepath(self.caver_root_folder,
                                                           self.parameters["caver_relative_profile_file"])
        return self._tunnel_profile_file

    @tunnel_profile_file.setter
    def tunnel_profile_file(self, value: str | None):
        self._tunnel_profile_file = value

    @property
    def bottleneck_file(self) -> str | None:
        """Path to the CAVER bottlenecks.csv when process_bottleneck_residues is set, else None;
        resolved on first access (used by _read_bottleneck_data at stage 2)."""
        if not self._bottleneck_resolved:
            if self.parameters["process_bottleneck_residues"]:
                self._bottleneck_file = utils.get_filepath(
                    self.caver_root_folder, self.parameters["caver_relative_bottleneck_file"])
            else:
                self._bottleneck_file = None
            self._bottleneck_resolved = True
        return self._bottleneck_file

    @bottleneck_file.setter
    def bottleneck_file(self, value: str | None):
        self._bottleneck_file = value
        self._bottleneck_resolved = True

    @property
    def starting_point_coords(self) -> np.ndarray | None:
        """Starting-point coordinates read from the CAVER v_origins.pdb, resolved on first access
        (used by read_tunnels_data at stage 2). None until then so a load-only rebuild never reads
        the raw CAVER folder."""
        if self._starting_point_coords is None:
            origin_file = utils.get_filepath(self.caver_root_folder,
                                             self.parameters["caver_relative_origin_file"])
            self._starting_point_coords = average_starting_point(origin_file)[0][0:3, :].reshape(1, 3)
        return self._starting_point_coords

    @starting_point_coords.setter
    def starting_point_coords(self, value: "np.ndarray | None"):
        self._starting_point_coords = value

    def read_tunnels_data(self):
        """
        Process tunnel_profile_file from CAVER to create TunnelClusters and their Tunnels
        """

        if self.transform_mat is None:
            raise ValueError("Variable 'self.transform_mat' is not specified")

        if self.starting_point_coords is None:
            raise ValueError("Variable 'self.starting_point_coords' is not specified")

        utils.test_file(self.tunnel_profile_file)
        with open(self.tunnel_profile_file) as in_stream:
            data = in_stream.readlines()

            for i in range(1, len(data), 7):
                current_id = int(data[i].rstrip("\n").split(",")[1])
                if not self.cluster_exists(current_id):
                    self.orig_entities.append(TunnelCluster(current_id, self.parameters, self.transform_mat,
                                                            self.starting_point_coords))

                tmp_tunnel = Tunnel(self.parameters, self.transform_mat)
                tmp_tunnel.fill_data(data[i:i + 7])
                cluster = self.orig_entities[current_id - 1]
                assert isinstance(cluster, TunnelCluster), f"Expected TunnelCluster but got {type(cluster).__name__}"
                cluster.add_tunnel(tmp_tunnel)

        self._validate_snapshot_sampling()

        if self.parameters["process_bottleneck_residues"]:
            self._read_bottleneck_data()

    def _validate_snapshot_sampling(self):
        """
        Cross-check the configured snapshot sampling against the CAVER snapshot IDs actually parsed.
        Runs for free at parse time - it only inspects IDs already read - and fails fast on a clear
        misconfiguration (wrong stride, or snapshots_per_simulation set to the trajectory length).
        """

        observed_ids = set()
        for cluster in self.orig_entities:
            if isinstance(cluster, TunnelCluster):
                observed_ids.update(cluster.tunnels.keys())
        if not observed_ids:
            return

        # cache the detected labelling on the parameters so it is derived once here at parse time and then
        # carried along (to workers and across a resume) instead of being re-derived downstream
        snapshot_map = utils.SnapshotFrameMap.resolve_into(self.parameters, observed_ids)
        num = self.parameters["snapshots_per_simulation"]
        max_id = max(observed_ids)

        if snapshot_map.by_frame:
            config_stride = self.parameters.get("caver_snapshot_stride") or 1
            if config_stride > 1 and config_stride != snapshot_map.id_stride:
                raise RuntimeError("CAVER snapshot IDs in {} are frame-numbered with a stride of {} "
                                   "(every {}th frame), but 'caver_snapshot_stride' is set to {}. "
                                   "Set 'caver_snapshot_stride' to {} or remove "
                                   "it.".format(self.md_label, snapshot_map.id_stride,
                                                snapshot_map.id_stride, config_stride,
                                                snapshot_map.id_stride))
        elif max_id * 2 < num:
            # losing more than half of a sequentially-numbered run to tunnel-less frames is implausible;
            # the usual cause is snapshots_per_simulation set to the trajectory length instead of the
            # CAVER snapshot count
            raise RuntimeError("Highest CAVER snapshot ID in {} is {}, far below 'snapshots_per_simulation' "
                               "({}). 'snapshots_per_simulation' must be the number of CAVER snapshots; if "
                               "CAVER analyzed a sparse trajectory, also set 'caver_snapshot_stride' "
                               "accordingly.".format(self.md_label, max_id, num))
        elif max_id < num:
            # a sequential run can legitimately end tunnel-less (e.g. the structure compacts), so a modest
            # shortfall is only a hint
            logger.warning("Highest CAVER snapshot ID in {} is {}, below 'snapshots_per_simulation' ({}). "
                           "If this is unexpected, check 'snapshots_per_simulation' and "
                           "'caver_snapshot_stride'.".format(self.md_label, max_id, num))

    def _read_reweighting_data(self):
        """
        TODO
        """
        pass

    def _read_bottleneck_data(self):
        """
        Process bottlenecks.csv from CAVER to fill info about Tunnels' bottlenecks
        """

        if self.bottleneck_file is None:
            raise ValueError("Variable 'self.bottleneck_file' is not specified")

        utils.test_file(self.bottleneck_file)
        with open(self.bottleneck_file) as in_stream:
            data = in_stream.readlines()

            for line in data[2:]:
                array = line.rstrip("\n").split(",")
                snapshot = array[0]
                snap_id = int(snapshot.split(self.parameters["snapshot_delimiter"])[self.parameters["snapshot_id_position"]])
                cluster_id = int(array[1])
                bottleneck_data = array[5:]
                if self.cluster_exists(cluster_id):
                    cluster = self.get_cluster(cluster_id)
                    if snap_id in cluster.tunnels.keys():
                        cluster.tunnels[snap_id].fill_bottleneck_data(bottleneck_data, self.transform_mat)
                    else:
                        raise RuntimeError("Bottleneck data - tunnel mismatch")
                else:
                    raise RuntimeError("Bottleneck data - tunnel mismatch")

    def get_cluster(self, cluster_id: int) -> TunnelCluster:
        """
        Retrieve cluster with given ID
        :param cluster_id: ID of retrieved cluster
        :return: the cluster with given ID
        """

        if cluster_id == 0:
            raise ValueError("Cluster with required ID (={}) does not exist in the network".format(cluster_id))
        try:
            cluster = self.orig_entities[cluster_id - 1]
        except IndexError:
            raise ValueError("Cluster with required ID (={}) does not exist in the network".format(cluster_id))

        assert isinstance(cluster, TunnelCluster), f"Expected TunnelCluster but got {type(cluster).__name__}"
        return cluster

    def cluster_exists(self, query_id: int) -> bool:
        """
        Verifies if cluster with given id exists
        :param query_id: ID of evaluated cluster
        """

        if query_id == 0:
            raise ValueError("Cluster with required ID (={}) does not exist in the network".format(query_id))
        if len(self.orig_entities) >= query_id:
            return True
        else:
            return False

    def get_clusters4layering(self) -> List[TunnelCluster]:
        """
        Get cluster in this tunnel network for layering
        :return: list of clusters for processing
        """

        clusters = list()

        for cls_id, cluster in enumerate(reversed(self.orig_entities)):  # reversed to start processing smaller clusters
            assert isinstance(cluster, TunnelCluster), f"Expected TunnelCluster but got {type(cluster).__name__}"
            if cluster.count_valid_tunnels() >= self.parameters["relevant_tunnel_cluster_min_size"]:
                clusters.append(cluster)
            else:
                logger.debug("Cluster {} of {} has no valid tunnels to layer".format(cluster.cluster_id, self.md_label))

        return clusters

    # parameters whose stage-2 values are baked into Tunnel.filters_passed (and therefore into
    # count_valid_tunnels); the sidecar records them so it can self-invalidate if these are
    # changed before stage 3 - today they have no stage-3 effect (the flag is frozen at stage 2),
    # but storing them protects the enumeration should that invariant ever change.
    _ORIG_SUMMARY_FILTER_KEYS = ("relevant_tunnel_min_radius", "relevant_tunnel_min_length",
                                 "relevant_tunnel_max_curvature")

    def _compute_orig_summary(self) -> dict:
        """
        Build the size sidecar payload: the per-cluster valid-tunnel counts in orig_entities
        order (so reversing reproduces get_clusters4layering's processing order), plus the
        per-tunnel filter parameters baked into those counts for self-validation on read.
        """

        counts = list()
        for cluster in self.orig_entities:
            assert isinstance(cluster, TunnelCluster), \
                f"Expected TunnelCluster but got {type(cluster).__name__}"
            counts.append((cluster.cluster_id, cluster.count_valid_tunnels()))
        return {
            "filter_params": tuple(self.parameters[k] for k in self._ORIG_SUMMARY_FILTER_KEYS),
            "counts": counts,
        }

    def _orig_summary_is_current(self, summary: dict) -> bool:
        """
        Whether a loaded size-sidecar payload was built with the per-tunnel filter parameters
        currently in effect. A mismatch means the sidecar's counts could no longer agree with a
        fresh get_clusters4layering() and must be regenerated from the loaded network.
        """

        current = tuple(self.parameters[k] for k in self._ORIG_SUMMARY_FILTER_KEYS)
        return summary.get("filter_params") == current

    def get_clusters4layering_ids(self) -> List[int]:
        """
        Enumerate the cluster_ids that get_clusters4layering() would return, reading the size
        sidecar to avoid the heavy orig_network load when possible. Used by the stage-3 SLURM
        launcher to build the (md_label, cluster_id) work enumeration; the returned order is
        identical to get_clusters4layering() so the SLURM and local backends emit byte-identical
        items.

        Falls back to a full load_orig_network() + get_clusters4layering() when the sidecar is
        missing (legacy checkpoint) or stale (the baked-in per-tunnel filter parameters no longer
        match the current ones), rewriting the sidecar so subsequent runs are fast, then drops
        orig_entities so the caller does not retain the heavy objects.
        :return: cluster_ids in get_clusters4layering() order
        """

        min_size = self.parameters["relevant_tunnel_cluster_min_size"]
        sidecar_path = self._orig_summary_path()
        if sidecar_path is not None and os.path.exists(sidecar_path):
            with open(sidecar_path, "rb") as in_stream:
                summary = pickle.load(in_stream)
            if self._orig_summary_is_current(summary):
                # mirror get_clusters4layering(): reversed orig order, keep clusters meeting the
                # (live) cluster-size threshold
                return [cls_id for cls_id, n_valid in reversed(summary["counts"])
                        if n_valid >= min_size]
            logger.debug("Orig size sidecar '{}' is stale (filter params changed); reloading "
                         "the full network.".format(sidecar_path))

        # Fallback: legacy checkpoint without sidecar, or stale sidecar. Load the heavy pickle,
        # rewrite the sidecar for subsequent runs, then drop orig_entities.
        self.load_orig_network()
        ids = [cluster.cluster_id for cluster in self.get_clusters4layering()]
        try:
            self._write_orig_summary_sidecar()
        except OSError as exc:
            logger.debug("Could not write orig size sidecar '{}': {}".format(sidecar_path, exc))
        self.orig_entities = list()
        return ids


# ---- AquaDuct related classes ----
class AquaductNetwork(Network):
    def __init__(self, parameters: dict, md_label: str, load_only: bool = False, root_path: str | None = None):
        """
        Class for processing of AQUA-DUCT results, containing AquaductPath objects
        :param parameters: job configuration parameters
        :param md_label: name of folder with the source MD simulation data
        :param load_only: object will be used to load already processed network, no need for pdb_file
        :param root_path: explicit root path containing this md_label; if None, uses first entry in
                          parameters["aquaduct_results_path"] that contains the md_label folder
        """

        Network.__init__(self, parameters, md_label)
        self.entity_pymol_abbreviation = "evt_"  # default prefix, overridden per-entity by visualization_prefix
        self.transformed_pdb_file_name = self.md_label + "_A_trans_rot.pdb"
        self.protein_pdb_filename = self.parameters["aquaduct_results_pdb_filename"]
        # number of source-trajectory frames AQUA-DUCT analyzed (parsed from the summary's
        # "Frames window: start:end step" line); the authoritative event frame-domain length used to
        # validate caver_snapshot_stride. None until read_raw_paths_data parses the summary.
        self.event_domain_length: int | None = None

        # Stage 9's SLURM launcher only needs the (md_label, event_label, traced_event)
        # key triples to enumerate items in submission order; the sidecar lets it skip the
        # full LayeredPathSet load on runs with millions of events.
        self._layered_summary_attr = "traced_event"
        self._layered_summary_suffix = ".traced_event"

        # Stage 8's SLURM launcher only needs the per-event entity_labels (in
        # get_events4layering order) to enumerate the (md_label, event_label) work; this orig
        # sidecar lets it skip the heavy full orig_network load of every md on a single node,
        # mirroring the tunnel side's .cluster_sizes sidecar consumed by
        # get_clusters4layering_ids(). save_orig_network() emits it automatically at stage 7.
        self._orig_summary_suffix = ".event_ids"

        # input paths - the raw AQUA-DUCT files (results tarball, summary) and the source-folder
        # location are read only when the network (re)parses events or extracts the protein PDB
        # (stage 7). They are resolved lazily on first access (see the tar_file and summary_file
        # properties below, which detect the root folder on demand), so a load-only rebuild from the
        # orig/layered dump - event layering at stage 8, assignment at stage 9, filtering at stage 10,
        # or a resume into any of them - does not require the original AQUA-DUCT folders to still exist.
        self._root_path = root_path
        self._root_folder: str | None = None
        self._tar_file: str | None = None
        self._summary_file: str | None = None
        self.fd = None

        # output paths
        self.transformation_folder = os.path.join(self.parameters["transformation_folder"],
                                                  self.parameters["aquaduct_foldername"])
        self.orig_viz_path = os.path.join(self.parameters["orig_aquaduct_vis_path"], self.md_label)
        self.orig_dump_file = os.path.join(self.parameters["orig_aquaduct_network_data_path"],
                                           self.md_label + "_aqua.dump")
        self.layered_viz_path = os.path.join(self.parameters["layered_aquaduct_vis_path"], self.md_label)
        self.layered_dump_file = os.path.join(self.parameters["layered_aquaduct_network_data_path"],
                                              self.md_label + "_layered_paths.dump")

        # initiation methods
        self._load_transformations()

        if not load_only:
            self.get_pdb_file()

    def _resolve_aquaduct_root_folder(self) -> str:
        """Locate the source AQUA-DUCT folder for this md_label, detecting it among
        aquaduct_results_path on demand when no explicit root_path was given. Called only when the
        raw tarball/summary are actually accessed (stage-7 parsing), so a load-only rebuild never
        touches the source folders."""
        if self._root_folder is None:
            root_path = self._root_path
            if root_path is None:
                root_path = next(p for p in self.parameters["aquaduct_results_path"]
                                 if os.path.isdir(os.path.join(p, self.md_label)))
            assert isinstance(root_path, str), "Must be string given the above if block"
            self._root_folder = os.path.join(root_path, self.md_label)
        return self._root_folder

    @property
    def tar_file(self) -> str:
        """Path to the AQUA-DUCT results tarball, resolved on first access (stage-7 parsing /
        PDB extraction)."""
        if self._tar_file is None:
            self._tar_file = utils.get_filepath(self._resolve_aquaduct_root_folder(),
                                                self.parameters["aquaduct_results_relative_tarfile"])
        return self._tar_file

    @tar_file.setter
    def tar_file(self, value: str | None):
        self._tar_file = value

    @property
    def summary_file(self) -> str:
        """Path to the AQUA-DUCT summary file, resolved on first access (stage-7 parsing)."""
        if self._summary_file is None:
            self._summary_file = utils.get_filepath(self._resolve_aquaduct_root_folder(),
                                                    self.parameters["aquaduct_results_relative_summaryfile"])
        return self._summary_file

    @summary_file.setter
    def summary_file(self, value: str | None):
        self._summary_file = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - cleanup temporary PDB file and file descriptor
        """
        if self.fd is not None:
            os.close(self.fd)
        if self.pdb_file is not None:
            os.remove(self.pdb_file)

    def clean_tempfile(self):
        """
        Removes tempfile
        """

        if self.pdb_file is None:
            raise ValueError("Variable 'self.pdb_file' is not specified")

        if self.fd is None:
            raise ValueError("Variable 'self.fd' is not specified")

        utils.test_file(self.pdb_file)
        os.close(self.fd)
        os.remove(self.pdb_file)

    def get_pdb_file(self):
        """
        Create temporary PDB file with protein structure from AQUA-DUCT tarfile
        """

        from tempfile import mkstemp
        self.fd, self.pdb_file = mkstemp(suffix=".pdb")
        with open(self.pdb_file, "wb") as out_pdb, tarfile.open(self.tar_file, "r:gz") as tar_handle:
            extracted_file = tar_handle.extractfile(self.protein_pdb_filename)
            if extracted_file is None:
                raise ValueError(f"File '{self.protein_pdb_filename}' not found in tarfile '{self.tar_file}'")
            out_pdb.write(extracted_file.read())

    @staticmethod
    def _path_sort(filename: str) -> int:
        """
        For raw_paths dump files get value based on their AQUA-DUCT ID embedded in their filename, for other files 0
        :param filename: path filename to evaluate
        :return: numeric value of filename
        """

        if search(r'^raw_paths_\d+\.dump', filename):
            return int(filename.split("_")[2].split(".")[0])
        return 0


    @staticmethod
    def _process_single_raw_path(path_label: str, parameters: dict, traced_residue: Tuple[str, int, Tuple[int, int],
                                                                                          Tuple[int, int]],
                                 transform_mat: np.ndarray, md_label: str, cgo_obj: list) -> AquaductPath:
        """
        Creation of AquaductPath object from a single raw_path for parallel processing
        :param path_label: the name of the path derived from AQUA-DUCT raw paths names
        :param parameters: job configuration parameters
        :param traced_residue: tuple containing resname & resid of ligand responsible for this path,
                                and beginning and last frames for entry and release event
        :param transform_mat: transformation matrix to be applied on the input coordinates
        :param md_label: name of folder with the source MD simulation data
        :param cgo_obj: AQUA-DUCT Compiled Graphics Object containing info on a raw path
        :return: processed AquaductPath object
        """

        tmp_path = AquaductPath(path_label, parameters, traced_residue, transform_mat, md_label)
        tmp_path.process_path(cgo_obj)

        return tmp_path

    def read_raw_paths_data(self, parallel_processing: bool = True):
        """
        Parallel processing of all raw_paths present in AQUA-DUCT tarfile to create corresponding AquaductPath object
        :param parallel_processing: if we process the raw_paths in parallel
        """

        if self.transform_mat is None:
            raise ValueError("Variable 'self.transform_mat' is not specified")

        # read residue info from summary text file from AquaDuct
        with open(self.summary_file) as sum_stream:
            summary_text = sum_stream.readlines()
        self.event_domain_length = utils.parse_aquaduct_frames_window(summary_text)
        try:
            start_line = summary_text.index("List of separate paths and properties\n") + 4
        except ValueError:
            raise RuntimeError("File {} seems to be incomplete, cannot parse information "
                               "on separate paths!".format(self.summary_file))

        header = summary_text[start_line - 2].strip().split()

        traced_residues = list()
        for line in summary_text[start_line:]:
            line = line.strip()
            if "-" * 80 in line:
                break
            line_fields = line.split()
            resid = int((line_fields[header.index("ID")]).split(":")[1])
            resname = line_fields[header.index("RES")]
            begin_frame = int(line_fields[header.index("BeginF")])
            inp_frame = int(line_fields[header.index("InpF")])
            end_frame = int(line_fields[header.index("EndF")])
            out_frame = int(line_fields[header.index("OutF")])

            entry_frames = (begin_frame, begin_frame + inp_frame)
            release_frames = (end_frame - out_frame, end_frame)
            traced_residues.append((resname, resid, entry_frames, release_frames))

        # read raw paths from tarfile
        resname_filter = self.parameters.get("aquaduct_traced_residues_filter")
        items2process = list()
        tar_handle = tarfile.open(self.tar_file, "r:gz")
        for filename in sorted(tar_handle.getnames(), key=self._path_sort):
            if search(r'^raw_paths_\d+\.dump', filename):
                path_label = filename.split(".")[0]
                path_id = int(path_label.split("_")[2])
                traced_residue = traced_residues[path_id - 1]
                if resname_filter is not None and traced_residue[0].upper() not in resname_filter:
                    continue
                extracted_file = tar_handle.extractfile(filename)
                if extracted_file is None:
                    raise ValueError(f"File '{filename}' not found in tarfile '{self.tar_file}'")
                cgo_object = pickle.load(extracted_file)
                items2process.append((path_label, traced_residue, cgo_object))
        tar_handle.close()

        if parallel_processing:
            num_cpus = self.parameters["num_cpus"]
            with get_context("spawn").Pool(processes=num_cpus,
                                           initializer=utils.cap_blas_threads_for_worker,
                                           initargs=(num_cpus, num_cpus)) as pool:
                tasks = [(path_label, self.parameters, traced_residue,
                          self.transform_mat, self.md_label, cgo_object)
                         for path_label, traced_residue, cgo_object in items2process]
                timeout = self.parameters["worker_task_timeout_s"]
                imap_iter = pool.imap_unordered(process_raw_path_worker, tasks)
                # The worker returns an AquaductPath whose .path_label is the unique identifier;
                # surface it on timeout to help operators pinpoint where the stall happened.
                results_iter = utils.iter_imap_results(
                    imap_iter, timeout=timeout, pool=pool,
                    label="raw-path[md={}]".format(self.md_label),
                    extract_task_label=lambda r: r.path_label)
                # imap_unordered yields in completion order, but self.orig_entities is appended
                # to a list whose order is observable downstream (e.g. it feeds visualization
                # scripts that reference 'raw_paths_<N>_cgo.dump.gz' in iteration order). Buffer
                # results by path_label and append in submission order so the produced state is
                # stable regardless of worker scheduling.
                buffered_paths: Dict[str, "AquaductPath"] = dict()
                for tmp_path in results_iter:
                    buffered_paths[tmp_path.path_label] = tmp_path

                for path_label, _, _ in items2process:
                    tmp_path = buffered_paths[path_label]
                    if tmp_path.has_transport_event():
                        self.orig_entities.append(tmp_path)
        else:
            for path_label, traced_residue, cgo_object in items2process:
                tmp_path = self._process_single_raw_path(path_label, self.parameters, traced_residue,
                                                         self.transform_mat, self.md_label, cgo_object)
                if tmp_path.has_transport_event():
                    self.orig_entities.append(tmp_path)

    def get_events4layering(self) -> List[TransportEvent]:
        """
        Get events in this aqua network for layering
        :return: list of events for processing
        """

        events = list()
        for path in self.orig_entities:
            assert isinstance(path, AquaductPath), f"Expected AquaductPath but got {type(path).__name__}"
            events += path.get_events4layering()

        return events

    def _compute_orig_summary(self) -> dict:
        """
        Build the orig event-id sidecar payload: the entity_labels that get_events4layering()
        would yield, in the same traversal order, so get_events4layering_ids() can reproduce
        the stage-8 enumeration without loading the full orig_network. has_transition() (the
        only filter get_events4layering applies) depends solely on the event type baked into
        the orig_network at stage 7, so no parameter self-invalidation is needed here.
        """

        return {"event_ids": [str(event.entity_label) for event in self.get_events4layering()]}

    def get_events4layering_ids(self) -> List[str]:
        """
        Enumerate the event entity_labels that get_events4layering() would return, reading the
        orig event-id sidecar to avoid the heavy full orig_network load when possible. Used by
        the stage-8 SLURM launcher to build the (md_label, event_label) work enumeration; the
        returned order is identical to get_events4layering() so the SLURM and local backends
        emit byte-identical items.

        Falls back to a full load_orig_network() + get_events4layering() when the sidecar is
        missing (legacy checkpoint) or malformed, rewriting the sidecar so subsequent runs are
        fast, then drops orig_entities so the caller does not retain the heavy objects.
        :return: event entity_labels in get_events4layering() order
        """

        sidecar_path = self._orig_summary_path()
        if sidecar_path is not None and os.path.exists(sidecar_path):
            with open(sidecar_path, "rb") as in_stream:
                summary = pickle.load(in_stream)
            if isinstance(summary, dict) and "event_ids" in summary:
                return [str(eid) for eid in summary["event_ids"]]
            logger.debug("Orig event-id sidecar '{}' is malformed; reloading the full "
                         "network.".format(sidecar_path))

        # Fallback: legacy checkpoint without sidecar, or malformed sidecar. Load the heavy
        # pickle, rewrite the sidecar for subsequent runs, then drop orig_entities.
        self.load_orig_network()
        ids = [str(event.entity_label) for event in self.get_events4layering()]
        try:
            self._write_orig_summary_sidecar()
        except OSError as exc:
            logger.debug("Could not write orig event-id sidecar '{}': {}".format(sidecar_path, exc))
        self.orig_entities = list()
        return ids


def process_raw_path_worker(task: Tuple[str, dict, Tuple[str, int, Tuple[int, int], Tuple[int, int]],
                                        np.ndarray, str, list]) -> "AquaductPath":
    """
    Top-level worker that unpacks a raw-path task tuple and delegates to
    AquaductNetwork._process_single_raw_path. Used with pool.imap_unordered() (which only supports
    single-argument workers); the underlying staticmethod takes six positional arguments and so
    cannot be passed directly.
    :param task: (path_label, parameters, traced_residue, transform_mat, md_label, cgo_object) tuple
    :return: same as AquaductNetwork._process_single_raw_path()
    """

    path_label, parameters, traced_residue, transform_mat, md_label, cgo_object = task
    return AquaductNetwork._process_single_raw_path(path_label, parameters, traced_residue,
                                                    transform_mat, md_label, cgo_object)


class TransportEvent:
    def __init__(self, event_type: str, path_label: str, parameters: dict, md_label: str,
                 traced_residue: Tuple[str, int, Tuple[int, int], Tuple[int, int]], transform_mat: np.ndarray):
        """
        Creates TransportEvent object - an elemental unit carrying info on a single transport event
        :param event_type: AQUA-DUCT type of events: "inside", "entry", "release", "outside"
        :param path_label: the name of the path containing this event derived from AQUA-DUCT raw paths names
        :param parameters: job configuration parameters
        :param md_label: name of folder with the source MD simulation data
        :param traced_residue: tuple containing resname & resid of ligand responsible for this path,
                                and beginning and last frames for entry and release event
        :param transform_mat: transformation matrix to be applied on the event coordinates
        """

        if event_type in ["inside", "entry", "release", "outside"]:
            self.type: str = event_type
        else:
            raise ValueError("Event type can only be one of 'inside', 'outside', 'entry' or 'release'")

        self.entity_label = "{}_{}_{}".format(traced_residue[0].lower(), path_label.split("_")[-1], self.type)
        self.entity_pymol_abbreviation = self.entity_label
        self.points: List[Point] = list()
        self.points_data: np.ndarray | None = None
        self.parameters = parameters
        self.traced_residue = traced_residue
        self.min_dist2starting_point: float | None = None
        self.penetration_depth = -1
        self.path_label = path_label
        self.md_label = md_label
        self.transform_mat = transform_mat
        self.modified = False

    def __str__(self):
        msg = "{}\n".format(self.type)
        for point in self.points:
            msg += "{}\n".format(point)

        return msg

    def is_same(self, other: TransportEvent) -> bool:
        """
        Test if the event is same as the other event
        :param other: other event to compare with
        :return: are two events same?
        """

        if self.type != other.type:
            return False

        if self.traced_residue != other.traced_residue:
            return False

        if len(self.points) != len(other.points):
            return False

        if len(self.points) == 0 and len(other.points) == 0:
            return True

        for point, other_point in zip(self.points, other.points):
            if not np.allclose(point.data, other_point.data, atol=1e-3):
                return False

        return True

    # ==== Methods to process CGO points forming the event's trajectory ===
    def add_point(self, point: Point):
        """
        Adds point to the event
        :param point: point to add
        """

        self.points.append(point)
        self.modified = True

    def extend_points_front(self, points2add: List[Point]):
        """
        Add points to the beginning of this transport event
        :param points2add: a list of points to add
        """

        tmp = self.points.copy()
        self.points = points2add.copy()
        self.points.extend(tmp)
        self.modified = True

    def extend_points_back(self, points2add: List[Point]):
        """
        Add points to the end of this transport event
        :param points2add: a list of points to add
        """

        self.points.extend(points2add)
        self.modified = True

    def get_num_points(self) -> int:
        """
        Count number of points in this transport event
        :return: number of points
        """

        return len(self.points)

    def assign_distances2point(self, dist_point: Point):
        """
        For all points in this event, calculate distance to dist_point and store it in point.data
        :param dist_point: reference point for calculations
        """

        for point in self.points:
            point.data[0, 3] = point.distance2point(dist_point)

    def get_min_distance(self) -> float:
        """
        Find smallest distance to other point stored for all points in this event
        :return: the smallest distance
        """

        if self.get_num_points() == 0:
            return -1
        distances = list()
        for point in self.points:
            distances.append(point.data[0, 3])

        return np.min(distances)

    def get_points_data(self) -> np.ndarray:
        """
        Convert data of points forming this event, adding info on order of points and identifying the event end point
        :return: augmented data for this event suitable for LayeredRepresentation.load_points() method
        """

        if self.points_data is None or self.modified:
            self.points_data = self.points[0].data
            for point in self.points[1:]:
                self.points_data = np.concatenate((self.points_data, point.data))

        num_points = self.points_data.shape[0]
        # store info on points order in the sixth column (id = 5)
        if self.type == "entry":
            order = np.arange(num_points - 1, -1, -1).reshape((num_points, 1)).astype(float)
            order[0] = -order[0]  # end point id is inverted
        else:
            order = np.arange(0, num_points).reshape((num_points, 1)).astype(float)
            order[-1] = -order[-1]  # end point id is inverted
        points_data = np.append(self.points_data, order, axis=1)
        # add event ID
        points_data = np.append(points_data, np.zeros((num_points, 1)), axis=1)
        data4transform = np.append(points_data[:, 0:3], np.full((points_data[:, 0:3].shape[0], 1), 1.), axis=1).T
        rev_trans_mat = np.linalg.inv(self.transform_mat)
        orig_dat = rev_trans_mat.dot(data4transform).T[:, 0:3]
        points_data[:, 0:3] = orig_dat

        return points_data

    # ===  Events methods ===
    def is_singleton(self) -> bool:
        """
        Test if this event is composed of more than one point
        """

        return len(self.points) <= 1

    def has_transition(self) -> bool:
        """
        Test if this event has transition between bulk solvent and active site
        """

        return self.type == "entry" or self.type == "release"

    def get_min_distance2starting_point(self) -> float:
        """
        Compute minimal distance of points in the event to the overall starting point [0, 0, 0]
        :return: minimal distance to starting point
        """

        if self.min_dist2starting_point is None:
            self.assign_distances2point(Point([0, 0, 0]))  # distance to origin == overall starting point
            self.min_dist2starting_point = self.get_min_distance()

        return self.min_dist2starting_point

    def merge_event(self, other_event: TransportEvent):
        """
        Add points from the other event to the beginning of this transport event
        :param other_event: event with points to merge
        """

        self.extend_points_front(other_event.points)
        self.modified = True

    def get_frame_trace(self) -> np.ndarray:
        """
        Reconstruct the per-frame AQUA-DUCT trace (the traced positions of the molecule, one per simulation
        frame) for this transition event, in the unified reference frame. Used by trace_matching to test the
        event against the actual per-snapshot CAVER tunnels without needing the source MD trajectory.

        The points are anchored to the frame range reported by AQUA-DUCT for this event type (entry events
        forward from their first frame, release events backward from their last frame) and then clipped to that
        range. The clipping drops the inside-point extensions appended toward the starting point during
        process_path() - those fall outside the reported transition frames - so only the genuine per-frame trace
        remains. The +-1 frame slips from duplicate-point removal in the source CGO stay within the range.
        :return: array of shape (n, 4) with columns (frame, x, y, z); empty for non-transition/empty events
        """

        if not self.has_transition() or not self.points:
            return np.empty((0, 4))

        coords = np.array([point.data[0, :3] for point in self.points])
        num_points = coords.shape[0]
        if self.type == "entry":
            first_frame, last_frame = self.traced_residue[2]
            frames = first_frame + np.arange(num_points)
        else:  # release: anchor the last point to the last reported frame
            first_frame, last_frame = self.traced_residue[3]
            frames = last_frame - (num_points - 1) + np.arange(num_points)

        within_range = (frames >= first_frame) & (frames <= last_frame)
        return np.column_stack((frames[within_range], coords[within_range]))

    def get_visualization_cgo(self) -> List[float]:
        """
        Converts event points to Pymol compiled graphics object(CGO) for visualization
        :return: CGO of event points
        """
        if self.points_data is None or self.modified:
            self.points_data = self.points[0].data
            for point in self.points[1:]:
                self.points_data = np.concatenate((self.points_data, point.data))

        return utils.convert_coords2cgo(self.points_data[:, 0:3], color_id=None)

    def create_layered_event(self) -> Tuple[str, str, LayeredPathSet]:
        """
        Loads event points into LayeredRepresentation and process them to get the
        actual set of Layered paths
        :return: set of layered paths representing this event
        """

        with threadpool_limits(limits=1, user_api=None):
            tmp_repre = LayeredRepresentationOfEvents(self.parameters, self.entity_label, self.traced_residue[0])
            tmp_repre.load_points(self)
            tmp_repre.split_points_to_layers()
            layered_pathset = tmp_repre.find_representative_paths(self.transform_mat, None,
                                                                  self.parameters["visualize_layered_events"])
            layered_pathset.set_traced_event(self.traced_residue)

        return self.entity_label, self.parameters["md_label"], layered_pathset


class AquaductPath:
    def __init__(self, path_label: str, parameters: dict,
                 traced_residue: Tuple[str, int, Tuple[int, int], Tuple[int, int]], transform_mat: np.ndarray,
                 md_label: str = ""):
        """
        Class for processing of transport paths produced by AQUA-DUCT, consisting of transport events
        :param path_label: the name of this path derived from AQUA-DUCT raw paths names
        :param parameters: job configuration parameters
        :param traced_residue: tuple containing resname & resid of ligand responsible for this path,
                                and beginning and last frames for entry and release event
        :param transform_mat: transformation matrix to be applied on the input coordinates
        :param md_label: name of folder with the source MD simulation data
        """

        self.md_label = md_label
        self.parameters = parameters
        self.traced_residue = traced_residue
        self.path_label = path_label
        self.entity_pymol_label = self.path_label
        self.vis_file = self.path_label + "_cgo.dump.gz"
        self.transform_mat = transform_mat
        self.required_min_distance = self.parameters["event_min_distance"]
        self.events: Dict[str, TransportEvent] = dict()

    def save_cgo(self, out_path: str):
        """
        Dump all events as single Pymol compiled graphics object(CGO) to gzipped files for visualization of this path
        :param out_path: folder where the CGO should be dumped to
        """

        path_cgos = list()
        for event_type in sorted(self.events.keys()):
            path_cgos.extend(self.events[event_type].get_visualization_cgo())

        with gzip.open(os.path.join(out_path, self.vis_file), "wb") as out_stream:
            pickle.dump(path_cgos, out_stream, self.parameters["pickle_protocol"])

    def get_events4layering(self) -> List[TransportEvent]:
        """
        Get events in this path for layering
        :return: list of events for processing
        """

        events = list()
        for event_type, event in self.events.items():
            if event.has_transition():
                events.append(event)

        return events

    def has_transport_event(self) -> bool:
        """
        Test if entry or release type event is among events forming this path
        """

        if "entry" in self.events.keys() or "release" in self.events.keys():
            return True
        return False

    def is_same(self, other: AquaductPath) -> bool:
        """
        Test if the path is same as the other path
        :param other: other path to compare with
        :return: are two paths same?
        """

        if set(self.events.keys()) != set(other.events.keys()):
            return False

        for event_label in self.events.keys():
            if not self.events[event_label].is_same(other.events[event_label]):
                return False

        return True

    # methods for path creation

    def process_path(self, cgo_object: list):
        """
        Process Compiled Graphics Object (CGO) representing a raw path from AQUA-DUCT to get initial transport events;
        these events are then processed as follows:
        1) Merge continuous sequences of inside events => max 3 events (inside, entry and release) in a single path
        2) Exclude events that does not reach close enough to the site of interest (starting point (SP))
        3) Extend transit events (entry and release) by points from inside events that are continuously closer to SP
        4) Extend transit events by overlapping inside points that form a shortest direct path to SP or close to it
        :param cgo_object: AQUA-DUCT CGO containing info on a raw path
        """

        input_events = self._parse_aquaduct_path(cgo_object)
        self._merge_repetitive_inside_events(input_events)
        self._prune_events_outside_required_min_distance()
        self._extend_transition_parts()
        self._find_overlapping_connections2starting_point()
        self._prune_singleton_events()

    def _prune_singleton_events(self):
        """
        Remove events that contain less then two points
        """
        singletons = list()
        for event in self.events.values():
            if event.has_transition() and event.is_singleton():
                logger.debug("Event {}_{} is composed of a single point only, skipping!".format(self.path_label,
                                                                                                event.type))
                singletons.append(event.type)

        for event_type in singletons:
            del self.events[event_type]

    def _parse_aquaduct_path(self, cgo_object: list) -> List[TransportEvent]:
        """
        Convert Compiled Graphics Object (CGO) representing a raw path from AQUA-DUCT to separate transport events
        of types inside, entry and release
        :param cgo_object: AQUA-DUCT CGO containing info on a raw path
        :return: list of TransportEvents read from cgo_object
        """

        in_events = list()
        obj = list()

        for element in cgo_object:
            if isinstance(element, tuple):
                for e in element:
                    obj.append(e)
            else:
                obj.append(element)
        obj = obj[2:-1]  # skip the first two and the last item that do inform about the start, type and the end of CGO
        prev_xyz = np.array([np.inf, np.inf, np.inf])

        event_type = ""
        has_transition = False

        for i in range(0, len(obj), 4):
            if obj[i] == 6.0:  # == color
                color = (obj[i + 1], obj[i + 2], obj[i + 3])
                if color == (0.0, 0.5, 0.0):  # AquaDuct type = "object"
                    event_type = "inside"
                elif color == (0.75, 0.75, 0.0):  # AquaDuct type = "outside object but not Outgoing yet"
                    event_type = "inside"
                elif color == (1.0, 0.0, 0.0):  # AquaDuct type = "incoming"
                    event_type = "entry"
                    has_transition = True
                elif color == (0.0, 0.0, 1.0):  # AquaDuct type = "outgoing"
                    event_type = "release"
                    has_transition = True
                in_events.append(TransportEvent(event_type, self.path_label, self.parameters, self.md_label,
                                                self.traced_residue, self.transform_mat))

            elif obj[i] == 4.0:
                xyz = self.transform_mat.dot(np.array([obj[i + 1], obj[i + 2], obj[i + 3], 1]))
                if np.array_equal(prev_xyz, xyz):
                    # this is to skip duplicate points in AquaDuct raw_paths dumps
                    prev_xyz = xyz
                else:
                    prev_xyz = xyz
                    in_events[-1].add_point(Point(xyz[0:3]))

        if not has_transition:
            logger.debug("Path '{}' does not have transport event".format(self.path_label))
            in_events = list()

        return in_events

    def _merge_repetitive_inside_events(self, in_events: List[TransportEvent]):
        """
        Merge continuous sequences of events of the same type (relevant only for inside events due their creation
        from AQUA-DUCT types 'object' and 'outside') to have only the major types of events
        :param in_events: events to evaluate
        """

        merged_events = list()
        prev_type = None
        prev_id = None
        last_event_id = len(in_events) - 1
        in_events.reverse()

        # merge duplicates
        for i, event in enumerate(in_events):
            if prev_type:
                assert isinstance(prev_id, int), f"Expected int but got {type(prev_id).__name__}"
                if prev_type == event.type:
                    in_events[prev_id].merge_event(event)
                    if i == last_event_id:
                        merged_events.append(in_events[prev_id])
                else:
                    merged_events.append(in_events[prev_id])
                    prev_type = event.type
                    prev_id = i
                    if i == last_event_id:
                        merged_events.append(event)
            else:
                prev_type = event.type
                prev_id = i
                if i == last_event_id:
                    merged_events.append(event)

        # store unique events
        for event in merged_events:
            self.events[event.type] = event

    def _prune_events_outside_required_min_distance(self):
        """
        Check if the transport events (possibly extended by inside points) have any point inside required radius
        from the starting point. If not such events are removed. If neither entry nor release events remains, also
        inside event is removed
        """

        inside_min_dist = -1

        if "inside" in self.events.keys():
            inside_min_dist = self.events["inside"].get_min_distance2starting_point()

        labels_to_remove = list()
        for event_type, event in self.events.items():
            if event_type == "inside":
                continue

            if inside_min_dist >= self.required_min_distance \
                    and event.get_min_distance2starting_point() >= self.required_min_distance:
                labels_to_remove.append(event_type)
                logger.debug("Event {}_{} does not penetrate to the required distance from starting point of {:.2f} A"
                             "(distance was {:.2f} A)".format(self.path_label, event_type, self.required_min_distance,
                                                              min(event.get_min_distance2starting_point(),
                                                                  inside_min_dist)))

        # remove distant events
        for label in labels_to_remove:
            del self.events[label]

        if not self.has_transport_event() and "inside" in self.events.keys():
            del self.events["inside"]

    def _extend_transition_parts(self):
        """
        Use inside points to extend transition events by a sequence of inside points that are between the event point
        closest to the starting point and the starting point.
        """

        if "inside" in self.events.keys():
            inside_points = self.events["inside"].points

            if "entry" in self.events.keys():
                entry_min_dist = self.events["entry"].get_min_distance2starting_point()
                add_points_entry = self._get_continuously_closer_points(inside_points, entry_min_dist)
                self.events["entry"].extend_points_back(add_points_entry)

            if "release" in self.events.keys():
                relase_min_dist = self.events["release"].get_min_distance2starting_point()
                add_points_release = self._get_continuously_closer_points(list(reversed(inside_points)),
                                                                          relase_min_dist)
                add_points_release.reverse()
                self.events["release"].extend_points_front(add_points_release)

    def _find_overlapping_connections2starting_point(self):
        """
        Extends transition events by interconnected(overlapping) points that form shortest path to the starting point
        or as close as possible without notable detour from the direct path)
        """

        if "inside" in self.events.keys():
            inside_points = self.events["inside"].points

            if "entry" in self.events.keys():
                border_point = self.events["entry"].points[-1]
                overlap, points2consider = self._get_points_from_intersection_between_starting_and_border_points(
                    inside_points, border_point)

                if overlap:  # border_point overlaps with SP -> adding SP to the event
                    overlapping_points = [Point([0, 0, 0], 0)]
                else:
                    overlapping_points = self._find_overlapping_path2starting_point(points2consider, border_point)

                self.events["entry"].extend_points_back(overlapping_points)

            if "release" in self.events.keys():
                border_point = self.events["release"].points[0]
                overlap, points2consider = self._get_points_from_intersection_between_starting_and_border_points(
                    inside_points, border_point)

                if overlap:  # border_point overlaps with SP -> adding SP to the event
                    overlapping_points = [Point([0, 0, 0], 0)]
                else:
                    overlapping_points = self._find_overlapping_path2starting_point(points2consider, border_point)
                    overlapping_points.reverse()

                self.events["release"].extend_points_front(overlapping_points)

    @staticmethod
    def _get_continuously_closer_points(points: List[Point], initial_distance: float) -> List[Point]:
        """
        Finds a sequence of points that are each closer to the starting point (SP) then the previous one
        :param points: points to evaluate
        :param initial_distance: distance of initial border point
        :return: list of continuously closer points
        """

        include_points = list()
        for point in points:
            current_dist = point.data[0, 3]
            if current_dist > initial_distance:
                break
            initial_distance = current_dist
            include_points.append(point)

        return include_points

    def _get_points_from_intersection_between_starting_and_border_points(self, points: List[Point],
                                                                         border_point: Point) -> Tuple[bool,
                                                                                                  List[Tuple[int,
                                                                                                             Point]]]:
        """
        Identify points that are in the intersection between starting point (SP) and the border point
        :param points: points to evaluate
        :param border_point: border point (BP) from the transit event that is closest to the SP
        :return: if BP overlaps with SP and list of points (and their original IDs) between SP and BP
        """

        points_dataset = list()
        border_overlaps_starting_point = False

        # previously run check_events_within_required_min_distance provided precomputed distances to SP
        radius = border_point.data[0, 3]  # distance between the starting and border points
        if radius <= self.parameters["aqauduct_ligand_effective_radius"]:
            border_overlaps_starting_point = True
            # BP of event efficiently overlaps with SP => no need to find any further connection, just add SP
            return border_overlaps_starting_point, points_dataset

        for i, point in enumerate(points):
            dist2start = point.data[0, 3]
            if dist2start <= radius:  # close to SP
                if 0 < point.distance2point(border_point) <= radius:  # close to BP but not the BP itself
                    points_dataset.append((i, point))

        return border_overlaps_starting_point, points_dataset

    def _find_overlapping_path2starting_point(self, points_dataset: List[Tuple[int, Point]],
                                              border_point: Point) -> List[Point]:
        """
        Identify points for extension of transition events by searching the shortest direct path of overlapping points
        between border point (BP) and starting point (SP) using A* algorithm
        :param points_dataset: points (and their original IDs) to evaluate
        :param border_point: BP from the transit event that is closest to the SP
        :return: list of overlapping points forming the shortest 'direct' path efficiently leading to the SP, or as
        close to it as possible
        """

        if not points_dataset:
            return []

        network = self._compute_network_of_overlapping_points(points_dataset, border_point)
        points = [x[1] for x in points_dataset]

        length_scores = dict()  # connectivity cost of reaching this point from BP
        full_scores = dict()  # length_score + distance to the target (SP)
        previous_nodes = dict()  # nodes visited on the way
        nodes2process = set()
        tested_nodes = set()

        # initiate all points being in infinity
        for point_id in network.keys():
            length_scores[point_id] = np.inf
            full_scores[point_id] = np.inf

        # add border point scores as the start of path search
        length_scores["BP"] = 0
        full_scores["BP"] = border_point.data[0, 3]
        nodes2process.add("BP")

        curr_node_id = None
        while nodes2process:
            curr_node_id = self._get_cheapest_point(nodes2process, full_scores)

            if curr_node_id == "SP":  # path stopped at SP
                break

            nodes2process.remove(curr_node_id)
            tested_nodes.add(curr_node_id)

            # search points overlapping with current node
            for neighbor_id in network[curr_node_id]:
                if neighbor_id in tested_nodes:
                    continue

                tmp_length_score = length_scores[curr_node_id] + 1  # next connected points  += one step
                if tmp_length_score < length_scores[neighbor_id]:
                    # this is shortest path know so far to get to reach the neighbor_id => update its scores
                    previous_nodes[neighbor_id] = curr_node_id
                    length_scores[neighbor_id] = tmp_length_score
                    full_scores[neighbor_id] = tmp_length_score + self._get_distance2starting_point(neighbor_id, points)
                    if neighbor_id not in nodes2process:
                        nodes2process.add(neighbor_id)

        if curr_node_id is None:
            return []

        points2consider = self._get_path(previous_nodes, curr_node_id, points)
        # next filter the points on the path to keep only those leading 'directly' towards SP
        return self._get_continuously_closer_points(points2consider, border_point.data[0, 3])

    @staticmethod
    def _get_path(previous_nodes: Dict[int | str, int | str], current_id: int | str, points: List[Point]) -> List[Point]:
        """
        Convert visited point ID on the way from border point (BP) towards starting point (SP) to actual list of points
        :param previous_nodes: nodes visited on the path form BP to the point with the current ID
        :param current_id: ID of point closest to the SP
        :param points: points to evaluate to get from ID to their values
        :return: list of points forming overlapping path from BP towards SP
        """

        path = [current_id]

        # trace the point ID path back up to BP that initiated the path
        while current_id != "BP":
            current_id = previous_nodes[current_id]
            path.insert(0, current_id)

        # convert ID path to the Points
        path_points = list()
        for path_point_id in path[1:]:  # excluding BP
            if "SP" == path_point_id:
                path_points.append(Point([0, 0, 0], 0))
            else:
                assert isinstance(path_point_id, int), f"Expected int but got {type(path_point_id).__name__}"
                path_points.append(points[path_point_id])

        return path_points

    @staticmethod
    def _get_distance2starting_point(node_id: int | str, points: List[Point]) -> float:
        """
        Get distance of evaluated point to the starting point (SP)
        :param node_id: ID of evaluated point
        :param points: points to evaluate to get the actual distanace to SP
        :return: distance of evaluated point to SP
        """

        if node_id == "SP":
            return 0
        else:
            assert isinstance(node_id, int), f"Expected int but got {type(node_id).__name__}"
            return points[node_id].data[0, 3]

    @staticmethod
    def _get_cheapest_point(nodes2evaluate: Set[str | int], full_scores: Dict[str | int, float]) -> str | int:
        """
        Selects next cheapest point according to its length cost and distance to the target == starting point (SP)
        :param nodes2evaluate: point IDs to evaluate for next stage
        :param full_scores: dictionary with path length cost of evaluated points ID and their distance to SP
        :return: ID of the cheapest point
        """

        if "SP" in nodes2evaluate:  # we can reach SP
            return "SP"

        tmp_list = list()
        for node_id in nodes2evaluate:
            tmp_list.append((full_scores[node_id], node_id))

        return min(tmp_list)[1]

    def _compute_network_of_overlapping_points(self, points_dataset: List[Tuple[int, Point]],
                                               border_point: Point) -> Dict[str | int, Set[str | int]]:
        """
        Define connectivity between points on the basis of their potential overlaps
        :param points_dataset: points to evaluate
        :param border_point: border point (BP) from the transit event that is closest to the starting point (SP)
        :return: dictionary with a set of IDs of overlapping (connected) points for each point
        """

        num_obs = len(points_dataset)
        network = dict()
        network["BP"] = set()
        network["SP"] = set()
        for id1 in range(num_obs):
            network[id1] = set()

        if num_obs == 0:
            return network

        radius = 2 * self.parameters["aqauduct_ligand_effective_radius"]
        orig_ids = np.array([entry[0] for entry in points_dataset], dtype=float)
        coords = np.concatenate([entry[1].data[:, 0:3] for entry in points_dataset], axis=0)
        dist2sp = np.array([entry[1].data[0, 3] for entry in points_dataset])

        # pairwise coordinate distances among points_dataset (upper triangle only, id1 < id2), via
        # the squared-norm expansion |a-b|^2 = |a|^2 + |b|^2 - 2 a.b so only NxN and Nx3 arrays are
        # ever materialised (no NxNx3 diff tensor)
        sq_norms = np.einsum('ij,ij->i', coords, coords)
        gram = coords.dot(coords.T)
        sq_dists = np.clip(sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * gram, 0, None)
        pair_dists = np.sqrt(sq_dists)
        consecutive_in_trace = np.abs(orig_ids[:, np.newaxis] - orig_ids[np.newaxis, :]) == 1
        overlapping = pair_dists <= radius
        connected = np.triu(consecutive_in_trace | overlapping, k=1)

        id1s, id2s = np.nonzero(connected)
        for id1, id2 in zip(id1s.tolist(), id2s.tolist()):
            network[id1].add(id2)

        sp_connected = np.nonzero(dist2sp <= self.parameters["aqauduct_ligand_effective_radius"])[0]
        for id1 in sp_connected.tolist():
            network[id1].add("SP")

        bp_diff = coords - border_point.data[0, 0:3]
        dist2bp = np.sqrt(np.einsum('ij,ij->i', bp_diff, bp_diff))
        bp_connected = np.nonzero(dist2bp <= radius)[0]
        for id1 in bp_connected.tolist():
            network[id1].add("BP")

        for point1, connections in network.items():  # make contacts symmetric
            for point2 in connections:
                network[point2].add(point1)

        return network


# Super Clusters
# process-local LRU registry of SuperCluster instances currently holding a loaded path_sets;
# bounds how many superclusters' path_sets stay resident at once in a single worker process
# (each load_path_sets() call, hit or miss, is an access and refreshes recency)
_LOADED_PATH_SETS_LRU: "OrderedDict[int, SuperCluster]" = OrderedDict()
_LOADED_PATH_SETS_LRU_CAP = 100


class SuperCluster:
    def __init__(self, sc_id: int, parameters: dict, total_num_md_sims: int):
        """
        Class for storing and operation on supercluster (SC) created from caver clusters across various MD simulation
        :param sc_id: initial/original supercluster ID
        :param parameters: job configuration parameters
        :param total_num_md_sims: number of all input MD simulation
        """

        self.sc_id = sc_id
        self.prioritized_sc_id: int | None = None

        self.parameters = parameters
        self.total_num_md_sims = total_num_md_sims
        self.dump_filename = "super_cluster_{:02d}.dump".format(self.sc_id)
        self.transformation_folder = os.path.join(self.parameters["transformation_folder"],
                                                  self.parameters["caver_foldername"])

        self.path_set_filename = os.path.join(self.parameters["super_cluster_path_set_folder"],
                                              self.dump_filename)

        # info on representative paths of caver clusters forming this SC, this is kept unmodified by filters
        self.tunnel_clusters: Dict[str, Dict[int, LayeredPathSet]] = dict()  # md_label -> cls_id -> pathset
        # info on which clusters fulfill active filters
        self.tunnel_clusters_valid: Dict[str, Dict[int, bool]] = dict()  # md_label -> cls_id -> validity

        # info on transport events assigned to this SC
        # event_type -> md_label -> list of events and their residue info
        self.transport_events: Dict[str, Dict[str, List[Tuple[str, Tuple[str, Tuple[int, int]]]]]] = dict()

        # average and md_label sepcific properties of the tunnel and event networks in SC from
        # TunnelProfile.get_properties()
        self.properties: Dict[str, Dict[str, float | int]] = dict()

        self.bottleneck_residue_freq: Dict[str, Dict[str, float]] = dict()

        self.num_events: Dict[str, Dict[str, int]] = {"overall": {"entry": 0, "release": 0}} 
        self.num_events_by_residue: Dict[str, Dict[str, Dict[str, int]]] = {"overall": {}}

        # space descriptors
        self.path_sets: Dict[str, LayeredPathSet] = dict()  # info on overall and per MD representative paths of the SC
        self.avg_direction: np.ndarray | None = None  # overall direction of this SC based on representative path ends

    def _get_csv_file(self, subfolder: str = "initial"):
        return os.path.join(self.parameters["super_cluster_csv_folder"], subfolder,
                            "super_cluster_{:02d}.csv".format(self.sc_id))

    def _get_bottleneck_file(self, subfolder: str = "initial"):
        return os.path.join(self.parameters["super_cluster_bottleneck_folder"], subfolder,
                            "super_cluster_{:02d}.csv".format(self.sc_id))

    def report_details(self, events_assigned: bool):
        """
        Prints detailed information on the content of supercluster (SC)
        :param events_assigned: were transport events  already assigned to decide if to report them
        :return: string with SC details
        """
        details_txt = "Supercluster ID {:d}\n".format(self.sc_id)
        # add tunnel data
        details_txt += "\nDetails on tunnel network:\n"
        details_txt += "Number of MD simulations = {:d}\n".format(len(self.get_md_labels()))
        details_txt += "Number of tunnel clusters = {:d}\n".format(len(self.get_caver_clusters_full_labels()))
        details_txt += "Tunnel clusters:\n"

        for md_label in sorted(self.get_md_labels()):
            details_txt += "from {}: ".format(md_label)
            for cls_id in sorted(self.get_caver_cluster_ids4md_label(md_label)):
                details_txt += "{}, ".format(cls_id)
            details_txt += "\n"

        if events_assigned:
            # add transport events data
            details_txt += "\nDetails on transport events:\n"
            details_txt += "Number of MD simulations = {:d}\n".format(self.count_md_labels4events())
            details_txt += "Number of entry events = {:d}\n".format(self.num_events["overall"]["entry"])
            details_txt += "Number of release events = {:d}\n".format(self.num_events["overall"]["release"])
            if self.num_events_by_residue["overall"]:
                details_txt += "Per-residue breakdown:\n"
                for resname in sorted(self.num_events_by_residue["overall"].keys()):
                    details_txt += "  {}: entry={:d}, release={:d}\n".format(
                        resname, self.num_events_by_residue["overall"][resname]["entry"],
                        self.num_events_by_residue["overall"][resname]["release"])

            for event_type in sorted(self.transport_events.keys()):
                details_txt += "{}: (from Simulation: AQUA-DUCT ID, (Resname:Residue), " \
                               "start_frame->end_frame; ... )\n".format(event_type)

                for md_label, paths in self.transport_events[event_type].items():
                    details_txt += "from {}: ".format(md_label)
                    for path_id, traced_event in paths:
                        details_txt += "{}, ({}), {}->{}; ".format(path_id, traced_event[0], traced_event[1][0],
                                                                   traced_event[1][1])
                    details_txt += "\n"

        return details_txt

    # === methods for super-cluster creation & modification ===
    def add_transport_event(self, md_label: str, path_id: str, event_type: str,
                            traced_event: Tuple[str, Tuple[int, int]]) -> bool:
        """
        Store information on transport event assigned to this supercluster (SC)
        :param md_label: name of folder with the MD simulation data that contain this transport event
        :param path_id: AQUA-DUCT path ID of this event
        :param event_type: type(release or entry) of this event
        :param traced_event: tuple containing identity of ligand responsible for this event,
                                and beginning and last frames of the event
        :return: if the event was locally assigned also to the group/particular md_label
        """

        if event_type not in self.transport_events.keys():
            self.transport_events[event_type] = dict()

        if md_label not in self.transport_events[event_type].keys():
            self.transport_events[event_type][md_label] = list()

        self.transport_events[event_type][md_label].append((path_id, traced_event))
        self.num_events["overall"][event_type] += 1
        resname = traced_event[0].split(":")[0]
        if resname not in self.num_events_by_residue["overall"]:
            self.num_events_by_residue["overall"][resname] = {"entry": 0, "release": 0}
        self.num_events_by_residue["overall"][resname][event_type] += 1

        stat_md_label = md_label
        if self.parameters["perform_comparative_analysis"] \
                and self.parameters["comparative_groups_definition"] is not None:
            membership = get_md_membership4groups(self.parameters["comparative_groups_definition"])
            if md_label in membership.keys():
                stat_md_label = membership[md_label]

        if stat_md_label not in self.properties.keys() or not self.properties[stat_md_label]:
            # event cannot be locally assigned withing this md_label
            return False
        else:
            if stat_md_label not in self.num_events.keys():
                self.num_events[stat_md_label] = {"entry": 0, "release": 0}
            self.num_events[stat_md_label][event_type] += 1
            if stat_md_label not in self.num_events_by_residue:
                self.num_events_by_residue[stat_md_label] = {}
            if resname not in self.num_events_by_residue[stat_md_label]:
                self.num_events_by_residue[stat_md_label][resname] = {"entry": 0, "release": 0}
            self.num_events_by_residue[stat_md_label][resname][event_type] += 1
            return True

    def add_caver_cluster(self, md_label: str, cls_id: int, path_set: LayeredPathSet):
        """
        Store information on tunnel cluster forming this supercluster (SC)
        :param md_label: name of folder with the MD simulation data that contain this cluster
        :param cls_id: ID of this cluster
        :param path_set: Layered paths representing this cluster
        """

        if md_label not in self.tunnel_clusters.keys():
            self.tunnel_clusters[md_label] = dict()
            self.tunnel_clusters_valid[md_label] = dict()

        self.tunnel_clusters[md_label][cls_id] = path_set
        self.tunnel_clusters_valid[md_label][cls_id] = True

    def set_properties(self, new_properties: dict):
        self.properties = new_properties.copy()

    def set_bottleneck_residue_freq(self, new_bottleneck_residue_freq: dict):
        self.bottleneck_residue_freq = new_bottleneck_residue_freq.copy()

    def update_caver_clusters_validity(self, retained_tunnel_clusters: Dict[str, List[int]]):
        """
        Update info on validity of clusters forming this SC, which is stored in self.tunnel_clusters_valid
        :param retained_tunnel_clusters: which clusters from which source MD simulations are valid
        """

        for md_label in self.tunnel_clusters_valid.keys():
            if md_label in retained_tunnel_clusters.keys():
                for cluster_id in self.tunnel_clusters_valid[md_label]:
                    self.tunnel_clusters_valid[md_label][cluster_id] = cluster_id in retained_tunnel_clusters[md_label]
            else:
                for cluster_id in self.tunnel_clusters_valid[md_label]:
                    self.tunnel_clusters_valid[md_label][cluster_id] = False

    def load_path_sets(self):
        if not self.path_sets:  # not already loaded in this worker
            with open(self.path_set_filename, "rb") as in_stream:
                self.path_sets: Dict[str, LayeredPathSet] = pickle.load(in_stream)

            for path_set in self.path_sets.values():
                path_set.parameters.update(self.parameters)  # to update config if needed

        # (re-)register on every access, hit or miss: this instance may have been unpickled with
        # path_sets already populated (e.g. it travelled into a fresh worker process via Pool
        # initargs, or came from a checkpoint) without ever registering here, so this process-local
        # LRU can lack an entry for it even though path_sets itself is present
        _LOADED_PATH_SETS_LRU[self.sc_id] = self
        _LOADED_PATH_SETS_LRU.move_to_end(self.sc_id)
        if len(_LOADED_PATH_SETS_LRU) > _LOADED_PATH_SETS_LRU_CAP:
            _, evicted = _LOADED_PATH_SETS_LRU.popitem(last=False)
            evicted.path_sets = dict()

    def compute_space_descriptors(self) -> Tuple[int, np.ndarray]:
        """
        Collect all unique nodes and paths from all caver clusters to a single PathSet and compute overall direction
        in which the end points of supercluster lay
        :return: id of supercluster and its average direction
        """

        # collect data for all nodes from caver clusters belonging to this super-cluster
        path_sets = dict()
        path_sets["overall"] = LayeredPathSet("TMP_{:03}".format(self.sc_id), "overall", self.parameters,
                                              np.array([0., 0., 0.]))

        for md_label in self.get_md_labels():
            path_sets[md_label] = LayeredPathSet("TMP_{:03}".format(self.sc_id), md_label, self.parameters,
                                                 np.array([0., 0., 0.]))
            for cls_id in self.get_caver_cluster_ids4md_label(md_label):
                path_sets[md_label] += self.tunnel_clusters[md_label][cls_id]  # add all PathSets together

            path_sets["overall"] += path_sets[md_label]
            if not self.parameters["perform_comparative_analysis"]:
                del path_sets[md_label]

        if self.parameters["perform_comparative_analysis"]:
            if self.parameters["comparative_groups_definition"] is not None:
                # replace all individual md_labels path_sets by corresponding group pathset
                for group, md_labels in self.parameters["comparative_groups_definition"].items():
                    path_sets[group] = LayeredPathSet("TMP_{:03}".format(self.sc_id), group, self.parameters,
                                                      np.array([0., 0., 0.]))
                    for _md_label in md_labels:
                        if _md_label not in path_sets.keys():
                            continue
                        path_sets[group] += path_sets[_md_label]
                        del path_sets[_md_label]

        # process all valid path_sets
        for path_set in path_sets.values():
            if path_set.is_empty():
                del path_set
                continue
            path_set.remove_duplicates()
            path_set.compute_node_depths()

        # get average direction of terminal nodes possibly used during a transport event
        if path_sets["overall"].nodes_data is None:
            raise RuntimeError("Variable 'nodes_data' is not specified for overall path set")
        avg_direction = np.average(path_sets["overall"].nodes_data[path_sets["overall"].nodes_data[:, 4] == 1, :3],
                                   axis=0)

        os.makedirs(os.path.dirname(self.path_set_filename), exist_ok=True)
        with open(self.path_set_filename, "wb") as out_stream:
            pickle.dump(path_sets, out_stream)

        return self.sc_id, avg_direction

    # ===  methods for data navigation  ===
    def has_passed_filter(self, consider_transport_events: bool = False, active_filters: dict | None = None) -> bool:
        """
        Test if the supercluster (SC) is valid under conditions defined by the active filters
        :param consider_transport_events: if related filters related to transport events should be considered
        :param active_filters: active filters (created by define_filters() function) to be evaluated
        for transport events
        """

        if self.properties:  # if the SC properties is not empty, the tunnel related filters are fulfilled
            if consider_transport_events:
                if active_filters is None:
                    raise RuntimeError("active_filters must be defined here")
                if self.num_events["overall"]["entry"] + self.num_events["overall"]["release"] >= active_filters["min_transport_events"] \
                        and self.num_events["overall"]["entry"] >= active_filters["min_entry_events"] \
                        and self.num_events["overall"]["release"] >= active_filters["min_release_events"]:
                    return True
            else:
                return True

        return False

    def count_md_labels4events(self):
        """
        Counts how many simulations contributed some event in this supercluster
        :return: number of simulations
        """

        md_labels = set()
        for event_type in self.transport_events.keys():
            md_labels.update(self.transport_events[event_type].keys())

        return len(md_labels)

    def get_md_labels(self, only_with_transport_events: bool = False) -> List[str]:
        """
        Enumerate names of folders with the MD simulations (md_labels) contributing with at least one valid tunnel
        cluster to this supercluster
        :consider_transport_events: if the md_labels should be listed also considering assigned events
        :return: list of md_labels with valid tunnel clusters
        """

        md_labels = list()

        for md_label in self.tunnel_clusters_valid.keys():
            event_assigned = False
            if only_with_transport_events:
                for event_type in self.transport_events.keys():
                    if md_label in self.transport_events[event_type].keys() \
                            and self.transport_events[event_type][md_label]:
                        # there is some event of any type for this md_label
                        event_assigned = True
                        break
                if not event_assigned:
                    # we go for the next md_label as this one is not valid even if tunnel clusters are present
                    continue

            label_valid = False
            for cls_id, validity in self.tunnel_clusters_valid[md_label].items():
                if validity and (not only_with_transport_events or event_assigned):
                    label_valid = True
                    break

            if label_valid:
                md_labels.append(md_label)

        return md_labels

    def get_caver_cluster_ids4md_label(self, md_label: str) -> List[int]:
        """
        Enumerate cluster IDs for given name of the source MD simulation folder that are valid and belong to this SC
        :param md_label: name of folder with the source MD simulation data
        :return:  list of IDs of valid tunnel clusters for this md_label
        """

        cls_ids = list()
        if md_label not in self.tunnel_clusters_valid:
            return cls_ids

        for cls_id, validity in self.tunnel_clusters_valid[md_label].items():
            if validity:
                cls_ids.append(cls_id)

        return cls_ids

    def get_caver_clusters_full_labels(self) -> List[str]:
        """
        Generate full names of valid tunnel clusters consisting of foldername of their source MD simulation and their ID
        :return: list of full names of valid clusters
        """

        caver_cluster_labels = list()
        for md_label in self.get_md_labels():
            for cluster_id in self.get_caver_cluster_ids4md_label(md_label):
                caver_cluster_labels.append("{}_{}".format(md_label, cluster_id))

        return caver_cluster_labels

    def get_caver_clusters(self, md_labels: List[str] | None = None,
                           snap_ids: List[int] | None = None) -> Dict[str, List[TunnelCluster]]:
        """
        Get tunnel clusters from this SC, possibly filtered for specified Snapshot IDs and particular MD simulations
        :param md_labels:
        :param snap_ids:
        :return: dictionary with list of requested tunnel clusters for each md_label
        """

        caver_clusters = dict()
        if md_labels is None:
            md_labels = self.get_md_labels()

        for md_label in md_labels:
            if md_label in self.get_md_labels():
                temp_network = TunnelNetwork(self.parameters, md_label)
                temp_network.load_orig_network()
                caver_clusters[md_label] = list()
                for cluster_id in self.get_caver_cluster_ids4md_label(md_label):
                    cluster = temp_network.orig_entities[cluster_id - 1]
                    assert isinstance(cluster, TunnelCluster), f"Expected TunnelCluster but got {type(cluster).__name__}"
                    if snap_ids is not None:
                        cluster = cluster.get_subcluster(snap_ids)
                    caver_clusters[md_label].append(cluster)

        return caver_clusters

    # === methods for assignment of transport events ===
    def is_directionally_aligned(self, other_direction: np.ndarray) -> bool:
        """
        Test if the supercluster direction is aligned to other_direction within directional_cutoff
        :param other_direction: other evaluated direction
        """

        directional_cutoff = self.parameters["directional_cutoff"]
        if self.avg_direction is None:
            raise RuntimeError(f"Variable 'avg_direction' is not specified for supercluster {self.sc_id}")
        angle = vector_angle(self.avg_direction, other_direction)
        if angle <= directional_cutoff or angle >= (2 * np.pi - directional_cutoff):
            return True
        else:
            return False

    def compute_distance2transport_event(self, transport_event: LayeredPathSet) -> Tuple[float, float, float]:
        """
        Computes the fraction of nodes from Layered path that are buried inside the supercluster, their maximal
        depth (counted towards starting point (SP) along shortest path), and the minimal depth among buried nodes
        :param transport_event: Layered path representing the transport event
        :return: path buriedness, max depth towards SP, min depth towards SP among buried nodes
        """

        self.load_path_sets()
        return transport_event.how_much_is_inside(self.path_sets["overall"])

    # === methods for data reporting ===
    def prepare_visualization(self,  md_label: str = "overall", flag: str = "") -> Tuple[List[str], Tuple[LayeredPathSet, Tuple[str, str, int, bool, str]] | None, Tuple[str, List[Tuple[str, str, Tuple[str, str], List[float]]]] | None]:
        """
        Prepare overall CGO files for visualization of paths representing this supercluster (SC) and generate lines
        for Pymol visualization script
        :param md_label: visualization of which simulations to prepare; by default 'overall' visualization
        :param flag: additional description enabling differentiation of cgo files among various results after filtering
        :return: lines to load visualization of this SC into Pymol; LayeredPathSet and parameters to generate the
                 SC tunnel CGO file; and (when bundling events) the event-bundle request as (bundle_basename,
                 selection) where selection is a list of (md_label, entity_key, (event_type, resname), rgb) for the
                 caller to materialise into the bundle - None when bundling is off or the SC has no events to show
        """

        plines = list()
        viz_data = None
        bundle_request = None

        if md_label not in self.path_sets.keys() or md_label not in self.properties.keys() \
                or not self.properties[md_label]:
            # pathset not created or invalid supercluster
            return plines, viz_data, bundle_request

        # dump CGO files for visualization of paths representing this SC
        os.makedirs(self.parameters["super_cluster_vis_path"], exist_ok=True)
        if self.prioritized_sc_id is None:
            raise RuntimeError(f"Variable 'prioritized_sc_id' was not assigned for supercluster {self.sc_id}")
        viz_data = (self.path_sets[md_label], (self.parameters["super_cluster_vis_path"],
                                               "SC{:02d}_{}".format(self.sc_id, md_label),
                                               self.prioritized_sc_id - 1, True, flag))

        root_folder = self.parameters["visualization_folder"]
        if "overall" not in md_label:
            root_folder = os.path.join(root_folder, "comparative_analysis", md_label)

        sc_vis_folder = os.path.relpath(self.parameters["super_cluster_vis_path"], root_folder)
        # CGO filepath for loading to Pymol
        filename = os.path.join(sc_vis_folder, "SC{:02d}_{}_pathset{}.dump.gz".format(self.sc_id, md_label, flag))
        filename_vol = os.path.join(sc_vis_folder, "SC{:02d}_{}_volume{}.dump.gz".format(self.sc_id, md_label, flag))

        # generate Pymol script of this SC
        plines.append("with gzip.open({}, 'rb') as in_stream:\n".format(utils.path_loader_string(filename)))
        plines.append("    pathset = pickle.load(in_stream)\n")
        plines.append("cmd.load_cgo(pathset, 'cluster_{:03d}')\n".format(self.sc_id))
        plines.append("cmd.set('cgo_line_width', {}, 'cluster_{:03d}')\n\n".format(5, self.sc_id))
        if self.parameters["visualize_super_cluster_volumes"] and \
                ("overall" in md_label or self.parameters["visualize_comparative_super_cluster_volumes"]):
            plines.append("with gzip.open({}, 'rb') as in_stream:\n".format(utils.path_loader_string(filename_vol)))
            plines.append("    volume = pickle.load(in_stream)\n")
            plines.append("cmd.load_cgo(volume, 'cluster_{:03d}_vol')\n\n".format(self.sc_id))

        comparative_groups_definition = {}
        if self.parameters["perform_comparative_analysis"] \
                and self.parameters["comparative_groups_definition"] is not None:
            comparative_groups_definition = self.parameters["comparative_groups_definition"]

        # collect sorted residue names across all event types for deterministic color assignment
        all_residues: List[str] = sorted({
            path[1][0].split(":")[0]
            for events in self.transport_events.values()
            for paths in events.values()
            for path in paths
        })

        if self.parameters["bundle_events_visualization"]:
            # Bundle mode: emit a single per-(SC, scope) load referencing a bundle file the caller writes,
            # holding {(event_type, resname): merged CGO} with residue colors already baked in. The selection
            # over event identities is computed here (no geometry) and returned for the caller to materialise.
            selection: List[Tuple[str, str, Tuple[str, str], List[float]]] = list()
            for event_type in sorted(self.transport_events.keys()):
                for _md_label, path_id, resname in subsample_events(self.transport_events[event_type],
                                                                    self.parameters["random_seed"],
                                                                    self.parameters["max_events_per_cluster4visualization"],
                                                                    md_label, comparative_groups_definition):
                    entity_key = "{}_{}_{}".format(resname.lower(), path_id, event_type)
                    rgb = utils.get_residue_color(all_residues.index(resname))
                    selection.append((_md_label, entity_key, (event_type, resname), rgb))

            if selection:
                bundle_basename = "SC{:02d}_{}_events{}.dump.gz".format(self.sc_id, md_label, flag)
                bundle_relpath = os.path.join(sc_vis_folder, bundle_basename)
                plines.append("with gzip.open({}, 'rb') as in_stream:\n".format(utils.path_loader_string(bundle_relpath)))
                plines.append("    sc_events = pickle.load(in_stream)\n")
                plines.append("for (event_type, resname), event_cgo in sc_events.items():\n")
                plines.append('    obj_name = "{{}}_{{}}_{:03d}".format(resname.lower(), event_type)\n'.format(self.sc_id))
                plines.append("    cmd.load_cgo(event_cgo, obj_name)\n")
                plines.append("    cmd.set('cgo_line_width', 2, obj_name)\n\n")
                bundle_request = (bundle_basename, selection)
        else:
            # Legacy mode: one CGO file per event (written at stage 8); recolor per residue at load time.
            vis_folder = os.path.relpath(self.parameters["layered_aquaduct_vis_path"], root_folder)
            for event_type in sorted(self.transport_events.keys()):
                events_by_residue: Dict[str, List[str]] = {}
                for _md_label, path_id, resname in subsample_events(self.transport_events[event_type],
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
                    obj_name = "{}_{:03d}".format(resname.lower() + "_" + event_type, self.sc_id)
                    plines.append("events = [{}]\n".format(",\n".join(events_by_residue[resname])))
                    plines.append("for event in events:\n")
                    plines.append("    with gzip.open(event, 'rb') as in_stream:\n")
                    plines.append("        pathset = pickle.load(in_stream)\n")
                    plines.append("        for path in pathset:\n")
                    plines.append("            path[3:6] = {}\n".format(color))
                    plines.append("            cmd.load_cgo(path, '{}')\n".format(obj_name))
                    plines.append("cmd.set('cgo_line_width', {}, '{}')\n\n".format(2, obj_name))

        return plines, viz_data, bundle_request

    def _get_per_sim_residue_event_counts(self, resname: str, sims2process: List[str]) -> Tuple[List[int], List[int]]:
        """
        Counts entry and release events of a given residue for each simulation in sims2process, using 0 for
        simulations that contributed no such event; used as the sample for per-residue mean/stdev reporting
        :param resname: residue name to count events for
        :param sims2process: md_labels of all simulations belonging to the group being summarized
        :return: (per-sim entry counts, per-sim release counts), aligned with sims2process
        """

        entry_events = self.transport_events.get("entry", {})
        release_events = self.transport_events.get("release", {})

        entry_counts = [sum(1 for _, traced_event in entry_events.get(sim, [])
                            if traced_event[0].split(":")[0] == resname) for sim in sims2process]
        release_counts = [sum(1 for _, traced_event in release_events.get(sim, [])
                              if traced_event[0].split(":")[0] == resname) for sim in sims2process]

        return entry_counts, release_counts

    def get_summary_line_data(self, print_transport_events: bool = False, md_label: str = "overall",
                              residue_names: List[str] | None = None,
                              sims2process: List[str] | None = None) -> List[str]:
        """
        Generates data for creation of line summarizing overall properties of this supercluster (SC)
        :param print_transport_events: if properties related to transport events should be reported
        :param md_label: summary of which simulations to report; by default report 'overall' stats
        :param residue_names: sorted list of residue names for per-residue event columns; None = skip
        :param sims2process: md_labels of all simulations belonging to md_label's group, used to compute
                             per-residue mean/stdev of event counts (0-filled for silent simulations);
                             required when residue_names is set
        :return: list of items for the summary line
        """
        data = ["{:d}".format(self.sc_id)]

        if md_label not in self.properties.keys() or not self.properties[md_label]:
            data.extend(["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"])
        else:
            for label in ["num_md_sims", "num_snaps"]:
                data.append("{:d}".format(self.properties[md_label][label]))

            data.append("{:.1f}".format(self.properties[md_label]["avg_snaps"]))

            for label in ["radii_avg", "radii_std", "radii_max", "length_avg", "length_std", "curvature_avg",
                          "curvature_std"]:
                data.append("{:.3f}".format(self.properties[md_label][label]))

            for label in ["throughput_avg", "throughput_std", "priority"]:
                data.append("{:.5f}".format(self.properties[md_label][label]))

        if print_transport_events:
            if md_label not in self.num_events.keys():
                data.extend(["-", "-", "-"])
            else:
                data.append("{:d}".format(self.num_events[md_label]["entry"] + self.num_events[md_label]["release"]))
                data.append("{:d}".format(self.num_events[md_label]["entry"]))
                data.append("{:d}".format(self.num_events[md_label]["release"]))
            if residue_names:
                if not sims2process:
                    raise ValueError("sims2process must be provided (non-empty) when residue_names is set")
                res_counts = self.num_events_by_residue.get(md_label, {})
                for resname in residue_names:
                    if resname in res_counts:
                        entry_counts, release_counts = self._get_per_sim_residue_event_counts(
                            resname, sims2process)
                        data.append("{:.1f}".format(np.average(entry_counts)))
                        data.append("{:.1f}".format(np.std(entry_counts)))
                        data.append("{:.1f}".format(np.average(release_counts)))
                        data.append("{:.1f}".format(np.std(release_counts)))
                    else:
                        data.extend(["-", "-", "-", "-"])

        return data

    def process_cluster_profile(self) -> Tuple[int, Dict[str, Dict[str, float | int]], Dict[str, Dict[str, float]],
                                          Dict[str, List[int]]]:
        """
        Merging of individual caver tunnel profiles into the new one for a single supercluster (SC)
        and computes SC properties.
        Note: due to parallel processing, these SC properties are assigned to this SC via set_properties() method
        called by TransportProcess.create_super_cluster_profiles() method.
        :return: ID of this SC to identify SC in TP.create_super_cluster_profiles(), overall properties of this SC,
        bottleneck residues frequency, and tunnel clusters that are valid after keeping only single tunnel per snapshot
        """

        # get all (since all are valid at this stage) caver clusters for md_label mapping
        tunnel_clusters = dict()
        for md_label in self.get_md_labels():
            tunnel_clusters[md_label] = self.get_caver_cluster_ids4md_label(md_label)

        # create cumulative tunnel profile for SC
        cumulative_tunnel_profile = CumulativeTunnelProfile4SuperCluster(self.sc_id, tunnel_clusters,
                                                                         self.parameters)
        merged_tunnel_clusters = cumulative_tunnel_profile.load_networks()
        logger.debug("Creating supercluster profile file: {}".format(self.dump_filename))

        with open(os.path.join(self.parameters["super_cluster_profiles_folder"], "initial",
                               self.dump_filename), "wb") as out:
            pickle.dump(cumulative_tunnel_profile, out)

        properties = cumulative_tunnel_profile.get_properties(self.total_num_md_sims)
        if self.parameters["save_super_cluster_profiles_csvs"]:
            cumulative_tunnel_profile.write_csv(self._get_csv_file())

        residues_freq = dict()
        if self.parameters["process_bottleneck_residues"]:
            cumulative_tunnel_profile.write_residues(self._get_bottleneck_file())
            residues_freq = cumulative_tunnel_profile.get_bottleneck_residues_frequency()
            for md_label in residues_freq.keys():
                for residue in residues_freq[md_label].keys():
                    residues_freq[md_label][residue] /= properties[md_label]["num_snaps"]

        return self.sc_id, properties, residues_freq, merged_tunnel_clusters

    def filter_super_cluster(self, consider_transport_events: bool, active_filters: dict,
                             flag: int) -> Tuple[int, Dict[str, Dict[str, float | int]], Dict[str, Dict[str, float]],
                                            Dict[str, List[int]]]:
        """
        Filtering of tunnels loaded from supercluster (SC) profile dumps generated by process_cluster_profile,
        recalculating the SC properties, and saving the filtered cumulative profile
        Note: due to parallel processing, these SC properties and valid clusters after filtering are assigned to this SC
        via set_properties() and update_caver_clusters_validity() methods called by
        TransportProcess.filter_super_clusters() method.
        :param consider_transport_events: if filters related to transport events should be used
        :param active_filters: filters to be applied (created by define_filters() function)
        :param flag: filtering ID for subfolder name to differentiate among various results after different steps
        :return: (ID of this SC to to identify SC in TP.filter_super_clusters(), SC properties after tunnel filtering,
        bottleneck residues frequency, and tunnel clusters that are valid after filtering)
        """

        logger.debug("Filtering supercluster profile dump: {}".format(self.dump_filename))

        filtered_dump_filename = os.path.join(self.parameters["super_cluster_profiles_folder"],
                                              "filtered{:02d}".format(flag), self.dump_filename)
        in_dump_filename = os.path.join(self.parameters["super_cluster_profiles_folder"], "initial", self.dump_filename)

        with open(in_dump_filename, "rb") as in_stream:
            cumulative_tunnel_profile: CumulativeTunnelProfile4SuperCluster = pickle.load(in_stream)
        cumulative_tunnel_profile.parameters.update(self.parameters)

        filtered_tunnel_clusters = cumulative_tunnel_profile.filter_clusters(active_filters, self.total_num_md_sims)
        filtered_props = cumulative_tunnel_profile.get_properties(self.total_num_md_sims)

        filtered_residues_freq = dict()
        if self.parameters["process_bottleneck_residues"]:
            filtered_residues_freq = cumulative_tunnel_profile.get_bottleneck_residues_frequency()
            for md_label in filtered_residues_freq.keys():
                for residue in filtered_residues_freq[md_label].keys():
                    filtered_residues_freq[md_label][residue] /= filtered_props[md_label]["num_snaps"]

        self.properties = filtered_props  # to enable self.has_passed_filter

        if self.has_passed_filter(consider_transport_events=consider_transport_events, active_filters=active_filters):
            # save valid profiles
            with open(filtered_dump_filename, "wb") as out:
                pickle.dump(cumulative_tunnel_profile, out)

            if self.parameters["save_super_cluster_profiles_csvs"]:
                cumulative_tunnel_profile.write_csv(self._get_csv_file("filtered{:02d}".format(flag)))

            if self.parameters["process_bottleneck_residues"]:
                cumulative_tunnel_profile.write_residues(self._get_bottleneck_file("filtered{:02d}".format(flag)))

        return self.sc_id, filtered_props, filtered_residues_freq, filtered_tunnel_clusters

    def get_property_time_evolution_data(self, property_name: str, active_filters: dict,
                                         missing_value_default: float = 0) -> Dict[str, np.ndarray]:
        """
        For each MD simulation return array containing values of given tunnel property for each simulation frame
        :param property_name: name of property to extract
        :param active_filters: filters to be applied (created by define_filters() function)
        :param missing_value_default: value to be used for frames where tunnels are missing or invalid in this cluster
        :return: mapping of tunnel property values for each MD simulation
        """

        in_dump_filename = os.path.join(self.parameters["super_cluster_profiles_folder"], "initial", self.dump_filename)

        with open(in_dump_filename, "rb") as in_stream:
            cumulative_tunnel_profile: CumulativeTunnelProfile4SuperCluster = pickle.load(in_stream)

        cumulative_tunnel_profile.filter_clusters(active_filters, self.total_num_md_sims)
        all_md_labels = [*self.tunnel_clusters_valid.keys()]
        return cumulative_tunnel_profile.get_property_time_evolution_data(property_name, all_md_labels,
                                                                          missing_value_default)


class CumulativeTunnelProfile4SuperCluster:
    def __init__(self, sc_id: int, tunnel_clusters: Dict[str, List[int]], parameters: dict):
        """

        Class to store and process the cumulative tunnel profile of all tunnels belonging to caver clusters
        from parent supercluster (SC), there is a single such profile per SC
        :param sc_id: initial/original ID of parent SC
        :param tunnel_clusters: mapping of clusters belonging to particular MD simulation
        :param parameters: job configuration parameters
        """

        self.super_cluster_id = sc_id
        self.tunnel_profiles4md: Dict[str, TunnelProfile4MD] = dict()  # md_label -> TunnelProfile4MD
        self.tunnel_clusters = tunnel_clusters  # md_label -> list of caver_cluster_ids belonging to parent SC
        self.parameters = parameters
        self.passed_filter = True
        # set path to folder with dump files of original TunnelNetworks containing all clusters
        self.dump_file_root_folder = self.parameters["orig_caver_network_data_path"]

    def write_csv(self, filename: str):
        """
        Saves the actual cumulative tunnel profile to CSV file formatted akin to CAVER tunnel profiles
        :param filename: path to the CSV formatted file to save the cumulative tunnel profile to
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as out_stream:
            out_stream.write("MD_traj, Snapshot, Tunnel cluster, Tunnel, Throughput, Cost, Bottleneck radius, Average"
                             " R error bound, Max. R error bound, Bottleneck R error bound, Curvature, Length, , Axis, "
                             "Values...\n")
            for tunnel_profile4md in self.tunnel_profiles4md.values():
                tunnel_profile4md.write_csv_section(out_stream)

    def write_residues(self, filename: str):
        """
        Saves the actual cumulative bottleneck data to CSV file formatted akin to CAVER bottlenecks
        :param filename: path to the CSV formatted file to save the cumulative bottleneck data  to
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as out_stream:
            out_stream.write("MD_traj, Snapshot, Tunnel cluster, Tunnel, Throughput, Cost, Bottleneck X, Bottleneck Y, "
                             "Bottleneck Z,  Bottleneck R, Bottleneck residues\n")
            for tunnel_profile4md in self.tunnel_profiles4md.values():
                tunnel_profile4md.write_residues(out_stream)

    def get_properties(self, total_num_md_sims: int = 1) -> Dict[str, Dict[str, float | int]]:
        """
        Compute overall properties of parent supercluster (SC) based on the cumulative tunnel profiles
        :param total_num_md_sims: number of all input MD simulation, not only in this SC
        :return: overall properties of parent SC
        """

        properties = dict()

        if not self.passed_filter:
            return properties

        lengths = dict()
        radii = dict()
        curvatures = dict()
        throughputs = dict()
        lengths["overall"] = list()
        radii["overall"] = list()
        curvatures["overall"] = list()
        throughputs["overall"] = list()

        for md_label, tunnel_profile4md in self.tunnel_profiles4md.items():
            lengths[md_label], radii[md_label], curvatures[md_label], throughputs[md_label] = \
                tunnel_profile4md.get_parameters()

            lengths["overall"].extend(lengths[md_label])
            radii["overall"].extend(radii[md_label])
            curvatures["overall"].extend(curvatures[md_label])
            throughputs["overall"].extend(throughputs[md_label])

            if self.parameters["perform_comparative_analysis"]:
                # here total number of considered MD and analyzed MDs are both = 1
                num_md_snaps = len(throughputs[md_label])
                throughput_md_avg = np.average(throughputs[md_label])
                properties[md_label] = {
                    "throughput_avg": throughput_md_avg,
                    "throughput_std": np.std(throughputs[md_label]),
                    "radii_avg": np.average(radii[md_label]),
                    "radii_std": np.std(radii[md_label]),
                    "radii_max": np.max(radii[md_label]),
                    "curvature_avg": np.average(curvatures[md_label]),
                    "curvature_std": np.std(curvatures[md_label]),
                    "length_avg": np.average(lengths[md_label]),
                    "length_std": np.std(lengths[md_label]),
                    "num_md_sims": 1,
                    "num_snaps": num_md_snaps,
                    "avg_snaps": num_md_snaps,
                    "priority": throughput_md_avg * (num_md_snaps / self.parameters["snapshots_per_simulation"])
                }

        if self.parameters["perform_comparative_analysis"] \
                and self.parameters["comparative_groups_definition"] is not None:
            for group, md_labels in self.parameters["comparative_groups_definition"].items():
                lengths[group] = list()
                radii[group] = list()
                curvatures[group] = list()
                throughputs[group] = list()
                num_md_sims = 0
                for md_label in md_labels:
                    if md_label not in self.tunnel_profiles4md.keys():  # this SC actually contains data from this MD
                        continue
                    num_md_sims += 1
                    lengths[group].extend(lengths[md_label])
                    radii[group].extend(radii[md_label])
                    curvatures[group].extend(curvatures[md_label])
                    throughputs[group].extend(throughputs[md_label])
                    # remove individual md record
                    del properties[md_label]

                if num_md_sims > 0:  # there are some simulations from this SC that belong to the group
                    group_total_num_md_sims = len(md_labels)
                    num_md_snaps = len(throughputs[group])
                    throughput_md_avg = np.average(throughputs[group])
                    properties[group] = {
                        "throughput_avg": throughput_md_avg,
                        "throughput_std": np.std(throughputs[group]),
                        "radii_avg": np.average(radii[group]),
                        "radii_std": np.std(radii[group]),
                        "radii_max": np.max(radii[group]),
                        "curvature_avg": np.average(curvatures[group]),
                        "curvature_std": np.std(curvatures[group]),
                        "length_avg": np.average(lengths[group]),
                        "length_std": np.std(lengths[group]),
                        "num_md_sims": num_md_sims,
                        "num_snaps": num_md_snaps,
                        "avg_snaps": num_md_snaps / group_total_num_md_sims,
                        "priority": throughput_md_avg * (num_md_snaps / self.parameters["snapshots_per_simulation"]) *
                        (num_md_sims / group_total_num_md_sims)
                    }
                else:
                    properties[group] = {}

        # overall statistics
        num_md_sims = len(self.tunnel_profiles4md.keys())  # simulations with given SC present
        num_snaps = len(throughputs["overall"])
        throughput_avg = np.average(throughputs["overall"])

        properties["overall"] = {
            "throughput_avg": throughput_avg,
            "throughput_std": np.std(throughputs["overall"]),
            "radii_avg": np.average(radii["overall"]),
            "radii_std": np.std(radii["overall"]),
            "radii_max": np.max(radii["overall"]),
            "curvature_avg": np.average(curvatures["overall"]),
            "curvature_std": np.std(curvatures["overall"]),
            "length_avg": np.average(lengths["overall"]),
            "length_std": np.std(lengths["overall"]),
            "num_md_sims": num_md_sims,
            "num_snaps": num_snaps,
            "avg_snaps": num_snaps / total_num_md_sims,
            "priority": throughput_avg * (num_snaps / self.parameters["snapshots_per_simulation"]) *
            (num_md_sims / total_num_md_sims)
        }

        return properties

    def get_bottleneck_residues_frequency(self) -> Dict[str, Dict[str, float]]:
        """
        Compute overall bottleneck residues frequency of supercluster (SC) based on its cumulative bottleneck data
        NOTE that this assumes residue numbering equivalency across analyzed simulations
        :return: bottleneck residues frequency of SC
        """

        residues_freq = dict()
        residues_freq["overall"] = dict()

        if not self.passed_filter:
            return residues_freq

        for md_label, tunnel_profile4md in self.tunnel_profiles4md.items():
            residues_freq[md_label] = tunnel_profile4md.get_bottleneck_residues_frequency()
            for reside, freq in residues_freq[md_label].items():
                if reside not in residues_freq["overall"].keys():
                    residues_freq["overall"][reside] = 0
                residues_freq["overall"][reside] += residues_freq[md_label][reside]
            if not self.parameters["perform_comparative_analysis"]:
                del residues_freq[md_label]

        if self.parameters["perform_comparative_analysis"] \
                and self.parameters["comparative_groups_definition"] is not None:
            for group, md_labels in self.parameters["comparative_groups_definition"].items():
                residues_freq[group] = dict()
                for md_label in md_labels:
                    if md_label not in self.tunnel_profiles4md.keys():  # this SC actually contains data from this MD
                        continue
                    for reside, freq in residues_freq[md_label].items():
                        if reside not in residues_freq[group].keys():
                            residues_freq[group][reside] = 0
                        residues_freq[group][reside] += residues_freq[md_label][reside]
                    # remove individual md record
                    del residues_freq[md_label]

        return residues_freq

    def has_no_tunnels(self):
        """
        Test if this profile has no valid tunnels
        """

        if len(self.tunnel_profiles4md.keys()) == 0:
            return True
        return False

    def filter_clusters(self, active_filters: dict, total_num_md_sims: int = 1) -> Dict[str, List[int]]:
        """
        Filtering of tunnels and their clusters and respective MD simulations; those clusters and MDs that do not
        retain any tunnel are removed from this cumulative tunnel profile; possibly flagging whole SC as
        failed (self.passed_filter = False)
        :param active_filters: filters to be applied (created by define_filters() function )
        :param total_num_md_sims: number of all input MD simulation
        :return: mapping of tunnel clusters from MD simulations that are valid after filtering
        """

        # reset the cluster content
        self.tunnel_clusters = dict()
        num_snaps = 0
        empty_md_profiles4removal = list()

        # filter tunnel profiles from individual MD simulations
        for md_label, tunnel_profile4md in self.tunnel_profiles4md.items():
            tunnel_profile4md.filter_tunnels(active_filters)
            count_snapshots = tunnel_profile4md.count_tunnels()  # in tunnel_profile we have max 1 tunnel per snapshot
            if count_snapshots == 0:
                empty_md_profiles4removal.append(md_label)
            else:
                num_snaps += count_snapshots

        # remove empty profiles
        for md_label in empty_md_profiles4removal:
            del self.tunnel_profiles4md[md_label]

        # does supercluster fulfil the filters
        if self.has_no_tunnels():
            logger.debug("Supercluster {:d} does not contain any tunnel fulfilling "
                         "the applied filters.".format(self.super_cluster_id))
            self.passed_filter = False
        else:
            num_md_sims = len(self.tunnel_profiles4md.keys())
            avg_num_snaps = num_snaps / total_num_md_sims

            if num_md_sims < active_filters["min_sims_num"] or num_snaps < active_filters["min_snapshots_num"] \
                    or avg_num_snaps < active_filters["min_avg_snapshots_num"]:
                logger.debug("Supercluster {:d} occurred only in {:d} simulations and in {:d} snapshots ({:.2f} on "
                             "average) -> skipping.".format(self.super_cluster_id, num_md_sims, num_snaps,
                                                            avg_num_snaps))
                self.passed_filter = False
            else:
                self.passed_filter = True
                # generate mapping of tunnel clusters remaining after filtering
                for md_label, tunnel_profile4md in self.tunnel_profiles4md.items():
                    self.tunnel_clusters[md_label] = tunnel_profile4md.enumerate_caver_cluster_ids()

        return self.tunnel_clusters

    def load_networks(self) -> Dict[str, List[int]]:
        """
        Load all tunnel profiles for all MD simulations that contribute to the parent supercluster(SC)
        :return: mapping of tunnel clusters from MD simulations that are valid after keeping only single tunnel
        per snapshot
        """

        for md_label in self.tunnel_clusters.keys():
            in_file = os.path.join(self.dump_file_root_folder, md_label + "_caver.dump")
            utils.test_file(in_file)
            self.tunnel_profiles4md[md_label] = TunnelProfile4MD(md_label, self.tunnel_clusters[md_label], in_file,
                                                                 self.parameters)
            self.tunnel_profiles4md[md_label].load_network()

        # generate mapping of tunnel clusters remaining after keeping only single tunnel per snapshot
        _tunnel_clusters = dict()
        for md_label, tunnel_profile4md in self.tunnel_profiles4md.items():
            _tunnel_clusters[md_label] = tunnel_profile4md.enumerate_caver_cluster_ids()

        return _tunnel_clusters

    def get_property_time_evolution_data(self, property_name: str, md_labels: List[str],
                                         missing_value_default: float = 0) -> Dict[str, np.ndarray]:
        """
        For each MD simulation return array containing values of given tunnel property for each simulation frame
        :param property_name: name of property to extract
        :param md_labels: list of all MD simulations contributing to the original supercluster
        :param missing_value_default: value to be used for frames where tunnels are missing or invalid in this cluster
        :return: mapping of tunnel property values for each MD simulation
        """

        data4md_label = dict()

        for md_label in md_labels:
            if md_label in self.tunnel_profiles4md.keys():
                tunnel_profile4md = self.tunnel_profiles4md[md_label]
                data4md_label[md_label] = tunnel_profile4md.get_property_time_evolution_data(property_name,
                                                                                             missing_value_default)
            else:  # no tunnels for this MD simulation after filtering
                data4md_label[md_label] = np.full(self.parameters["snapshots_per_simulation"], missing_value_default)

        return data4md_label


class TunnelProfile4MD:
    def __init__(self, md_label: str, caver_clusters: List[int], dump_file: str, parameters: dict):
        """
        Class to store and process the tunnel profile of tunnels belonging to caver clusters of given MD simulation;
        there can be multiple such profiles per one cumulative tunnel profile of parent supercluster (SC)
        :param md_label: name of folder with the source MD simulation data
        :param caver_clusters: list of valid clusters from this MD simulation belonging to parent SC
        :param dump_file: dump file of original TunnelCluster from TunnelNetwork of this MD simulation
        :param parameters: job configuration parameters
        """

        self.md_label = md_label
        self.records: Dict[int, Tunnel] = dict()  # snapshot_ID with tunnel -> Tunnel object
        self.dump_file = dump_file
        self.parameters = parameters
        self.sc_caver_clusters = caver_clusters

    def load_network(self):
        """
        Extract all clusters from this MD simulation belonging to parent SC and create tunnel records single (or none)
        tunnel per snapshot to represent parent SC in given snapshot, here also distances of tunnel spheres are adjusted
        to general starting point at the origin [0,0,0]
        """

        with open(self.dump_file, "rb") as in_stream:
            clusters: List[TunnelCluster] = pickle.load(in_stream)

        for cluster in clusters:  # load all clusters from this MD simulation
            cluster.parameters.update(self.parameters)
            if cluster.cluster_id in self.sc_caver_clusters:  # cluster belongs to supercluster
                for snapshot_id, loaded_tunnel in cluster.tunnels.items():
                    try:
                        # calculate actual distances from the general starting point (0,0,0) shared across
                        # transformed MD simulations
                        loaded_tunnel.spheres_data[:, 3] = einsum_dist(loaded_tunnel.spheres_data[:, 0:3], np.array([0, 0, 0]))  # type: ignore[index]
                        if snapshot_id in self.records.keys():
                            # only single tunnel can exist in one cluster for any given snapshot
                            existing_tunnel = self.records[snapshot_id]
                            if loaded_tunnel.has_better_throughput(existing_tunnel):  # keep tunnel with better throughput
                                self.records[snapshot_id] = loaded_tunnel
                        else:
                            self.records[snapshot_id] = loaded_tunnel
                    except (TypeError, AttributeError) as e:
                        raise RuntimeError(f"Variable 'spheres_data' is not specified for tunnel in cluster {cluster.cluster_id} snapshot {snapshot_id}") from e

    def write_csv_section(self, file_handler: TextIO):
        """
        Writes tunnel data for tunnels from this profile to CSV file
        :param file_handler: file object for opened CSV file to write tunnel profile data to
        """

        for tunnel in self.records.values():
            file_handler.writelines(tunnel.get_csv_lines(self.md_label))

    def write_residues(self, file_handler: TextIO):
        """
        Writes bottleneck residues data for tunnels from this profile to CSV file
        :param file_handler: file object for opened CSV file to write bottleneck residues data to
        """

        for tunnel in self.records.values():
            file_handler.writelines(tunnel.get_bottleneck_line(self.md_label))

    def count_tunnels(self):
        """
        Count number of tunnels in this tunnel profile
        """

        return len(self.records.keys())

    def filter_tunnels(self, active_filters: dict):
        """
        Filtering of tunnels in this tunnel profiles; tunnels that failed to pass are removed for this profile
        :param active_filters: filters to be applied (created by define_filters() function)
        """

        empty_snapshots4removal = list()
        for snapshot_id, tunnel in self.records.items():
            if not tunnel.does_tunnel_pass_filters(active_filters):
                empty_snapshots4removal.append(snapshot_id)

        for snapshot_id in empty_snapshots4removal:
            del self.records[snapshot_id]

    def get_parameters(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Collect tunnel parameters (length, radius, curvature, throughput) for all tunnels present in tunnel profile
        from this MD simulation
        :return: lists of tunnel parameters
        """

        lengths = list()
        radii = list()
        curvatures = list()
        throughputs = list()

        for tunnel in self.records.values():
            length, radius, curvature, throughput = tunnel.get_parameters()
            lengths.append(length)
            radii.append(radius)
            curvatures.append(curvature)
            throughputs.append(throughput)

        return lengths, radii, curvatures, throughputs

    def get_bottleneck_residues_frequency(self) -> Dict[str, float]:
        """
        Collect bottleneck residues frequency for all tunnels present in tunnel profile from this MD simulation
        :return: frequency of bottleneck residues occurrence
        """

        residues_freq = dict()

        for tunnel in self.records.values():
            for residue in tunnel.bottleneck_residues:
                if residue in residues_freq.keys():
                    residues_freq[residue] += 1
                else:
                    residues_freq[residue] = 1

        return residues_freq

    def enumerate_caver_cluster_ids(self) -> List[int]:
        """
        Report tunnel cluster IDs that are present (valid) in tunnel profile from this MD simulation
        :return:
        """

        caver_cluster_ids = set()
        for tunnel in self.records.values():
            caver_cluster_ids.add(tunnel.caver_cluster_id)

        return list(caver_cluster_ids)

    def get_property_time_evolution_data(self, property_name: str, missing_value_default: float = 0) -> np.ndarray:
        """
        Returns array containing values of given tunnel property for each simulation frame
        :param property_name: name of property to extract
        :param missing_value_default: value to be used for frames where tunnels are missing or invalid in this cluster
        :return: array of tunnel property values adhering to filters
        """

        values = dict()
        for snapshot_id, tunnel in self.records.items():
            values[snapshot_id] = getattr(tunnel, property_name)
        array = list()
        # iterate the analysed-snapshot domain (1..N for dense/sequential CAVER output, the actual sparse
        # IDs for strided by-frame output) rather than a dense 0..N-1 frame range plus a fixed offset; the
        # profile's own snapshot IDs let the map detect the labelling
        snapshot_map = utils.SnapshotFrameMap.from_parameters(self.parameters, observed_ids=self.records.keys())
        for caver_id in snapshot_map.snapshot_ids():
            if caver_id in values.keys():
                array.append(values[caver_id])
            else:
                array.append(missing_value_default)

        return np.array(array)


def subsample_events(transport_events: Dict[str, List[Tuple[str, Tuple[str, Tuple[int, int]]]]], random_seed: int,
                     max_events: int, md_label: str = "overall",
                     comparative_groups_definition: Dict[str, List[str]] | None = None) \
        -> List[Tuple[str, str, str]]:
    """
    Randomly selects limited number of transport events for visualization, subsampled per residue type separately
    :param transport_events: evaluated information about transport events
    :param random_seed: value to initiate the random number generator
    :param max_events: maximum number of events to keep per residue type in supercluster
    :param md_label: subsampling from which simulations to prepare; by default 'overall' from all
    :param comparative_groups_definition: definition of groups and belonging MD simulations for comparative analyses
    :return: retained information on transport events as list of (md_label, path_id, resname)
    """

    # collect all events with resname info
    all_events = list()
    if md_label == "overall":
        for _md_label, paths in transport_events.items():
            for path in paths:
                resname = path[1][0].split(":")[0]
                all_events.append((_md_label, path[0], resname))
    elif comparative_groups_definition is not None and md_label in comparative_groups_definition.keys():
        for _md_label in comparative_groups_definition[md_label]:
            if _md_label in transport_events.keys():
                for path in transport_events[_md_label]:
                    resname = path[1][0].split(":")[0]
                    all_events.append((_md_label, path[0], resname))
    else:
        if md_label in transport_events.keys():
            for path in transport_events[md_label]:
                resname = path[1][0].split(":")[0]
                all_events.append((md_label, path[0], resname))

    # group by residue type and subsample each group independently
    from collections import defaultdict
    import random
    events_by_residue: Dict[str, list] = defaultdict(list)
    for event in all_events:
        events_by_residue[event[2]].append(event)

    result = list()
    for resname in sorted(events_by_residue.keys()):
        residue_events = events_by_residue[resname]
        if len(residue_events) > max_events:
            random.seed(random_seed)
            random.shuffle(residue_events)
            result.extend(residue_events[:max_events])
        else:
            result.extend(residue_events)

    return result


def define_filters(min_length: float = -1, max_length: float = -1, min_bottleneck_radius: float = -1,
                   max_bottleneck_radius: float = -1, min_curvature: float = -1, max_curvature: float = -1,
                   min_sims_num: int = -1, min_snapshots_num: int = -1, min_avg_snapshots_num: float = -1,
                   min_total_events: int = -1, min_entry_events: int = -1, min_release_events: int = -1):
    """
    Defines filters to be used for filtering; keeping them in acceptable ranges
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

    if 0 < max_length <= min_length:
        raise ValueError("max_length filter must be >  min_length filter")
    if 0 < max_bottleneck_radius <= min_bottleneck_radius:
        raise ValueError("max_radius filter must be > min_radius filter")
    if 0 < max_curvature <= min_curvature:
        raise ValueError("max_curvature filter must be > min_curvature filter")
    if 0 < max_curvature <= 1:
        raise ValueError("max_curvature filter must be > 1")
    if 0 < min_curvature < 1:
        raise ValueError("min_curvature filter must be >= 1")

    if min_length <= 0:
        pr_min_length = 0.
    else:
        pr_min_length = min_length

    if min_bottleneck_radius <= 0:
        pr_min_radius = 0.
    else:
        pr_min_radius = min_bottleneck_radius

    if min_curvature <= 1:
        pr_min_curvature = 1.
    else:
        pr_min_curvature = min_curvature

    if max_curvature < 0:
        pr_max_curvature = 999.
    else:
        pr_max_curvature = max_curvature

    if max_length < 0:
        pr_max_length = 999.
    else:
        pr_max_length = max_length

    if max_bottleneck_radius <= 0:
        pr_max_radius = 999.
    else:
        pr_max_radius = max_bottleneck_radius

    if min_avg_snapshots_num <= 0:
        pr_min_avg_snapshots_num = 0.
    else:
        pr_min_avg_snapshots_num = min_avg_snapshots_num

    if min_total_events <= 0:
        pr_min_transport_events = 0
    else:
        pr_min_transport_events = min_total_events

    if min_entry_events <= 0:
        pr_min_entry_events = 0
    else:
        pr_min_entry_events = min_entry_events

    if min_release_events <= 0:
        pr_min_release_events = 0
    else:
        pr_min_release_events = min_release_events

    if min_sims_num <= 0:
        pr_min_sims_num = 0
    else:
        pr_min_sims_num = min_sims_num

    if min_snapshots_num <= 0:
        pr_min_snapshots_num = 0
    else:
        pr_min_snapshots_num = min_snapshots_num

    return {
        "length": (pr_min_length, pr_max_length),
        "radius": (pr_min_radius, pr_max_radius),
        "curvature": (pr_min_curvature, pr_max_curvature),
        "min_sims_num": pr_min_sims_num,
        "min_snapshots_num": pr_min_snapshots_num,
        "min_avg_snapshots_num": pr_min_avg_snapshots_num,
        "min_transport_events": pr_min_transport_events,
        "min_entry_events": pr_min_entry_events,
        "min_release_events": pr_min_release_events
    }


def get_md_membership4groups(comparative_groups_definition: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Converts group definitions to membership of MD simulations to groups
    :param comparative_groups_definition: definition of groups and belonging MD simulations for comparative analyses
    :return: membership of MD simulations to groups
    """

    membership = dict()
    for group, md_labels in comparative_groups_definition.items():
        for md_label in md_labels:
            if md_label in membership.keys():
                raise ValueError("Folder '{}' is assigned to multiple groups. Please check "
                                 "'comparative_groups_definition' parameter in the config file.\n".format(md_label))
            membership[md_label] = group

    return membership
