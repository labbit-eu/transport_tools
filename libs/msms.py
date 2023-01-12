# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2022  Jan Brezovsky, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta <janbre@amu.edu.pl>
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

__version__ = '0.9.5'
__author__ = 'Jan Brezovsky, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta'
__mail__ = 'janbre@amu.edu.pl'

import os
import tempfile
import subprocess
import numpy as np
from typing import List, Tuple
import transport_tools.libs.utils as tt_utils


def _convert_msms_to_cgo(meshfile: str, color_id: int or None) -> List[float]:
    """
    Converts mesh file generated by msms program to complied graphics object for Pymol
    :param meshfile: location of meshfile from msms program
    :param color_id: Pymol color id
    :return: CGO of spheres surface
    """

    with open(meshfile + ".vert", "r") as instream:
        instream.readline()
        instream.readline()
        vertnum = int(instream.readline().split()[0])
        vertices = np.zeros([vertnum, 3])
        normals = np.zeros([vertnum, 3])
        i = 0
        for line in instream:
            chunks = [float(x) for x in line.strip().split()]
            vertices[i,:] = chunks[:3]
            normals[i,:] = chunks[3:6]
            i += 1
    # Create and populate the cgo to use in pymol
    cgo_obj = [2.0, 4.0]
    cgo_obj.extend([6.0, *tt_utils.get_caver_color(color_id)])
    with open(meshfile + ".face", "r") as instream:
        instream.readline()
        instream.readline()
        instream.readline()
        for line in instream:
            chunks = [int(x)-1 for x in line.strip().split()]
            n = np.zeros(3)
            n += normals[chunks[0]]
            n += normals[chunks[1]]
            n += normals[chunks[2]]
            n /= 3.0
            for i in range(3):
                cgo_obj.append(5.0)
                cgo_obj.extend([n[0], n[1], n[2]])
                cgo_obj.append(4.0)
                cgo_obj.append(vertices[chunks[i]][0])
                cgo_obj.append(vertices[chunks[i]][1])
                cgo_obj.append(vertices[chunks[i]][2])
    cgo_obj.append(3.0)
    return cgo_obj


def msms_surface(msms_binary_path: str, spheres: List[Tuple[np.array, float]], color_id: int or None,
                 probe_radius: float = 5.0) -> List[float]:
    """
    Converts spheres (tuples that contain XYZ coords and radius) to surface complied graphics object for Pymol using
    msms program from https://ccsb.scripps.edu/msms/
    :param msms_binary_path: path to binary file of msms program
    :param spheres: list of spheres to generate their approximate surface
    :param color_id: Pymol color id
    :param probe_radius: probe sphere radius [in Angstroms] used by msms program
    :return: CGO of spheres surface
    """

    def _save_xyzr_file(spheres: List[Tuple[np.array, float]], filename: str):
        """
        Converts list of spheres (tuples that contain XYZ coords and radius) to file suitable as input for msms program
        :param spheres: list of spheres to save to file
        :param filename: location of file with spheres set for msms program
        """

        output = ""
        for sphere in spheres:
            output += "{:>7.3f} {:>7.3f} {:>7.3f} {:>5.3f}\n".format(sphere[0][0], sphere[0][1], sphere[0][2],
                                                                     sphere[1])
        with open(filename, "w") as outstream:
            outstream.write(output)

    with tempfile.TemporaryDirectory(prefix="TT_") as tmp:
        xyzr_file = os.path.join(tmp, next(tempfile._get_candidate_names()))
        xyzr_file += ".xyzr"
        _save_xyzr_file(spheres, xyzr_file)
        meshfile = os.path.join(tmp, next(tempfile._get_candidate_names()))
        command = [msms_binary_path, "-probe_radius", str(probe_radius), "-if", xyzr_file, "-of", meshfile]
        output = subprocess.run(command, capture_output= True,  universal_newlines=True)

        return _convert_msms_to_cgo(meshfile, color_id)


def filter_spheres(spheres: List[Tuple[np.array, float]]) -> List[Tuple[np.array, float]]:
    """
    Filters possibly duplicated spheres (tuples that contain XYZ coords and radius) from the list
    :param spheres: list of spheres to filter
    :return: list of filtered spheres
    """

    tmp_spheres = []
    for sphere in spheres:
        tmp_spheres.append(tuple([*sphere[0], sphere[1]]))
    filtered_spheres = set()
    for sphere in tmp_spheres:
        filtered_spheres.add(sphere)
    final_spheres = []
    for sphere in filtered_spheres:
        s = (np.array(sphere[:3]), sphere[3])
        final_spheres.append(s)
    return final_spheres
