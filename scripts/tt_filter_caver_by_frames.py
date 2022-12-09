#!/usr/bin/env python3
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

__version__ = '0.9.4'
__author__ = 'Jan Brezovsky, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta'
__mail__ = 'janbre@amu.edu.pl'

import os
import time
import shutil
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass(eq=False, repr=True, unsafe_hash=False)
class CVTunnel:
    """
    Class that represents a tunnel, parsed from a Caver tunnel profile
    frame: the name of the odb file containing information about the frame number
    cluster_id: the cluster id to which the tunnel belongs
    tunnel_id: the tunnel id
    throughput: the throughput of the tunnel
    nodes: list of CVNode objects that form the tunnel
    connections: list of strings to repesent the bonds in the tunnel in pdb format
    csvlines: original content of the csv input file, useful to recreate the csv profile
    """
    frame: str = field(repr=True, hash=True)
    cluster_id: int = field(repr=True, hash=True)
    tunnel_id: int = field(repr=True, hash=True)
    throughput: float = field(repr=False, hash=False)
    nodes: List = field(repr=False, default_factory=list, hash=False)
    connections: List = field(repr=False, default_factory=list, hash=False)
    csvlines: List = field(repr=False, default_factory=list, hash=False)
    
    def __eq__(self, other):
        a = int(self.frame.split(".")[-2])
        b = int(other.frame.split(".")[-2])
        return a == b
    def __lt__(self, other):
        a = int(self.frame.split(".")[-2])
        b = int(other.frame.split(".")[-2])
        return a < b
    def __str__(self):
        outstr = self.frame + "," + str(self.cluster_id) + "," + self.csvlines[0]
        for csvline in self.csvlines[1:]:
            outstr += self.frame + "," + str(self.cluster_id) + "," + csvline
        return outstr
    
    def _build_connections(self, offset):
        """
        Create the bonds definition in pdb format for the cluster
        """
        self.connections = []
        for n in self.nodes:
            self.connections.append("CONECT{:>5d}{:>5d}".format(n.node_id+offset, n.node_id+offset+1))
        self.connections.pop()
    
    def to_pdb(self, offset=0) -> str:
        """
        Function to export the cluster in pdb format, useful for visualization
        """
        output = ""
        for node in self.nodes:
            output += node.to_pdb(offset) + "\n"
        self._build_connections(offset)
        for connection in self.connections:
            output += connection + "\n"
        output += "\n"
        return output

@dataclass(order=True, repr=True, unsafe_hash=False)
class CVNode:
    """
    Class representing a node in a tunnel
    node_id: id of the current node
    node_x: X coordinate of the node
    node_y: Y coordinate of the node
    node_z: Z coordinate of the node
    node_radius: radius of the node
    """
    node_id: int = field(repr=True, hash=True)
    node_x: float = field(repr=True, hash=True)
    node_y: float = field(repr=True, hash=True)
    node_z: float = field(repr=True, hash=True)
    node_radius: float = field(repr=True, hash=True)
    
    def to_pdb(self, offset=0) -> str:
        """
        Function to export the node in pdb format, useful for visualization
        """
        return "ATOM  {:>5d}  H   CVT A   1    {:>8.3f}{:>8.3f}{:>8.3f}  0.00{:>6.2f}              ".format(
                self.node_id + offset, self.node_x, self.node_y, self.node_z, self.node_radius)

@dataclass(eq=False, repr=True, unsafe_hash=False)
class CVBottleneck:
    """
    Class representing the bottleneck info of a tunnel
    frame: the name of the odb file containing information about the frame number
    cluster_id: the cluster id to which the bottleneck belongs
    tunnel_id: the tunnel id of the bottleneck
    content: the original content of the bottleneck from the input csv file
    """
    frame: str = field(repr=True, hash=True)
    cluster_id: int = field(repr=True, hash=True)
    tunnel_id: int = field(repr=True, hash=True)
    content: str = field(repr=False, hash=False, init=False)
    
    def __eq__(self, other):
        a = int(self.frame.split(".")[-2])
        b = int(other.frame.split(".")[-2])
        return a == b
    def __lt__(self, other):
        a = int(self.frame.split(".")[-2])
        b = int(other.frame.split(".")[-2])
        return a < b
    def __str__(self):
        outstr = self.frame + "," + str(self.cluster_id) + "," + self.content
        return outstr

@dataclass(eq=False, repr=True, unsafe_hash=False)
class CVCluster:
    """
    Class representing a cluster of tunnels
    cluster_id: the cluster id
    original_id: the original cluster id assigned by Caver. This value can change
                 depending on the reordering of the tunnels if a filter is applied
    csvlines: list of lines from the original input csv file, useful to recreate the csv profile
    tunnels: list of CVTunnel objects representing all the tunnels that are present in the cluster
    """
    cluster_id: int = field(repr=True, hash=True)
    original_id: int = field(repr=True)
    csvlines: List = field(repr=False, default_factory=list, hash=False)
    tunnels: List = field(repr=False, default_factory=list, hash=False)
    
    def __eq__(self, other):
        a = int(self.frame.split(".")[-2])
        b = int(other.frame.split(".")[-2])
        return a == b
    def __lt__(self, other):
        a = int(self.frame.split(".")[-2])
        b = int(other.frame.split(".")[-2])
        return a < b
    def __str__(self):
        output = ""
        for tunnel in self.tunnels:
            output += str(tunnel)
        return output


def get_clusters(csvfile: str) -> List[CVCluster]:
    """
    Parse the Caver tunnel profiles csvfile into a list of CVCluster objects.
    :param csvfile: input csv file from Caver with the tunnel profiles
    :return: a list of CVCluster objects
    """
    def _find_pos(line: str, char: str) -> int:
        """
        Helper function to add the csvline correctly
        :param line: input csv line from the tunnel profiles
        :param char: character used to split the data, usually a comma
        :return: the position from which the data is useful for our usage
        """
        i = 0
        for j in range(2):
            i = line.find(char, i) + 1
        return i
    
    cvclusters = []
    orig_cluster = None
    with open(csvfile, "r") as csv_in:
        csv_in.readline()
        for line in csv_in:
            chunks = line.strip().split(",")
            if orig_cluster is None:
                orig_cluster = CVCluster(int(chunks[1]), int(chunks[1]))
            elif orig_cluster.cluster_id != int(chunks[1]):
                orig_cluster.tunnels.sort()
                cvclusters.append(orig_cluster)
                orig_cluster = CVCluster(int(chunks[1]), int(chunks[1]))
            cv_tunnel = CVTunnel(frame=chunks[0].strip(), cluster_id=int(chunks[1]),
                                 tunnel_id=int(chunks[2]), throughput=float(chunks[3]))
            
            x = [float(_x) for _x in line.split(",")[13:]]
            i = _find_pos(line, ",")
            cv_tunnel.csvlines.append(line[i:])
            line = csv_in.readline()
            
            y = [float(_y) for _y in line.split(",")[13:]]
            i = _find_pos(line, ",")
            cv_tunnel.csvlines.append(line[i:])
            line = csv_in.readline()
            
            z = [float(_z) for _z in line.split(",")[13:]]
            i = _find_pos(line, ",")
            cv_tunnel.csvlines.append(line[i:])
            line = csv_in.readline()
            
            for _ in range(2):
                i = _find_pos(line, ",")
                cv_tunnel.csvlines.append(line[i:])
                line = csv_in.readline()
            
            r = [float(_r) for _r in line.split(",")[13:]]
            i = _find_pos(line, ",")
            cv_tunnel.csvlines.append(line[i:])
            line = csv_in.readline()
            
            i = _find_pos(line, ",")
            cv_tunnel.csvlines.append(line[i:])
            
            for i in range(len(x)):
                cv_tunnel.nodes.append(CVNode(i+1, x[i], y[i], z[i], r[i]))
            orig_cluster.tunnels.append(cv_tunnel)
            
        orig_cluster.tunnels.sort()
        cvclusters.append(orig_cluster)
    return cvclusters

def get_bottlenecks(csvfile: str) -> Dict[Tuple[str, int, int], CVBottleneck]:
    """
    Parse the Caver bottleneck profiles csvfile into a dictionary of CVBottleneck objects,
    using as key a tuple of (pdb name, cluster id, tunnel id)
    :param csvfile: input csv file from Caver with the bottlenecks
    :return: a dictionary of CVBottleneck objects
    """
    bottlenecks = {}
    with open(csvfile, "r") as csvinf:
        csvinf.readline()
        csvinf.readline()
        for line in csvinf:
            chunks = line.strip().split(",")
            bneck = CVBottleneck(frame=chunks[0].strip(), cluster_id=int(chunks[1]), tunnel_id=int(chunks[2]))
            i = 0
            for j in range(2):
                i = line.find(",", i) + 1
            bneck.content = line[i:]
            bottlenecks[(chunks[0], int(chunks[1]), int(chunks[2]))] = bneck
    return bottlenecks

def prioritize_clusters(clusters: List[CVCluster], num_frames: int) -> List[CVCluster]:
    """
    Reorders the clusters depending on their throughput
    :param clusters: list of CVCluster objects after filtering
    :param num_frames: total number of frames of the MD trajectory
    :return: list of sorted CVCluster objects with renumbered ids
    """
    priority_list = []
    for cluster in clusters:
        priority = 0.0
        for tunnel in cluster.tunnels:
            priority += tunnel.throughput
        priority /= num_frames
        priority_list.append((cluster, priority))
    priority_list.sort(key=lambda x: x[1], reverse=True)
    new_clusters = []
    for i, element in enumerate(priority_list):
        element[0].cluster_id = i+1
        for tunnel in element[0].tunnels:
            tunnel.cluster_id = i+1
        new_clusters.append(element[0])
    return new_clusters

def save_clusters(clusters: List[CVCluster], outfile: str):
    """
    Write the new clusters to a csv file following the Caver format
    :param clusters: list of CVCluster objects
    :param outfile: path to the csv file to write
    """
    with open(outfile, "w") as fout:
        fout.write("Snapshot, Tunnel cluster, Tunnel, Throughput, Cost, Bottleneck radius, Average"
                      " R error bound, Max. R error bound, Bottleneck R error bound, Curvature,"
                      " Length, , Axis, Values...\n")
        for cluster in clusters:
            fout.write(str(cluster))

def refactor_bottlenecks(clusters: List[CVCluster], infile: str, outfile: str):
    """
    Reorders the bottlenecks depending on the new order of the CVCluster list.
    During the reordering, the cluster ids of the bottlenecks will be updated.
    :param clusters: list of CVCluster objects
    :param infile: path to the input csv file with the bottleneck information
    :param outfile: path to the output csv file with the bottleneck information
    """
    output = []
    bottlenecks = get_bottlenecks(infile)
    for cluster in clusters:
        for tunnel in cluster.tunnels:
            key = (tunnel.frame, cluster.original_id, tunnel.tunnel_id)
            try:
                bneck = bottlenecks[key]
                bneck.cluster_id = cluster.cluster_id
                output.append(bneck)
            except KeyError:
                print("Error, missing bottleneck for tunnel:", tunnel)
                print("\twith key:", key)
    with open(outfile, "w") as outstream:
        outstream.write("Bottleneck residues:, Residues in distance <= 3.0 A from the bottleneck, sorted from the closest.\n")
        outstream.write("Snapshot,Tunnel cluster,Tunnel,Throughput,Cost,Bottleneck X,Bottleneck Y,Bottleneck Z,Bottleneck R,Bottleneck residues\n")
        for b in output:
            outstream.write(str(b))

def save_cluster_pdb(cluster: CVCluster, outdir: str):
    """
    Exports the data from a CVCluster object to a pdb file in the specified directory
    :param cluster: the CVCluster object to export
    :param outdir: path to save the pdb file
    """
    pdb_i = 1
    pdb_out = ""
    atoms = 0
    cluster_name = os.path.join(outdir, "data", "clusters_timeless", "tun_cl_{:0>3}".format(cluster.cluster_id))
    for tunnel in cluster.tunnels:
        if atoms + len(tunnel.nodes) >= 99999:
            _name = "{}_{}.pdb".format(cluster_name, pdb_i)
            with open(_name, "w") as outstream:
                outstream.write(pdb_out)
            pdb_i += 1
            pdb_out = tunnel.to_pdb()
            atoms = len(tunnel.nodes)
            continue
        pdb_out += tunnel.to_pdb(offset=atoms)
        atoms += len(tunnel.nodes)
    _name = "{}_{}.pdb".format(cluster_name, pdb_i)
    with open(_name, "w") as outstream:
        outstream.write(pdb_out)

def make_vmd_script(number_of_clusters: int, outdir: str):
    """
    Prepares a VMD script to be used similarly to the one prepared by Caver
    :param number_of_clusters: number of clusters present in the analysis
    :param outdir: path to save the visualization files
    """
    timeless_script = """mol representation CPK
set dir "../data/clusters_timeless"
set ext "1.pdb"
set color 0
set molecule 0
"""
    color_list = ["0","7","1","10","4","27","3","5","9","11","12","13","14","15",
                  "16","17","18","19","20","21","22","23","24","25","26","28","29",
                  "30","31","32"]
    _factor = number_of_clusters // len(color_list) + 1
    colors = color_list * _factor
    colors = colors[:number_of_clusters]
    colors = " ".join(colors)
    timeless_script += "set colors [list {}]\n".format(colors)
    timeless_script += """set contents [glob -directory $dir *$ext]
set contents [lsort $contents]
foreach item $contents {
  mol load pdb $item 
  set tunnel [atomselect top "all"]
  $tunnel set radius [$tunnel get beta]
  mol modcolor 0 $molecule "ColorID" [lindex $colors $color]
  if {[expr {$color - 1 < [llength $colors]}]} {incr color}
  incr molecule 
}
source "./scripts/vmd_load_structure.tcl"
"""
    with open(os.path.join(outdir, "vmd", "scripts", "view_timeless.tcl"), "w") as fout:
        fout.write(timeless_script)
    
    _pdb = [f for f in os.listdir(os.path.join(outdir, "data")) if f.endswith(".pdb")
                     and f != "origins.pdb" and f != "v_origins.pdb"]
    if len(_pdb) != 1:
        print("Posible error copying structure pdb, please check the data folder in the output")
    with open(os.path.join(outdir, "vmd", "scripts", "vmd_load_structure.tcl"), "w") as fout:
        structure_script = "mol load pdb ../data/{}\n".format(_pdb[0])
        structure_script += """after idle { 
  mol representation NewCartoon 
  mol delrep 0 top
  mol addrep top
  mol modcolor 0 top "ColorID" 8
} 
"""
        fout.write(structure_script)
    
    with open(os.path.join(outdir, "vmd", "vmd_timeless.sh"), "w") as fout:
        fout.write("#!/bin/bash\n")
        fout.write("vmd -e scripts/view_timeless.tcl\n")
    os.chmod(os.path.join(outdir, "vmd", "vmd_timeless.sh"), 0o775)

def make_pymol_scripts(outdir: str):
    """
    Prepares a PyMOL script to be used similarly to the one prepared by Caver
    :param outdir: path to save the visualization files
    """
    _pdb = [f for f in os.listdir(os.path.join(outdir, "data")) if f.endswith(".pdb")
            and f != "origins.pdb" and f != "v_origins.pdb"]
    if len(_pdb) != 1:
        print("Posible error copying structure pdb, please check the data folder in the output")
    timeless_script = """from pymol import cmd

sys.path.insert(1, os.path.join(os.getcwd(), "modules"))
import caver

try:
    execfile("./modules/rgb.py")
except NameError:
    exec(compile(open("./modules/rgb.py", "rb").read(), "./modules/rgb.py", "exec"))

color = 1
pdb_files = [f for f in os.listdir("../data/clusters_timeless") if f.endswith("_1.pdb")]
pdb_files.sort()
name = ""
for fn in pdb_files:
    old_name = name
    name = fn

    if color < 1000 and caver.new_cluster(old_name, name):
        color += 1
    else:
        color = 1

    cmd.load("../data/clusters_timeless/" + fn, name)
    cmd.color("caver" + str(color), name)
    cmd.alter(name, "vdw=b")
    #cmd.show("spheres", name)
cmd.do("set all_states, 1")

cmd.set("two_sided_lighting", "on")
cmd.set("transparency", "0.2")

cmd.load("../data/origins.pdb", "origins")
cmd.load("../data/v_origins.pdb", "v_origins")
cmd.show("nb_spheres", "origins")
cmd.show("nb_spheres", "v_origins")

cmd.load("../data/{}", "structure")
cmd.hide("lines", "structure")
cmd.show("cartoon", "structure")
cmd.color("gray", "structure")

""".format(_pdb[0])
    with open(os.path.join(outdir, "pymol", "view_timeless.py"), "w") as fout:
        fout.write(timeless_script)
    caver_script = """def new_cluster(a, b):
    if len(a) != len(b):
        return False # do not increase color in first iteration
    else:
        ai = a[11]
        bi = b[11]
        return int(ai) + 1 != int(bi)
        # equality would indicates only a next file of the same cluster
"""
    with open(os.path.join(outdir, "pymol", "modules", "caver.py"), "w") as fout:
        fout.write(caver_script)
    rgb_cols = ["[0.0,0.0,1.0]","[0.0,1.0,0.0]","[1.0,0.0,0.0]","[0.0,1.0,1.0]","[1.0,1.0,0.0]","[1.0,0.0,1.0]",
        "[0.71,0.71,0.97]","[0.5,0.99,0.42]","[0.99,0.5,0.42]","[0.21,0.5,0.85]","[0.5,0.06,0.87]",
        "[0.0,0.87,0.5]","[0.86,0.01,0.5]","[0.58,0.82,0.0]","[0.89,0.89,0.6]","[1.0,0.41,0.84]",
        "[0.41,1.0,0.85]","[0.87,0.49,0.0]","[0.62,0.42,0.69]","[0.66,0.67,0.35]","[0.4,0.71,0.62]",
        "[0.32,1.0,0.13]","[0.72,0.3,1.0]","[1.0,0.82,0.28]","[0.96,0.21,0.25]","[0.24,0.22,1.0]",
        "[0.0,0.74,0.79]","[0.4,0.71,1.0]","[0.75,1.0,0.2]","[0.79,0.07,0.79]","[0.87,0.29,0.6]",
        "[0.14,1.0,0.73]","[0.45,0.42,1.0]","[0.86,0.59,0.66]","[0.61,0.89,0.66]","[0.0,0.36,0.99]",
        "[0.86,0.77,0.03]","[0.23,0.99,0.39]","[1.0,0.26,0.0]","[0.0,0.98,0.27]","[0.15,0.79,1.0]",
        "[0.53,0.62,0.81]","[0.88,0.61,0.22]","[0.75,0.47,0.47]","[0.73,0.89,0.42]","[0.34,0.94,0.6]",
        "[0.77,0.49,0.87]","[0.49,0.85,0.22]","[0.69,0.0,1.0]","[1.0,0.1,0.68]","[0.24,0.78,0.78]",
        "[0.94,0.23,0.99]","[0.87,0.7,0.46]","[0.01,0.6,1.0]","[0.33,0.0,1.0]","[1.0,0.42,0.17]",
        "[0.61,0.89,0.89]","[0.99,0.0,0.32]","[0.22,0.99,0.96]","[0.49,0.28,0.83]","[0.48,0.77,0.43]",
        "[1.0,0.17,0.47]","[0.91,1.0,0.36]","[0.76,0.28,0.78]","[0.7,0.79,0.18]","[0.57,1.0,0.1]",
        "[0.79,0.97,0.0]","[0.4,0.47,0.79]","[0.19,0.81,0.56]","[0.88,0.33,0.4]","[0.62,0.53,1.0]",
        "[1.0,0.45,0.62]","[0.43,0.8,0.81]","[1.0,0.73,0.65]","[0.91,0.64,0.84]","[1.0,0.63,0.06]",
        "[0.44,0.19,1.0]","[0.94,0.97,0.18]","[0.23,0.61,1.0]","[0.89,0.41,1.0]","[0.05,0.18,0.99]",
        "[0.33,0.64,0.84]","[0.94,0.24,0.8]","[0.26,0.4,1.0]","[0.01,0.93,0.83]","[0.13,0.99,0.13]",
        "[0.42,0.91,0.0]","[0.66,0.16,0.88]","[0.78,0.13,0.62]","[0.37,0.9,1.0]","[0.13,0.99,0.54]",
        "[0.74,1.0,0.59]","[0.32,0.84,0.45]","[1.0,0.86,0.46]","[0.96,0.0,0.82]","[0.04,0.57,0.82]",
        "[0.62,0.4,0.88]","[0.86,0.42,0.73]","[0.85,0.08,0.96]","[0.82,0.79,0.31]","[0.05,0.86,0.67]",
        "[0.37,0.96,0.29]","[0.61,0.95,0.29]","[0.48,0.99,0.69]","[0.82,0.48,0.31]","[1.0,0.08,0.15]",
        "[0.63,0.0,0.78]","[0.78,0.62,0.07]","[1.0,0.65,0.34]","[0.55,0.77,1.0]","[1.0,0.82,0.11]",
        "[0.16,0.05,0.99]","[0.48,0.88,0.55]","[0.63,0.8,0.52]","[0.98,0.61,0.54]","[0.34,0.3,0.9]",
        "[0.14,0.68,0.88]","[0.27,0.94,0.81]","[0.11,0.48,0.99]","[0.0,0.84,0.97]","[0.88,0.0,0.66]",
        "[0.71,0.31,0.63]","[1.0,0.54,0.75]","[0.56,0.3,1.0]","[0.45,0.57,0.94]","[0.13,0.69,0.72]",
        "[0.22,0.98,0.0]","[1.0,0.33,0.51]","[0.57,0.11,1.0]","[0.78,0.8,0.54]","[0.67,0.16,0.73]",
        "[0.05,1.0,0.42]","[0.75,0.46,0.63]","[0.6,0.8,0.33]","[0.14,0.32,0.94]","[0.59,1.0,0.55]",
        "[0.85,0.85,0.17]","[0.28,0.79,0.92]","[0.78,0.58,1.0]","[0.14,0.88,0.88]","[1.0,0.36,0.32]",
        "[0.89,0.44,0.51]","[0.85,0.18,0.5]","[0.73,0.65,0.21]","[1.0,0.3,0.67]","[0.61,0.99,0.77]",
        "[0.7,0.72,0.0]","[0.0,1.0,0.63]","[0.23,0.94,0.24]","[0.81,0.22,0.91]","[0.28,0.66,0.7]",
        "[0.68,0.91,0.08]","[0.42,0.59,0.72]","[1.0,0.41,0.02]","[0.8,0.62,0.35]","[0.5,0.8,0.68]",
        "[0.35,0.83,0.69]","[0.36,0.99,0.46]","[0.98,0.13,0.89]","[1.0,0.01,0.56]","[0.67,0.6,0.88]",
        "[0.99,0.7,0.2]","[0.86,0.35,0.87]","[0.14,0.89,0.44]","[0.99,0.53,0.27]","[0.87,0.48,0.15]",
        "[0.58,0.75,0.86]","[0.84,0.58,0.52]","[0.85,0.99,0.49]","[0.62,0.29,0.76]","[0.74,0.44,1.0]",
        "[0.49,1.0,0.22]","[0.32,0.52,0.94]","[0.92,0.51,0.9]","[0.93,0.12,0.36]","[0.47,0.0,0.99]",
        "[0.95,0.29,0.14]","[0.91,0.9,0.03]","[0.21,0.89,0.68]","[0.86,0.85,0.43]","[0.73,0.14,1.0]",
        "[0.54,0.49,0.8]","[0.87,0.15,0.71]","[0.74,0.68,0.47]","[0.33,0.13,0.94]","[0.77,1.0,0.34]",
        "[0.73,0.01,0.68]","[0.47,0.89,0.91]","[0.56,0.65,0.95]","[0.01,0.46,0.89]","[0.99,0.74,0.51]",
        "[0.34,1.0,0.72]","[0.64,1.0,0.42]","[0.02,0.71,0.93]","[0.72,0.88,0.28]","[0.72,0.41,0.78]",
        "[0.46,0.88,0.36]","[0.97,0.2,0.59]","[0.79,0.35,0.51]","[0.99,0.13,0.03]","[0.88,0.7,0.59]",
        "[0.52,0.89,0.77]","[0.83,0.72,0.16]","[0.58,0.82,0.14]","[0.12,0.94,0.32]","[0.86,0.92,0.28]",
        "[0.89,0.39,0.24]","[0.1,1.0,0.91]","[0.0,0.99,0.14]","[0.25,0.93,0.51]","[0.99,0.75,0.0]",
        "[0.57,0.88,0.43]","[0.42,0.91,0.13]","[0.32,0.79,0.57]","[0.62,0.04,0.9]","[0.3,0.42,0.86]",
        "[0.53,0.19,0.9]","[0.11,0.81,0.77]","[0.86,0.54,0.78]","[0.76,0.84,0.0]","[0.48,0.4,0.87]",
        "[0.89,0.62,0.0]","[0.71,0.96,0.71]","[0.46,0.71,0.89]","[0.99,0.04,0.44]","[0.44,1.0,0.55]",
        "[0.64,0.99,0.0]","[0.84,1.0,0.11]","[0.2,0.61,0.79]","[0.11,0.91,1.0]","[0.9,0.09,0.58]",
        "[0.96,0.76,0.39]","[0.7,0.77,0.39]","[0.99,0.25,0.37]","[0.76,0.0,0.9]","[0.98,0.51,0.09]",
        "[1.0,0.29,0.89]","[0.12,0.66,0.99]","[0.62,0.41,1.0]","[0.36,0.88,0.87]","[0.94,0.65,0.73]",
        "[0.38,0.71,0.74]","[0.62,0.28,0.9]","[0.74,0.35,0.89]","[0.91,0.6,0.41]","[0.84,0.3,1.0]",
        "[0.91,0.71,0.28]","[0.54,0.49,0.93]","[0.86,0.29,0.72]","[0.01,0.74,0.67]","[0.24,0.88,1.0]",
        "[0.7,0.89,0.56]","[0.71,0.57,0.42]","[0.23,1.0,0.62]","[0.98,0.94,0.29]","[0.59,0.72,0.43]",
        "[0.48,0.76,0.55]","[0.88,0.14,0.84]","[0.44,0.31,0.96]","[0.12,0.57,0.91]","[0.35,1.0,0.0]",
        "[0.63,0.21,0.99]","[0.66,0.51,0.8]","[0.53,0.37,0.77]","[0.51,0.99,0.0]","[0.88,0.48,0.63]",
        "[0.21,0.75,0.67]","[0.8,0.61,0.89]","[0.33,0.99,0.93]","[0.98,0.41,0.73]","[0.59,0.11,0.8]",
        "[0.4,0.91,0.77]","[0.31,0.53,0.79]","[0.28,0.9,0.35]","[0.78,0.23,0.68]","[0.1,0.88,0.56]",
        "[0.52,0.89,0.06]","[0.9,0.8,0.53]","[0.44,0.89,0.65]","[0.53,0.97,0.85]","[0.65,0.91,0.19]",
        "[0.64,0.73,0.25]","[0.97,0.11,1.0]","[0.85,0.5,0.42]","[0.3,0.68,0.94]","[0.75,0.17,0.81]",
        "[0.41,0.21,0.89]","[0.99,0.1,0.26]","[0.94,0.72,0.1]","[0.97,0.55,0.64]","[0.92,0.39,0.09]",
        "[0.03,0.99,0.74]","[0.99,0.19,0.13]","[0.76,0.5,0.73]","[0.24,0.6,0.89]","[0.92,0.44,0.36]",
        "[0.65,0.8,0.94]","[0.0,0.99,0.52]","[0.9,0.6,0.11]","[0.14,0.42,0.9]","[0.33,0.75,0.83]",
        "[0.51,0.73,0.77]","[0.05,0.82,0.87]","[0.45,0.81,0.99]","[0.71,0.69,0.12]","[0.11,1.0,0.0]",
        "[0.99,0.58,0.18]","[0.61,1.0,0.65]","[0.81,0.37,0.65]","[0.9,0.28,0.49]","[0.36,0.61,1.0]",
        "[0.44,0.57,0.82]","[0.97,0.0,0.71]","[1.0,0.14,0.78]","[0.22,0.51,0.99]","[0.92,0.01,0.92]",
        "[0.74,0.99,0.48]","[0.1,0.78,0.62]","[1.0,0.93,0.09]","[0.67,0.49,0.91]","[0.18,0.78,0.88]",
        "[0.92,0.78,0.21]","[0.76,0.11,0.9]","[0.91,0.3,0.3]","[0.79,0.56,0.26]","[0.18,1.0,0.84]",
        "[0.42,0.9,0.46]","[0.67,0.8,0.08]","[0.79,0.91,0.63]","[0.92,0.37,0.65]","[0.4,0.04,0.92]",
        "[0.36,0.37,0.99]","[0.35,0.26,1.0]","[0.4,0.78,0.49]","[0.17,0.91,0.78]","[0.78,0.9,0.09]",
        "[0.1,0.96,0.64]","[0.99,0.5,0.52]","[0.01,0.48,1.0]","[0.93,0.87,0.35]","[0.75,0.7,0.31]",
        "[0.97,0.21,0.7]","[0.9,0.55,0.31]","[0.41,0.48,0.9]","[0.24,0.34,0.91]","[0.24,0.69,0.84]",
        "[0.77,0.78,0.1]","[0.66,1.0,0.14]","[0.92,0.8,0.63]","[0.58,0.0,1.0]","[0.0,0.81,0.59]",
        "[1.0,0.39,0.42]","[0.43,1.0,0.07]","[0.54,0.92,0.17]","[0.68,0.62,1.0]","[0.9,1.0,0.0]",
        "[0.87,0.19,0.62]","[0.57,0.21,0.8]","[0.69,0.08,0.83]","[0.8,0.9,0.5]","[0.33,0.92,0.2]",
        "[1.0,0.88,0.2]","[0.0,0.84,0.76]","[0.89,0.05,0.76]","[0.59,0.89,0.54]","[0.87,0.45,0.83]",
        "[0.1,0.99,0.23]","[0.15,0.16,0.99]","[0.47,0.1,0.96]","[0.85,0.18,1.0]","[0.85,0.0,0.84]",
        "[1.0,0.86,0.01]","[0.92,0.93,0.45]","[0.89,0.22,0.4]","[1.0,0.11,0.55]","[0.25,0.11,1.0]",
        "[0.38,0.79,0.91]","[0.8,0.77,0.43]","[0.78,0.24,0.57]","[0.92,0.11,0.48]","[0.99,0.65,0.45]",
        "[0.94,0.34,0.8]","[0.95,0.33,0.97]","[0.07,0.66,0.8]","[0.47,0.97,0.32]","[0.99,0.55,0.0]",
        "[0.99,0.3,0.23]","[0.81,0.41,0.93]","[0.22,0.72,0.99]","[0.53,0.93,0.62]","[0.42,0.81,0.6]",
        "[0.81,0.4,0.43]","[0.91,0.25,0.9]","[0.53,0.56,1.0]","[0.75,0.88,0.19]","[0.72,0.24,0.87]",
        "[0.64,0.9,0.36]","[0.84,0.23,0.81]","[0.02,0.93,0.93]","[0.24,0.98,0.72]","[0.15,0.99,0.45]",
        "[0.59,0.83,0.23]","[0.85,1.0,0.21]","[0.26,0.89,0.9]","[0.81,0.57,0.16]","[0.55,0.38,0.94]",
        "[0.07,0.08,1.0]","[0.8,0.88,0.35]","[0.81,0.02,0.59]","[0.85,0.5,0.98]","[0.31,0.85,0.79]",
        "[0.4,0.51,1.0]","[0.05,0.28,0.97]","[0.27,0.87,0.59]","[0.64,0.69,0.91]","[0.98,0.0,0.22]",
        "[0.79,0.01,1.0]","[1.0,0.51,0.84]","[0.79,0.78,0.22]","[0.39,1.0,0.19]","[0.47,0.51,0.73]",
        "[0.54,0.88,0.3]","[0.57,0.57,0.89]","[0.68,0.4,0.62]","[0.81,0.69,0.0]","[0.0,0.55,0.91]",
        "[0.75,0.1,0.71]","[0.31,0.74,0.65]","[0.66,0.93,0.49]","[0.69,0.35,0.71]","[0.69,1.0,0.28]",
        "[0.81,0.48,0.55]","[0.92,0.41,0.9]","[0.34,0.79,1.0]","[0.01,0.91,0.59]","[0.91,0.49,0.24]",
        "[0.16,0.39,1.0]","[0.84,0.71,0.36]","[0.55,0.81,0.6]","[0.51,0.8,0.91]","[0.23,0.98,0.15]",
        "[0.71,0.99,0.06]","[0.39,0.37,0.85]","[0.53,0.2,1.0]","[0.62,0.71,1.0]","[0.37,0.87,0.54]",
        "[0.09,0.97,0.82]","[0.58,1.0,0.21]","[0.41,0.87,0.27]","[1.0,0.0,0.1]","[0.49,0.7,1.0]",
        "[0.95,0.65,0.63]","[0.81,0.36,0.79]","[0.72,0.78,0.27]","[0.69,0.25,0.7]","[0.9,0.16,0.93]",
        "[0.92,0.5,0.71]","[0.39,0.08,1.0]","[0.28,0.99,0.31]","[0.92,0.36,0.55]","[0.96,0.63,0.26]",
        "[0.84,0.94,0.41]","[0.39,0.65,0.91]","[0.16,0.73,0.79]","[0.93,1.0,0.09]","[0.1,0.77,0.92]",
        "[0.0,0.93,0.36]","[0.61,0.9,0.77]","[0.98,0.81,0.57]","[0.18,0.91,0.59]","[0.91,0.82,0.09]",
        "[0.26,0.31,1.0]","[0.54,0.81,0.49]","[0.23,0.87,0.44]","[0.14,0.84,0.68]","[0.67,0.33,0.83]",
        "[0.6,0.91,0.02]","[0.66,0.09,0.97]","[0.1,0.49,0.86]","[1.0,0.33,0.07]","[0.71,0.82,0.48]",
        "[0.08,0.9,0.75]","[0.37,0.92,0.38]","[0.81,0.67,0.52]","[0.3,0.91,0.7]","[0.82,0.68,0.24]",
        "[0.44,0.66,0.8]","[0.5,0.8,0.34]","[0.07,0.37,0.93]","[0.23,0.44,0.92]","[0.91,0.54,0.48]",
        "[0.4,1.0,0.64]","[0.05,0.64,0.89]","[0.07,0.94,0.49]","[0.73,0.0,0.79]","[0.53,0.29,0.91]",
        "[0.62,0.43,0.79]","[0.62,0.81,0.42]","[0.57,0.96,0.37]","[0.8,0.62,0.44]","[0.55,0.81,0.79]",
        "[0.33,0.61,0.75]","[0.58,0.33,0.83]","[0.24,0.0,0.97]","[0.51,0.96,0.5]","[0.9,0.55,0.57]",
        "[0.9,0.64,0.52]","[0.9,0.91,0.12]","[0.56,0.0,0.84]","[0.74,0.42,0.55]","[0.13,0.57,0.82]",
        "[0.77,0.53,0.38]","[0.99,0.17,0.33]","[0.66,0.35,0.95]","[0.82,0.0,0.73]","[0.47,0.99,0.78]",
        "[0.46,0.72,0.69]","[0.04,0.76,1.0]","[0.36,0.94,0.06]","[0.99,0.57,0.35]","[1.0,0.35,0.6]",
        "[0.66,0.22,0.8]","[0.3,0.22,0.93]","[0.93,0.03,0.37]","[1.0,0.46,0.33]","[0.71,0.53,0.99]",
        "[0.73,0.62,0.3]","[0.0,0.67,0.74]","[0.9,0.8,0.3]","[0.83,0.08,0.68]","[0.7,0.44,0.7]",
        "[0.56,0.85,0.95]","[0.84,0.4,0.57]","[0.92,0.37,0.46]","[0.91,0.87,0.22]","[0.57,0.94,0.71]",
        "[0.14,0.25,1.0]","[0.84,0.54,0.07]","[0.91,0.0,0.58]","[0.4,0.83,0.4]","[0.26,1.0,0.88]",
        "[0.86,0.67,0.08]","[0.16,0.57,0.99]","[0.33,0.45,0.99]","[0.65,0.96,0.59]","[0.08,0.74,0.84]",
        "[0.94,0.88,0.52]","[0.43,0.83,0.72]","[0.26,0.83,0.51]","[0.71,0.92,0.0]","[0.78,0.23,0.99]",
        "[0.93,0.68,0.39]","[0.78,0.36,0.99]","[0.32,0.77,0.75]","[0.33,1.0,0.82]","[0.99,0.73,0.29]",
        "[0.91,0.99,0.28]","[1.0,0.04,0.89]","[0.53,0.0,0.92]","[0.1,0.0,1.0]","[0.06,1.0,0.07]",
        "[0.22,0.85,0.84]","[0.98,0.61,0.8]","[0.48,1.0,0.13]","[0.91,0.7,0.02]","[0.41,0.94,0.93]",
        "[0.21,0.93,0.33]","[0.26,0.82,0.7]","[0.37,0.56,0.88]","[0.3,1.0,0.53]","[0.91,0.67,0.17]",
        "[0.77,0.94,0.27]","[0.83,0.97,0.58]","[0.99,0.21,0.86]","[0.73,0.64,0.93]","[0.71,0.42,0.87]",
        "[0.99,0.25,0.52]","[0.46,0.92,0.83]","[0.98,0.42,0.5]","[0.93,0.81,0.01]","[0.82,0.93,0.17]",
        "[0.76,0.31,0.7]","[0.3,0.93,0.43]","[0.59,0.12,0.92]","[0.17,1.0,0.28]","[0.94,0.47,0.79]",
        "[0.92,0.08,0.68]","[0.6,0.49,0.87]","[0.84,0.85,0.04]","[0.46,0.14,0.88]","[0.92,0.07,0.86]",
        "[0.19,0.84,0.94]","[0.37,0.65,0.67]","[0.97,0.5,0.18]","[0.41,1.0,0.39]","[0.54,0.86,0.85]",
        "[0.94,0.35,0.0]","[0.67,0.07,0.74]","[0.41,0.39,0.93]","[0.8,0.85,0.26]","[0.66,0.84,0.0]",
        "[0.8,0.29,0.86]","[0.6,0.92,0.11]","[0.96,0.3,0.44]","[1.0,0.29,0.76]","[0.08,0.86,0.93]",
        "[0.48,0.51,0.86]","[0.56,0.85,0.72]","[0.81,0.51,0.66]","[0.0,1.0,0.88]","[0.88,0.59,0.91]",
        "[0.76,0.71,0.39]","[0.88,0.79,0.38]","[0.88,0.01,1.0]","[0.68,0.74,0.46]","[0.49,0.86,0.43]",
        "[0.5,0.63,0.89]","[0.54,0.72,0.93]","[0.97,0.94,0.38]","[0.78,0.76,0.0]","[0.44,0.92,0.21]",
        "[0.97,0.42,0.25]","[0.92,0.27,0.66]","[0.88,0.64,0.32]","[0.22,0.69,0.75]","[0.21,0.27,0.94]",
        "[0.6,0.72,0.35]","[0.05,1.0,0.33]","[0.69,0.01,0.86]","[0.31,1.0,0.38]","[0.26,0.97,0.07]",
        "[0.84,0.08,0.88]","[0.98,0.06,0.76]","[0.08,0.76,0.71]","[0.22,0.67,0.92]","[0.77,0.41,0.72]",
        "[0.43,0.3,0.88]","[0.8,0.16,0.74]","[0.92,0.16,0.54]","[0.29,1.0,0.21]","[1.0,0.12,0.4]",
        "[0.07,0.55,0.98]","[0.73,0.55,0.91]","[0.92,0.46,0.44]","[1.0,0.57,0.46]","[0.98,0.04,0.63]",
        "[0.71,0.21,0.95]","[0.29,0.94,0.99]","[0.0,0.92,0.71]","[0.17,0.93,0.94]","[0.8,0.46,0.79]",
        "[0.31,0.69,0.78]","[0.44,0.64,0.99]","[0.53,0.44,0.72]","[0.35,0.86,0.62]","[0.7,0.96,0.37]",
        "[0.76,0.56,0.49]","[0.73,0.95,0.14]","[0.75,0.72,0.19]","[0.91,0.47,0.08]","[0.78,0.65,0.15]",
        "[0.93,0.37,0.17]","[0.82,0.27,0.5]","[0.49,0.86,0.14]","[0.85,0.82,0.59]","[0.62,0.77,0.18]",
        "[0.85,0.09,0.52]","[0.19,0.88,0.51]","[0.16,0.5,0.92]","[0.58,0.99,0.47]","[0.92,0.56,0.83]",
        "[0.0,0.94,0.44]","[0.77,0.54,0.8]","[0.93,0.81,0.45]","[0.32,0.18,1.0]","[0.25,0.8,1.0]",
        "[1.0,0.61,0.69]","[0.09,0.92,0.39]","[0.87,0.37,0.33]","[0.41,0.95,0.7]","[0.63,0.59,0.95]",
        "[0.74,0.06,0.96]","[0.03,0.68,1.0]","[0.94,0.04,0.51]","[0.82,0.58,0.0]","[0.54,0.44,0.99]",
        "[0.95,0.28,0.59]","[0.08,0.89,0.83]","[0.07,0.41,1.0]","[0.58,0.67,0.86]","[0.77,0.29,0.94]",
        "[0.14,0.92,0.7]","[0.73,0.85,0.12]","[0.84,0.74,0.52]","[0.48,0.23,0.94]","[0.99,0.43,0.09]",
        "[0.99,0.15,0.2]","[0.85,0.63,0.58]","[0.3,0.6,0.95]","[0.95,0.74,0.58]","[1.0,0.17,0.96]",
        "[0.39,0.76,0.68]","[0.85,0.53,0.87]","[0.87,0.92,0.52]","[0.91,0.42,0.01]","[0.38,0.55,0.77]",
        "[0.47,0.42,0.78]","[0.91,0.52,0.38]","[0.94,0.15,0.65]","[0.7,0.91,0.64]","[0.24,0.88,0.76]",
        "[0.5,0.36,1.0]","[0.33,0.86,0.95]","[1.0,0.83,0.36]","[0.19,1.0,0.07]","[0.92,0.55,0.04]",
        "[0.68,0.83,0.34]","[0.26,1.0,0.46]","[0.99,0.37,0.92]","[0.43,0.98,0.47]","[0.25,0.8,0.62]",
        "[0.43,0.92,0.58]","[0.48,0.5,0.98]","[0.08,0.99,0.99]","[0.41,0.73,0.83]","[0.64,0.29,1.0]",
        "[0.79,0.32,0.59]","[0.9,0.58,0.73]","[0.34,0.48,0.85]","[0.92,0.21,0.32]","[0.84,0.94,0.06]",
        "[0.0,0.68,0.84]","[0.06,0.96,0.56]","[0.37,0.96,0.53]","[0.56,0.42,0.83]","[0.99,0.25,0.08]",
        "[0.93,0.33,0.71]","[0.99,0.24,0.19]","[0.5,0.57,0.77]","[0.87,0.35,0.95]","[0.99,0.49,0.69]",
        "[0.84,0.01,0.92]","[0.79,0.7,0.08]","[0.64,0.87,0.28]","[0.92,0.13,0.77]","[0.53,0.94,0.26]",
        "[0.73,0.63,0.38]","[1.0,0.78,0.19]","[0.21,1.0,0.54]","[0.93,0.09,0.93]","[0.3,0.39,0.93]",
        "[0.89,0.22,0.74]","[0.92,0.2,0.47]","[0.68,0.42,0.94]","[0.36,0.72,0.9]","[0.93,0.43,0.59]",
        "[0.96,0.48,0.02]","[0.15,0.71,0.95]","[0.3,0.92,0.28]","[0.31,0.99,0.65]","[0.91,0.56,0.17]",
        "[0.58,0.55,0.81]","[0.86,0.82,0.24]","[0.83,0.99,0.3]","[0.73,0.17,0.67]","[1.0,0.68,0.57]",
        "[0.87,0.3,0.8]","[0.55,0.81,0.41]","[0.6,0.46,0.94]","[0.28,0.52,0.87]","[0.26,0.78,0.85]",
        "[0.84,0.51,0.22]","[0.5,1.0,0.59]","[0.16,0.78,0.73]","[0.0,0.12,0.99]","[0.83,0.55,0.59]",
        "[0.74,0.22,0.75]","[0.53,0.13,0.84]","[0.87,0.23,0.55]","[0.46,0.82,0.5]","[0.97,0.66,0.13]",
        "[0.3,0.73,1.0]","[0.94,0.85,0.15]","[0.17,0.93,0.39]","[0.69,1.0,0.53]","[0.66,0.98,0.22]",
        "[0.91,0.0,0.44]","[0.77,0.49,0.95]","[0.5,0.77,0.84]","[0.53,0.81,0.27]","[0.51,0.92,0.4]",
        "[0.99,0.08,0.34]","[0.94,0.34,0.87]","[0.77,0.81,0.36]","[0.85,0.34,0.47]","[0.3,0.06,0.96]",
        "[0.65,0.85,0.14]","[0.17,1.0,0.2]","[0.95,0.33,0.36]","[0.45,0.89,0.98]","[0.59,0.94,0.83]",
        "[0.09,0.34,1.0]","[0.66,0.14,0.8]","[0.68,0.66,0.42]","[0.85,0.78,0.12]","[0.66,0.87,0.44]",
        "[0.51,0.9,0.69]","[0.78,0.41,0.85]","[0.46,0.0,0.89]","[0.12,0.99,0.38]","[0.76,0.94,0.55]",
        "[0.87,0.24,0.96]","[0.41,0.15,0.94]","[0.85,0.56,0.37]","[0.68,0.74,0.33]","[0.4,0.98,0.77]",
        "[0.83,0.14,0.91]","[0.48,0.21,0.85]","[0.27,0.59,0.82]","[0.82,0.15,0.56]","[0.6,0.22,0.87]",
        "[0.75,0.75,0.49]","[0.92,0.72,0.52]","[0.82,0.42,0.5]","[0.47,0.35,0.81]","[0.83,0.44,0.66]",
        "[0.62,0.85,0.07]","[0.95,0.72,0.45]","[0.33,0.92,0.77]","[0.2,0.92,0.86]","[0.13,0.63,0.77]",
        "[0.08,0.83,1.0]","[0.69,0.99,0.64]","[0.57,0.24,0.95]","[0.16,0.63,0.94]","[0.86,0.65,0.41]",
        "[0.41,0.74,0.55]","[0.68,0.09,0.9]","[0.97,0.93,0.0]","[0.46,0.93,0.07]","[0.82,0.29,0.66]",
        "[0.31,0.94,0.87]","[1.0,0.26,0.3]","[0.89,0.93,0.35]","[0.03,0.9,1.0]","[0.99,0.67,0.0]",
        "[0.67,0.83,0.24]","[0.71,0.37,1.0]","[0.63,0.98,0.07]","[0.79,0.85,0.44]","[0.0,0.76,0.87]",
        "[0.86,0.85,0.32]","[0.43,0.85,0.86]","[0.86,0.74,0.23]","[0.16,0.99,0.65]","[0.93,0.56,0.24]",
        "[0.84,0.83,0.49]","[0.79,1.0,0.43]","[0.8,0.12,1.0]","[0.14,0.84,0.61]","[0.3,0.83,0.86]",
        "[0.37,0.82,0.76]","[0.46,0.97,0.89]","[0.86,0.46,0.9]","[0.61,0.82,0.88]","[0.19,0.32,1.0]",
        "[0.73,0.48,0.81]","[0.61,0.36,0.74]","[0.12,0.81,0.85]","[0.64,1.0,0.34]","[0.17,0.61,0.87]",
        "[0.64,0.94,0.71]","[0.44,0.76,0.75]","[0.93,0.44,0.16]","[0.46,0.75,0.62]","[0.73,0.38,0.66]",
        "[0.88,0.38,0.79]","[0.54,0.99,0.31]","[1.0,0.35,0.14]","[0.18,0.83,0.77]","[0.32,0.92,0.51]",
        "[1.0,0.07,0.08]","[0.54,0.92,0.9]","[0.52,0.05,1.0]","[0.4,0.25,0.95]","[0.47,0.91,0.28]",
        "[0.84,0.49,0.73]","[0.78,0.87,0.57]","[0.0,1.0,0.81]","[0.07,0.49,0.93]","[0.28,0.55,1.0]",
        "[0.34,0.45,0.92]","[0.48,0.84,0.61]","[0.45,0.76,0.94]","[0.82,0.35,0.72]","[0.08,0.69,0.94]",
        "[0.82,0.56,0.93]","[0.92,0.33,0.23]","[0.43,1.0,0.27]","[0.94,0.44,0.67]","[0.8,0.64,0.95]",
        "[0.84,0.43,0.37]","[0.95,0.16,0.83]","[0.27,0.95,0.57]","[1.0,0.71,0.06]","[0.87,0.78,0.46]",
        "[1.0,0.7,0.4]","[0.15,0.93,0.5]","[0.7,0.54,0.85]","[0.69,0.67,0.28]","[0.8,0.01,0.66]",
        "[0.24,0.74,0.91]","[0.93,0.64,0.06]","[0.96,0.18,0.39]","[0.81,0.25,0.74]","[0.66,0.47,1.0]",
        "[0.37,0.94,0.83]","[0.64,0.87,0.59]","[0.85,0.11,0.78]","[0.94,0.49,0.3]","[0.37,0.9,0.67]",
        "[0.53,0.89,0.49]","[0.2,0.56,0.94]","[0.85,0.59,0.83]","[0.01,0.88,0.89]","[0.73,0.9,0.49]",
        "[0.38,0.86,0.34]","[0.73,0.18,0.89]","[0.26,0.93,0.65]","[0.82,0.16,0.67]","[0.19,0.97,0.78]",
        "[0.99,0.09,0.83]","[0.49,0.92,0.0]","[1.0,0.33,0.83]","[0.7,0.29,0.93]","[0.64,0.77,0.01]",
        "[0.4,0.86,0.93]","[1.0,0.38,0.67]","[0.95,0.5,0.59]","[0.0,0.23,0.99]","[0.06,0.62,0.96]",
        "[0.57,0.98,0.04]","[0.09,0.93,0.88]","[0.85,0.29,0.92]","[0.97,0.58,0.11]","[1.0,0.78,0.45]",
        "[0.81,0.44,1.0]","[0.93,0.72,0.65]","[0.72,0.94,0.23]","[1.0,0.95,0.22]","[0.67,0.75,0.13]",
        "[0.47,0.44,0.93]","[0.75,0.67,0.03]","[0.74,0.35,0.81]","[0.51,0.69,0.84]","[0.9,0.13,1.0]",
        "[0.61,0.53,0.93]","[1.0,0.55,0.58]","[0.0,0.64,0.94]","[0.9,0.14,0.42]","[0.85,0.22,0.68]",
        "[0.96,0.87,0.08]","[0.8,0.13,0.84]","[0.72,0.75,0.07]","[0.52,0.56,0.93]","[0.15,1.0,0.97]",
        "[0.93,0.41,0.83]","[0.75,0.05,0.85]","[0.84,0.51,0.49]","[0.14,0.74,0.67]","[0.07,0.99,0.16]",
        "[0.79,1.0,0.53]","[0.0,0.81,0.69]","[0.62,0.05,0.83]","[0.59,0.9,0.24]","[0.85,0.64,0.15]",
        "[1.0,0.68,0.7]","[0.78,0.82,0.16]","[0.61,0.75,0.93]","[0.54,0.07,0.93]","[1.0,0.74,0.13]",
        "[0.4,0.01,0.99]","[0.81,0.62,0.28]","[0.86,0.58,0.45]","[0.72,0.13,0.77]","[0.45,0.66,0.73]",
        "[0.39,0.61,0.8]","[0.86,0.26,0.44]","[0.8,0.7,0.45]","[0.72,0.89,0.35]","[0.98,0.24,0.44]",
        "[0.59,0.77,0.27]","[1.0,0.11,0.48]","[0.46,0.83,0.3]","[0.18,0.67,0.81]","[0.54,0.98,0.77]",
        "[0.99,0.23,0.93]","[0.91,0.46,0.95]","[0.26,0.16,0.95]","[0.69,0.28,0.77]","[0.49,0.35,0.91]",
        "[0.63,0.22,0.74]","[0.96,0.78,0.07]","[0.94,0.04,0.99]","[0.54,1.0,0.65]","[0.28,0.29,0.92]"]
    rgb_script = "from pymol import cmd\n"
    for i,col in enumerate(rgb_cols):
        rgb_script += "cmd.set_color(\"caver{}\",{})\n".format(i+1, col)
    with open(os.path.join(outdir, "pymol", "modules", "rgb.py"), "w") as fout:
        fout.write(rgb_script)
    with open(os.path.join(outdir, "pymol", "modules", "__init__.py"), "w") as fout:
        fout.write("\n")


def main():
    start = time.time()
    # Make output directories
    os.makedirs(args.outdir, exist_ok=True)
    for d in ["analysis", "data"]:
        os.makedirs(os.path.join(args.outdir, d), exist_ok=True)
    # Read tunnel profiles
    input_profiles = os.path.join(args.inputdir, "analysis", "tunnel_profiles.csv")
    out_profiles = os.path.join(args.outdir, "analysis", "tunnel_profiles.csv")
    clusters = get_clusters(input_profiles)
    # Filter tunnel profiles using the limit defined. Default 1
    new_clusters = []
    while clusters:
        _nc = clusters.pop(0)
        if len(_nc.tunnels) >= args.occurence_limit:
            new_clusters.append(_nc)
        else:
            print("Removed cluster {} since it has less than {} tunnel members ({})".format(_nc.cluster_id,
                                                                                            args.occurence_limit,
                                                                                            len(_nc.tunnels)))
    # Sort the clusters by priority, in this case the average cluster throughput
    # divided by the total number of frames
    prioritized_clusters = prioritize_clusters(new_clusters, args.frames)
    for cl in prioritized_clusters:
        print("Old Caver cluster id {} is now {}".format(cl.original_id, cl.cluster_id))
    save_clusters(prioritized_clusters, out_profiles)
    print("Time used for filtering: {:>6.3f} seconds".format(time.time() - start))
    # Refactor bottlenecks
    inbneck = os.path.join(args.inputdir, "analysis", "bottlenecks.csv")
    if os.path.isfile(inbneck):
        start = time.time()
        print("Refactoring bottlenecks")
        outbneck = os.path.join(args.outdir, "analysis", "bottlenecks.csv")
        refactor_bottlenecks(prioritized_clusters, inbneck, outbneck)
        print("Time used for refactoring bottlenecks: {:>6.3f} seconds".format(time.time() - start))
    else:
        print("No bottlenecks.csv file found, skipping...")

    # Copy protein and origins pdbs
    pdbs = [f for f in os.listdir(os.path.join(args.inputdir, "data")) if f.endswith(".pdb")]
    for pdb in pdbs:
        if "origins" in pdb:
            shutil.copy(os.path.join(args.inputdir, "data", pdb), os.path.join(args.outdir, "data", pdb))
        else:
            shutil.copy(os.path.join(args.inputdir, "data", pdb), os.path.join(args.outdir, "data", "caver_ref.1.pdb"))

    # Build vmd visualization files
    if args.vis:
        start = time.time()
        print("Preparing VMD and PyMOL visualization files")
        os.makedirs(os.path.join(args.outdir, "vmd", "scripts"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "pymol", "modules"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "data", "clusters_timeless"), exist_ok=True)
        for cluster in prioritized_clusters:
            if cluster.cluster_id == 0:
                continue
            save_cluster_pdb(cluster, args.outdir)
        make_vmd_script(len(prioritized_clusters), args.outdir)
        make_pymol_scripts(args.outdir)
        print("Time used for produce the visualization files: {} seconds".format(time.time() - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Caver cluster results by number of frames")
    parser.add_argument("-i", "--in", action="store", required=True, dest="inputdir",
        help="The input folder where the caver results are. It should point to the main output folder of caver")
    parser.add_argument("-o", "--out", action="store", required=True, dest="outdir",
        help="Specifies the name for the output folder where the new reclustered files will be located")
    parser.add_argument("-f", "--frames", action="store", required=True, dest="frames", type=int,
        help="The number of frames of the trajectory used to produce the Caver results")
    parser.add_argument("-v", "--vis", action="store_true", required=False, dest="vis",
        help="Add this argument if VMD and PyMOL visualization scripts are desired")
    parser.add_argument("-l", "--limit", action="store", required=False, default=1, dest="occurence_limit",
        type=int, help="Defines a threshold for the clusters. If a cluster has less tunnels than "
        "this especified number, the cluster is not added to the final results. The default or "
        "negative values are the same as not using this parameter.")
    args = parser.parse_args()
    main()
