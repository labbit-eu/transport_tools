#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright 2022 Carlos Eduardo Sequeiros Borja <carseq@amu.edu.pl>
#  

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
            print("Removed cluster {} since it has less than {} tunnel members ({})".format(_nc.cluster_id, args.occurence_limit, len(_nc.tunnels)))
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
            shutil.copy(os.path.join(args.inputdir, "data", pdb), os.path.join(args.outdir, "data", "caver_ref.pdb"))
    # Build vmd visualization files
    if args.vmd:
        start = time.time()
        print("Preparing vmd visualization files")
        os.makedirs(os.path.join(args.outdir, "vmd", "scripts"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "data", "clusters_timeless"), exist_ok=True)
        for cluster in prioritized_clusters:
            if cluster.cluster_id == 0:
                continue
            save_cluster_pdb(cluster, args.outdir)
        make_vmd_script(len(prioritized_clusters), args.outdir)
        print("Time used for produce the visualization files: {} seconds".format(time.time() - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Caver cluster results by number of frames")
    parser.add_argument("-i", "--in", action="store", required=True, dest="inputdir",
        help="The input folder where the caver results are. It should point to the main output folder of caver")
    parser.add_argument("-o", "--out", action="store", required=True, dest="outdir",
        help="Specifies the name for the output folder where the new reclustered files will be located")
    parser.add_argument("-f", "--frames", action="store", required=True, dest="frames", type=int,
        help="The number of frames of the trajectory used to produce the Caver results")
    parser.add_argument("-v", "--vmd", action="store_true", required=False, dest="vmd",
        help="Add this argument if vmd vsualization scripts are desired")
    parser.add_argument("-l", "--limit", action="store", required=False, default=1, dest="occurence_limit",
        type=int, help="Defines a threshold for the clusters. If a cluster has less tunnels than "
        "this especified number, the cluster is not added to the final results. The default or "
        "negative values are the same as not using this parameter.")
    args = parser.parse_args()
    main()
