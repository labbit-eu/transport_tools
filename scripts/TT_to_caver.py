#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright 2022 Carlos Eduardo Sequeiros Borja <carseq@amu.edu.pl>
#  

import os
import time
import shutil
import argparse
import configparser as cfp
from sys import argv
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


def read_sc_profile_csv(incsv: str, sc_id: int) -> CVCluster:
    """
    Parse the TransportTools tunnel profile csvfile into a CVCluster object.
    :param incsv: input csv file from TransportTools with the tunnel profile
    :param sc_id: the id number of the cluster
    :return: a CVCluster object
    """
    def _find_pos(line: str, char: str) -> int:
        """
        Helper function to add the csvline correctly
        :param line: input csv line from the tunnel profiles
        :param char: character used to split the data, usually a comma
        :return: the position from which the data is useful for our usage
        """
        i = 0
        for j in range(3):
            i = line.find(char, i) + 1
        return i
    
    cluster = None
    with open(incsv, "r") as csv_in:
        csv_in.readline()
        line = csv_in.readline()
        chunks = line.strip().split(",")
        cluster = CVCluster(sc_id, int(chunks[2]))
        while line:
            chunks = line.strip().split(",")
            cv_tunnel = CVTunnel(frame=chunks[1].strip(), cluster_id=sc_id,
                         tunnel_id=int(chunks[3]), throughput=float(chunks[4]))
            
            x = [float(_x) for _x in line.split(",")[14:]]
            i = _find_pos(line, ",")
            cv_tunnel.csvlines.append(line[i:])
            line = csv_in.readline()
            
            y = [float(_y) for _y in line.split(",")[14:]]
            i = _find_pos(line, ",")
            cv_tunnel.csvlines.append(line[i:])
            line = csv_in.readline()
            
            z = [float(_z) for _z in line.split(",")[14:]]
            i = _find_pos(line, ",")
            cv_tunnel.csvlines.append(line[i:])
            line = csv_in.readline()
            
            for _ in range(2):
                i = _find_pos(line, ",")
                cv_tunnel.csvlines.append(line[i:])
                line = csv_in.readline()
            
            r = [float(_r) for _r in line.split(",")[14:]]
            i = _find_pos(line, ",")
            cv_tunnel.csvlines.append(line[i:])
            line = csv_in.readline()
            
            i = _find_pos(line, ",")
            cv_tunnel.csvlines.append(line[i:])
            line = csv_in.readline()
            
            for i in range(len(x)):
                cv_tunnel.nodes.append(CVNode(i+1, x[i], y[i], z[i], r[i]))
            cluster.tunnels.append(cv_tunnel)
            
        cluster.tunnels.sort()
    return cluster

def merge_tt_superclusters(indir: str, outdir: str) -> List[CVCluster]:
    """
    Export the TransportTools superclusters into one csv file using the Caver
    format for the tunnel profiles. It returns the superclusters as a list
    of CVCluster objects
    :param indir: path to the TransportTools supercluster csv files
    :param outdir: path to export the new csv file
    :return: a list of CVCluster objects
    """
    scs = [d for d in os.listdir(indir) if d.startswith("super") and d.endswith(".csv")]
    clusters = []
    for i in range(len(scs)):
        sc_file = "super_cluster_{:0>2}.csv".format(i+1)
        cluster = read_sc_profile_csv(os.path.join(indir, sc_file), i+1)
        clusters.append(cluster)
    with open(os.path.join(outdir, "tunnel_profiles.csv"), "w") as outfile:
        outfile.write("Snapshot, Tunnel cluster, Tunnel, Throughput, Cost, Bottleneck radius, Average"
                      " R error bound, Max. R error bound, Bottleneck R error bound, Curvature,"
                      " Length, , Axis, Values...\n")
        for cluster in clusters:
            outfile.write(str(cluster))
    return clusters

def read_sc_bneck_csv(incsv: str, sc_id: int) -> List[CVBottleneck]:
    """
    Parse the TransportTools bottleneck csvfile into a list of CVBottleneck objects
    :param incsv: input csv file from TransportTools with the bottleneck profiles
    :param sc_id: the id number of the cluster
    :return: a list of CVBottleneck objects
    """
    bnecks = []
    with open(incsv, "r") as csv_in:
        csv_in.readline()
        for line in csv_in:
            chunks = line.strip().split(",")
            bneck = CVBottleneck(frame=chunks[1].strip(), cluster_id=sc_id, tunnel_id=int(chunks[3]))
            i = 0
            for j in range(3):
                i = line.find(",", i) + 1
            bneck.content = line[i+1:]
            bnecks.append(bneck)
    bnecks.sort()
    return bnecks

def merge_tt_bottlenecks(indir: str, outdir: str):
    """
    Export the TransportTools bottlenecks into one csv file using the Caver
    format
    :param indir: path to the TransportTools bottleneck csv files
    :param outdir: path to export the new csv file
    """
    bns = [d for d in os.listdir(indir) if d.startswith("super") and d.endswith(".csv")]
    new_clusters = []
    for i in range(len(bns)):
        bn_file = "super_cluster_{:0>2}.csv".format(i+1)
        cluster = read_sc_bneck_csv(os.path.join(indir, bn_file), i+1)
        new_clusters.append(cluster)
    with open(os.path.join(outdir, "bottlenecks.csv"), "w") as outfile:
        outfile.write("Bottleneck residues:, Residues in distance <= 3.0 A from the bottleneck, sorted from the closest.\n")
        outfile.write("Snapshot, Tunnel cluster, Tunnel, Throughput, Cost, Bottleneck X,"
                      " Bottleneck Y, Bottleneck Z,  Bottleneck R, Bottleneck residues\n")
        for cluster in new_clusters:
            for bneck in cluster:
                outfile.write(str(bneck))

def merge_v_origins(indir: str, outdir: str, v_pdbs: str, frame: int):
    """
    Joins the v_origins.pdb files. These files set the starting point for tunnel
    calculation in Caver for each frame
    :param indir: path to the TransportTools internal folder where the v_origin are
    :param outdir: path to save the new pdb file
    :param v_pdbs: list of ordered v_origin filenames
    :param frame: frame number to define the reference pdb file, it usually in Caver
                  corresponds to the frame in the middle of the trajectory, however
                  in TransportTools is the reference pdb file used in the run
    """
    model = 1
    v_origins = []
    for v in v_pdbs:
        with open(os.path.join(indir, v), "r") as opdb:
            for line in opdb:
                if line.startswith("ATOM"):
                    v_origins.append("MODEL        {}".format(model))
                    v_origins.append("ATOM {:>6}  {}".format(model, line[11:].strip()))
                    v_origins.append("ENDMDL")
                    model += 1
    with open(os.path.join(outdir, "v_origins.pdb"), "w") as outfile:
        outfile.write("\n".join(v_origins))
    shutil.copy(os.path.join(indir, "..", "ref_transformed.pdb"),
                os.path.join(outdir, "stripped_system.{}.pdb".format(frame)))

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
    ttconfig = cfp.ConfigParser()
    ttconfig.read(args.tt_config_ini)
    tt_folder = os.path.dirname(args.tt_config_ini)
    frames = int(ttconfig['CALCULATIONS_SETTINGS']['snapshots_per_simulation'])
    
    outpdb_path = os.path.join(args.outdir, "data")
    os.makedirs(outpdb_path, exist_ok=True)
    pdb_dir = os.path.join(tt_folder, ttconfig["OUTPUT_SETTINGS"]["output_path"],
                          "_internal", "transformations", "caver")
    v_pdbs = [f for f in os.listdir(pdb_dir) if f.startswith("v_origin")]
    v_pdbs.sort()
    num_frames = int(ttconfig["CALCULATIONS_SETTINGS"]["snapshots_per_simulation"]) * len(v_pdbs)
    num_frames //= 2
    num_frames += 1
    merge_v_origins(pdb_dir, outpdb_path, v_pdbs, int(num_frames))
    
    start = time.time()
    outcsv_path = os.path.join(args.outdir, "analysis")
    os.makedirs(outcsv_path, exist_ok=True)
    sc_dir = os.path.join(tt_folder, ttconfig["OUTPUT_SETTINGS"]["output_path"],
                          "data", "super_clusters", "CSV_profiles", "initial")
    bn_dir = os.path.join(tt_folder, ttconfig["OUTPUT_SETTINGS"]["output_path"],
                          "data", "super_clusters", "bottlenecks", "initial")
    new_clusters = merge_tt_superclusters(sc_dir, outcsv_path)
    print("Time used for creating tunnel profiles: {:>6.3f} seconds".format(time.time() - start))
    start = time.time()
    merge_tt_bottlenecks(bn_dir, outcsv_path)
    print("Time used for creating bottleneck profiles: {:>6.3f} seconds".format(time.time() - start))
    # Build vmd visualization files
    if args.vmd:
        start = time.time()
        print("Preparing vmd visualization files")
        os.makedirs(os.path.join(args.outdir, "vmd", "scripts"), exist_ok=True)
        os.makedirs(os.path.join(args.outdir, "data", "clusters_timeless"), exist_ok=True)
        for cluster in new_clusters:
            if cluster.cluster_id == 0:
                continue
            save_cluster_pdb(cluster, args.outdir)
        make_vmd_script(len(new_clusters), args.outdir)
        print("Time used for produce the visualization files: {} seconds".format(time.time() - start))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge TransportTools supercluster profiles in Caver format")
    parser.add_argument("-c", "--ttconfig", action="store", required=True, dest="tt_config_ini",
        help="The ini file used in the TransportTools run.")
    parser.add_argument("-o", "--out", action="store", required=True, dest="outdir",
        help="Specifies the name for the output folder where the new merged profiles and bottlenecks \
        will be located. Be careful, since the files will be overwritten without warnings.")
    parser.add_argument("-v", "--vmd", action="store_true", required=False, dest="vmd",
        help="Add this argument if vmd vsualization scripts are desired")
    args = parser.parse_args()
    main()
