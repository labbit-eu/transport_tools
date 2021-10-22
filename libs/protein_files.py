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

from logging import getLogger
import numpy as np
import os
from typing import Optional, Tuple, Dict
from transport_tools.libs.utils import test_file, get_filepath
import Bio.PDB
import Bio.Align


logger = getLogger(__name__)

THREE2ONE_CODES_MAP = {
    # standard aa
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    # other forms from simulation packages
    'HISA': 'H', 'HISB': 'H', 'HSE': 'H', 'HSD': 'H', 'HID': 'H', 'HIE': 'H', 'HIS1': 'H', 'HIP': 'H', 'HIS2': 'H',
    'ASPH': 'D', 'ASH': 'D',
    'GLUH': 'E', 'GLH': 'E',
    'LYSH': 'K', 'LYN': 'K',
    'ARGN': 'R',
    'CYSH': 'C', 'CYS1': 'C', 'CYS2': 'C', 'CYX': 'C'
}


def TrajectoryFactory(parameters: dict, md_label: str, superpose_mask: str = None):
    if parameters["trajectory_engine"] == "mdtraj":
        if superpose_mask is not None:
            return TrajectoryMdtraj(parameters, md_label, superpose_mask)
        else:
            return TrajectoryMdtraj(parameters, md_label)
    elif parameters["trajectory_engine"] == "pytraj":
        if superpose_mask is not None:
            return TrajectoryPytraj(parameters, md_label, superpose_mask)
        else:
            return TrajectoryPytraj(parameters, md_label)
    else:
        raise RuntimeError("Trajectory processing engine '{}' is not "
                           "supported.".format(parameters["trajectory_engine"]))


class TrajectoryTT:
    def __init__(self, parameters: dict, md_label: str, superpose_mask: str):
        """
        Generic class for handling MD trajectories
        :param parameters: job configuration parameters
        :param md_label: name of folder with the source MD simulation data
        :param superpose_mask: mask for selection to get same system used in CAVER, if None, all atoms are used
        """

        self.parameters = parameters
        self.md_label = md_label
        root_folder = os.path.join(parameters["trajectory_path"], md_label)
        self.traj = get_filepath(root_folder, parameters["trajectory_relative_file"])
        self.top = get_filepath(root_folder, parameters["topology_relative_file"])
        self.ref_frame = None
        self.superpose_mask = superpose_mask
        if self.inputs_exists():
            self._get_ref_frame_from_caver_ref()

    def inputs_exists(self) -> bool:
        """
        Tests if inputs exists
        """

        try:
            test_file(self.traj)
            test_file(self.top)
        except FileNotFoundError:
            return False

        return True

    def _get_ref_frame_from_caver_ref(self):
        raise NotImplementedError("Provide implementation of this method.")

    def get_coords(self, start_frame: int, end_frame: int, keep_mask: Optional[str] = None,
                   out_file: Optional[str] = None) -> np.array:
        """
        Get coordinates of system across specified frames, potentially after removing some of its parts. And if
        out_file is provided, the resulting structure is saved too
        :param start_frame: start frame to consider
        :param end_frame: end frame to consider
        :param keep_mask: mask to select part of system to keep
        :param out_file: file where to save specified frames as MULTIMODEL PDB file
        :return: coordinates of the kept system in selected frames
        """

        raise NotImplementedError("Provide implementation of this method.")

    def write_frames(self, start_frame: int, end_frame: int, out_file: str, keep_mask: Optional[str] = None):
        """
        Write specified frames to MULTIMODEL PDB file
        :param start_frame: start frame to consider
        :param end_frame: end frame to consider
        :param out_file: file where to save specified frames
        :param keep_mask: mask to select part of system to keep
        """

        self.get_coords(start_frame, end_frame, keep_mask, out_file)


class TrajectoryMdtraj(TrajectoryTT):
    def __init__(self, parameters: dict, md_label: str, superpose_mask: str = "name CA"):
        """
        Class for handling MD trajectories with MDtraj package
        :param parameters: job configuration parameters
        :param md_label: name of folder with the source MD simulation data
        :param superpose_mask: MDtraj mask for selection to get same system used in CAVER, if None, all atoms are used
        """

        TrajectoryTT.__init__(self, parameters, md_label, superpose_mask)

    def _get_ref_frame_from_caver_ref(self):
        """
        For CAVER we have aligned to first simulation frame. Now to get back, we can align first frame to reference
        PDB file from tunnel analyses and use this frame as valid reference structure for further snapshots
        """

        import mdtraj
        reference_pdb_file = os.path.join(self.parameters["transformation_folder"], "ref_transformed.pdb")
        caver_pdb = mdtraj.load(reference_pdb_file)
        frame1 = mdtraj.load_frame(self.traj, index=0, top=self.top)
        self.ref_frame = frame1.superpose(reference=caver_pdb, atom_indices=frame1.topology.select(self.superpose_mask),
                                          ref_atom_indices=caver_pdb.topology.select(self.superpose_mask),
                                          parallel=False)

    def get_coords(self, start_frame: int, end_frame: int, keep_mask: Optional[str] = None,
                   out_file: Optional[str] = None) -> np.array:
        """
        Get coordinates of system across specified frames, potentially after removing some of its parts
        :param start_frame: start frame to consider
        :param end_frame: end frame to consider
        :param keep_mask: MDtraj mask to select part of system to keep
        :param out_file: file where to save specified frames as MULTIMODEL PDB file
        :return: coordinates of the kept system in selected frames
        """

        import mdtraj
        if keep_mask is None:
            keep_mask = "all"

        align_indices = self.ref_frame.topology.select(self.superpose_mask)
        keep_indices = self.ref_frame.topology.select(keep_mask)

        fitted_traj = None
        chunk_size = 1000
        start_chunk = start_frame // chunk_size
        end_chunk = end_frame // chunk_size
        for i, md_frame in enumerate(mdtraj.iterload(self.traj, top=self.top, chunk=chunk_size)):

            if i not in range(start_chunk, end_chunk + 1):
                continue

            slice_start = 0
            slice_end = chunk_size - 1
            chunk_start_frame = i * chunk_size

            if i == start_chunk:
                slice_start = start_frame - chunk_start_frame
            if i == end_chunk:
                slice_end = end_frame - chunk_start_frame

            md_frame = md_frame[slice_start: slice_end + 1]
            md_frame.superpose(reference=self.ref_frame, atom_indices=align_indices, parallel=False)
            md_frame.restrict_atoms(keep_indices)

            if fitted_traj is None:
                fitted_traj = md_frame
            else:
                fitted_traj = mdtraj.join((fitted_traj, md_frame), check_topology=False)

        if out_file is not None:
            fitted_traj.save_pdb(out_file)

        return fitted_traj.xyz * 10  # nm -> A


class TrajectoryPytraj(TrajectoryTT):
    def __init__(self, parameters: dict, md_label: str, superpose_mask: str = "@CA"):
        """
        Class for handling MD trajectories with Pytraj package
        :param parameters: job configuration parameters
        :param md_label: name of folder with the source MD simulation data
        :param superpose_mask: Pytraj mask for selection to get same system used in CAVER, if None, all atoms are used
        """

        TrajectoryTT.__init__(self, parameters, md_label, superpose_mask)

    def _get_ref_frame_from_caver_ref(self):
        """
        For CAVER we have aligned to first simulation frame. Now to get back, we can align first frame to reference
        PDB file from tunnel analyses and use this frame as valid reference structure for further snapshots
        """

        try:
            import pytraj as pt
        except ModuleNotFoundError:
            raise RuntimeError("Requested to use 'pytaj' as 'trajectory_engine' but pytraj package cannot be "
                               "imported. Please check that it is properly installed in the current environment.")
        reference_pdb_file = os.path.join(self.parameters["transformation_folder"], "ref_transformed.pdb")
        caver_pdb = pt.iterload(reference_pdb_file, frame_slice=(0, 1))
        frame1 = pt.iterload(self.traj, self.top, frame_slice=(0, 1))
        self.ref_frame = frame1.superpose(mask=self.superpose_mask, ref=caver_pdb, ref_mask=self.superpose_mask)

    def get_coords(self, start_frame: int, end_frame: int, keep_mask: Optional[str] = None,
                   out_file: Optional[str] = None) -> np.array:

        """
        Get coordinates of system across specified frames, potentially after removing some of its parts
        :param start_frame: start frame to consider
        :param end_frame: end frame to consider
        :param keep_mask: inverted AMBER mask to select part of system to keep
        :param out_file: file where to save specified frames as MULTIMODEL PDB file
        :return: coordinates of the kept system in selected frames
        """

        try:
            import pytraj as pt
        except ModuleNotFoundError:
            raise RuntimeError("Requested to use 'pytaj' as 'trajectory_engine' but pytraj package cannot be "
                               "imported. Please check that it is properly installed in the current environment.")
        md_traj = pt.iterload(self.traj, self.top, frame_slice=(start_frame, end_frame + 1)).autoimage()
        fitted_traj = md_traj.superpose(mask=self.superpose_mask, ref=self.ref_frame)
        if keep_mask is not None:
            fitted_traj = fitted_traj.strip(keep_mask)

        if out_file is not None:
            pt.write_traj(out_file, fitted_traj, overwrite=True, options="model")

        return fitted_traj.xyz


class AtomFromPDB:
    def __init__(self, line: str):
        """
        Generating Atoms from pdb lines
        :param line: line from PDB file starting with "ATOM" field
        """

        self.AtomID = int(line[7:11])
        self.AtomName = line[12:16].strip()
        self.ResName = line[17:20]
        self.ResID = int(line[22:26])
        self.Chain = line[21]
        self.x = float(line[31:38])
        self.y = float(line[39:46])
        self.z = float(line[47:54])

    def __str__(self):
        return "ATOM{:7d} {:>4s} {:3s}{:6d}{:12.3f}{:8.3f}{:8.3f}\n".format(self.AtomID, self.AtomName, self.ResName,
                                                                            self.ResID, self.x, self.y, self.z)

    def isprotein(self):
        """
        Test if the atom is from protein residues
        """
        if self.ResName in THREE2ONE_CODES_MAP.keys():
            return True
        else:
            return False


class VizAtom:
    def __init__(self, array: list, use_hetatm: bool = True):
        """
        Creation of PDB line from variables
        :param array: list of variables to create PDB line
        :param use_hetatm: if the PDB line should use HETATM
        """

        self.AtomID = int(array[0])
        self.AtomName = str(array[1])
        self.ResName = str(array[2])
        self.ResID = int(array[3])
        self.x = float(array[4])
        self.y = float(array[5])
        self.z = float(array[6])
        if len(array) == 8:

            self.R = "{:12.2f}".format(float(array[7]))
        else:
            self.R = ""
        self.use_hetatm = use_hetatm

    def __str__(self):
        if self.use_hetatm:
            return "HETATM{:5d}  {:<3s} {:3s} T{:4d}{:12.3f}{:8.3f}{:8.3f}{}\n".format(self.AtomID, self.AtomName,
                                                                                       self.ResName, self.ResID, self.x,
                                                                                       self.y, self.z, self.R)
        else:
            return "ATOM{:7d}  {:<3s} {:3s} T{:4d}{:12.3f}{:8.3f}{:8.3f}{}\n".format(self.AtomID, self.AtomName,
                                                                                     self.ResName, self.ResID, self.x,
                                                                                     self.y, self.z, self.R)


def _seq_align_proteins(target_structure: Bio.PDB.Structure.Structure,
                        moved_structure: Bio.PDB.Structure.Structure) -> Bio.Align.PairwiseAlignment:
    """
    Performs a sequence alignment of two proteins loaded previously with Bio.PDB.Structure.
    :param target_structure: target structure to which we align
    :param moved_structure: structure to be moved
    :return: an alignment structure, which contains the aligned sequences and the indices of the corresponding residues
    """

    target_seq = [residue.get_resname() for residue in target_structure.get_residues()]
    moved_seq = [residue.get_resname() for residue in moved_structure.get_residues()]
    target_seq = [THREE2ONE_CODES_MAP[residue] for residue in target_seq if residue in THREE2ONE_CODES_MAP.keys()]
    moved_seq = [THREE2ONE_CODES_MAP[residue] for residue in moved_seq if residue in THREE2ONE_CODES_MAP.keys()]
    target_seq = "".join(target_seq)
    moved_seq = "".join(moved_seq)

    # Pymol like settings
    aligner = Bio.Align.PairwiseAligner()
    aligner.substitution_matrix = Bio.Align.substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    alignments = aligner.align(target_seq, moved_seq)

    return alignments[0]


def _get_atoms2superpose(alignment: Bio.Align.PairwiseAlignment, target_structure: Bio.PDB.Structure.Structure,
                         moved_structure: Bio.PDB.Structure.Structure) -> Tuple[Dict[int, Bio.PDB.Atom.Atom],
                                                                                Dict[int, Bio.PDB.Atom.Atom]]:
    """
    Selects atoms for structure superposition based on sequence alignment. Only alpha Carbon atoms of sequentially
    aligned protein residues are considered.
    :param alignment: sequential alignment generated by PairwiseAligner()
    :param target_structure: target structure to which we align
    :param moved_structure: structure to be moved
    :return: dictionary of residues' alpha Carbons of the target structure, and dictionary of residues' alpha Carbons of
    structure to be moved
    """

    # get atoms mapped to ResIDs
    a, b = 0, 0
    target_ca = dict()
    moved_ca = dict()
    for residue in target_structure.get_residues():
        if residue.get_resname() not in THREE2ONE_CODES_MAP.keys():
            logger.warning("Unknown residue name '{}' encountered in '{}'.\n This residues will not be considered "
                           "during structural alignment.".format(residue.get_resname(), target_structure.get_id()))
            continue

        for atom in residue.get_atoms():
            if atom.get_name() == "CA":
                target_ca[a] = atom
                a += 1
                break

    for residue in moved_structure.get_residues():
        if residue.get_resname() not in THREE2ONE_CODES_MAP.keys():
            logger.warning("Unknown residue name '{}' encountered in '{}'.\n This residues will not be considered "
                           "during structural alignment.".format(residue.get_resname(), moved_structure.get_id()))
            continue

        for atom in residue.get_atoms():
            if atom.get_name() == "CA":
                moved_ca[b] = atom
                b += 1
                break

    # get ResIDs of aligned residues
    target_inds, moved_inds = alignment.aligned
    target_aligned_resids = set()
    moved_aligned_resids = set()
    for ti, si in zip(target_inds, moved_inds):
        for i in range(ti[0], ti[1]):
            target_aligned_resids.add(i + 1)
        for i in range(si[0], si[1]):
            moved_aligned_resids.add(i + 1)

    # remove misaligned residues
    for residue in set(target_ca.keys()) - target_aligned_resids:
        del target_ca[residue]

    for residue in set(moved_ca.keys()) - moved_aligned_resids:
        del moved_ca[residue]

    return target_ca, moved_ca


def _superpose_moved2target(target_atoms: Dict[int, Bio.PDB.Atom.Atom],
                            moved_atoms: Dict[int, Bio.PDB.Atom.Atom]) -> Tuple[float, np.array]:
    """
    Superposition of the structure to the target.
    :param target_atoms: dictionary of residues' alpha Carbons of the target structure
    :param moved_atoms: dictionary of residues' alpha Carbons of the structure to be moved
    :return: RMSD of the superposition and the 4x4 transformation matrix
    """

    superimposer = Bio.PDB.Superimposer()
    superimposer.set_atoms([*target_atoms.values()], [*moved_atoms.values()])
    rot, trans = superimposer.rotran
    rotran = np.identity(4)
    rotran[:3, :3] = rot
    rotran[3, :3] = trans

    return superimposer.rms, rotran


def _get_atom_rmsds(target_atoms: Dict[int, Bio.PDB.Atom.Atom], moved_atoms: Dict[int, Bio.PDB.Atom.Atom],
                    transform_mat: np.array) -> np.array:
    """
    Modifies the coordinates of the atoms from the moved protein using defined transformation matrix and
    computes the distance corresponding target atoms.
    :param target_atoms: dictionary of residues' alpha Carbons of the target structure
    :param moved_atoms: dictionary of residues' alpha Carbons of the structure to be moved
    :param transform_mat:
    :return: a numpy array with the distances of the atoms after movement.
    """

    target_at = np.array([a.coord for a in target_atoms.values()])
    moved_at = np.array([a.coord for a in moved_atoms.values()])
    _mvd = np.ones((len(target_atoms), 4))
    _mvd[:, :3] = moved_at
    _mvd = np.matmul(_mvd, transform_mat)
    diff = target_at - _mvd[:, :3]

    return np.sqrt((diff*diff).sum(axis=1)/3)


def _update_atoms(rmsds: np.array, target_atoms: dict, moved_atoms: dict,
                  outlier_cutoff: float = 2.0) -> Tuple[Dict[int, Bio.PDB.Atom.Atom], Dict[int, Bio.PDB.Atom.Atom]]:
    """
    Sorts atoms depending on their RMSD and then removes outliers with too high RMSD.
    :param rmsds: pairwise atomic RMSD values
    :param target_atoms: dictionary of residues' alpha Carbons of the target structure
    :param moved_atoms: dictionary of residues' alpha Carbons of the structure to be moved
    :param outlier_cutoff: RMSD cutoff for atom removal as outlier; based on Pymol
    :return: updated dictionaries of residues' alpha Carbons of the target structure and of the structure to be moved
    """

    id_rmsd = dict()
    for rmsd, t_resid, m_resid in zip(rmsds, target_atoms.keys(), moved_atoms.keys()):
        id_rmsd[rmsd] = (t_resid, m_resid)

    for rmsd in sorted(id_rmsd.keys(), reverse=True):
        if rmsd > outlier_cutoff:
            t_resid, m_resid = id_rmsd[rmsd]
            del target_atoms[t_resid]
            del moved_atoms[m_resid]

    return target_atoms, moved_atoms


def get_transform_matrix(moved_protein: str, target_protein: str, md_label: str = "", max_iter: int = 5,
                         rmsd_cutoff: float = 0.1) -> (np.array, str):
    """
    Performs a sequence alignment of the target and moved proteins, then tries to reduce the RMSD by a series of
    iterations removing the atoms with the higher difference before and after alignment.
    :param moved_protein: input PDB file, to be moved
    :param target_protein: reference PDB file to which we align
    :param md_label: name of folder with the source MD simulation data
    :param max_iter: maximum number of iterations
    :param rmsd_cutoff: rmsd convergence cutoff to stop the iterations
    :return: 4x4 transformation matrix describing the alignment, similar to default Pymol align command & md_label
    """

    n_iter = 0
    rmsd_diff = 10.0 * rmsd_cutoff

    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    target = pdb_parser.get_structure(target_protein, target_protein)
    moved = pdb_parser.get_structure(moved_protein, moved_protein)

    seq_alignment = _seq_align_proteins(target, moved)

    target_atoms, moved_atoms = _get_atoms2superpose(seq_alignment, target, moved)
    rmsd, rotran_mat = _superpose_moved2target(target_atoms, moved_atoms)
    while (n_iter < max_iter) and (rmsd_diff > rmsd_cutoff):
        atom_rmsds = _get_atom_rmsds(target_atoms, moved_atoms, rotran_mat)
        target_atoms, moved_atoms = _update_atoms(atom_rmsds, target_atoms, moved_atoms)

        _rmsd, rotran_mat = _superpose_moved2target(target_atoms, moved_atoms)
        rmsd_diff = np.abs(rmsd - _rmsd)
        rmsd = _rmsd
        n_iter += 1

    return rotran_mat.T, md_label


def transform_pdb_file(in_pdb_file: str, out_pdb_file: str, transform_mat: np.array):
    """
    Translates and rotates atoms in PDB file by transformation matrix
    :param in_pdb_file: input pdb file path
    :param out_pdb_file: path to the transformed pdb file
    :param transform_mat: 4x4 transformation matrix to be applied on the input coordinates
    """
    import mdtraj
    to_move = mdtraj.load(in_pdb_file)
    to_move.xyz = to_move.xyz * 10
    coords = np.ones((to_move.xyz.shape[1], 4))
    coords[:, :3] = to_move.xyz[0]
    to_move.xyz[0] = np.matmul(coords, transform_mat.T)[:, :3]
    to_move.xyz = to_move.xyz / 10
    to_move.save_pdb(out_pdb_file)


def get_general_rot_mat_from_2_ca_atoms(in_pdb_file: str) -> np.array:
    """
    Arbitrary selects Calpha atoms from 1/4 and 3/4 of sequence and orients the 1st one along z-axis,
    and 2nd one into yz-plane
    :param in_pdb_file:  input PDB file
    :return: 4x4 transformation matrix describing the re-orientation
    """

    from transport_tools.libs.geometry import cart2spherical
    structure = Bio.PDB.PDBParser(QUIET=True).get_structure("", in_pdb_file)
    ca_atoms_coords = np.array([atom.get_coord() for atom in structure.get_atoms() if atom.get_name() == "CA"])

    num_resids = ca_atoms_coords.shape[0]
    atom1 = ca_atoms_coords[int(num_resids / 4) - 1, :]  # -1 as it starts from 0 unlike residues IDs
    atom2 = ca_atoms_coords[int(num_resids * 3 / 4) - 1, :]  # -1 as it starts from 0 unlike residues IDs

    logger.debug("Using CA atoms of residues {:d} & {:d} to align the system "
                 "coordinates".format(int(num_resids / 4), int(num_resids * 3 / 4)))

    # orient the first atom along z-axis
    xyz1 = np.array([atom1[0], atom1[1], atom1[2]])
    r1, theta1, phi1 = cart2spherical(xyz1)
    rot1 = np.array([[np.cos(-phi1), -np.sin(-phi1), 0],
                     [np.sin(-phi1), np.cos(-phi1), 0],
                     [0, 0, 1]])

    rot2 = np.array([[np.cos(-theta1), 0, np.sin(-theta1)],
                     [0, 1, 0],
                     [-np.sin(-theta1), 0, np.cos(-theta1)]])

    rot_mat1 = np.matmul(rot2, rot1)

    # orient the second atom after 1st transformation into yz-plane
    xyz2 = np.array([atom2[0], atom2[1], atom2[2]])
    rot_xyz2 = np.matmul(rot_mat1, xyz2)
    r2, theta2, phi2 = cart2spherical(rot_xyz2)
    rot3 = np.array([[np.cos(-phi2+np.pi/2), -np.sin(-phi2+np.pi/2), 0],
                     [np.sin(-phi2+np.pi/2), np.cos(-phi2+np.pi/2), 0],
                     [0, 0, 1]])
    rot_mat_full = np.matmul(rot3, rot_mat1)

    # transform 3x3 rotation_matrix to 4x4 transformation matrix
    trans_mat = np.identity(4, dtype=float)
    trans_mat[:3, :3] = rot_mat_full

    return trans_mat


def transform_aquaduct(md_label: str, tar_file: str, aquaduct_results_pdb_filename: str,
                       reference_pdb_file: str) -> (np.array, str, int):
    """
    Prepares temporary files from AQUA-DUCT data and gets transformation matrix
    :param md_label: name of folder with the source MD simulation data
    :param tar_file: tarfile with AQUA-DUCT results
    :param aquaduct_results_pdb_filename: name of PDB file with protein structure in the AQUA-DUCT tarfile
    :param reference_pdb_file: reference PDB file to which we align
    :return: transformation matrix, md_label, number_of_raw_paths in tar_file
    """

    import tarfile
    from tempfile import mkstemp
    from re import search

    number_of_raw_paths = 0
    fd, pdb_filename_from_tar = mkstemp(suffix=".pdb")
    tar_handle = tarfile.open(tar_file, "r:gz")
    with open(pdb_filename_from_tar, "wb") as out_stream:
        data = tar_handle.extractfile(aquaduct_results_pdb_filename).read()
        out_stream.write(data)
    _matrix, _md_label = get_transform_matrix(pdb_filename_from_tar, reference_pdb_file, md_label)
    os.close(fd)
    os.remove(pdb_filename_from_tar)

    for filename in tar_handle.getnames():
        if search(r'^raw_paths_\d+\.dump', filename):
            number_of_raw_paths += 1
    tar_handle.close()

    return _matrix, _md_label, number_of_raw_paths
