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

import unittest
import os


def set_paths(*args):
    from transport_tools.libs.utils import splitall
    cwd = os.getcwd()
    all_parts = splitall(cwd)
    if "transport_tools" not in all_parts:
        raise RuntimeError("Must be executed from the 'transport_tools' folder")
    root_index = all_parts.index("transport_tools")
    root = os.path.join(*all_parts[:root_index + 1], *args)

    return root


def prep_config(root: str):
    in_config_file = os.path.join(root, "tmp_config.ini")
    out_config_file = os.path.join(root, "config.ini")
    update_parameters = ["caver_results_path", "aquaduct_results_path", "trajectory_path"]
    with open(in_config_file) as in_stream, open(out_config_file, "w") as out_stream:
        for line in in_stream.readlines():
            for param in update_parameters:
                if param in line:
                    line = "{} = {}\n".format(param, os.path.join(root, "simulations"))
            out_stream.write(line)


class TestProteinFiles(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        from shutil import rmtree

        rmtree(set_paths("tests", "test_results", "TestProteinFiles"))

    def _compare_files(self, out_file: str, res_file: str,):
        import gzip
        import pickle

        if res_file.endswith(".dump.gz"):
            with gzip.open(res_file, 'rb') as res_in, gzip.open(out_file, 'rb') as out_in:
                res_lines = pickle.load(res_in)
                out_lines = pickle.load(out_in)
        elif ".gz" in res_file:
            with gzip.open(res_file, 'r') as res_in, gzip.open(out_file, 'r') as out_in:
                res_lines = res_in.readlines()
                out_lines = out_in.readlines()
        else:
            with open(res_file, "r") as res_in, open(out_file, "r") as out_in:
                res_lines = res_in.readlines()
                out_lines = out_in.readlines()

        self.assertTrue(len(res_lines) == len(out_lines),
                        msg="Different length of files '{}' and '{}':".format(out_file, res_file))
        for res_line, out_line in zip(res_lines, out_lines):
            if ".pdb" in res_file and "REMARK   1 CREATED WITH MDTraj" in res_line:
                continue
            self.assertEqual(out_line, res_line, msg="In files '{}' and '{}':".format(out_file, res_file))

    def setUp(self):
        from transport_tools.libs.config import AnalysisConfig

        self.root = set_paths("tests", "data")
        prep_config(self.root)
        self.out_path = set_paths("tests", "test_results", "TestProteinFiles")
        self.saved_data = os.path.join(self.root, "saved_outputs")
        os.makedirs(self.out_path, exist_ok=True)
        configuration = AnalysisConfig(os.path.join(self.root, "config.ini"), logging=False)
        configuration.set_parameter("transformation_folder", os.path.join(self.root, "saved_outputs",
                                                                          "_internal", "transformations"))
        self.parameters = configuration.get_parameters()

    def test_Trajectory(self):
        from transport_tools.libs.protein_files import TrajectoryFactory
        import numpy as np

        traj1 = TrajectoryFactory(self.parameters, "md1", superpose_mask="protein")
        coords1 = traj1.get_coords(1, 10)
        self.assertTrue(np.allclose([18.325684, -12.265163, -7.21877], coords1[1, 0, :], atol=1e-7))
        traj1.write_frames(1, 10, os.path.join(self.out_path, "traj1.pdb"))
        self._compare_files(os.path.join(self.saved_data, "trajs", "traj1.pdb"),
                            os.path.join(self.out_path, "traj1.pdb"))

        traj2 = TrajectoryFactory(self.parameters, "md1", superpose_mask="name CA")
        coords2 = traj2.get_coords(1, 10)
        self.assertTrue(np.allclose([18.393047, -12.21746, -7.220956], coords2[1, 0, :], atol=1e-7))
        traj2.write_frames(1, 10, os.path.join(self.out_path, "traj2.pdb"), keep_mask="protein")
        self._compare_files(os.path.join(self.saved_data, "trajs", "traj2.pdb"),
                            os.path.join(self.out_path, "traj2.pdb"))

    def test_AtomFromPDB(self):
        from transport_tools.libs.protein_files import AtomFromPDB

        line = "ATOM      4  H3  ILE     1      18.979  -8.370  -4.718  1.00  0.00           H  "
        self.assertEqual(18.979, AtomFromPDB(line).x)
        self.assertEqual(4, AtomFromPDB(line).AtomID)
        self.assertEqual(1, AtomFromPDB(line).ResID)
        self.assertTrue(AtomFromPDB(line).isprotein())

    def test_VizAtom(self):
        from transport_tools.libs.protein_files import VizAtom

        array1 = [13, 'H', 'FIL', 1, -0.7111502155332865, 3.8558356309220017, -1.3426065261705347, 1.6715386140422581]
        array2 = [7, 'UNK', 'A14', 7, 17.62480917654898, -4.341797414586716, 10.878080663249264]
        self.assertEqual("HETATM   13  H   FIL T   1      -0.711   3.856  -1.343        1.67\n", str(VizAtom(array1)))
        self.assertEqual("HETATM    7  UNK A14 T   7      17.625  -4.342  10.878\n", str(VizAtom(array2)))
        self.assertEqual("ATOM      7  UNK A14 T   7      17.625  -4.342  10.878\n", str(VizAtom(array2, False)))

    def test_get_transform_matrix(self):
        from transport_tools.libs.protein_files import get_transform_matrix
        import numpy as np

        out_matrix1 = np.array([[0.82479408, 0.06869789, -0.56124444, 0.03520347],
                                [-0.17422391, 0.97517527, -0.13667194, -0.03265645],
                                [0.53792262, 0.21050841, 0.81628761, 0.00983951],
                                [0.0, 0.0, 0.0, 1.0]])
        identity = np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])

        matrix1 = get_transform_matrix(os.path.join(self.saved_data, "pdbs", "file1.pdb"),
                                       os.path.join(self.saved_data, "pdbs", "file2.pdb"))[0]
        matrix2 = get_transform_matrix(os.path.join(self.saved_data, "pdbs", "file1.pdb"),
                                       os.path.join(self.saved_data, "pdbs", "file1.pdb"))[0]
        matrix3 = get_transform_matrix(os.path.join(self.saved_data, "pdbs", "file2.pdb"),
                                       os.path.join(self.saved_data, "pdbs", "file2.pdb"))[0]

        self.assertTrue(np.allclose(out_matrix1, matrix1, atol=1e-7))
        self.assertTrue(np.allclose(identity, matrix2, atol=1e-7))
        self.assertTrue(np.allclose(identity, matrix3, atol=1e-7))

    def test_transform_pdb_file(self):
        from transport_tools.libs.protein_files import transform_pdb_file, get_transform_matrix

        matrix = get_transform_matrix(os.path.join(self.saved_data, "pdbs", "file2.pdb"),
                                      os.path.join(self.saved_data, "pdbs", "file1.pdb"))[0]
        transform_pdb_file(os.path.join(self.saved_data, "pdbs", "file2.pdb"),
                           os.path.join(self.out_path, "transformed_file2.pdb"), matrix)
        self._compare_files(os.path.join(self.saved_data, "pdbs", "transformed_file2.pdb"),
                            os.path.join(self.out_path, "transformed_file2.pdb"))

    def test_get_general_rot_mat_from_2_ca_atoms(self):
        from transport_tools.libs.protein_files import get_general_rot_mat_from_2_ca_atoms
        import numpy as np

        out_matrix1 = np.array([[-0.132786841, -0.080647613,  0.987858096, 0.0],
                                [-0.613131089,  0.789777455, -0.017939899, 0.0],
                                [-0.778741242, -0.608068692, -0.154319613, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])
        out_matrix2 = np.array([[-0.584202291, -0.108392954,  0.804337398, 0.0],
                                [-0.425154702,  0.885055655, -0.189525633, 0.0],
                                [-0.691340119, -0.452689136, -0.563135316, 0.0],
                                [0.0, 0.0, 0.0, 1.0]])

        out_matrix3 = np.array([[-0.02963773,  0.02323531,  0.99929061, 0.0],
                                [-0.60542614,  0.79506673, -0.03644294, 0.0],
                                [-0.79534948, -0.60607675, -0.00949671, 0.0],
                                [0.0, 0.0, 0.0, 1.0]]
                               )

        matrix1 = get_general_rot_mat_from_2_ca_atoms(os.path.join(self.saved_data, "pdbs", "file1.pdb"))
        matrix2 = get_general_rot_mat_from_2_ca_atoms(os.path.join(self.saved_data, "pdbs", "file2.pdb"))
        matrix3 = get_general_rot_mat_from_2_ca_atoms(os.path.join(self.saved_data, "pdbs", "transformed_file2.pdb"))

        self.assertTrue(np.allclose(out_matrix1, matrix1, atol=1e-7))
        self.assertTrue(np.allclose(out_matrix2, matrix2, atol=1e-7))
        self.assertTrue(np.allclose(out_matrix3, matrix3, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
