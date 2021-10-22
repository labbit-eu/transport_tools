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


class TestFileProcessing(unittest.TestCase):
    def setUp(self):
        self.root = set_paths("tests", "data")
        prep_config(self.root)

    def tearDown(self):
        os.remove(os.path.join(self.root, "config.ini"))

    def test_average_starting_point(self):
        import numpy as np
        from transport_tools.libs.geometry import average_starting_point

        in_val1 = os.path.join(self.root, "simulations", "md1", "caver", "data", "v_origins.pdb")
        result1 = average_starting_point(in_val1)[0]
        in_val2 = os.path.join(self.root, "simulations", "md1", "caver", "data", "origins.pdb")
        result2 = average_starting_point(in_val2)[0]
        out_val1 = np.array([[42.78924099999999], [42.17802499999996], [30.81321299999997], [1.0]])
        out_val2 = np.array([[42.97486599999998], [40.84702699999997], [31.67991399999999], [1.0]])
        self.assertTrue(np.allclose(out_val1, result1, atol=1e-7))
        self.assertTrue(np.allclose(out_val2, result2, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
