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
import numpy as np


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_get_caver_color(self):
        from transport_tools.libs.utils import get_caver_color
        from transport_tools.tests.units.data.data_utils import color_pallete

        for i in range(1006):
            self.assertListEqual(color_pallete[i], get_caver_color(i))
        self.assertListEqual([1.0, 1.0, 1.0], get_caver_color(None))

    def test_convert_coords2cgo(self):
        from transport_tools.libs.utils import convert_coords2cgo
        from transport_tools.tests.units.data.data_utils import coorinates, cgo

        for item1, item2 in zip(cgo, convert_coords2cgo(coorinates, 1)):
            self.assertAlmostEqual(item1, item2)

    def test_get_boundaries(self):
        from transport_tools.libs.utils import _get_boundaries
        from transport_tools.tests.units.data.data_utils import spheres

        x_lims, y_lims, z_lims = _get_boundaries(spheres)
        self.assertAlmostEqual(-1.9707431512941527, x_lims[0])
        self.assertAlmostEqual(1.7799817245391525, x_lims[1])
        self.assertAlmostEqual(-1.8790397812941526, y_lims[0])
        self.assertAlmostEqual(5.182953051294152, y_lims[1])
        self.assertAlmostEqual(-2.9844158212941525, z_lims[0])
        self.assertAlmostEqual(1.7790397812941525, z_lims[1])

    def test_build_grid(self):
        from transport_tools.libs.utils import _build_grid
        from transport_tools.tests.units.data.data_utils import spheres, x_points, y_points, z_points, grid
        self.assertTrue(np.allclose(_build_grid(spheres, x_points, y_points, z_points), grid, atol=1e-7))

    def test__get_mesh(self):
        from transport_tools.libs.utils import _get_mesh
        from transport_tools.tests.units.data.data_utils import x_points, y_points, z_points, grid, vertices, normals, \
            triangles

        results = _get_mesh(grid, x_points, y_points, z_points)
        self.assertTrue(np.allclose(vertices, results[0], atol=1e-7))
        self.assertTrue(np.allclose(normals, results[1], atol=1e-7, equal_nan=True))
        self.assertTrue(np.allclose(triangles, results[2], atol=1e-7))

    def test_convert_spheres2cgo_surface(self):
        from transport_tools.libs.utils import convert_spheres2cgo_surface
        from transport_tools.tests.units.data.data_utils import spheres, cgo_surf

        try:
            import mcubes
        except ModuleNotFoundError:
            self.assertTrue(True)
        for item1, item2 in zip(cgo_surf, convert_spheres2cgo_surface(spheres, 1, resolution=0.5)):
            self.assertAlmostEqual(item1, item2)

    def test_node_labels_split(self):
        from transport_tools.libs.utils import node_labels_split

        self.assertSequenceEqual((0, 1), node_labels_split("0_1"))
        self.assertSequenceEqual((10, 3), node_labels_split("10_3"))
        self.assertSequenceEqual((8, 105), node_labels_split("8_105"))
        self.assertSequenceEqual((-99, -99), node_labels_split("SP"))


if __name__ == "__main__":
    unittest.main()
