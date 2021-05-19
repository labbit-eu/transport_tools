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

__version__ = '0.8.5'
__author__ = 'Jan Brezovsky, Aravind Selvaram Thirunavukarasu, Carlos Eduardo Sequeiros-Borja, Bartlomiej Surpeta, ' \
             'Nishita Mandal, Cedrix Jurgal Dongmo Foumthuim, Dheeraj Kumar Sarkar, Nikhil Agrawal'
__mail__ = 'janbre@amu.edu.pl'

import unittest


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

        self.assertListEqual(cgo, convert_coords2cgo(coorinates, 1))

    def test_node_labels_split(self):
        from transport_tools.libs.utils import node_labels_split

        self.assertSequenceEqual((0, 1), node_labels_split("0_1"))
        self.assertSequenceEqual((10, 3), node_labels_split("10_3"))
        self.assertSequenceEqual((8, 105), node_labels_split("8_105"))
        self.assertSequenceEqual((-99, -99), node_labels_split("SP"))


if __name__ == "__main__":
    unittest.main()
