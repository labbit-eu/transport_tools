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

import numpy as np
from typing import List, Tuple

# Test data for filter_spheres tests
sample_spheres_no_duplicates: List[Tuple[np.ndarray, float]] = [
    (np.array([1.0, 2.0, 3.0]), 1.5),
    (np.array([4.0, 5.0, 6.0]), 2.0),
    (np.array([7.0, 8.0, 9.0]), 2.5),
]

sample_spheres_with_duplicates: List[Tuple[np.ndarray, float]] = [
    (np.array([1.0, 2.0, 3.0]), 1.5),
    (np.array([4.0, 5.0, 6.0]), 2.0),
    (np.array([1.0, 2.0, 3.0]), 1.5),  # duplicate
    (np.array([7.0, 8.0, 9.0]), 2.5),
    (np.array([4.0, 5.0, 6.0]), 2.0),  # duplicate
]

sample_spheres_all_duplicates: List[Tuple[np.ndarray, float]] = [
    (np.array([1.0, 2.0, 3.0]), 1.5),
    (np.array([1.0, 2.0, 3.0]), 1.5),
    (np.array([1.0, 2.0, 3.0]), 1.5),
]

sample_spheres_single: List[Tuple[np.ndarray, float]] = [
    (np.array([1.0, 2.0, 3.0]), 1.5),
]

sample_spheres_empty: List[Tuple[np.ndarray, float]] = []

# Mock MSMS output files content matching actual MSMS format
# Format based on real MSMS 2.6.1 output
mock_msms_vert_content = """# MSMS solvent excluded surface vertices
#vertex #sphere density probe_r
3 10 1.00 1.40
1.000 2.000 3.000 0.577 0.577 0.577 0 1 2
4.000 5.000 6.000 0.707 0.707 0.000 0 2 2
7.000 8.000 9.000 0.000 0.707 0.707 0 3 2
"""

mock_msms_face_content = """# MSMS solvent excluded surface faces
#faces #sphere density probe_r
1 10 1.00 1.40
1 2 3 1 1
"""

# Simple 1-triangle mesh for testing
mock_msms_vert_simple = """# MSMS solvent excluded surface vertices
#vertex #sphere density probe_r
3 5 1.00 1.40
0.000 0.000 0.000 0.000 0.000 1.000 0 1 2
1.000 0.000 0.000 0.000 0.000 1.000 0 2 2
0.000 1.000 0.000 0.000 0.000 1.000 0 3 2
"""

mock_msms_face_simple = """# MSMS solvent excluded surface faces
#faces #sphere density probe_r
1 5 1.00 1.40
1 2 3 1 1
"""

# Larger mesh with 2 triangles
mock_msms_vert_two_triangles = """# MSMS solvent excluded surface vertices
#vertex #sphere density probe_r
4 8 1.00 1.40
0.000 0.000 0.000 0.000 0.000 1.000 0 1 2
1.000 0.000 0.000 0.000 0.000 1.000 0 2 2
0.000 1.000 0.000 0.000 0.000 1.000 0 3 2
1.000 1.000 0.000 0.000 0.000 1.000 0 4 2
"""

mock_msms_face_two_triangles = """# MSMS solvent excluded surface faces
#faces #sphere density probe_r
2 8 1.00 1.40
1 2 3 1 1
2 3 4 1 1
"""

# Test data for msms_surface function
sample_spheres_for_msms: List[Tuple[np.ndarray, float]] = [
    (np.array([10.123, 20.456, 30.789]), 1.500),
    (np.array([15.000, 25.000, 35.000]), 2.000),
]

expected_xyzr_output = """ 10.123  20.456  30.789 1.500
 15.000  25.000  35.000 2.000
"""
