# -*- coding: utf-8 -*-

# TransportTools, a library for massive analyses of internal voids in biomolecules and ligand transport through them
# Copyright (C) 2021 The TransportTools Authors
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

import unittest
import tempfile
import os
import numpy as np
from unittest.mock import patch, MagicMock
from typing import List, Tuple

from transport_tools.libs.msms import filter_spheres, _convert_msms_to_cgo, msms_surface

from transport_tools.tests.units.data.data_msms import (
    sample_spheres_no_duplicates,
    sample_spheres_with_duplicates,
    sample_spheres_all_duplicates,
    sample_spheres_single,
    sample_spheres_empty,
    mock_msms_vert_content,
    mock_msms_face_content,
    mock_msms_vert_simple,
    mock_msms_face_simple,
    mock_msms_vert_two_triangles,
    mock_msms_face_two_triangles,
    sample_spheres_for_msms,
    expected_xyzr_output,
)


class TestFilterSpheres(unittest.TestCase):
    """Test cases for filter_spheres function"""

    def test_filter_spheres_no_duplicates(self):
        """Test filtering when there are no duplicates"""
        result = filter_spheres(sample_spheres_no_duplicates)
        self.assertEqual(len(result), 3)
        # Verify all original spheres are present
        for sphere in sample_spheres_no_duplicates:
            found = False
            for result_sphere in result:
                if (np.allclose(result_sphere[0], sphere[0]) and
                    np.isclose(result_sphere[1], sphere[1])):
                    found = True
                    break
            self.assertTrue(found, f"Original sphere {sphere} not found in result")

    def test_filter_spheres_with_duplicates(self):
        """Test filtering when duplicates are present"""
        result = filter_spheres(sample_spheres_with_duplicates)
        self.assertEqual(len(result), 3, "Should have 3 unique spheres")
        # Verify each unique sphere appears exactly once
        unique_coords = set()
        for sphere in result:
            coord_tuple = tuple([*sphere[0], sphere[1]])
            self.assertNotIn(coord_tuple, unique_coords, "Duplicate found in filtered result")
            unique_coords.add(coord_tuple)

    def test_filter_spheres_all_duplicates(self):
        """Test filtering when all spheres are duplicates"""
        result = filter_spheres(sample_spheres_all_duplicates)
        self.assertEqual(len(result), 1, "Should have only 1 unique sphere")
        # Verify it's the correct sphere
        self.assertTrue(np.allclose(result[0][0], np.array([1.0, 2.0, 3.0])))
        self.assertAlmostEqual(result[0][1], 1.5)

    def test_filter_spheres_single_sphere(self):
        """Test filtering with a single sphere"""
        result = filter_spheres(sample_spheres_single)
        self.assertEqual(len(result), 1)
        self.assertTrue(np.allclose(result[0][0], np.array([1.0, 2.0, 3.0])))
        self.assertAlmostEqual(result[0][1], 1.5)

    def test_filter_spheres_empty_list(self):
        """Test filtering with an empty list"""
        result = filter_spheres(sample_spheres_empty)
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, list)

    def test_filter_spheres_preserves_type(self):
        """Test that filtered spheres maintain correct types"""
        result = filter_spheres(sample_spheres_no_duplicates)
        for sphere in result:
            self.assertIsInstance(sphere, tuple)
            self.assertIsInstance(sphere[0], np.ndarray)
            self.assertIsInstance(sphere[1], (float, np.floating))
            self.assertEqual(sphere[0].shape, (3,))

    def test_filter_spheres_different_radii_same_coords(self):
        """Test that spheres with same coords but different radii are not filtered"""
        spheres = [
            (np.array([1.0, 2.0, 3.0]), 1.5),
            (np.array([1.0, 2.0, 3.0]), 2.0),  # Same coords, different radius
        ]
        result = filter_spheres(spheres)
        self.assertEqual(len(result), 2, "Spheres with different radii should not be filtered")


class TestConvertMsmsToCgo(unittest.TestCase):
    """Test cases for _convert_msms_to_cgo function"""

    def setUp(self):
        """Set up temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp(prefix="TT_test_msms_")

    def tearDown(self):
        """Clean up temporary directory"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_mesh_files(self, basename: str, vert_content: str, face_content: str):
        """Helper to create mock msms mesh files"""
        vert_file = os.path.join(self.temp_dir, basename + ".vert")
        face_file = os.path.join(self.temp_dir, basename + ".face")

        with open(vert_file, "w") as f:
            f.write(vert_content)

        with open(face_file, "w") as f:
            f.write(face_content)

        return os.path.join(self.temp_dir, basename)

    @patch('transport_tools.libs.utils.get_caver_color')
    def test_convert_msms_to_cgo_basic(self, mock_get_color):
        """Test basic conversion of msms mesh to CGO"""
        mock_get_color.return_value = [1.0, 0.0, 0.0]  # Red color

        meshfile = self._create_mesh_files("test_mesh", mock_msms_vert_simple,
                                           mock_msms_face_simple)

        result = _convert_msms_to_cgo(meshfile, color_id=1)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # Check CGO header
        self.assertEqual(result[0], 2.0)  # BEGIN
        self.assertEqual(result[1], 4.0)  # TRIANGLES
        self.assertEqual(result[2], 6.0)  # COLOR
        # Check color values
        self.assertEqual(result[3], 1.0)
        self.assertEqual(result[4], 0.0)
        self.assertEqual(result[5], 0.0)
        # Check for END marker
        self.assertEqual(result[-1], 3.0)

    @patch('transport_tools.libs.utils.get_caver_color')
    def test_convert_msms_to_cgo_with_none_color(self, mock_get_color):
        """Test conversion with None color_id"""
        mock_get_color.return_value = [0.5, 0.5, 0.5]  # Default gray

        meshfile = self._create_mesh_files("test_mesh_none", mock_msms_vert_simple,
                                           mock_msms_face_simple)

        result = _convert_msms_to_cgo(meshfile, color_id=None)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        mock_get_color.assert_called_once_with(None)

    @patch('transport_tools.libs.utils.get_caver_color')
    def test_convert_msms_to_cgo_structure(self, mock_get_color):
        """Test CGO structure with single triangle"""
        mock_get_color.return_value = [1.0, 1.0, 1.0]

        meshfile = self._create_mesh_files("test_structure", mock_msms_vert_simple,
                                           mock_msms_face_simple)

        result = _convert_msms_to_cgo(meshfile, color_id=5)

        # For 1 triangle with 3 vertices, we expect:
        # - Header: BEGIN (2.0), TRIANGLES (4.0), COLOR (6.0), R, G, B
        # - For each of 3 vertices: NORMAL (5.0), nx, ny, nz, VERTEX (4.0), x, y, z
        # - END (3.0)
        expected_min_length = 6 + 3 * 8 + 1  # 31 elements
        self.assertGreaterEqual(len(result), expected_min_length)

    @patch('transport_tools.libs.utils.get_caver_color')
    def test_convert_msms_to_cgo_two_triangles(self, mock_get_color):
        """Test CGO with two triangles"""
        mock_get_color.return_value = [0.0, 1.0, 0.0]

        meshfile = self._create_mesh_files("test_two_tri", mock_msms_vert_two_triangles,
                                           mock_msms_face_two_triangles)

        result = _convert_msms_to_cgo(meshfile, color_id=2)

        # For 2 triangles with 6 vertices total:
        # Header (6) + 2 triangles * 3 vertices * 8 elements + END (1)
        expected_min_length = 6 + 6 * 8 + 1  # 55 elements
        self.assertGreaterEqual(len(result), expected_min_length)

    @patch('transport_tools.libs.utils.get_caver_color')
    def test_convert_msms_to_cgo_vertices_and_normals(self, mock_get_color):
        """Test that vertices and normals are correctly read"""
        mock_get_color.return_value = [1.0, 0.0, 0.0]

        meshfile = self._create_mesh_files("test_vn", mock_msms_vert_content,
                                           mock_msms_face_content)

        result = _convert_msms_to_cgo(meshfile, color_id=1)

        self.assertIsInstance(result, list)
        # Check that we have float values (vertices and normals)
        for val in result:
            self.assertIsInstance(val, (float, int, np.floating))


class TestMsmsSurface(unittest.TestCase):
    """Test cases for msms_surface function using real MSMS binary"""

    @classmethod
    def setUpClass(cls):
        """Check if MSMS binary is available"""
        import shutil

        # Check multiple possible locations for MSMS binary
        # Priority: environment variable > PATH > common locations
        possible_paths = [
            os.environ.get('MSMS_BIN'),  # Environment variable
        ]

        # Check for msms in system PATH
        msms_in_path = shutil.which('msms')
        if msms_in_path:
            possible_paths.append(msms_in_path)

        # Add common installation locations
        possible_paths.extend([
            "/usr/local/bin/msms",  # Common system location
            "/usr/bin/msms",  # System bin
            "/opt/msms/msms.x86_64Linux2.2.6.1",  # Alternative system location
            os.path.expanduser("~/software/msms/msms.x86_64Linux2.2.6.1"),  # User home
            # Check for other common platform-specific names
            "/usr/local/bin/msms.x86_64Linux2.2.6.1",
            "/opt/msms/bin/msms",
        ])

        cls.msms_binary = None
        cls.msms_available = False

        for path in possible_paths:
            if path and os.path.exists(path) and os.access(path, os.X_OK):
                cls.msms_binary = path
                cls.msms_available = True
                break

        if not cls.msms_available:
            print(f"\nWarning: MSMS binary not found. Checked locations:")
            for path in possible_paths:
                if path:
                    print(f"  - {path}")
            print("  Set MSMS_BIN environment variable to specify custom location.")
            print("  Or ensure 'msms' is in your system PATH.")

    def setUp(self):
        """Set up test environment"""
        if not self.msms_available:
            self.skipTest("MSMS binary not available")
        # Assert for type checker - we know msms_binary is not None if available
        assert self.msms_binary is not None, "MSMS binary should be set when available"

    def test_msms_surface_basic_execution(self):
        """Test basic msms_surface execution with real MSMS"""
        # Use realistic protein-like sphere coordinates
        spheres = [
            (np.array([10.0, 20.0, 30.0]), 1.7),  # C atom radius
            (np.array([11.5, 20.5, 30.5]), 1.55), # N atom radius
            (np.array([9.0, 19.5, 29.5]), 1.52),  # O atom radius
            (np.array([10.5, 21.0, 31.0]), 1.2),  # H atom radius
        ]

        result = msms_surface(self.msms_binary, spheres, color_id=1, probe_radius=1.4)  # type: ignore[arg-type]

        # Verify result structure
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # Check CGO header elements
        self.assertEqual(result[0], 2.0)  # BEGIN
        self.assertEqual(result[1], 4.0)  # TRIANGLES
        self.assertEqual(result[2], 6.0)  # COLOR
        # Check for END marker
        self.assertEqual(result[-1], 3.0)  # END

    def test_msms_surface_single_sphere(self):
        """Test msms_surface with a single sphere"""
        single_sphere = [(np.array([15.0, 15.0, 15.0]), 2.0)]

        result = msms_surface(self.msms_binary, single_sphere, color_id=2, probe_radius=1.4)  # type: ignore[arg-type]

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0], 2.0)  # BEGIN
        self.assertEqual(result[-1], 3.0)  # END

    def test_msms_surface_multiple_spheres(self):
        """Test msms_surface with multiple spheres creating a surface"""
        # Create a cluster of spheres
        spheres = [
            (np.array([0.0, 0.0, 0.0]), 2.0),
            (np.array([2.5, 0.0, 0.0]), 2.0),
            (np.array([0.0, 2.5, 0.0]), 2.0),
            (np.array([0.0, 0.0, 2.5]), 2.0),
            (np.array([1.25, 1.25, 1.25]), 1.5),
        ]

        result = msms_surface(self.msms_binary, spheres, color_id=3, probe_radius=1.4)  # type: ignore[arg-type]

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 50)  # Should have substantial CGO data
        # Verify structure
        self.assertEqual(result[0], 2.0)
        self.assertEqual(result[1], 4.0)
        self.assertEqual(result[2], 6.0)
        self.assertEqual(result[-1], 3.0)

    def test_msms_surface_different_probe_radii(self):
        """Test msms_surface with different probe_radius values"""
        spheres = [
            (np.array([5.0, 5.0, 5.0]), 2.0),
            (np.array([7.5, 5.0, 5.0]), 2.0),
        ]

        for probe_radius in [1.0, 1.4, 2.0, 3.0]:
            result = msms_surface(self.msms_binary, spheres, color_id=1,  # type: ignore[arg-type]
                                probe_radius=probe_radius)

            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            # Different probe radii should potentially give different surface sizes
            # but all should produce valid CGO

    def test_msms_surface_with_none_color(self):
        """Test msms_surface with None color_id"""
        spheres = [(np.array([10.0, 10.0, 10.0]), 2.0)]

        result = msms_surface(self.msms_binary, spheres, color_id=None, probe_radius=1.4)  # type: ignore[arg-type]

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        # Should still produce valid CGO with default color
        self.assertEqual(result[0], 2.0)
        self.assertEqual(result[-1], 3.0)

    def test_msms_surface_realistic_protein_atoms(self):
        """Test msms_surface with realistic protein atom coordinates"""
        # Simulate a small peptide fragment with realistic coordinates
        spheres = [
            # Backbone atoms
            (np.array([24.350, 32.450, 15.230]), 1.7),   # CA
            (np.array([25.120, 31.890, 14.030]), 1.7),   # C
            (np.array([24.920, 32.380, 12.910]), 1.52),  # O
            (np.array([23.890, 31.480, 16.130]), 1.7),   # CB
            (np.array([26.010, 30.920, 14.250]), 1.55),  # N
            # Add some hydrogens
            (np.array([24.920, 33.480, 15.010]), 1.2),   # H
            (np.array([23.120, 32.010, 16.890]), 1.2),   # H
        ]

        result = msms_surface(self.msms_binary, spheres, color_id=5, probe_radius=1.4)  # type: ignore[arg-type]

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 100)  # Should generate substantial surface
        # Verify CGO structure is intact
        self.assertEqual(result[0], 2.0)
        self.assertEqual(result[1], 4.0)
        self.assertEqual(result[-1], 3.0)


if __name__ == '__main__':
    unittest.main()
