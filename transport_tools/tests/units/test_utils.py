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

__version__ = '0.9.8'
__author__ = 'Jan Brezovsky'
__mail__ = 'janbre@amu.edu.pl'

import unittest
import os
import tempfile
import shutil
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
        self.assertTrue(np.allclose(_build_grid(spheres, x_points, y_points, z_points), grid, atol=1e-3))

    def test__get_mesh(self):
        try:
            import mcubes  # type: ignore[import-untyped]
        except ModuleNotFoundError:
            self.skipTest("mcubes not installed")
            return
        from transport_tools.libs.utils import _get_mesh
        from transport_tools.tests.units.data.data_utils import x_points, y_points, z_points, grid, vertices, normals, \
            triangles

        results = _get_mesh(grid, x_points, y_points, z_points)
        self.assertTrue(np.allclose(vertices, results[0], atol=1e-3))
        self.assertTrue(np.allclose(normals, results[1], atol=1e-3, equal_nan=True))
        self.assertTrue(np.allclose(triangles, results[2], atol=1e-3))

    def test_convert_spheres2cgo_surface(self):
        try:
            import mcubes  # type: ignore[import-untyped]
        except ModuleNotFoundError:
            self.skipTest("mcubes not installed")
            return

        from transport_tools.libs.utils import convert_spheres2cgo_surface
        from transport_tools.tests.units.data.data_utils import spheres, cgo_surf

        for item1, item2 in zip(cgo_surf, convert_spheres2cgo_surface(spheres, 1, resolution=0.5)):
            self.assertAlmostEqual(item1, item2)

    def test_node_labels_split(self):
        from transport_tools.libs.utils import node_labels_split

        self.assertSequenceEqual((0, 1), node_labels_split("0_1"))
        self.assertSequenceEqual((10, 3), node_labels_split("10_3"))
        self.assertSequenceEqual((8, 105), node_labels_split("8_105"))
        self.assertSequenceEqual((-99, -99), node_labels_split("SP"))

    def test_test_file_exists(self):
        """Verify test_file() does not raise for existing files"""
        from transport_tools.libs.utils import test_file

        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        try:
            test_file_path = os.path.join(temp_dir, "test.txt")
            with open(test_file_path, "w") as f:
                f.write("test")

            # Should not raise
            test_file(test_file_path)
        finally:
            shutil.rmtree(temp_dir)

    def test_test_file_not_exists(self):
        """Verify test_file() raises FileNotFoundError for non-existing files"""
        from transport_tools.libs.utils import test_file

        with self.assertRaises(FileNotFoundError) as context:
            test_file("/nonexistent/path/to/file.txt")

        self.assertIn("does not exist", str(context.exception))

    def test_get_filepath_single_match(self):
        """Verify get_filepath() returns path for unique file match"""
        from transport_tools.libs.utils import get_filepath

        temp_dir = tempfile.mkdtemp()
        try:
            # Create a single test file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test")

            # Should return the file path
            result = get_filepath(temp_dir, "test.txt")
            self.assertTrue(result.endswith("test.txt"))
            self.assertTrue(os.path.exists(result))
        finally:
            shutil.rmtree(temp_dir)

    def test_get_filepath_pattern_match(self):
        """Verify get_filepath() works with glob patterns"""
        from transport_tools.libs.utils import get_filepath

        temp_dir = tempfile.mkdtemp()
        try:
            # Create a single test file with specific extension
            test_file = os.path.join(temp_dir, "test.dat")
            with open(test_file, "w") as f:
                f.write("test")

            # Should find file with pattern
            result = get_filepath(temp_dir, "*.dat")
            self.assertTrue(result.endswith(".dat"))
        finally:
            shutil.rmtree(temp_dir)

    def test_get_filepath_no_match(self):
        """Verify get_filepath() raises RuntimeError when no file matches"""
        from transport_tools.libs.utils import get_filepath

        temp_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(RuntimeError) as context:
                get_filepath(temp_dir, "nonexistent.txt")

            self.assertIn("does not exist", str(context.exception))
        finally:
            shutil.rmtree(temp_dir)

    def test_get_filepath_multiple_matches(self):
        """Verify get_filepath() raises RuntimeError when multiple files match"""
        from transport_tools.libs.utils import get_filepath

        temp_dir = tempfile.mkdtemp()
        try:
            # Create multiple matching files
            for i in range(3):
                with open(os.path.join(temp_dir, f"test{i}.txt"), "w") as f:
                    f.write("test")

            with self.assertRaises(RuntimeError) as context:
                get_filepath(temp_dir, "*.txt")

            self.assertIn("more then 1 file", str(context.exception))
        finally:
            shutil.rmtree(temp_dir)

    def test_splitall_absolute_path(self):
        """Verify splitall() correctly splits absolute paths"""
        from transport_tools.libs.utils import splitall

        if os.name == 'nt':  # Windows
            result = splitall("C:\\Users\\test\\file.txt")
            self.assertIn("Users", result)
            self.assertIn("test", result)
            self.assertEqual(result[-1], "file.txt")
        else:  # Unix-like
            result = splitall("/home/user/test/file.txt")
            self.assertEqual(result[0], "/")
            self.assertIn("home", result)
            self.assertIn("user", result)
            self.assertIn("test", result)
            self.assertEqual(result[-1], "file.txt")

    def test_splitall_relative_path(self):
        """Verify splitall() correctly splits relative paths"""
        from transport_tools.libs.utils import splitall

        result = splitall("folder/subfolder/file.txt")
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "folder")
        self.assertEqual(result[1], "subfolder")
        self.assertEqual(result[2], "file.txt")

    def test_splitall_single_component(self):
        """Verify splitall() handles single path component"""
        from transport_tools.libs.utils import splitall

        result = splitall("file.txt")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "file.txt")

    def test_path_loader_string_simple(self):
        """Verify path_loader_string() generates os.path.join code"""
        from transport_tools.libs.utils import path_loader_string

        result = path_loader_string("folder/file.txt")
        self.assertIn("os.path.join", result)
        self.assertIn("'folder'", result)
        self.assertIn("'file.txt'", result)

    def test_path_loader_string_nested(self):
        """Verify path_loader_string() handles nested paths"""
        from transport_tools.libs.utils import path_loader_string

        result = path_loader_string("a/b/c/file.txt")
        self.assertIn("os.path.join", result)
        self.assertIn("'a'", result)
        self.assertIn("'b'", result)
        self.assertIn("'c'", result)
        self.assertIn("'file.txt'", result)

    def test_set_paths_from_package_root(self):
        """Verify set_paths_from_package_root() constructs correct paths"""
        from transport_tools.libs.utils import set_paths_from_package_root
        import transport_tools

        # Should return path within package
        result = set_paths_from_package_root("tests", "data")

        # Check it contains package directory
        package_dir = os.path.dirname(transport_tools.__file__)
        self.assertTrue(result.startswith(package_dir))
        self.assertTrue(result.endswith(os.path.join("tests", "data")))

    def test_set_paths_from_package_root_no_args(self):
        """Verify set_paths_from_package_root() works with no arguments"""
        from transport_tools.libs.utils import set_paths_from_package_root
        import transport_tools

        result = set_paths_from_package_root()
        package_dir = os.path.dirname(transport_tools.__file__)
        self.assertEqual(result, package_dir)


class TestGetResidueColor(unittest.TestCase):
    """
    Unit tests for libs.utils.get_residue_color() — the per-residue palette introduced
    so transport-event paths can be visually separated by residue type in PyMOL.
    """

    def test_returns_list_of_three_floats(self):
        """Output must be a list of three floats (RGB)"""
        from transport_tools.libs.utils import get_residue_color
        color = get_residue_color(0)
        self.assertIsInstance(color, list)
        self.assertEqual(len(color), 3)
        for component in color:
            self.assertIsInstance(component, float)

    def test_all_values_in_0_1_range(self):
        """All RGB components for ranks up to wrap-around must lie in [0.0, 1.0]"""
        from transport_tools.libs.utils import get_residue_color
        for rank in range(0, 200):
            color = get_residue_color(rank)
            for component in color:
                self.assertGreaterEqual(component, 0.0)
                self.assertLessEqual(component, 1.0)

    def test_rank_0_returns_cyan(self):
        """Rank 0 maps to offset 3 in the palette = cyan, intentionally distinct from
        the first CAVER color (blue) so residue paths stand out from SC tunnels."""
        from transport_tools.libs.utils import get_residue_color
        self.assertListEqual([0.0, 1.0, 1.0], get_residue_color(0))

    def test_consecutive_ranks_differ(self):
        """Consecutive residue ranks must produce distinct colors"""
        from transport_tools.libs.utils import get_residue_color
        for rank in range(0, 10):
            self.assertNotEqual(get_residue_color(rank), get_residue_color(rank + 1))

    def test_wraps_around_for_large_rank(self):
        """Function must remain valid (no IndexError) for ranks larger than the palette
        and continue returning well-formed colors via modulo wrap-around."""
        from transport_tools.libs.utils import get_residue_color
        for rank in (0, 50, 100, 500, 1000, 10000):
            color = get_residue_color(rank)
            self.assertEqual(len(color), 3)
            for component in color:
                self.assertGreaterEqual(component, 0.0)
                self.assertLessEqual(component, 1.0)

    def test_differs_from_caver_color_for_first_three_ranks(self):
        """First three residue colors must differ from the first three CAVER colors —
        the offset is the whole point of having a separate palette."""
        from transport_tools.libs.utils import get_residue_color, get_caver_color
        for rank in range(3):
            self.assertNotEqual(get_residue_color(rank), get_caver_color(rank))


class TestSnapshotFrameMap(unittest.TestCase):
    """Frame<->snapshot mapping under CAVER snapshot sparsity (caver_snapshot_stride)."""

    def test_default_stride1_matches_legacy(self):
        """At stride 1 the map reproduces the legacy 'frame + offset' behaviour exactly."""
        from transport_tools.libs.utils import SnapshotFrameMap
        m = SnapshotFrameMap(5, offset=1, stride=1)
        self.assertEqual([m.frame_to_snap(f) for f in range(5)], [1, 2, 3, 4, 5])
        self.assertIsNone(m.frame_to_snap(5))      # frame 5 -> snap 6 is outside 1..5
        self.assertIsNone(m.frame_to_snap(-1))
        self.assertEqual(m.snapshot_ids(), [1, 2, 3, 4, 5])
        self.assertEqual(m.num_analyzed_snapshots, 5)
        self.assertEqual([m.snap_to_frame(s) for s in (1, 2, 5)], [0, 1, 4])

    def test_sequential_stride_exact(self):
        """Sequential, stride 20: only frames on the analysed grid map; the rest are None."""
        from transport_tools.libs.utils import SnapshotFrameMap
        m = SnapshotFrameMap(1000, offset=1, stride=20, by_frame=False, id_stride=1)
        self.assertEqual(m.frame_to_snap(0), 1)
        self.assertEqual(m.frame_to_snap(20), 2)
        self.assertEqual(m.frame_to_snap(40), 3)
        self.assertEqual(m.frame_to_snap(19980), 1000)
        for f in (1, 5, 19, 21, 39):               # between analysed snapshots -> no tunnel
            self.assertIsNone(m.frame_to_snap(f))
        self.assertIsNone(m.frame_to_snap(20000))  # past the analysed domain
        self.assertEqual(m.snapshot_ids()[:3], [1, 2, 3])

    def test_sequential_stride_interpolate(self):
        """interpolate=True returns the nearest analysed snapshot (half-up), clamped to the domain."""
        from transport_tools.libs.utils import SnapshotFrameMap
        m = SnapshotFrameMap(1000, offset=1, stride=20, by_frame=False, id_stride=1)
        self.assertEqual(m.frame_to_snap(5, interpolate=True), 1)    # closest to frame 0
        self.assertEqual(m.frame_to_snap(10, interpolate=True), 2)   # tie -> half-up to frame 20
        self.assertEqual(m.frame_to_snap(15, interpolate=True), 2)
        self.assertEqual(m.frame_to_snap(19, interpolate=True), 2)
        self.assertEqual(m.frame_to_snap(30, interpolate=True), 3)
        self.assertEqual(m.frame_to_snap(10 ** 9, interpolate=True), 1000)  # clamped to last
        self.assertEqual(m.frame_to_snap(-5, interpolate=True), 1)          # clamped to first

    def test_by_frame_exact(self):
        """by_frame: IDs are frame numbers; only grid frames map, to the real observed IDs."""
        from transport_tools.libs.utils import SnapshotFrameMap
        m = SnapshotFrameMap(1000, offset=1, stride=20, by_frame=True, id_stride=20)
        self.assertEqual(m.frame_to_snap(0), 1)
        self.assertEqual(m.frame_to_snap(20), 21)
        self.assertEqual(m.frame_to_snap(40), 41)
        for f in (1, 5, 19):
            self.assertIsNone(m.frame_to_snap(f))
        self.assertEqual(m.snapshot_ids()[:3], [1, 21, 41])

    def test_by_frame_interpolate(self):
        from transport_tools.libs.utils import SnapshotFrameMap
        m = SnapshotFrameMap(1000, offset=1, stride=20, by_frame=True, id_stride=20)
        self.assertEqual(m.frame_to_snap(5, interpolate=True), 1)
        self.assertEqual(m.frame_to_snap(15, interpolate=True), 21)
        self.assertEqual(m.frame_to_snap(25, interpolate=True), 21)

    def test_sequential_and_by_frame_match_same_physical_frames(self):
        """Same physical sampling, two labellings: both match the same frames and the same physical time."""
        from transport_tools.libs.utils import SnapshotFrameMap
        seq = SnapshotFrameMap(1000, offset=1, stride=20, by_frame=False, id_stride=1)
        byf = SnapshotFrameMap(1000, offset=1, stride=20, by_frame=True, id_stride=20)
        for f in range(0, 200):
            matched_seq = seq.frame_to_snap(f) is not None
            matched_byf = byf.frame_to_snap(f) is not None
            self.assertEqual(matched_seq, matched_byf, "labellings disagree on frame {}".format(f))
            if matched_seq:
                # the two snapshot IDs label the same physical source-trajectory frame
                self.assertEqual(seq.snap_to_frame(seq.frame_to_snap(f)),
                                 byf.snap_to_frame(byf.frame_to_snap(f)))

    def test_snap_to_frame_roundtrip(self):
        from transport_tools.libs.utils import SnapshotFrameMap
        for m in (SnapshotFrameMap(50, offset=1, stride=20, by_frame=False, id_stride=1),
                  SnapshotFrameMap(50, offset=1, stride=20, by_frame=True, id_stride=20),
                  SnapshotFrameMap(50, offset=0, stride=1)):
            for snap in m.snapshot_ids():
                self.assertEqual(m.frame_to_snap(m.snap_to_frame(snap)), snap)

    def test_detect_sequential_from_dense_ids(self):
        """Dense observed IDs => sequential; the configured stride is honoured, id_stride stays 1."""
        from transport_tools.libs.utils import SnapshotFrameMap
        p = {"snapshots_per_simulation": 1000, "caver_traj_offset": 1, "caver_snapshot_stride": 20}
        m = SnapshotFrameMap.from_parameters(p, observed_ids=list(range(1, 1001)))
        self.assertFalse(m.by_frame)
        self.assertEqual(m.id_stride, 1)
        self.assertEqual(m.map_stride, 20)             # the user's stride drives the mapping
        self.assertEqual(m.frame_to_snap(20), 2)
        self.assertEqual(m.snapshot_ids()[:3], [1, 2, 3])

    def test_detect_by_frame_from_spaced_ids(self):
        """Spaced observed IDs => by_frame; stride is read from the data gap."""
        from transport_tools.libs.utils import SnapshotFrameMap
        p = {"snapshots_per_simulation": 1000, "caver_traj_offset": 1, "caver_snapshot_stride": 1}
        m = SnapshotFrameMap.from_parameters(p, observed_ids=[1 + 20 * i for i in range(1000)])
        self.assertTrue(m.by_frame)
        self.assertEqual(m.id_stride, 20)
        self.assertEqual(m.frame_to_snap(20), 21)

    def test_detect_sequential_with_trailing_holes(self):
        """A sequential run ending tunnel-less (max ID < N) must still detect as sequential, not by_frame."""
        from transport_tools.libs.utils import SnapshotFrameMap
        p = {"snapshots_per_simulation": 1000, "caver_traj_offset": 1, "caver_snapshot_stride": 1}
        m = SnapshotFrameMap.from_parameters(p, observed_ids=list(range(1, 996)))  # 996..1000 absent
        self.assertFalse(m.by_frame)
        self.assertEqual(m.id_stride, 1)

    def test_detect_sequential_with_interior_holes(self):
        """Interior gaps keep gcd at 1 => sequential."""
        from transport_tools.libs.utils import SnapshotFrameMap
        p = {"snapshots_per_simulation": 10, "caver_traj_offset": 1, "caver_snapshot_stride": 1}
        m = SnapshotFrameMap.from_parameters(p, observed_ids=[1, 2, 3, 5, 6, 7])  # 4 missing
        self.assertFalse(m.by_frame)

    def test_resolve_into_caches_and_is_idempotent(self):
        """resolve_into caches the labelling on the parameters dict and keeps an already-cached one."""
        from transport_tools.libs.utils import SnapshotFrameMap as M
        p = {"snapshots_per_simulation": 100, "caver_traj_offset": 1, "caver_snapshot_stride": 1}
        m = M.resolve_into(p, observed_ids=[1 + 20 * i for i in range(100)])
        self.assertTrue(m.by_frame)
        self.assertEqual(p[M.MODE_KEY], True)
        self.assertEqual(p[M.ID_STRIDE_KEY], 20)
        # a second call with contradicting observed IDs keeps the cached labelling
        m2 = M.resolve_into(p, observed_ids=list(range(1, 101)))
        self.assertTrue(m2.by_frame)

    def test_from_parameters_uses_cached_mode_without_observed_ids(self):
        from transport_tools.libs.utils import SnapshotFrameMap as M
        p = {"snapshots_per_simulation": 100, "caver_traj_offset": 1, "caver_snapshot_stride": 20,
             M.MODE_KEY: True, M.ID_STRIDE_KEY: 20}
        m = M.from_parameters(p)
        self.assertTrue(m.by_frame)
        self.assertEqual(m.id_stride, 20)

    def test_from_parameters_fallback_sequential(self):
        """With neither observed IDs nor a cached mode, fall back to sequential (back-compatible)."""
        from transport_tools.libs.utils import SnapshotFrameMap
        p = {"snapshots_per_simulation": 100, "caver_traj_offset": 1}
        m = SnapshotFrameMap.from_parameters(p)
        self.assertFalse(m.by_frame)
        self.assertEqual(m.id_stride, 1)
        self.assertEqual(m.map_stride, 1)

    def test_derive_stride(self):
        from transport_tools.libs.utils import SnapshotFrameMap
        self.assertEqual(SnapshotFrameMap.derive_stride(20000, 1000), (20, True))
        self.assertEqual(SnapshotFrameMap.derive_stride(20001, 1000), (20, False))
        self.assertEqual(SnapshotFrameMap.derive_stride(500, 0), (1, False))


class TestParseAquaductFramesWindow(unittest.TestCase):
    def test_parses_window_step1(self):
        from transport_tools.libs.utils import parse_aquaduct_frames_window
        self.assertEqual(parse_aquaduct_frames_window(["Frames window: 0:19999 step 1"]), 20000)

    def test_parses_window_with_step(self):
        from transport_tools.libs.utils import parse_aquaduct_frames_window
        self.assertEqual(parse_aquaduct_frames_window(["Frames window: 0:19998 step 2"]), 10000)

    def test_missing_line_returns_none(self):
        from transport_tools.libs.utils import parse_aquaduct_frames_window
        self.assertIsNone(parse_aquaduct_frames_window(["some other summary line", "no window here"]))

    def test_step_zero_returns_none(self):
        from transport_tools.libs.utils import parse_aquaduct_frames_window
        self.assertIsNone(parse_aquaduct_frames_window(["Frames window: 0:100 step 0"]))


if __name__ == "__main__":
    unittest.main()
