/*
 * Compiled CPU distance kernel for the stage-4 cluster-cluster distance calculation
 * (the 'local'/'slurm' stage04_backend - see geometry.calc_distance_batch_compiled). Built by
 * setup.py as the optional 'transport_tools.libs._distance_kernel' extension; geometry.py falls
 * back to the pure-Python implementation when it fails to import (no compiler at install time).
 *
 * Same algorithm and node/path layout as the CUDA backends (_cuda_distance_kernel.py /
 * _cuda_distance_rawkernel.py) and their entry-point geometry.calc_distance_batch_kernel, kept in
 * sync by hand across all three implementations:
 *   - 'nodes' is a flat, row-major float64 array, 7 columns per node: x, y, z, layer_id,
 *     is_terminal, radius, rmsf (see geometry._prepare_compiled_path_sets for how it is built).
 *   - every other array is an integer offset/index into this shared node pool or into each other,
 *     avoiding any per-pair Python object overhead - the whole batch is processed in one call
 *     while the GIL is released (see Py_BEGIN_ALLOW_THREADS in distance_batch below).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <math.h>
#include <stdint.h>

/* Wrap a read-only Python buffer-protocol object (e.g. a NumPy array) as a Py_buffer, checking its
 * per-item size matches what the caller expects. */
static int get_read_buffer(PyObject *object, Py_ssize_t item_size, const char *name, Py_buffer *view) {
    if (PyObject_GetBuffer(object, view, PyBUF_CONTIG_RO) < 0) {
        return 0;
    }
    if (view->itemsize != item_size) {
        PyErr_Format(PyExc_TypeError, "%s has item size %zd, expected %zd", name, view->itemsize, item_size);
        PyBuffer_Release(view);
        return 0;
    }
    return 1;
}

/* Same as get_read_buffer, but for a buffer the kernel writes results into (e.g. the output array);
 * additionally rejects a read-only buffer. */
static int get_write_buffer(PyObject *object, Py_ssize_t item_size, const char *name, Py_buffer *view) {
    if (PyObject_GetBuffer(object, view, PyBUF_CONTIG) < 0) {
        return 0;
    }
    if (view->readonly) {
        PyErr_Format(PyExc_TypeError, "%s must be writable", name);
        PyBuffer_Release(view);
        return 0;
    }
    if (view->itemsize != item_size) {
        PyErr_Format(PyExc_TypeError, "%s has item size %zd, expected %zd", name, view->itemsize, item_size);
        PyBuffer_Release(view);
        return 0;
    }
    return 1;
}

/* True when a query node is close enough in "layer" (radial shell around the tunnel's starting
 * point) to a candidate node to be treated as a valid surface-distance correspondence; identical
 * rule to _cuda_distance_kernel.py's adjacent() device function - see that module's docstring for
 * the full rationale. Asymmetric in (query, candidate): path_pair_distance below calls it with
 * both node orders. */
static inline int adjacent(
    const double query_layer,
    const double candidate_layer,
    const double query_last_layer,
    const double candidate_last_layer,
    const double query_first_terminal
) {
    return fabs(query_layer - candidate_layer) <= 1.0
        || candidate_layer >= query_first_terminal
        || (query_layer == query_last_layer && candidate_layer > query_layer)
        || (query_layer > candidate_last_layer
            && (candidate_layer == candidate_last_layer
                || candidate_layer == candidate_last_layer - 1.0))
        || candidate_layer < 0.0;
}

/* Euclidean center-to-center distance between two nodes minus both their radii (columns 0-2 and 5
 * of the 7-column node layout - see the file header), clamped at 0 for overlapping/touching nodes. */
static inline double surface_distance(
    const double *nodes,
    const int64_t node_a,
    const int64_t node_b
) {
    const double *node_data_a = nodes + node_a * 7;
    const double *node_data_b = nodes + node_b * 7;
    const double dx = node_data_a[0] - node_data_b[0];
    const double dy = node_data_a[1] - node_data_b[1];
    const double dz = node_data_a[2] - node_data_b[2];
    const double distance = sqrt(dx * dx + dy * dy + dz * dz) - node_data_a[5] - node_data_b[5];
    return distance > 0.0 ? distance : 0.0;
}

/*
 * Average closest-node surface distance between one effective path from cluster A (path_a,
 * truncated to length_a nodes) and one from cluster B (path_b, length_b nodes) - the same quantity
 * computed by _cuda_distance_kernel.py's path_pair_node_distances kernel for one task, just
 * sequentially on the CPU instead of one CUDA block per task.
 *
 * Two passes: for each A-node find its closest adjacent B-node (below), then for each B-node find
 * its closest adjacent A-node (further down) - together a symmetric average over both paths' nodes.
 * *invalid is set when some node has no valid adjacent counterpart at all (mirrors task_invalid in
 * the CUDA kernels); the caller (compute_pair) then reports the pair as unresolved (NaN) so
 * geometry.calc_distance_batch_compiled can fall back to the exact reference implementation for it.
 */
static double path_pair_distance(
    const double *nodes,
    const int64_t *path_nodes,
    const int64_t *path_node_offsets,
    const int64_t path_a,
    const int64_t length_a,
    const int64_t path_b,
    const int64_t length_b,
    const double last_layer_a,
    const double last_layer_b,
    const double first_terminal_layer_a,
    const double first_terminal_layer_b,
    int *invalid
) {
    const int64_t offset_a = path_node_offsets[path_a];
    const int64_t offset_b = path_node_offsets[path_b];
    double path_sum = 0.0;

    for (int64_t index_a = 0; index_a < length_a; ++index_a) {
        const int64_t node_a = path_nodes[offset_a + index_a];
        const double layer_a = nodes[node_a * 7 + 3];
        if (layer_a < 0.0) {
            continue;
        }

        double minimum = INFINITY;
        for (int64_t index_b = 0; index_b < length_b; ++index_b) {
            const int64_t node_b = path_nodes[offset_b + index_b];
            const double layer_b = nodes[node_b * 7 + 3];
            if (adjacent(layer_a, layer_b, last_layer_a, last_layer_b, first_terminal_layer_a)) {
                minimum = fmin(minimum, surface_distance(nodes, node_a, node_b));
            }
        }
        if (!isfinite(minimum)) {
            *invalid = 1;
            return 0.0;
        }
        path_sum += minimum;
    }

    for (int64_t index_b = 0; index_b < length_b; ++index_b) {
        const int64_t node_b = path_nodes[offset_b + index_b];
        const double layer_b = nodes[node_b * 7 + 3];
        if (layer_b < 0.0) {
            continue;
        }

        /* adjacent() is asymmetric; probe under A's perspective first (as above) - if no A-node
         * qualifies that way, reverse_fallback stays set and the loop below re-evaluates
         * adjacency from B's own perspective instead, so this B-node is not spuriously marked
         * invalid just because the asymmetric test was evaluated from the "wrong" side. */
        int reverse_fallback = 1;
        for (int64_t index_a = 0; index_a < length_a; ++index_a) {
            const int64_t node_a = path_nodes[offset_a + index_a];
            const double layer_a = nodes[node_a * 7 + 3];
            if (adjacent(layer_a, layer_b, last_layer_a, last_layer_b, first_terminal_layer_a)) {
                reverse_fallback = 0;
                break;
            }
        }

        double minimum = INFINITY;
        for (int64_t index_a = 0; index_a < length_a; ++index_a) {
            const int64_t node_a = path_nodes[offset_a + index_a];
            const double layer_a = nodes[node_a * 7 + 3];
            const int is_adjacent = reverse_fallback
                ? adjacent(layer_b, layer_a, last_layer_b, last_layer_a, first_terminal_layer_b)
                : adjacent(layer_a, layer_b, last_layer_a, last_layer_b, first_terminal_layer_a);
            if (is_adjacent) {
                minimum = fmin(minimum, surface_distance(nodes, node_a, node_b));
            }
        }
        if (!isfinite(minimum)) {
            *invalid = 1;
            return 0.0;
        }
        path_sum += minimum;
    }

    /* -2 excludes each path's starting-point (SP) node from the average, matching
     * dists2evaluate in geometry.LayeredPathSet._avg_distance2path_set_reference */
    const int64_t denominator = length_a + length_b - 2;
    if (denominator <= 0) {
        *invalid = 1;
        return 0.0;
    }
    return path_sum / (double) denominator;
}

/*
 * Average distance for one cluster pair over every combination of their effective paths (nested
 * path/prefix loops below, mirroring the array layout built by geometry._prepare_compiled_path_sets)
 * - equivalent to what finalize_cluster_pair_distances averages from task_values in the CUDA
 * backends, just computed directly instead of from precomputed per-task results.
 * Early-outs: misaligned pairs skip straight to the 999.0 sentinel; once the running sum for a
 * non-exact pair exceeds 3x the cutoff, the pair is reported as 999.0 without finishing the
 * remaining path combinations, since callers with exact=False only care whether it is in range.
 */
static double compute_pair(
    const double *nodes,
    const int64_t *path_nodes,
    const int64_t *path_node_offsets,
    const int64_t *prefix_lengths,
    const int64_t *prefix_offsets,
    const int64_t *cluster_path_offsets,
    const int64_t *num_effective_paths,
    const double *last_layers,
    const double *first_terminal_layers,
    const int64_t cluster_a,
    const int64_t cluster_b,
    const uint8_t exact,
    const uint8_t misaligned,
    const double distance_cutoff
) {
    if (misaligned) {
        return 999.0;
    }

    const int64_t num_path_pairs = num_effective_paths[cluster_a] * num_effective_paths[cluster_b];
    if (num_path_pairs <= 0) {
        return NAN;
    }

    const double too_distant = 3.0 * distance_cutoff * (double) num_path_pairs;
    double sum = 0.0;

    for (int64_t path_a = cluster_path_offsets[cluster_a];
         path_a < cluster_path_offsets[cluster_a + 1];
         ++path_a) {
        for (int64_t prefix_a = prefix_offsets[path_a];
             prefix_a < prefix_offsets[path_a + 1];
             ++prefix_a) {
            const int64_t length_a = prefix_lengths[prefix_a];
            for (int64_t path_b = cluster_path_offsets[cluster_b];
                 path_b < cluster_path_offsets[cluster_b + 1];
                 ++path_b) {
                for (int64_t prefix_b = prefix_offsets[path_b];
                     prefix_b < prefix_offsets[path_b + 1];
                     ++prefix_b) {
                    const int64_t length_b = prefix_lengths[prefix_b];
                    int invalid = 0;
                    sum += path_pair_distance(
                        nodes,
                        path_nodes,
                        path_node_offsets,
                        path_a,
                        length_a,
                        path_b,
                        length_b,
                        last_layers[cluster_a],
                        last_layers[cluster_b],
                        first_terminal_layers[cluster_a],
                        first_terminal_layers[cluster_b],
                        &invalid);
                    if (invalid) {
                        return NAN;
                    }
                    if (!exact && sum > too_distant) {
                        return 999.0;
                    }
                }
            }
        }
    }

    return sum / (double) num_path_pairs;
}

/*
 * Python entry point (transport_tools.libs._distance_kernel.distance_batch), called from
 * geometry.calc_distance_batch_compiled with the arrays built by
 * geometry._prepare_compiled_path_sets() plus per-pair cluster_pairs/exact_pairs/misaligned_pairs
 * and the pre-allocated 'output' array it fills in place. Every array argument must support the
 * buffer protocol (a contiguous NumPy array); this function itself does not allocate or return any
 * new array - it only validates buffer sizes, unwraps them to raw pointers, and loops
 * compute_pair() over every requested pair with the GIL released.
 */
static PyObject *distance_batch(PyObject *self, PyObject *args) {
    (void) self;

    PyObject *nodes_object;
    PyObject *path_nodes_object;
    PyObject *path_node_offsets_object;
    PyObject *prefix_lengths_object;
    PyObject *prefix_offsets_object;
    PyObject *cluster_path_offsets_object;
    PyObject *num_effective_paths_object;
    PyObject *last_layers_object;
    PyObject *first_terminal_layers_object;
    PyObject *cluster_pairs_object;
    PyObject *exact_pairs_object;
    PyObject *misaligned_pairs_object;
    PyObject *output_object;
    double distance_cutoff;

    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOOOOOOdO",
            &nodes_object,
            &path_nodes_object,
            &path_node_offsets_object,
            &prefix_lengths_object,
            &prefix_offsets_object,
            &cluster_path_offsets_object,
            &num_effective_paths_object,
            &last_layers_object,
            &first_terminal_layers_object,
            &cluster_pairs_object,
            &exact_pairs_object,
            &misaligned_pairs_object,
            &distance_cutoff,
            &output_object)) {
        return NULL;
    }

    Py_buffer nodes_view = {0};
    Py_buffer path_nodes_view = {0};
    Py_buffer path_node_offsets_view = {0};
    Py_buffer prefix_lengths_view = {0};
    Py_buffer prefix_offsets_view = {0};
    Py_buffer cluster_path_offsets_view = {0};
    Py_buffer num_effective_paths_view = {0};
    Py_buffer last_layers_view = {0};
    Py_buffer first_terminal_layers_view = {0};
    Py_buffer cluster_pairs_view = {0};
    Py_buffer exact_pairs_view = {0};
    Py_buffer misaligned_pairs_view = {0};
    Py_buffer output_view = {0};

    if (!get_read_buffer(nodes_object, sizeof(double), "nodes", &nodes_view)
        || !get_read_buffer(path_nodes_object, sizeof(int64_t), "path_nodes", &path_nodes_view)
        || !get_read_buffer(
            path_node_offsets_object, sizeof(int64_t), "path_node_offsets", &path_node_offsets_view)
        || !get_read_buffer(prefix_lengths_object, sizeof(int64_t), "prefix_lengths", &prefix_lengths_view)
        || !get_read_buffer(prefix_offsets_object, sizeof(int64_t), "prefix_offsets", &prefix_offsets_view)
        || !get_read_buffer(
            cluster_path_offsets_object, sizeof(int64_t), "cluster_path_offsets", &cluster_path_offsets_view)
        || !get_read_buffer(
            num_effective_paths_object, sizeof(int64_t), "num_effective_paths", &num_effective_paths_view)
        || !get_read_buffer(last_layers_object, sizeof(double), "last_layers", &last_layers_view)
        || !get_read_buffer(
            first_terminal_layers_object, sizeof(double), "first_terminal_layers", &first_terminal_layers_view)
        || !get_read_buffer(cluster_pairs_object, sizeof(int64_t), "cluster_pairs", &cluster_pairs_view)
        || !get_read_buffer(exact_pairs_object, sizeof(uint8_t), "exact_pairs", &exact_pairs_view)
        || !get_read_buffer(misaligned_pairs_object, sizeof(uint8_t), "misaligned_pairs", &misaligned_pairs_view)
        || !get_write_buffer(output_object, sizeof(double), "output", &output_view)) {
        goto release_buffers;
    }

    /* num_pairs/num_clusters are derived from the output/cluster_path_offsets buffer lengths
     * (rather than taken as explicit arguments) since geometry.py always sizes those two to match
     * the batch; every other buffer is then checked to be at least as large as what that implies,
     * catching a caller bug (mismatched array) here instead of an out-of-bounds C read below. */
    const Py_ssize_t num_pairs = output_view.len / (Py_ssize_t) sizeof(double);
    const Py_ssize_t num_clusters =
        cluster_path_offsets_view.len / (Py_ssize_t) sizeof(int64_t) - 1;
    if (nodes_view.len % (7 * (Py_ssize_t) sizeof(double)) != 0
        || num_clusters < 0
        || num_effective_paths_view.len / (Py_ssize_t) sizeof(int64_t) < num_clusters
        || last_layers_view.len / (Py_ssize_t) sizeof(double) < num_clusters
        || first_terminal_layers_view.len / (Py_ssize_t) sizeof(double) < num_clusters
        || cluster_pairs_view.len / (Py_ssize_t) sizeof(int64_t) < num_pairs * 2
        || exact_pairs_view.len / (Py_ssize_t) sizeof(uint8_t) < num_pairs
        || misaligned_pairs_view.len / (Py_ssize_t) sizeof(uint8_t) < num_pairs) {
        PyErr_SetString(PyExc_ValueError, "inconsistent compiled distance-kernel buffer sizes");
        goto release_buffers;
    }

    const double *nodes = nodes_view.buf;
    const int64_t *path_nodes = path_nodes_view.buf;
    const int64_t *path_node_offsets = path_node_offsets_view.buf;
    const int64_t *prefix_lengths = prefix_lengths_view.buf;
    const int64_t *prefix_offsets = prefix_offsets_view.buf;
    const int64_t *cluster_path_offsets = cluster_path_offsets_view.buf;
    const int64_t *num_effective_paths = num_effective_paths_view.buf;
    const double *last_layers = last_layers_view.buf;
    const double *first_terminal_layers = first_terminal_layers_view.buf;
    const int64_t *cluster_pairs = cluster_pairs_view.buf;
    const uint8_t *exact_pairs = exact_pairs_view.buf;
    const uint8_t *misaligned_pairs = misaligned_pairs_view.buf;
    double *output = output_view.buf;

    /* the whole batch runs without the GIL: no Python objects are touched again until the loop
     * finishes, letting the multiprocessing Pool workers that call this run their C loops
     * concurrently instead of serialising on the GIL like the pure-Python fallback would */
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t pair_id = 0; pair_id < num_pairs; ++pair_id) {
        output[pair_id] = compute_pair(
            nodes,
            path_nodes,
            path_node_offsets,
            prefix_lengths,
            prefix_offsets,
            cluster_path_offsets,
            num_effective_paths,
            last_layers,
            first_terminal_layers,
            cluster_pairs[pair_id * 2],
            cluster_pairs[pair_id * 2 + 1],
            exact_pairs[pair_id],
            misaligned_pairs[pair_id],
            distance_cutoff);
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&nodes_view);
    PyBuffer_Release(&path_nodes_view);
    PyBuffer_Release(&path_node_offsets_view);
    PyBuffer_Release(&prefix_lengths_view);
    PyBuffer_Release(&prefix_offsets_view);
    PyBuffer_Release(&cluster_path_offsets_view);
    PyBuffer_Release(&num_effective_paths_view);
    PyBuffer_Release(&last_layers_view);
    PyBuffer_Release(&first_terminal_layers_view);
    PyBuffer_Release(&cluster_pairs_view);
    PyBuffer_Release(&exact_pairs_view);
    PyBuffer_Release(&misaligned_pairs_view);
    PyBuffer_Release(&output_view);
    Py_RETURN_NONE;

/* error path: release only the buffers that were actually acquired above (a Py_buffer left
 * zero-initialised, per the {0} initialisers at declaration, has obj == NULL) */
release_buffers:
    if (nodes_view.obj != NULL) PyBuffer_Release(&nodes_view);
    if (path_nodes_view.obj != NULL) PyBuffer_Release(&path_nodes_view);
    if (path_node_offsets_view.obj != NULL) PyBuffer_Release(&path_node_offsets_view);
    if (prefix_lengths_view.obj != NULL) PyBuffer_Release(&prefix_lengths_view);
    if (prefix_offsets_view.obj != NULL) PyBuffer_Release(&prefix_offsets_view);
    if (cluster_path_offsets_view.obj != NULL) PyBuffer_Release(&cluster_path_offsets_view);
    if (num_effective_paths_view.obj != NULL) PyBuffer_Release(&num_effective_paths_view);
    if (last_layers_view.obj != NULL) PyBuffer_Release(&last_layers_view);
    if (first_terminal_layers_view.obj != NULL) PyBuffer_Release(&first_terminal_layers_view);
    if (cluster_pairs_view.obj != NULL) PyBuffer_Release(&cluster_pairs_view);
    if (exact_pairs_view.obj != NULL) PyBuffer_Release(&exact_pairs_view);
    if (misaligned_pairs_view.obj != NULL) PyBuffer_Release(&misaligned_pairs_view);
    if (output_view.obj != NULL) PyBuffer_Release(&output_view);
    return NULL;
}

static PyMethodDef methods[] = {
    {"distance_batch", distance_batch, METH_VARARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_distance_kernel",
    .m_size = -1,
    .m_methods = methods
};

PyMODINIT_FUNC PyInit__distance_kernel(void) {
    return PyModule_Create(&module);
}
