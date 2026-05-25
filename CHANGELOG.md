# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [Unreleased] - v0.9.7
### Features
- support multiple AQUA-DUCT input paths for multi-residue analyses
- per-residue event tracking and unique entity labels for multi-residue merging
- optional `aquaduct_traced_residues_filter` parameter (comma-separated uppercase resnames, e.g. `WAT, O2`) to skip unwanted residues; `None` (default) processes all residues, preserving backwards compatibility
- per-residue colors and object names in PyMOL event visualization
- optional skipping of empty AQUA-DUCT folders via new `aquaduct_allow_empty_folders` calculations setting (default `False`); when enabled, folders matching `aquaduct_results_folder_pattern` that are missing required tar/summary files are pruned with a one-time warning instead of raising `FileNotFoundError`
- replaced hardcoded `wat_` prefix with dynamic residue abbreviation to enable multi-residue inputs from AQUA-DUCT
- point clusters in layers are added in spatially sorted order to ensure stable, deterministic labeling
- optional SLURM backend for the stage-4 cluster-cluster distance calculation via the new `compute_backend` setting (`local` default, or `slurm`) plus the per-stage `stage04_backend` override; the `slurm` backend distributes the pairwise distances over SLURM array jobs using the optional `submitit` package and is configured through the new `slurm_partition`, `slurm_account`, `slurm_timeout_min`, `slurm_mem_gb`, `slurm_cpus_per_task`, `slurm_num_shards`, `slurm_root_folder`, and `slurm_array_parallelism` parameters; each shard persists its result, so an interrupted or partially failed run is resumed on re-run without recomputing the shards that already finished. Per-stage shards land in `<slurm_root_folder>/stage<NN>_<name>/` (e.g. `stage04_distances/`) so multiple SLURM-capable stages can coexist on disk
- reusable `SlurmShardStage` framework (`transport_tools.libs.slurm`) that factors the resumability, fingerprint-based stale-result detection, completion-order polling with per-shard runtime timeouts, and SLURM submission out of the stage-4 launcher into a stage-agnostic `run_stage_on_slurm()` driver; stage 4 is the first adapter (`DistanceShardStage`), and the framework is the prerequisite for SLURM analogues of the must-have stages (tunnel layering, aquaduct layering, event assignment). Each stage's effective backend resolves via `AnalysisConfig.resolve_stage_backend(<stage_knob>)` from the per-stage override and the global `compute_backend` default; config validation fails fast with a clear error when any stage is set to `slurm` and the optional `submitit` package is not installed
- optional SLURM backend for the stage-3 tunnel-layering work (`create_layered_description4tunnel_networks`) via the new `stage03_backend` override (`None` → `compute_backend`); shards land in `<slurm_root_folder>/stage03_tunnel_layering/` and each shard takes a strided subset of the flat `(md_label, cluster_id)` enumeration so sklearn-heavy clusters balance across shards. The launcher persists the enumeration to `<slurm_folder>/items.json` via the framework's new `prepare_state()` hook so every shard reads one artifact instead of re-deriving the work-item list from every orig_network.dump file. Per-shard results are pickled as `[(cls_id, md_label, LayeredPathSet), …]`; the launcher assembles them in original submission order so `layered_network.dump` is byte-identical to what the local backend produces
- new `worker_task_timeout_s` calculations setting (default `3600` s; set to `None` to disable) that caps the wall-clock time the launcher waits on any single multiprocessing worker task; on timeout the offending task is logged and the pool is terminated

### Bug Fixes
- eliminate intermittent fork-with-threads deadlocks by pinning the multiprocessing start method to `spawn` in every process entry point (the launcher and every SLURM shard) before any `Pool`/`Process`/`Queue` is constructed; previously, worker processes could inherit numpy/BLAS internal locks from a multi-threaded parent and silently hang
- guard `_pre_process_single_aquaduct_network` against nested-pool deadlocks: passing `parallel_processing=True` from inside a multiprocessing worker now raises `RuntimeError` immediately instead of silently deadlocking on the inner pool's children

### Misc
- cap BLAS thread budget per worker so that `pool_size × threads_per_worker ≤ allocated_CPUs` (with allocated CPUs being `num_cpus` locally or `slurm_cpus_per_task` inside a SLURM shard); avoids the `num_cpus × BLAS_threads` oversubscription that previously caused stalls resembling deadlocks. Pre-existing `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `NUMEXPR_NUM_THREADS` / `BLIS_NUM_THREADS` values from the user shell, SLURM, or cluster prolog are respected via `setdefault`
- worker exceptions and timeouts now surface with the failing task's identity (md_label / sc_id / event_specification / cluster_id / shard label) so a stage failure can be triaged from a single log line instead of an anonymous traceback; every `apply_async`/`imap_unordered` call site in `tools.py` and the raw-paths loop in `networks.py` was converted to label each submitted task with a stable identifier and iterate via the new `utils.iter_pool_results()` / `utils.iter_imap_results()` helpers
- stage-4 SLURM launcher now polls shards in completion order (rather than submission order) and applies a per-shard runtime timeout of `2 × slurm_timeout_min + 5 min` (measured from the SLURM `RUNNING` state, so queued shards are never abandoned for queueing alone); a shard that runs past its deadline is cancelled and abandoned, and the existing per-shard atomic result files plus resumability path let the next stage-4 invocation pick up the abandoned shards without recomputing the finished ones. Replaces the previous submission-order `job.result()` loop that could block indefinitely on a stuck SLURM node
- overlap the independent `average_starting_point` and `transform_aquaduct` batches in `compute_transformations`: both are submitted up front so pool workers interleave the fast PDB reads with the slower tar.gz extractions, dropping the transformations-stage wall-clock from sum-of-batches to max-of-batches on CAVER+AQUA-DUCT analyses
- share the (potentially hundreds-of-MB) supercluster dict and active filter set with `assign_transport_events` workers exactly once via a `Pool` initializer instead of pickling them per task; per-task pickle drops from `O(EventAssigner + super_clusters + active_filters + event_path_set)` to `O(event_specification + event_path_set)`, and total state shipping from `O(num_workers × num_events)` to `O(num_workers)` for large studies under the `spawn` start method
- mirror the SLURM-ready worker pattern across `create_super_cluster_profiles`, `filter_super_cluster_profiles`, `create_layered_description4tunnel_networks`, `create_layered_description4aquaduct_networks`, `process_aquaduct_networks`, and `read_raw_paths_data`: each pool now uses `get_context("spawn").Pool(...)` with `utils.cap_blas_threads_for_worker` as the initializer (so the BLAS thread cap is enforced on every pool that lacked it), and submits a top-level worker function instead of a bound method. The per-task pickle no longer drags in the surrounding `TransportProcesses` instance, and each worker is now picklable as a top-level function — the prerequisite for adding SLURM analogues for these stages
- migrate the layering / event-assignment / aquaduct-network / raw-path / supercluster-profile pools from `apply_async` + ordered drain to `pool.imap_unordered(...)`: pool workers stream results in completion order, so a slow task no longer blocks progress reporting for the others and the inter-completion timeout fires on real stalls rather than per-task drain time. `utils.iter_imap_results` was extended with an optional `extract_task_label` callable that records the last few completed task identities; on timeout/failure the error log now reads e.g. `"event-assignment timed out; 1247 task(s) completed before the stall. Last completed: 5e9_e1/path_88/release, 5e9_e1/path_84/release, 5e9_e1/path_82/release."` so operators have a region to investigate
- preserve submission-order side effects across the `imap_unordered` migration for the sites whose downstream artifacts depend on insertion order (`create_layered_description4tunnel_networks`, `create_layered_description4aquaduct_networks`, `assign_transport_events`, `read_raw_paths_data`): each site now buffers worker results during streaming, then commits them in the original submission order (per md_label for the layering sites, by `path_sets.keys()` for event assignment, by the original `items2process` ordering for raw paths). The pickled `layered_entities` dict layout, the lists inside `_super_clusters[sc_id]` / `_outlier_transport_events`, and `AquaductNetwork.orig_entities` are therefore stable regardless of worker scheduling and match the integration-test reference data. Per-md_label save+free memory profile of the layering sites is unchanged (commit fires as soon as the last result for that md_label arrives)
- sped up the stage-4 cluster-cluster distance calculation: path-sets are now shared with the worker processes once via a `Pool` initializer instead of being re-pickled for every pairwise comparison; removed the now-obsolete `n_jobs_per_cpu_batch` parameter
- vectorized the path-set distance kernel (`LayeredPathSet.avg_distance2path_set`): the per-node closest-node Python loops were replaced by NumPy operations over dense node-to-node distance matrices
- reworked stage-4 distance assembly to use the condensed (upper-triangle) form end-to-end: the SLURM shards' results are concatenated as NumPy arrays instead of Python tuple lists, the pairwise distances are scattered straight into a 1-D condensed vector (no dense N×N matrix is ever materialized), and stage 5 feeds it directly to `fastcluster` without rebuilding triangular indices; the saved matrix file is now `caver_clusters_clustering_distances.npy` (older dense `caver_clusters_clustering_matrix.npy` files are still read for backwards compatibility), roughly halving disk usage and peak clustering memory
- updated install requirements to match new environment
- refactored type checking and validations; resolved several `DeprecationWarning`s
- improved type hints and added validation checks throughout
- updated to current numpy/scipy/sklearn/biopython/hdbscan packages with minor node-cluster adjustments; updated test fixtures accordingly
- added tests to reach >70% coverage of library code; more robust test comparisons
- new unit tests for tunnel profiles processing
- corrected formatting of produced `bottleneck.csv` files (no extra spaces) to match original CAVER output format
- updated package structure to support future deployment


## [v0.9.6](https://github.com/labbit-eu/transport_tools/releases/tag/v0.9.6) - 2023-03-30
### Bug Fixes
- improving error message for autofinding Caver PDB files
- using `print` instead of logging during early startup before the logger is initialized


## [v0.9.5](https://github.com/labbit-eu/transport_tools/releases/tag/v0.9.5) - 2023-01-12
### Features
- new parameter `folder_pattern4exact_matching_analysis` enables selection of specific folders for exact matching analysis (default `"*"` — all folders)

### Bug Fixes
- correcting selection of residues for visualization of exact matching analysis when using pytraj

### Misc
- updated references


## [v0.9.4](https://github.com/labbit-eu/transport_tools/releases/tag/v0.9.4) - 2022-12-09
### Features
- updating scripts for divide-and-conquer approach for tunnel analyses in long trajectories
- new parameter `caver_pdb_file_pattern` allows the matched filename pattern to be defined in the config file
- reference caver filename now includes `.1.` to conform to the default file matching pattern
- supercluster volumes can optionally be generated with the msms program, which is significantly faster than mcubes
- avoiding detection of CAVER origin and zone files as a relative PDB file for the system
- handling of `FutureWarning`s from sklearn

### Bug Fixes
- in case there are 0 items to process, do not trigger progressbar to avoid division by zero

### Documentation
- updated documentation and docstrings
- updated readme formatting

### Misc
- updated documentation


## [v0.9.3](https://github.com/labbit-eu/transport_tools/releases/tag/v0.9.3) - 2022-10-18
### Features
- adding scripts to perform spitting and joining of caver data from trajectory parts
- saving transformation of caver starting points to enable splitting and joining of caver data from trajectory parts
- enabling exact calculation of distances between all paths (even very remote ones) instead of skipping those calculations 
  (previously default behavior) => new parameter _calculate_exact_path_distances_ 


### Bug Fixes
- updating test reference files to match cross-OS filepaths
- avoiding duplicate keys for tunnel clusters during distance calculation when same parts of simulations are reused
- skipping assigment of singleton events instead of dying
- detecting and removing events that are formed by less then two points 
- preserving also original connectivity from AquaDuct trace among points in coarse-grained events


## [v0.9.0](https://github.com/labbit-eu/transport_tools/releases/tag/v0.9.0) - 2021-11-05
### Features
- default trajectory engine is now MDTraj to enable easier cross-platform installation
- using pymcubes to visualize surfaces of superclusters
- report max buriedness also for unassigned events to facilitate optimization of cutoff
- only manually not set features will be auto-activated

### Misc
- enabling conda build for Windows OS

### Documentation
- updated user guide 
- updated technical documentation 

### Bug Fixes
- corrected handling of temp files
- avoid processing of visualization data for empty superclusters


## [v0.8.5](https://github.com/labbit-eu/transport_tools/releases/tag/v0.8.5) - 2021-05-19
### Features
- verification of caver input data

### Bug Fixes
- replacing pytraj's load with iterload, to avoid random instabilities in created reference frame
- in aquaduct parallel mode + more stable selection of reference structure 

### Documentation
- fixed some typos and formatting issues 



## [v0.8.3](https://github.com/labbit-eu/transport_tools/releases/tag/v0.8.3) - 2021-05-14
### Documentation
- added complete user guide 
- added technical documentation 

### Misc
- License file added
