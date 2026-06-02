# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [Unreleased] - v0.9.8
### Features
- renamed tunnel-relevance filter parameters for clarity: `min_tunnel_radius4clustering` → `relevant_tunnel_min_radius`, `min_tunnel_length4clustering` → `relevant_tunnel_min_length`, `max_tunnel_curvature4clustering` → `relevant_tunnel_max_curvature`; old names are accepted as legacy aliases with a deprecation warning to preserve backwards compatibility
- new `relevant_tunnel_cluster_min_size` calculations setting (default `1`): tunnel clusters with fewer relevant tunnels than this threshold are not layered


## [Unreleased] - v0.9.7
### Features
- stage-5 connected-components partition driver for agglomerative clustering: tunnel clusters are first partitioned by sub-cutoff connectivity and each component is clustered independently (single/complete/average), dropping linkage cost from O(N²) to O(max|C|²); ward stays on vanilla full-matrix linkage
- optional HDBSCAN stage-5 clustering method via new `clustering_method=hdbscan` setting; runs density-based clustering per connected component with `clustering_cutoff` as a hard merge floor; noise points become singleton superclusters
- optional multi-scale stage-5 clustering to prevent spatially-overlapping superclusters via new Phase A (decoupled partition/epsilon/noise handling) and Phase B (orphan consolidation) parameters, both default-off
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
- optional SLURM backend for the stage-8 aquaduct-layering work (`create_layered_description4aquaduct_networks`) via the new `stage08_backend` override (`None` → `compute_backend`); mirror of the stage-3 backend with AQUA-DUCT transport events instead of tunnel clusters as the work-item unit. Shards land in `<slurm_root_folder>/stage08_aquaduct_layering/` and each shard takes a strided subset of the flat `(md_label, event_label)` enumeration so sklearn-heavy events balance across shards. The launcher persists the enumeration to `<slurm_folder>/items.json` via the framework's `prepare_state()` hook so every shard reads one artifact instead of re-deriving the work-item list from every aqua_orig_network.dump file. Per-shard results are pickled as `[(event_label, md_label, LayeredPathSet), …]`; the launcher assembles them in original submission order so `aqua_layered_network.dump` is byte-identical to what the local backend produces. Fingerprint deliberately omits upstream filters (`event_min_distance`, `aquaduct_traced_residues_filter`) - their effect is baked into the stage-7 orig_networks and reflected via `items_hash`, so changes to them invalidate stage-8 shards only after stage 7 is rerun
- optional SLURM backend for the stage-7 aquaduct-networks work (`process_aquaduct_networks`) via the new `stage07_backend` override (`None` → `compute_backend`); shards land in `<slurm_root_folder>/stage07_aquaduct_networks/` and each shard takes a strided subset of the flat md_label enumeration (auto-derived at ~50 md_labels per shard, matching the tunnel-networks granularity). The launcher persists the md_label list to `<slurm_folder>/items.json` via the framework's `prepare_state()` hook so every shard reads one artifact instead of re-deriving it from the config. The SLURM shard runs the per-MD work in an inner `spawn`-Pool with `parallel_processing=False`, so the raw-paths pool inside `_pre_process_single_aquaduct_network` is never spawned from a worker (the nested-pool guard would otherwise raise); the shard always takes the outer-pool path rather than the local backend's `_aquaduct_single_event_inputs` heuristic, since a SLURM shard always benefits from outer parallelism (one node per MD batch). Per-shard pickles hold a tiny list of processed md_labels for completeness verification; the real outputs - `<md_label>_aqua.dump` in `orig_aquaduct_network_data_path` plus, when `visualize_transformed_transport_events` is True, the per-md visualisation folder under `orig_aquaduct_vis_path/<md_label>/` - are tracked as side-effects via the framework's manifest + tar.gz backup so a wiped output folder can be restored on resume without re-extracting the AQUA-DUCT tarballs. Fingerprint keys (`aquaduct_results_folder_pattern`, `aquaduct_results_pdb_filename`, `aquaduct_results_relative_tarfile`, `aquaduct_traced_residues_filter`, `event_min_distance`, `aquaduct_allow_empty_folders`, `visualize_transformed_transport_events`) invalidate stale shards on any parameter change that affects results; `items_hash` invalidates them when the set of AQUA-DUCT input folders changes
- optional SLURM backend for the stage-9 event-assignment work (`assign_transport_events`) via the new `stage09_backend` override (`None` → `compute_backend`); shards land in `<slurm_root_folder>/stage09_event_assignment/` and each shard takes a strided subset of the flat `(md_label, event_label)` enumeration. Unlike the layering stages, stage 9 needs non-trivial pre-shard state (the supercluster dict + active filter set produced by stages 5/6); the launcher writes a single `state.pkl` artifact alongside `items.json` via `prepare_state()`, stream-hashed in one pass via a new `_HashingWriter` helper so the SHA-256 of the state becomes a fingerprint key (`state_hash`) without an extra read-back of the (potentially hundreds-of-MB) pickle. Per-shard results are pickled as `[(event_specification, assigned_sc_ids, max_buriedness, max_depth), …]`; the launcher applies them in original submission order through a shared `_apply_event_assignments` helper so the produced `_super_clusters` / outlier-event state is byte-identical to the local backend. Changing the supercluster set or the filter pass invalidates stage-9 shards automatically via `state_hash`; changing the event set invalidates them via `items_hash`
- per-shard side-effects manifest + tar.gz backup for SLURM-capable layering stages (`TunnelLayeringShardStage`, `AquaductLayeringShardStage`): when a layering shard runs with `visualize_layered_clusters` / `visualize_layered_events` enabled, the framework bundles every worker-written file (per-cluster / per-event `.py` scripts + `nodes/cls***_layer*_pts.pdb` family) into `<slurm_folder>/shard_result_NNNNN.side_effects.tar.gz` alongside the existing `shard_result_NNNNN.pkl` and writes a tiny `shard_result_NNNNN.manifest.json` listing the expected file paths. On resume, the launcher validates every side-effect file from the manifest; if any is missing on disk (e.g. the user wiped their visualisation folder between runs) the framework restores from the tar.gz backup at the original absolute paths so the heavy per-cluster sklearn stack does NOT need to re-run, and only falls back to invalidating the cached pickle + resubmitting the shard when the backup is also unrecoverable. The `visualize_layered_clusters` / `visualize_layered_events` knobs are now part of each layering stage's fingerprint so flipping them between runs correctly invalidates the cache (a False→True transition would otherwise reuse a shard that produced no side-effects). New base-class API: `SlurmShardStage.has_side_effects` flag (default False, opt-in per stage), `shard_manifest_path()` / `shard_side_effects_archive_path()` / `write_side_effects_archive()` / `restore_side_effects_from_archive()` / `expected_side_effect_files()` hooks. The `DistanceShardStage` keeps `has_side_effects=False` and preserves its existing fast resume path (result `.npy` is the only artefact)
- new `worker_task_timeout_s` calculations setting (default `3600` s; set to `None` to disable) that caps the wall-clock time the launcher waits on any single multiprocessing worker task; on timeout the offending task is logged and the pool is terminated
- new `slurm_additional_parameters` calculations setting (default `None`): comma-separated `key=value` string parsed into a dict at INI load time and forwarded verbatim to `submitit.AutoExecutor.update_parameters(slurm_additional_parameters=...)`, which emits one `#SBATCH --key=value` header per entry. Use for cluster-specific knobs (`--qos`, `--constraint`, `--gres`, `--reservation`, `--exclude`, ...) that are not exposed as dedicated `slurm_*` parameters, e.g. `slurm_additional_parameters = qos=high, constraint=icelake, gres=gpu:1`. Distinct from `slurm_srun_args` (which adds args to the `srun` step inside the job): this adds `#SBATCH` headers at the allocation level. An empty value (e.g. `gres=`) is accepted as a legitimate SLURM header; a non-empty token without `=` raises `ValueError` at config load so a typo surfaces immediately rather than three stages in when sbatch sees a malformed header
- post-run reporter for every SLURM stage (`_report_run_stats()` in `transport_tools.libs.slurm`): after the launcher's poll loop completes, the reporter emits two diagnostic blocks. (a) **Failure categorisation**: each failed shard is classified into one of `oom` / `timeout` / `cancelled` / `node_failure` / `preempted` / `error` from its SLURM state (with a kernel-OOM-marker stderr-fallback scan for the case where SLURM only reports `FAILED`) and the failure summary lists shard IDs per category alongside a remediation hint (e.g. `"3 shard(s) ended as 'oom' (shards [3, 17, 41]). Bump slurm_mem_gb."`). The per-shard failure line in the polling loop is also extended with the SLURM state, category, hint, and absolute stderr file path, so users have everything needed to triage in one log line. (b) **Per-shard runtime statistics**: median / p95 / max / skew over the timed shards, total array wall time, and parallel efficiency. Shards that completed between poll cycles (no observed `RUNNING -> done` transition) are reported as fast-uncounted so they don't skew timing. Tuning hints fire only on the relevant signal: skew > 3 suggests increasing the stage's `target_items_per_shard`; efficiency < 50% suggests reducing `slurm_array_parallelism`. The reporter never raises - any internal formatting bug is caught and logged as a warning so a successful stage stays successful
- recursive-SLURM diagnostics in the framework launcher (`run_stage_on_slurm`): when the launcher detects it is itself running inside a SLURM allocation (`SLURM_JOB_ID` set in the env), it logs the parent job id along with the parent's CPU count and time limit (parsed from `SLURM_CPUS_ON_NODE` / `SLURM_JOB_CPUS_PER_NODE` and `SLURM_TIMELIMIT` / `SBATCH_TIMELIMIT`), and warns when the driver's allocation looks too small to outlive the array it is about to submit (driver wall-time limit shorter than `slurm_timeout_min`, or single-CPU driver about to poll ≥16 shards - submitit holds an in-memory Job object per shard). Both signals are warn-only (the launcher still submits) so users who know what they are doing are not blocked
- `--cpu-bind=none` workaround for nested-SLURM execution is now guaranteed even when the user overrides `slurm_srun_args` (e.g. to pass `--mpi=pmix` or `--exact`): the executor wraps the resolved srun-args list through `_ensure_cpu_bind_none()`, which prepends the workaround whenever no explicit `--cpu-bind=...` is already present. An explicit user override (e.g. `--cpu-bind=cores`) is still honoured. Previously the workaround only fired when `slurm_srun_args` was left at its `None` default
- new `slurm_keep_shard_results` calculations setting (default `False`) that controls whether the per-stage SLURM folder is cleaned up automatically after a successful stage assembly. When false (default), every `_slurm` assembler ends with `slurm.cleanup_stage_folder(stage, parameters)` and the whole `<slurm_root_folder>/<stage.folder_name>/` tree - per-shard pickles, manifests, side-effects tarballs, `items.json` / `state.pkl` / `partial_shards_info.json`, and submitit per-job artifacts - is removed in one shot; without this the `_slurm/` tree grows without bound across many stages × runs × resubmissions. Set to `True` to keep the cache after a successful run (useful for debugging shard outputs or for iterative re-runs that change only the assembly step). Resumability of an interrupted run is unaffected either way: cleanup only fires after a successful end-to-end assembly

### Bug Fixes
- ensure logging levels are maintained consistently across the whole codebase so that the level set at startup is honoured in every module without being silently reset
- eliminate intermittent fork-with-threads deadlocks by pinning the multiprocessing start method to `spawn` in every process entry point (the launcher and every SLURM shard) before any `Pool`/`Process`/`Queue` is constructed; previously, worker processes could inherit numpy/BLAS internal locks from a multi-threaded parent and silently hang
- guard `_pre_process_single_aquaduct_network` against nested-pool deadlocks: passing `parallel_processing=True` from inside a multiprocessing worker now raises `RuntimeError` immediately instead of silently deadlocking on the inner pool's children

### Misc
- separate `std_level` (standard output) and `log_level` (logfile) configuration parameters to allow independent control over verbosity of console vs. file logging; previously `log_level` governed both outputs
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
