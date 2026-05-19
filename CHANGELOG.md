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
- optional SLURM backend for the stage-4 cluster-cluster distance calculation via the new `distance_backend` setting (`local` default, or `slurm`); the `slurm` backend distributes the pairwise distances over SLURM array jobs using the optional `submitit` package and is configured through the new `slurm_partition`, `slurm_account`, `slurm_timeout_min`, `slurm_mem_gb`, `slurm_cpus_per_task`, `slurm_num_shards`, `slurm_folder`, and `slurm_array_parallelism` parameters; each shard persists its result, so an interrupted or partially failed run is resumed on re-run without recomputing the shards that already finished

### Misc
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
