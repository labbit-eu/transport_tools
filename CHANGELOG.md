# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [v0.9.3](https://github.com/labbit-eu/transport_tools/releases/tag/v0.9.3) - 2022-10-18
### Features
- scripts to perform spitting and joining of caver data from trajectory parts
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
