# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [v0.9.0](https://github.com/labbit-eu/transport_tools/releases/tag/v0.8.5) - 2021-10-22
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
