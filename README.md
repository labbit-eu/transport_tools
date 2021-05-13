TransportTools
==============

*TransportTools* library builds on and extends results of [CAVER](http://caver.cz) and [AQUA-DUCT](http://aquaduct.pl) analyses to provide comprehensive insights into molecular details of transport processes related to biomolecules. 

Visit our [homepage](http://labbit.eu/software).

## Overview

Using the *TransportTools engine*, end-users can get access to 
* efficient analyses of molecular tunnels in extensive MD simulations, including those originating from massively parallel calculations or very long simulations,
* information on molecular voids in the biomolecules with their actual utilization by small molecules, 
* rigorous comparison of transport processes in different simulation sets, e.g., contrasting transport in the original system and system perturbed by mutations, different solvents, presence of ligands, etc.

<br>

From a programmatic point of view, the library will also help to 
* simplify custom-made analyses of transport processes, application of various filters on explored molecular voids, 
* facilitate further development of tools focusing on the study of transport processes by providing a rich environment and interface to the popular packages in the field.

<br>

Currently, TransportTools engine can perform its action in 10 consecutive stages:
1. Preparatory stage including defining transformations needed for unified analyses
2. Processing of input datasets of tunnel networks
3. Layering tunnel clusters to get their simplified representation
4. Computing distances among the layered clusters
5. Clustering the layered clusters into superclusters and creating initial outputs
6. Filtering superclusters and creating filtered outputs
7. Processing datasets of transport events of ligands
8. Layering transport events to get their simplified representation
9. Assigning transport events to tunnel networks in superclusters and creating initial outputs 
10. Filtering supercluster with events and creating filtered outputs with events

Note that stages 7-10 depend on the availability of data on transport events from AQUA-DUCT and hence are optional.    
    
    
## Availability

*TransportTools* is licensed under: [GNU GPL v3 license](https://www.gnu.org/licenses/gpl-3.0.en.html), and 
the source code is available at [github.com](https://github.com/labbit-eu/transport_tools).

## Installation

*TransportTools* can be installed either:
<br>

via CONDA:
1. if you do not have already, get [conda management system](https://conda.io/projects/conda/en/latest/user-guide/install/download.html)
2. create new environment for transport tools:  conda create --name transport_tools python=3.8
3. activate the environment: conda activate transport_tools 
4. install TransportTools and all required dependencies: conda install transport_tools -c labbit -c conda-forge

or PyPi:
1. install [AmberTools](http://ambermd.org/AmberTools.php) according to the [instructions](http://ambermd.org/Installation.php)
2. source the amber.sh file
3. pip install transport_tools
    
To start using *TransportTools* type:

    tt_engine.py --help

## Troubleshooting

If you encounter any problems with installation or the use of *TransportTools* please contact us at [github.com](https://github.com/labbit-eu/transport_tools/issues).

## Documentation

Documentation can be found on [github.com](https://github.com/labbit-eu/transport_tools).


