Authors and Contributors
========================

*TransportTools* is developed at the [Laboratory of Biomolecular Interactions and Transport](http://labbit.eu),
Adam Mickiewicz University in Poznan.

The `__author__` field in the source files names the principal author of the package as a whole;
credit for individual modules and features is recorded here.


## Maintainer

**Jan Brezovsky** &lt;janbre@amu.edu.pl&gt; — original author, design and implementation of the library, and current maintainer.


## Contributors

Listed in reversed-chronological order.

* **Igor Marchlewski**
  * tunnel-profile pruning: detection of CAVER tunnel tails that inflate into bulk solvent and their in-place truncation
* **Bartlomiej Surpeta**
  * conceived and written the tutorial included in the user guide
  * contributed user testing and ideas througout 2021-2025
* **Carlos Eduardo Sequeiros-Borja** 
  * authored of divide-and-conquer scripts for analyses of long trajectories
  * authored structural alignment 
  * authored surface visualizations of clusters with MSMS/PyCUBES.
  * contributed user testing and ideas to its initial public release
* **Aravind Selvaram Thirunavukarasu** 
  * contributed user testing and ideas to its initial public release
  * evaluated initial performance of the tools
  * generated use-case 1
* **Nishita Mandal**
  * contributed user testing and ideas to its initial public release
  * generated use-case 2
* **Dheeraj Kumar Sarkar**
  * contributed user testing and ideas to its initial public release
  * generated use-case 3
* **Nikhil Agrawal**
  * contributed user testing and ideas to its initial public release
* **Cedrix Jurgal Dongmo Foumthuim**
  * contributed user testing and ideas to its initial public release
  
   
## Module authorship

Modules whose primary author is someone other than the maintainer:

| Module | Primary author |
| --- | --- |
| [transport_tools/libs/tunnel_pruning.py](transport_tools/libs/tunnel_pruning.py) | Igor Marchlewski |
| [transport_tools/tests/units/test_tunnel_pruning.py](transport_tools/tests/units/test_tunnel_pruning.py) | Igor Marchlewski |
| [transport_tools/scripts/tt_convert_to_caver.py](transport_tools/scripts/tt_convert_to_caver.py) | Carlos Eduardo Sequeiros-Borja  |
| [transport_tools/scripts/tt_filter_caver_by_frames.py](transport_tools/scripts/tt_filter_caver_by_frames.py) | Carlos Eduardo Sequeiros-Borja |
| [transport_tools/libs/msms.py](transport_tools/libs/msms.py) | Carlos Eduardo Sequeiros-Borja |

All other modules were written primarily by Jan Brezovsky, with contributions from the people listed above.


## Publications

The scientific contributions behind the library are credited in its publications, listed in the
[README](README.md#references).


## Adding yourself

New contributors are welcome to add themselves to the *Contributors* section as part of the pull request
that carries their contribution, giving their name,and a one-line description of the
area they worked on. When a contribution makes someone the primary author of a whole module, add that
module to the *Module authorship* table as well.
