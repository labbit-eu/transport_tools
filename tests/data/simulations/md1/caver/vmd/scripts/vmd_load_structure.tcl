#set dir "/home/briza/PycharmProjects/transport_tools/tests/data/simulations/md1/caver/pdbs"

mol load pdb ../data/stripped_system.000501.pdb

after idle { 
  mol representation NewCartoon 
  mol delrep 0 top
  mol addrep top
  mol modcolor 0 top "ColorID" 8
} 

