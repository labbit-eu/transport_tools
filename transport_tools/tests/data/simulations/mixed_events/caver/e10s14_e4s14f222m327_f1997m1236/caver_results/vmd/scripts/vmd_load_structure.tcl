#set dir "/mnt/storage_5/scratch/pl0252-02/igomar/ups/linb-wt_be/workdir/epoch_10_runs/e10s14_e4s14f222m327_f1997m1236/pdb"

mol load pdb ../data/stripped_system.501.pdb

after idle { 
  mol representation NewCartoon 
  mol delrep 0 top
  mol addrep top
  mol modcolor 0 top "ColorID" 8
} 

