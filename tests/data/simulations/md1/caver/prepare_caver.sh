#!/bin/bash
PARMTOP=../structure.parm7
INPTRAJ=../trajectory.nc
STRIP_MASK=":WAT,Na+,Cl-"

# create pdb files for CAVER calculations
cat > cpptraj.in <<EOF1
trajin ${INPTRAJ} 1 last ${STEP}
strip ${STRIP_MASK}
autoimage
rmsd @N,CA,C mass first
trajout pdbs/stripped_system.pdb pdb multi
go
EOF1

mkdir -p pdbs
cpptraj ${PARMTOP} < cpptraj.in


#rename pdb files to have optimal format for CAVER calculations
cat > rename.py <<EOF2
import os
os.chdir("pdbs")
pdb_files = os.listdir(".")
for f in pdb_files:
    if ".pdb." in f:
        fname = f.split(".", 3)
        new_name = fname[0] + ".{:0>6d}.pdb".format(int(fname[2]))
        os.rename(f, new_name)
EOF2

python rename.py

# clean up
rm -f cpptraj.in rename.py
