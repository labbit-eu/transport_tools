#!/bin/bash
CAVER_HOME=~/software/caver_3.0/caver
STARTING_POINT_ATOMS="1608 1634 2602 3867"

cat > config.txt <<EOF
starting_point_atom $STARTING_POINT_ATOMS

probe_radius 0.7
shell_radius 3
shell_depth 4 
clustering_threshold 3.5
one_tunnel_in_snapshot cheapest
save_dynamics_visualization no
generate_summary yes
generate_tunnel_characteristics no
generate_tunnel_profiles yes
compute_bottleneck_residues yes
frame_clustering yes
do_approximate_clustering no
seed 1
EOF


java -Xmx8000m -cp ${CAVER_HOME}/lib -jar ${CAVER_HOME}/caver.jar -home ${CAVER_HOME} -pdb ./pdbs -conf ./config.txt -out .

# clean up of temp data
rm -rf pdbs config.txt data/tunnel_edges data/tunnels
