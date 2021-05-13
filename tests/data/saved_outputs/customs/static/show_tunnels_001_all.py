cmd.load('md1_Cluster_1.pdb.gz', 'Cls_1_md1')
cmd.set_color('caver0', [0.0, 0.0, 1.0])
cmd.color('caver0', "Cls_1_md1")
cmd.alter('Cls_*', 'vdw=b')
cmd.show_as('spheres', 'Cls_*')
cmd.load('../../_internal/transformations/ref_transformed.pdb', 'structure')
cmd.show_as('cartoon', 'structure')
cmd.show('lines', 'structure')
