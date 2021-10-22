cmd.load('Cluster_3.pdb.gz', 'Cluster_3')
cmd.set_color('caver2', [1.0, 0.0, 0.0])
cmd.color('caver2', "Cluster_3")
cmd.alter('Cluster_*', 'vdw=b')
cmd.show_as('spheres', 'Cluster_*')
cmd.set('sphere_transparency', 0.6, 'Cluster_*')
cmd.load('event_326.pdb.gz', 'event_326')
cmd.show_as('spheres', 'event_*')
cmd.load('structure.pdb.gz', 'structure')
cmd.show_as('cartoon', 'structure')
cmd.show('lines', 'structure')
