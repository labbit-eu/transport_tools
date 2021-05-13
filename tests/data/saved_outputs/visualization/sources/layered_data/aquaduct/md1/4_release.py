import pickle, gzip

with gzip.open('../../../network_data/aquaduct/md1/raw_paths_4_cgo.dump.gz', 'rb') as in_stream:
    orig_cluster = pickle.load(in_stream)
cmd.load_cgo(orig_cluster, 'org_data')
cmd.disable('org_data')

cmd.load('nodes/wat_4_release-3-0.pdb')
cmd.set_color('caver4', [1.0, 1.0, 0.0])
cmd.color('4', "wat_4_release-3-0")
cmd.load('nodes/wat_4_release-4-0.pdb')
cmd.set_color('caver5', [1.0, 0.0, 1.0])
cmd.color('5', "wat_4_release-4-0")
cmd.load('nodes/wat_4_release-5-0.pdb')
cmd.set_color('caver6', [0.71, 0.71, 0.97])
cmd.color('6', "wat_4_release-5-0")
cmd.load('nodes/wat_4_release-6-0.pdb')
cmd.set_color('caver7', [0.5, 0.99, 0.42])
cmd.color('7', "wat_4_release-6-0")
cmd.load('nodes/wat_4_release-8-0.pdb')
cmd.set_color('caver9', [0.21, 0.5, 0.85])
cmd.color('9', "wat_4_release-8-0")
cmd.load('nodes/wat_4_release-10-0.pdb')
cmd.set_color('caver11', [0.0, 0.87, 0.5])
cmd.color('11', "wat_4_release-10-0")
cmd.load('nodes/wat_4_release-11-0.pdb')
cmd.set_color('caver12', [0.86, 0.01, 0.5])
cmd.color('12', "wat_4_release-11-0")
cmd.show_as('wire', 'all')
cmd.show_as('spheres', 'r. A*')

with gzip.open('paths/wat_4_release_pathset.dump.gz', 'rb') as in_stream:
    pathset = pickle.load(in_stream)
    for path in pathset:
        cmd.load_cgo(path, 'path_wat_4_release')
cmd.set('cgo_line_width', 10, 'path_wat_4_release')

cmd.load('origin.pdb', 'starting_point')
cmd.show_as('spheres', 'starting_point')
cmd.do('set all_states, 1')
cmd.set('sphere_scale', 0.25)
cmd.zoom('all')
cmd.show('cgo')
