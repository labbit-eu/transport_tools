import pickle, gzip

with gzip.open('../../../network_data/caver/md1/cls_032_cgo.dump.gz', 'rb') as in_stream:
    orig_cluster = pickle.load(in_stream)
cmd.load_cgo(orig_cluster, 'org_data')
cmd.disable('org_data')

cmd.load('nodes/cls032-0-0.pdb')
cmd.set_color('caver1', [0.0, 1.0, 0.0])
cmd.color('1', "cls032-0-0")
cmd.load('nodes/cls032-1-0.pdb')
cmd.set_color('caver2', [1.0, 0.0, 0.0])
cmd.color('2', "cls032-1-0")
cmd.load('nodes/cls032-2-0.pdb')
cmd.set_color('caver3', [0.0, 1.0, 1.0])
cmd.color('3', "cls032-2-0")
cmd.load('nodes/cls032-2-1.pdb')
cmd.set_color('caver3', [0.0, 1.0, 1.0])
cmd.color('3', "cls032-2-1")
cmd.load('nodes/cls032-3-0.pdb')
cmd.set_color('caver4', [1.0, 1.0, 0.0])
cmd.color('4', "cls032-3-0")
cmd.load('nodes/cls032-3-1.pdb')
cmd.set_color('caver4', [1.0, 1.0, 0.0])
cmd.color('4', "cls032-3-1")
cmd.load('nodes/cls032-4-0.pdb')
cmd.set_color('caver5', [1.0, 0.0, 1.0])
cmd.color('5', "cls032-4-0")
cmd.load('nodes/cls032-4-1.pdb')
cmd.set_color('caver5', [1.0, 0.0, 1.0])
cmd.color('5', "cls032-4-1")
cmd.load('nodes/cls032-5-0.pdb')
cmd.set_color('caver6', [0.71, 0.71, 0.97])
cmd.color('6', "cls032-5-0")
cmd.load('nodes/cls032-6-0.pdb')
cmd.set_color('caver7', [0.5, 0.99, 0.42])
cmd.color('7', "cls032-6-0")
cmd.load('nodes/cls032-6-1.pdb')
cmd.set_color('caver7', [0.5, 0.99, 0.42])
cmd.color('7', "cls032-6-1")
cmd.load('nodes/cls032-7-0.pdb')
cmd.set_color('caver8', [0.99, 0.5, 0.42])
cmd.color('8', "cls032-7-0")
cmd.load('nodes/cls032-7-1.pdb')
cmd.set_color('caver8', [0.99, 0.5, 0.42])
cmd.color('8', "cls032-7-1")
cmd.load('nodes/cls032-8-0.pdb')
cmd.set_color('caver9', [0.21, 0.5, 0.85])
cmd.color('9', "cls032-8-0")
cmd.load('nodes/cls032-8-1.pdb')
cmd.set_color('caver9', [0.21, 0.5, 0.85])
cmd.color('9', "cls032-8-1")
cmd.load('nodes/cls032-9-0.pdb')
cmd.set_color('caver10', [0.5, 0.06, 0.87])
cmd.color('10', "cls032-9-0")
cmd.load('nodes/cls032-9-1.pdb')
cmd.set_color('caver10', [0.5, 0.06, 0.87])
cmd.color('10', "cls032-9-1")
cmd.load('nodes/cls032-9-3.pdb')
cmd.set_color('caver10', [0.5, 0.06, 0.87])
cmd.color('10', "cls032-9-3")
cmd.load('nodes/cls032-9-4.pdb')
cmd.set_color('caver10', [0.5, 0.06, 0.87])
cmd.color('10', "cls032-9-4")
cmd.show_as('wire', 'all')
cmd.show_as('spheres', 'r. A*')

with gzip.open('paths/cls032_pathset.dump.gz', 'rb') as in_stream:
    pathset = pickle.load(in_stream)
    for path in pathset:
        cmd.load_cgo(path, 'path_cls032')
cmd.set('cgo_line_width', 10, 'path_cls032')

cmd.load('origin.pdb', 'starting_point')
cmd.show_as('spheres', 'starting_point')
cmd.do('set all_states, 1')
cmd.set('sphere_scale', 0.25)
cmd.zoom('all')
cmd.show('cgo')
