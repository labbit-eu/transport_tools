import pickle, gzip

with gzip.open('../../../network_data/caver/md1/cls_049_cgo.dump.gz', 'rb') as in_stream:
    orig_cluster = pickle.load(in_stream)
cmd.load_cgo(orig_cluster, 'org_data')
cmd.disable('org_data')

cmd.load('nodes/cls049-0-0.pdb')
cmd.set_color('caver1', [0.0, 1.0, 0.0])
cmd.color('1', "cls049-0-0")
cmd.load('nodes/cls049-1-0.pdb')
cmd.set_color('caver2', [1.0, 0.0, 0.0])
cmd.color('2', "cls049-1-0")
cmd.load('nodes/cls049-2-0.pdb')
cmd.set_color('caver3', [0.0, 1.0, 1.0])
cmd.color('3', "cls049-2-0")
cmd.load('nodes/cls049-3-0.pdb')
cmd.set_color('caver4', [1.0, 1.0, 0.0])
cmd.color('4', "cls049-3-0")
cmd.load('nodes/cls049-4-0.pdb')
cmd.set_color('caver5', [1.0, 0.0, 1.0])
cmd.color('5', "cls049-4-0")
cmd.load('nodes/cls049-5-0.pdb')
cmd.set_color('caver6', [0.71, 0.71, 0.97])
cmd.color('6', "cls049-5-0")
cmd.load('nodes/cls049-6-0.pdb')
cmd.set_color('caver7', [0.5, 0.99, 0.42])
cmd.color('7', "cls049-6-0")
cmd.load('nodes/cls049-6-1.pdb')
cmd.set_color('caver7', [0.5, 0.99, 0.42])
cmd.color('7', "cls049-6-1")
cmd.load('nodes/cls049-7-0.pdb')
cmd.set_color('caver8', [0.99, 0.5, 0.42])
cmd.color('8', "cls049-7-0")
cmd.load('nodes/cls049-7-1.pdb')
cmd.set_color('caver8', [0.99, 0.5, 0.42])
cmd.color('8', "cls049-7-1")
cmd.load('nodes/cls049-7-2.pdb')
cmd.set_color('caver8', [0.99, 0.5, 0.42])
cmd.color('8', "cls049-7-2")
cmd.load('nodes/cls049-7-3.pdb')
cmd.set_color('caver8', [0.99, 0.5, 0.42])
cmd.color('8', "cls049-7-3")
cmd.load('nodes/cls049-7-4.pdb')
cmd.set_color('caver8', [0.99, 0.5, 0.42])
cmd.color('8', "cls049-7-4")
cmd.load('nodes/cls049-8-0.pdb')
cmd.set_color('caver9', [0.21, 0.5, 0.85])
cmd.color('9', "cls049-8-0")
cmd.load('nodes/cls049-8-1.pdb')
cmd.set_color('caver9', [0.21, 0.5, 0.85])
cmd.color('9', "cls049-8-1")
cmd.load('nodes/cls049-8-2.pdb')
cmd.set_color('caver9', [0.21, 0.5, 0.85])
cmd.color('9', "cls049-8-2")
cmd.load('nodes/cls049-8-3.pdb')
cmd.set_color('caver9', [0.21, 0.5, 0.85])
cmd.color('9', "cls049-8-3")
cmd.load('nodes/cls049-9-0.pdb')
cmd.set_color('caver10', [0.5, 0.06, 0.87])
cmd.color('10', "cls049-9-0")
cmd.load('nodes/cls049-9-1.pdb')
cmd.set_color('caver10', [0.5, 0.06, 0.87])
cmd.color('10', "cls049-9-1")
cmd.load('nodes/cls049-9-2.pdb')
cmd.set_color('caver10', [0.5, 0.06, 0.87])
cmd.color('10', "cls049-9-2")
cmd.load('nodes/cls049-10-0.pdb')
cmd.set_color('caver11', [0.0, 0.87, 0.5])
cmd.color('11', "cls049-10-0")
cmd.load('nodes/cls049-10-1.pdb')
cmd.set_color('caver11', [0.0, 0.87, 0.5])
cmd.color('11', "cls049-10-1")
cmd.load('nodes/cls049-10-2.pdb')
cmd.set_color('caver11', [0.0, 0.87, 0.5])
cmd.color('11', "cls049-10-2")
cmd.load('nodes/cls049-11-0.pdb')
cmd.set_color('caver12', [0.86, 0.01, 0.5])
cmd.color('12', "cls049-11-0")
cmd.load('nodes/cls049-11-1.pdb')
cmd.set_color('caver12', [0.86, 0.01, 0.5])
cmd.color('12', "cls049-11-1")
cmd.load('nodes/cls049-11-2.pdb')
cmd.set_color('caver12', [0.86, 0.01, 0.5])
cmd.color('12', "cls049-11-2")
cmd.load('nodes/cls049-12-0.pdb')
cmd.set_color('caver13', [0.58, 0.82, 0.0])
cmd.color('13', "cls049-12-0")
cmd.load('nodes/cls049-12-1.pdb')
cmd.set_color('caver13', [0.58, 0.82, 0.0])
cmd.color('13', "cls049-12-1")
cmd.show_as('wire', 'all')
cmd.show_as('spheres', 'r. A*')

with gzip.open('paths/cls049_pathset.dump.gz', 'rb') as in_stream:
    pathset = pickle.load(in_stream)
    for path in pathset:
        cmd.load_cgo(path, 'path_cls049')
cmd.set('cgo_line_width', 10, 'path_cls049')

cmd.load('origin.pdb', 'starting_point')
cmd.show_as('spheres', 'starting_point')
cmd.do('set all_states, 1')
cmd.set('sphere_scale', 0.25)
cmd.zoom('all')
cmd.show('cgo')
