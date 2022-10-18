import pickle, gzip, os

cmd.load(os.path.join('sources', 'super_cluster_CGOs', 'ref_transformed.pdb'), 'protein_structure')
cmd.show_as('cartoon', 'protein_structure')
cmd.color('gray', 'protein_structure')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC01_overall_pathset3.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_001')
cmd.set('cgo_line_width', 5, 'cluster_001')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC02_overall_pathset3.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_002')
cmd.set('cgo_line_width', 5, 'cluster_002')

events = [os.path.join('sources', 'layered_data', 'aquaduct', 'md1', 'paths', 'wat_1_entry_pathset.dump.gz')]
for event in events:
    with gzip.open(event, 'rb') as in_stream:
        pathset = pickle.load(in_stream)
        for path in pathset:
            path[3:6] = [0.0, 1.0, 0.0]
            cmd.load_cgo(path, 'entry_002')
cmd.set('cgo_line_width', 2, 'entry_002')

events = [os.path.join('sources', 'layered_data', 'aquaduct', 'md1', 'paths', 'wat_1_release_pathset.dump.gz')]
for event in events:
    with gzip.open(event, 'rb') as in_stream:
        pathset = pickle.load(in_stream)
        for path in pathset:
            path[3:6] = [0.0, 1.0, 0.0]
            cmd.load_cgo(path, 'release_002')
cmd.set('cgo_line_width', 2, 'release_002')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC03_overall_pathset3.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_003')
cmd.set('cgo_line_width', 5, 'cluster_003')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC04_overall_pathset3.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_004')
cmd.set('cgo_line_width', 5, 'cluster_004')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC05_overall_pathset3.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_005')
cmd.set('cgo_line_width', 5, 'cluster_005')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC07_overall_pathset3.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_007')
cmd.set('cgo_line_width', 5, 'cluster_007')

events = [os.path.join('sources', 'layered_data', 'aquaduct', 'md1', 'paths', 'wat_4_entry_pathset.dump.gz'),
os.path.join('sources', 'layered_data', 'aquaduct', 'md1', 'paths', 'wat_6_entry_pathset.dump.gz')]
for event in events:
    with gzip.open(event, 'rb') as in_stream:
        pathset = pickle.load(in_stream)
        for path in pathset:
            path[3:6] = [1.0, 1.0, 1.0]
            cmd.load_cgo(path, 'entry_outlier')
cmd.set('cgo_line_width', 2, 'entry_outlier')

events = [os.path.join('sources', 'layered_data', 'aquaduct', 'md1', 'paths', 'wat_4_release_pathset.dump.gz')]
for event in events:
    with gzip.open(event, 'rb') as in_stream:
        pathset = pickle.load(in_stream)
        for path in pathset:
            path[3:6] = [1.0, 1.0, 1.0]
            cmd.load_cgo(path, 'release_outlier')
cmd.set('cgo_line_width', 2, 'release_outlier')

cmd.do('set all_states, 1')
cmd.show('cgo')
cmd.disable('release_*')
cmd.disable('entry_*')
cmd.zoom()
