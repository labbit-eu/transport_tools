import pickle, gzip, os

cmd.load(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'ref_transformed.pdb'), 'protein_structure')
cmd.show_as('cartoon', 'protein_structure')
cmd.color('gray', 'protein_structure')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC01_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_001')
cmd.set('cgo_line_width', 5, 'cluster_001')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC02_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_002')
cmd.set('cgo_line_width', 5, 'cluster_002')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC04_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_004')
cmd.set('cgo_line_width', 5, 'cluster_004')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC03_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_003')
cmd.set('cgo_line_width', 5, 'cluster_003')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC05_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_005')
cmd.set('cgo_line_width', 5, 'cluster_005')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC07_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_007')
cmd.set('cgo_line_width', 5, 'cluster_007')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC08_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_008')
cmd.set('cgo_line_width', 5, 'cluster_008')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC09_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_009')
cmd.set('cgo_line_width', 5, 'cluster_009')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC10_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_010')
cmd.set('cgo_line_width', 5, 'cluster_010')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC12_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_012')
cmd.set('cgo_line_width', 5, 'cluster_012')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC13_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_013')
cmd.set('cgo_line_width', 5, 'cluster_013')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC14_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_014')
cmd.set('cgo_line_width', 5, 'cluster_014')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC17_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_017')
cmd.set('cgo_line_width', 5, 'cluster_017')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC18_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_018')
cmd.set('cgo_line_width', 5, 'cluster_018')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC19_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_019')
cmd.set('cgo_line_width', 5, 'cluster_019')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC20_group2_pathset2.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_020')
cmd.set('cgo_line_width', 5, 'cluster_020')

cmd.do('set all_states, 1')
cmd.show('cgo')
cmd.disable('*_release*')
cmd.disable('*_entry*')
cmd.zoom()
