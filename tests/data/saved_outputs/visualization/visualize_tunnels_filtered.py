import pickle, gzip

cmd.load('sources/super_cluster_CGOs/ref_transformed.pdb', 'protein_structure')
cmd.show_as('cartoon', 'protein_structure')
cmd.color('gray', 'protein_structure')

with gzip.open('sources/super_cluster_CGOs/SC01_overall_pathset2.dump.gz', 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_001')
cmd.set('cgo_line_width', 5, 'cluster_001')

with gzip.open('sources/super_cluster_CGOs/SC02_overall_pathset2.dump.gz', 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_002')
cmd.set('cgo_line_width', 5, 'cluster_002')

with gzip.open('sources/super_cluster_CGOs/SC03_overall_pathset2.dump.gz', 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_003')
cmd.set('cgo_line_width', 5, 'cluster_003')

with gzip.open('sources/super_cluster_CGOs/SC04_overall_pathset2.dump.gz', 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_004')
cmd.set('cgo_line_width', 5, 'cluster_004')

with gzip.open('sources/super_cluster_CGOs/SC05_overall_pathset2.dump.gz', 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_005')
cmd.set('cgo_line_width', 5, 'cluster_005')

with gzip.open('sources/super_cluster_CGOs/SC07_overall_pathset2.dump.gz', 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_007')
cmd.set('cgo_line_width', 5, 'cluster_007')

cmd.do('set all_states, 1')
cmd.show('cgo')
cmd.disable('release_*')
cmd.disable('entry_*')
