import pickle, gzip

cmd.load('md1_A_trans_rot.pdb', 'protein_structure')
cmd.show_as('cartoon', 'protein_structure')
cmd.color('gray', 'protein_structure')

with gzip.open('raw_paths_1_cgo.dump.gz', 'rb') as in_stream:
    load_cluster = pickle.load(in_stream)
cmd.load_cgo(load_cluster, 'raw_paths_1')
with gzip.open('raw_paths_4_cgo.dump.gz', 'rb') as in_stream:
    load_cluster = pickle.load(in_stream)
cmd.load_cgo(load_cluster, 'raw_paths_4')
with gzip.open('raw_paths_6_cgo.dump.gz', 'rb') as in_stream:
    load_cluster = pickle.load(in_stream)
cmd.load_cgo(load_cluster, 'raw_paths_6')
cmd.disable('raw*')

