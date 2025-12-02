import pickle, gzip, os

cmd.load(os.path.join('sources', 'super_cluster_CGOs', 'ref_transformed.pdb'), 'protein_structure')
cmd.show_as('cartoon', 'protein_structure')
cmd.color('gray', 'protein_structure')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC01_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_001')
cmd.set('cgo_line_width', 5, 'cluster_001')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC02_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_002')
cmd.set('cgo_line_width', 5, 'cluster_002')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC03_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_003')
cmd.set('cgo_line_width', 5, 'cluster_003')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC04_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_004')
cmd.set('cgo_line_width', 5, 'cluster_004')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC05_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_005')
cmd.set('cgo_line_width', 5, 'cluster_005')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC06_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_006')
cmd.set('cgo_line_width', 5, 'cluster_006')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC09_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_009')
cmd.set('cgo_line_width', 5, 'cluster_009')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC07_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_007')
cmd.set('cgo_line_width', 5, 'cluster_007')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC08_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_008')
cmd.set('cgo_line_width', 5, 'cluster_008')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC10_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_010')
cmd.set('cgo_line_width', 5, 'cluster_010')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC12_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_012')
cmd.set('cgo_line_width', 5, 'cluster_012')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC11_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_011')
cmd.set('cgo_line_width', 5, 'cluster_011')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC13_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_013')
cmd.set('cgo_line_width', 5, 'cluster_013')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC14_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_014')
cmd.set('cgo_line_width', 5, 'cluster_014')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC15_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_015')
cmd.set('cgo_line_width', 5, 'cluster_015')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC16_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_016')
cmd.set('cgo_line_width', 5, 'cluster_016')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC17_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_017')
cmd.set('cgo_line_width', 5, 'cluster_017')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC18_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_018')
cmd.set('cgo_line_width', 5, 'cluster_018')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC19_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_019')
cmd.set('cgo_line_width', 5, 'cluster_019')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC20_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_020')
cmd.set('cgo_line_width', 5, 'cluster_020')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC21_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_021')
cmd.set('cgo_line_width', 5, 'cluster_021')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC22_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_022')
cmd.set('cgo_line_width', 5, 'cluster_022')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC23_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_023')
cmd.set('cgo_line_width', 5, 'cluster_023')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC24_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_024')
cmd.set('cgo_line_width', 5, 'cluster_024')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC25_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_025')
cmd.set('cgo_line_width', 5, 'cluster_025')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC26_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_026')
cmd.set('cgo_line_width', 5, 'cluster_026')

with gzip.open(os.path.join('sources', 'super_cluster_CGOs', 'SC27_overall_pathset1.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_027')
cmd.set('cgo_line_width', 5, 'cluster_027')

cmd.do('set all_states, 1')
cmd.show('cgo')
cmd.disable('release_*')
cmd.disable('entry_*')
cmd.zoom()
