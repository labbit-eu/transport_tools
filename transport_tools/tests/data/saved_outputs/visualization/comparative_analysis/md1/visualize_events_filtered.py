import pickle, gzip, os

cmd.load(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'ref_transformed.pdb'), 'protein_structure')
cmd.show_as('cartoon', 'protein_structure')
cmd.color('gray', 'protein_structure')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC02_md1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_002')
cmd.set('cgo_line_width', 5, 'cluster_002')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC02_md1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_002".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'outliers_md1_events4.dump.gz'), 'rb') as in_stream:
    outlier_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in outlier_events.items():
    obj_name = "{}_{}_outlier".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

cmd.do('set all_states, 1')
cmd.show('cgo')
cmd.disable('*_release*')
cmd.disable('*_entry*')
cmd.zoom()
