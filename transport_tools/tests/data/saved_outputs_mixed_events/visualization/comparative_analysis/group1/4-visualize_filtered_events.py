import pickle, gzip, os

cmd.load(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'ref_transformed.pdb'), 'protein_structure')
cmd.show_as('cartoon', 'protein_structure')
cmd.color('gray', 'protein_structure')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC01_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_001')
cmd.set('cgo_line_width', 5, 'cluster_001')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC01_group1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_001".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC02_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_002')
cmd.set('cgo_line_width', 5, 'cluster_002')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC02_group1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_002".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC04_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_004')
cmd.set('cgo_line_width', 5, 'cluster_004')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC04_group1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_004".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC03_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_003')
cmd.set('cgo_line_width', 5, 'cluster_003')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC03_group1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_003".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC05_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_005')
cmd.set('cgo_line_width', 5, 'cluster_005')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC05_group1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_005".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC07_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_007')
cmd.set('cgo_line_width', 5, 'cluster_007')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC08_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_008')
cmd.set('cgo_line_width', 5, 'cluster_008')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC08_group1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_008".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC06_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_006')
cmd.set('cgo_line_width', 5, 'cluster_006')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC06_group1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_006".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC09_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_009')
cmd.set('cgo_line_width', 5, 'cluster_009')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC09_group1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_009".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC10_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_010')
cmd.set('cgo_line_width', 5, 'cluster_010')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC11_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_011')
cmd.set('cgo_line_width', 5, 'cluster_011')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC11_group1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_011".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC12_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_012')
cmd.set('cgo_line_width', 5, 'cluster_012')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC13_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_013')
cmd.set('cgo_line_width', 5, 'cluster_013')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC15_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_015')
cmd.set('cgo_line_width', 5, 'cluster_015')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC15_group1_events4.dump.gz'), 'rb') as in_stream:
    sc_events = pickle.load(in_stream)
for (event_type, resname), event_cgo in sc_events.items():
    obj_name = "{}_{}_015".format(resname.lower(), event_type)
    cmd.load_cgo(event_cgo, obj_name)
    cmd.set('cgo_line_width', 2, obj_name)

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'SC16_group1_pathset4.dump.gz'), 'rb') as in_stream:
    pathset = pickle.load(in_stream)
cmd.load_cgo(pathset, 'cluster_016')
cmd.set('cgo_line_width', 5, 'cluster_016')

with gzip.open(os.path.join('..', '..', 'sources', 'super_cluster_CGOs', 'outliers_group1_events4.dump.gz'), 'rb') as in_stream:
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
