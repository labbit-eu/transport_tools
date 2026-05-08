from pymol import cmd

if not os.path.exists("../data/clusters_timeless"):
	cmd.cd("/mnt/storage_5/scratch/pl0252-02/igomar/ups/linb-wt_be/workdir/epoch_10_runs/e10s14_e4s14f222m327_f1997m1236/caver_results/pymol")

cmd.cd("modules")
import caver
cmd.cd("..")

execfile('./modules/rgb.py')

color = 1
list = os.listdir("../data/clusters_timeless")
list.sort()
name = ''
for fn in list:
	old_name = name
	name = fn

	if color < 1000 and caver.new_cluster(old_name, name):
		color += 1

	cmd.load('../data/clusters_timeless/' + fn, name)
	cmd.color('caver' + str(color), name)
	cmd.alter(name, 'vdw=b')
cmd.do('set all_states,1')

cmd.set('two_sided_lighting', 'on')
cmd.set('transparency', '0.2')

cmd.load('../data/origins.pdb', 'origins')
cmd.load('../data/v_origins.pdb', 'v_origins')
cmd.show('nb_spheres', 'origins')
cmd.show('nb_spheres', 'v_origins')

cmd.load('../data/stripped_system.501.pdb', 'structure')
cmd.hide('lines', 'structure')
cmd.show('cartoon', 'structure')
cmd.color('gray', 'structure')

