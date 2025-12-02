from pymol import cmd
from chempy.models import Indexed
from chempy import Bond, Atom
from pymol import cmd

if not os.path.exists("../data/clusters"):
	cmd.cd("/home/briza/PycharmProjects/transport_tools/tests/data/simulations/md1/caver/pymol")

execfile('./modules/rgb.py')

def create_bond(model, a1, a2): # one-based atom serial numbers
	b = Bond()
	b.index = [a1 - 1, a2 - 1] # zero-based indices!
	model.bond.append(b)

def create_spheres(name, spheres, links, frame):
	
	cmd.delete(name)
	model = Indexed()
	for i in (range(len(spheres) / 4)):
		ai = i * 4
		r = spheres[ai + 3]
		cluster = int(name[7:10])
		if r <> 0.5:
			a=Atom()
			a.name = "X" + str(i)
			a.resi = str(cluster)
			a.vdw = spheres[ai + 3]
			a.coord = [spheres[ai], spheres[ai + 1], spheres[ai + 2]]
			model.atom.append(a)

	for i in (range(len(links) / 2)):
		li = i * 2
		a1 = links[li]
		a2 = links[li + 1]
		create_bond(model, a1, a2)

	cmd.load_model(model, name,frame)
	cmd.hide("lines", name)
	cmd.show("spheres", name)

def computeSpheres(frame):
	#starttime = time.time();
	tunnels = {}
	conects = {}
	
	for tunnelName in tunnelNames:  # for each cluster
		path = "../data/clusters/" + tunnelName
		infile = open(path, "r")    # open PDB file with cluster
		modelNumber = 0;
		spheres = []
		links = []
		unfinished = True
		while infile and unfinished:
			line = infile.readline()

			if(len(line) == 0): 
				break

			if(line[0:5] == "MODEL"):
				modelNumber += 1;
				
				if(modelNumber == frame): # find line for actual frame
					while infile and unfinished:
						line = infile.readline()
						if(line[0:4] == "ATOM"):
							spheres.append(float(line[30:38]))
							spheres.append(float(line[38:46]))
							spheres.append(float(line[46:54]))
							spheres.append(float(line[62:66]))
						if(line[0:6] == "CONECT"):
							links.append(int(line[6:11]))
							links.append(int(line[11:16]))
						if(line[0:6] == "ENDMDL"):
						 	tunnels[tunnelName] = spheres
							conects[tunnelName] = links
						 	unfinished = False
		infile.close()

	color = 1
	view = cmd.get_view()
	for tn in tunnelNames:  # for each cluster
		create_spheres(tn, tunnels[tn], conects[tn], frame)
		sc = 'caver' + str(color)
		cmd.color(color, tn)
		cmd.color(sc, tn)
		if color < 1000:
			color += 1
	cmd.set_view(view)
	#endtime = time.time();
	#print str(endtime - starttime)



if not os.path.exists("../data/clusters"):
	cmd.cd("c:/data/caver/testing_9/out/pymol")

color = 1
list = os.listdir("../data/clusters")

list.sort()

tunnelNames=[]
for fn in list:
	name = fn
	tunnelNames.append(name)



cmd.do('set all_states,0')

cmd.set('two_sided_lighting', 'on')
cmd.set('transparency', '0.2')

cmd.load('../data/origins.pdb', 'origins')
cmd.load('../data/v_origins.pdb', 'v_origins')
cmd.show('nb_spheres', 'origins')
cmd.show('nb_spheres', 'v_origins')

load_trajectory = False

if load_trajectory:
	cmd.load("../data/trajectory.pdb", "structure", discrete=1)
	cmd.dss("structure")
	
else:
	cmd.load('../data/stripped_system.000501.pdb', 'structure')

cmd.hide('lines', 'structure')
cmd.show('cartoon', 'structure')
cmd.color('gray', 'structure')

cmd.mset("1 -%d" % cmd.count_states())
for frame in range(1,cmd.count_states()+1):
	cmd.mdo(frame, "computeSpheres(" + str(frame) + ")")
cmd.frame(1)
