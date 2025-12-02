import argparse
parser=argparse.ArgumentParser(description="Aqua-Duct visualization script")
parser.add_argument("--save-session",action="store",dest="session",required=False,default=None,help="Pymol session file name.")
parser.add_argument("--discard",action="store",dest="discard",required=False,default='',help="Objects to discard.")
parser.add_argument("--keep",action="store",dest="keep",required=False,default='',help="Objects to keep.")
parser.add_argument("--force-color",action="store",dest="fc",required=False,default='',help="Force specific color.")
parser.add_argument("--fast",action="store_true",dest="fast",required=False,help="Hides all objects while loading.")
args,unknown=parser.parse_known_args()
import sys
if unknown: print("WARNING: Unknown options were used: "+" ".join(unknown), sys.stderr)
def _kd_order():
    if args.keep=='' and args.discard!='': return 'd'
    if args.keep!='' and args.discard=='': return 'k'
    if args.keep=='' and args.discard=='': return None
    if sys.argv.index('--keep')<sys.argv.index('--discard'): return 'k'
    return 'd'
kd_order = _kd_order()
def discard(name):
    if len([d for d in args.discard.split() if d in name])>0: return True
    return False
def keep(name):
    if len([k for k in args.keep.split() if k in name])>0: return True
    return False
def proceed(name):
    if kd_order == 'k':
        if not keep(name): return False
        elif discard(name): return False
    elif kd_order == 'd':
        if discard(name):
            if not keep(name): return False
    return True
from pymol import cmd,finish_launching
finish_launching()
print("Loading Aqua-Duct visualization...")
cmd.set("cgo_line_width",2)
cmd.set("line_smooth","off")
from os import close,unlink
from os.path import splitext,isfile
import tarfile
import pickle as pickle
import json
from tempfile import mkstemp
fd, pdb_filename = mkstemp(suffix=".pdb")
close(fd)
max_state=0
arch_file="6_visualize_results.tar.gz"
if not isfile(arch_file):
    import pymol
    if pymol.IS_WINDOWS:
        print("Please open visualization script using 'Open with' context menu and choose PyMol executable.")
        print("Alternatively, if you have PyMol installed as Python module, open visulaization script with Python executable.")
    while (pymol._ext_gui is None): pymol = reload(pymol)
    while (not hasattr(pymol._ext_gui,'root')): pymol = reload(pymol)
    import tkFileDialog
    arch_file=tkFileDialog.askopenfilename(filetypes=[("AQ Vis Arch","*.tar.gz")],title="Select AQ visualization archive",parent=pymol._ext_gui.root)
data_fh=tarfile.open(arch_file,"r:gz")
def decode_color(cgo_object,fc=None):
    for element in cgo_object:
        if isinstance(element,tuple):
            if fc is None:
                for e in element: yield e
            else:
                for e in fc: yield e
        else:
            yield element
def load_object(filename,name,state):
    if not proceed(name): return
    global max_state
    print("Loading %s" % splitext(filename)[0])
    obj=pickle.load(data_fh.extractfile(filename))
    #obj=json.load(data_fh.extractfile(filename))
    if name in args.fc.split():
        forced_color=args.fc.split()[args.fc.split().index(name)+1]
        forced_color=cmd.get_color_tuple(forced_color)
        obj=decode_color(obj,fc=forced_color)
    else:
        obj=decode_color(obj)
    cmd.load_cgo(obj,name,state)
    if state<2:
        if args.fast: cmd.disable("all")
        else: cmd.refresh()
    if state>max_state:
        max_state=state
def load_pdb(filename,name,state):
    if not proceed(name): return
    global max_state
    with open(pdb_filename,'wb') as fpdb:
        fpdb.write(data_fh.extractfile(filename).read())
    cmd.load(pdb_filename,state=state,object=name)
    if state>max_state:
        max_state=state
load_pdb("molecule0_1.pdb","molecule0",1)
if proceed("molecule0"): cmd.show_as('cartoon','molecule0')
if proceed("molecule0"): cmd.color('silver','molecule0')
load_object("raw_paths_1.dump","raw_paths",1)
load_object("raw_paths_2.dump","raw_paths",2)
load_object("raw_paths_3.dump","raw_paths",3)
load_object("raw_paths_4.dump","raw_paths",4)
load_object("raw_paths_5.dump","raw_paths",5)
load_object("raw_paths_6.dump","raw_paths",6)
if proceed("molecule0"): cmd.orient("molecule0")
data_fh.close()
unlink(pdb_filename)
if args.fast: cmd.enable("all")
print("Aqua-Duct visualization loaded.")
if args.session:
    print("Preparing data to save session...")
    for state in range(max_state):
        cmd.set_frame(state+1)
        cmd.refresh()
        if (state+1)%100==0:
            print("wait... %d of %d done..." % (state+1,max_state))
    print("%d of %d done." % (state+1,max_state))
    print("Saving session...")
    cmd.set_frame(1)
    cmd.save(args.session,state=0)
    print("Let the Valve be always open!")
    print("Goodbye!")
    cmd.quit()

