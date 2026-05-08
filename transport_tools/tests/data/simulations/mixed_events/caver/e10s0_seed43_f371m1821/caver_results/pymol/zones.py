from pymol import cmd

cmd.load('../data/start_zone.pdb' , 'starts')
cmd.color('green', 'starts')

cmd.load('../data/end_zone.pdb' , 'ends')
cmd.color('red', 'ends')

cmd.load('../data/surface.pdb' , 'sur')
cmd.color('grey', 'sur')

cmd.load('../data/surface_definition.pdb' , 'sur_def')
cmd.color('white', 'sur')
