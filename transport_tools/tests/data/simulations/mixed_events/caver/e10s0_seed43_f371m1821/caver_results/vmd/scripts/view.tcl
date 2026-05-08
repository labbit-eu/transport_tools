mol representation Lines
set dir "../data/clusters"
set ext ".pdb"
set color 0
set molecule 0
set colors [list 0 7 1 10 4 27 3 5 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 28 29 30 31 32]
set contents [glob -directory $dir *$ext]
set contents [lsort $contents]
foreach item $contents {
  mol load pdb $item 
  set tunnel [atomselect top "all"]
  $tunnel set radius [$tunnel get beta]
  mol modcolor 0 $molecule "ColorID" [lindex $colors $color]
  if {[expr {$color - 1 < [llength $colors]}]} {incr color}
  incr molecule 
}

source "./scripts/vmd_load_structures.tcl"
source "./scripts/radii.tcl"

