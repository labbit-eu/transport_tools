set dir "../data/cluster_radii"
set ext ".r"
set clusters 0
set contents [lsort [glob -directory $dir *$ext]]

foreach item $contents {
	set f [open $item]
	set frame 0
	foreach line [split [read $f] \n] {
		append radius($clusters,$frame) $line
		incr frame
	}
	incr clusters
	close $f
}

proc enabletrace {} {
    global vmd_frame
	upvar radius radius
    trace variable vmd_frame(1) w drawcounter
    #graphics top cylinder {0 0 0} {100 0 0} radius 1.0
}

proc disabletrace {} {
    global vmd_frame
    trace vdelete vmd_frame([molinfo top]) w drawcounter
}

proc drawcounter { name element op } {
	uplevel {redraw}
}
proc redraw {} {
	global vmd_frame
	upvar radius radius
	upvar clusters clusters
	for {set i 0} {$i<$clusters} {incr i} {
		if {[molinfo $i get "drawn"] == 0} {
			continue
		}
		set as [atomselect $i "all"]
		for {set m 0} {$m<[$as num]} {incr m} {
			set atom [atomselect $i "index $m"]
			set frame [molinfo $i get "frame"]
			$atom set radius [string range $radius($i,$vmd_frame(1)) [expr $m*4] [expr $m*4+3]]
		} 
	}
}


for {set i 0} {$i<$clusters} {incr i} {
	mol modstyle 0 $i VDW 1.0 25.0
}

enabletrace
redraw

