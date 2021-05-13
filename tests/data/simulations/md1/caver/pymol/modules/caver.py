def new_cluster(a, b):
	if len(a) <> len(b):
		return False # do not increase color in first iteration
	else:
		ai = a[11]
		bi = b[11]
		return int(ai) + 1 <> int(bi)
		# equality would indicates only a next file of the same cluster

