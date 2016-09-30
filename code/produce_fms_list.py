import os,sys,pdb
import numpy as np

dataset_path = '/cs/vml2/xla193/cluster_video/datasets/UCF-101'

with open('/cs/vml2/xla193/cluster_video/output/UCF-101/input-UCF-101-20-fmcount.txt', 'w') as ouf:
	cnames = os.listdir(dataset_path)
	for cname in cnames[:20]:
		vpath = os.path.join(dataset_path,cname)
		vnames = os.listdir(vpath)
		for vname in vnames:
			if not os.path.isdir(os.path.join(vpath, vname)):
				continue
			fpath = os.path.join(vpath, vname)
			fnames= os.listdir(fpath)
			content = len(fnames)
			print "frame number of ", vname, ":", content
			ouf.write(str(content)+'\n')

