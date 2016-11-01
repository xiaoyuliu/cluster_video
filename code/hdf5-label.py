import scipy.io as sio
import os,pdb
import h5py
import numpy as np
import random

data_root = '/cs/vml2/xla193/cluster_video/output/UCF-101'
pathlabel = 'list_frm-0labelvect.txt'

input_file= os.path.join(data_root, pathlabel)
with open(input_file, 'r') as inf:
	lines = inf.readlines()
path_list = []
label_list= np.zeros([len(lines), 10])
for idx, line in enumerate(lines):
	path, label = line.strip().split(' ')
	path_list.append([path])
	for cid,char in enumerate(label.split(',')):
		label_list[idx][cid] = int(char)

order        = range(len(lines))
random.shuffle(order)
path_random  = np.ndarray.tolist(np.array(path_list)[order,:])
label_random = label_list[order,:]

with open(os.path.join(data_root, 'list_frm-0labelvect-random.txt'), 'w') as ouf:
	for p in path_random:
		ouf.write(p[0]+' 0\n')

train_filename = os.path.join(data_root, 'train_labelvect0.h5')
test_filename  = os.path.join(data_root, 'test_labelvect0.h5')


# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
# To show this off, we'll list the same data file twice.
with h5py.File(train_filename, 'w') as hf:
    hf.create_dataset('train_label',data = label_random  )
with h5py.File(test_filename, 'w') as vhf:
    vhf.create_dataset('test_label',data = label_random  )

with open(os.path.join(data_root, 'train_label_0.txt'), 'w') as f:
    f.write(train_filename + '\n')
with open(os.path.join(data_root, 'test_label_0.txt'), 'w') as vf:
    vf.write(test_filename + '\n')