import scipy.io as sio
import os,pdb
import h5py
import numpy as np
import random

data_root = '/cs/vml4/xla193/cross1_list'
output_root = '/local-scratch/xla193/cluster_video_/output/UCF-101/'
pathlabel = 'list_video-fix-veri-7ft.txt'

K = 10
input_file= os.path.join(output_root, pathlabel)
with open(input_file, 'r') as inf:
	lines = inf.readlines()
path_list = []
label_list= np.zeros([len(lines), 10])
for idx, line in enumerate(lines):
	path, label = line.strip().split(' ')
	path_list.append([path])
	for cid,char in enumerate(label.split(',')):
		label_list[idx][cid] = int(char)

# new_length = len(lines) - (len(lines) % K);
# new_lines  = [path_list[x] for x in range(new_length)]
# new_labels = label_list[range(new_length),:]

# order        = range(new_length)
# random.shuffle(order)
# path_random  = np.array(new_lines)[order,:]
# label_random = new_labels[order,:]

all_order = range(len(lines))
random.shuffle(all_order)
all_path_random = (np.array(path_list)[all_order,:])
all_label_random= label_list[all_order,:]
all_path_fms  = []
all_label_fms = []
vcount = 0
fcount = 0
for vpath_ in all_path_random:
	vpath = os.path.join('/cs/vml2/xla193/cluster_video/datasets/UCF-101/', vpath_[0])
	fms = os.listdir(vpath)
	for fm in fms:
		fpath = os.path.join(vpath, fm)
		content = fpath
		all_path_fms.append(content)
		all_label_fms.append(all_label_random[vcount][:])
		fcount += 1
	vcount += 1
all_order_fms = range(len(all_label_fms))
random.shuffle(all_order_fms)
all_fpath_rand = np.ndarray.tolist(np.array(all_path_fms)[all_order_fms])
all_flabel_rand= np.array(all_label_fms)[all_order_fms, :]
with open(os.path.join(data_root, 'list_frm-labelvect-fix-veri-7ft-random.txt'), 'w') as allf:
	for x in all_fpath_rand:
		content = x + ' 0\n'
		allf.write(content)

all_label_filename = os.path.join(data_root, 'train_labelvect_fix-veri7.h5')
with h5py.File(all_label_filename, 'w') as hf:
	    hf.create_dataset('train_label',data = all_flabel_rand  )
with open(os.path.join(data_root, 'train_label_fix_veri_7.txt'), 'w') as f:
	    f.write(all_label_filename + '\n')
	    all_label_filename = os.path.join(data_root, 'valid_labelvect_fix-veri7.h5')
with h5py.File(all_label_filename, 'w') as hf:
	    hf.create_dataset('valid_label',data = all_flabel_rand  )
with open(os.path.join(data_root, 'valid_label_fix_veri_7.txt'), 'w') as f:
	    f.write(all_label_filename + '\n')

pdb.set_trace()
bin = len(lines) / K
for i in range(K):
	start_index = i*bin
	end_index = (i+1)*bin
	if i == K-1:
		end_index = len(lines)-1
	train_label_set = np.delete(all_label_random, range(start_index, end_index), axis=0)
	valid_label_set = all_label_random[start_index:end_index,:]
	train_path_set = np.delete(all_path_random, range(start_index, end_index), axis=0)
	valid_path_set = all_path_random[start_index:end_index, :]
	train_path = []
	train_label = []
	valid_path = []
	valid_label = []
	vcount = 0
	fcount = 0
	for video in train_path_set:
		vpath = os.path.join('/cs/vml2/xla193/cluster_video/datasets/UCF-101/', video[0])
		fms = os.listdir(vpath)
		for fm in fms:
			fpath = os.path.join(vpath, fm)
			content = fpath
			train_path.append([content])
			train_label.append(train_label_set[vcount][:])
			fcount += 1
		vcount += 1
	vcount = 0
	fcount = 0
	for video in valid_path_set:
		vpath = os.path.join('/cs/vml2/xla193/cluster_video/datasets/UCF-101/', video[0])
		fms = os.listdir(vpath)
		for fm in fms:
			fpath = os.path.join(vpath, fm)
			content = fpath
			valid_path.append([content])
			valid_label.append(valid_label_set[vcount][:])
			fcount += 1
		vcount += 1
	train_label_filename = os.path.join(data_root, 'train_labelvect_fix0-'+str(i+1)+'.h5')
	valid_label_filename = os.path.join(data_root, 'valid_labelvect_fix0-'+str(i+1)+'.h5')
	train_path_filename = os.path.join(data_root, 'list_frm-veri-fix-train-shuffle-0-'+str(i+1)+'.txt')
	valid_path_filename = os.path.join(data_root, 'list_frm-veri-fix-valid-shuffle-0-'+str(i+1)+'.txt')
	range_train = range(len(train_path))
	range_valid = range(len(valid_path))
	random.shuffle(range_train)
	random.shuffle(range_valid)

	train_path_rand = np.array(train_path)[range_train]
	train_label_rand= np.array(train_label)[range_train][:]
	valid_path_rand = np.array(valid_path)[range_valid]
	valid_label_rand= np.array(valid_label)[range_valid][:]
	with open(train_path_filename, 'w') as ouf:
		for x in train_path_rand:
			ouf.write(x[0]+' 0\n')
	with open(valid_path_filename, 'w') as ouf:
		for x in valid_path_rand:
			ouf.write(x[0]+' 0\n')

	# HDF5DataLayer source should be a file containing a list of HDF5 filenames.
	# To show this off, we'll list the same data file twice.
	with h5py.File(train_label_filename, 'w') as hf:
	    hf.create_dataset('train_label',data = train_label_rand  )
	with h5py.File(valid_label_filename, 'w') as vhf:
	    vhf.create_dataset('valid_label',data = valid_label_rand  )

	with open(os.path.join(data_root, 'train_label_fix_0-'+str(i+1)+'.txt'), 'w') as f:
	    f.write(train_label_filename + '\n')
	with open(os.path.join(data_root, 'valid_label_fix_0-'+str(i+1)+'.txt'), 'w') as vf:
	    vf.write(valid_label_filename + '\n')