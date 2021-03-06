import random
import time
import numpy as np 
import math
import pdb
from scipy.spatial.distance import euclidean
import scipy.io as sio
import optparse
import os

optparser = optparse.OptionParser()
optparser.add_option("--df", "--datafile", dest ="datafile", default='UCF-101-gtlabel-10.mat', help="Input data file name")
optparser.add_option("--cf", "--consfile", dest ="consfile", default=None, help="Input const file name")
optparser.add_option("-k", "--ncluster",   dest ="ncluster", default=10, help="Input the number of clusters")
optparser.add_option("--of", "--outfile",  dest ="outfile",  default=None, help="Input output file name")
optparser.add_option("--rp", "--repeat",   dest ="repeat", default=1, help="Input repeat number of clustering")
optparser.add_option("--sc", "--savecons", dest ="savecons", default='const1.mat', help="Input name of save const file")
optparser.add_option("--sv", "--savevects",dest ="savevects", default=None, help="Input name of save labelvector file")
(opts, _) = optparser.parse_args()

data_root = '/local-scratch/xla193/cluster_video_/output/UCF-101'

def cop_kmeans(dataset, kk, repeat, savecons, savevects, ml=[], cl=[]):
	ml, cl = transitive_closure(ml, cl, len(dataset)) # return two dictionaries
	finding_count   = 0
	converged_count = 0
	break_num       = 13740
	init_num        = 2 * repeat
	smallest_num    = 3

	sumd = float("inf")
	centers, centers_id = initialize_centers(dataset, kk)
	while not check_initialization(centers_id, ml):
		centers, centers_id = initialize_centers(dataset, kk)
	print "initialization check passed"
	clusters = [0] * len(dataset)
	while(converged_count < repeat):
		centers, centers_id = initialize_centers(dataset, kk)
		while not check_initialization(centers_id, ml):
			centers, centers_id = initialize_centers(dataset, kk)
		print "initialization check passed"
		cantfind_flag    = False
		converged = False
		clusters  = [0] * len(dataset)
		s_num = 0
		while (not converged):
			vialate_count   = 0
			print centers_id
			# print ("-%d-th finding clustering - %f " % (finding_count, sumd))
			clusters_ = [0] * len(dataset)
			for l, c_id in enumerate(centers_id):
				clusters_[c_id] = l+1
				for k in ml[c_id]:
					clusters_[k] = l+1
			for i, d in enumerate(dataset): # d:datapoint i:index in dataset
				if clusters_[i] != 0 and not cl[i]:
					continue
				if len(cl[i]) > 0:
					if clusters_[i] == 0:
						indices = closet_clusters(centers, d)
						counter = 0
						found_cluster = False
						while (not found_cluster) and (counter < len(indices)):
							index = indices[counter]
							if not violate_constraints(i, centers_id[index], index, clusters_, ml, cl):
								found_cluster = True
								clusters_[i]  = index+1
								break
							else:
								vialate_count += 1
								counter += 1
								print "violate, try again, number: ", vialate_count, " total allowed: ", break_num, "\r",
								if vialate_count == break_num:
									break
							if not found_cluster and counter==len(indices):
								# raise Exception("can't find cluster for %d-th datapoint" % i)
								print ("can't find cluster for %d-th datapoint" % i)
								cantfind_flag = True
								break
					for item in cl[i]:
						if clusters_[item]==0:
							indices = closet_clusters(centers, dataset[item])
							counter = 0
							found_cluster = False
							while (not found_cluster) and (counter < len(indices)):
								index = indices[counter]
								if not violate_constraints(item, centers_id[index], index, clusters_, ml, cl):
									found_cluster = True
									clusters_[item]  = index+1
								else:
									vialate_count += 1
									counter += 1
									print "violate, try again, number: ", vialate_count, " total allowed: ", break_num, "\r",
									if vialate_count == break_num:
										break
								if not found_cluster and counter==len(indices):
									# raise Exception("can't find cluster for %d-th datapoint" % i)
									print ("can't find cluster for %d-th datapoint" % i)
									cantfind_flag = True
									break
				if clusters_[i] != 0:
					continue
				indices = closet_clusters(centers, d)
				counter = 0
				found_cluster = False
				while (not found_cluster) and (counter < len(indices)):
					index = indices[counter]
					if not violate_constraints(i, centers_id[index], index, clusters_, ml, cl):
						found_cluster = True
						clusters_[i]  = index+1
						if i == len(dataset)-1:
							print("%d-th data point clustered." % i)
					else:
						vialate_count += 1
						counter += 1
						print "violate, try again, number: ", vialate_count, " total allowed: ", break_num, "\r",
						if vialate_count == break_num:
							break
					if not found_cluster and counter==len(indices):
						# raise Exception("can't find cluster for %d-th datapoint" % i)
						print ("can't find cluster for %d-th datapoint" % i)
						cantfind_flag = True
						break
				if vialate_count >= break_num or cantfind_flag:
					break
				if not found_cluster and (converged_count == 0):
					return None
			if cantfind_flag:
				break
			if vialate_count == break_num:
				if init_count == init_num:
					repeat = converged_count
				break
			clusters_, centers, centers_id, sumd_ = compute_centers(clusters_, dataset)
			converged     = True
			i = 0
			while converged and (i < len(dataset)):
				if clusters[i] != clusters_[i] or clusters_[i] == 0:
					converged = False
					break
				i += 1

			# if sumd_ == sumd:
			# 	s_num += 1
			# 	if s_num == smallest_num:
			# 		converged = True
			# if sumd_ < sumd:
			# 	s_num = 0
			# 	sumd = sumd_

			clusters = clusters_
			print centers_id
			if converged == True:
				converged_count += 1
				if sumd_ < sumd:
					repeat = converged_count
					print ("--%d-th clustering - smallest: %f - now: %f " % (converged_count, sumd, sumd_))
					print time.ctime()
					s_clusters   = clusters
					s_centers    = centers_id
					sumd         = sumd_					

			else:
				print "not converged yet."
				print sumd_

	gt_dict = sio.loadmat(os.path.join(data_root, 'UCF-101-gtlabel-10.mat'))
	gt_array = gt_dict['label']
	clusters_train = []
	for i in range(len(s_clusters)):
		clusters_train.append(s_clusters[i])
	clusters_train, ml, cl, new_centers = update_const(clusters_train, s_centers, gt_array, dataset, ml, cl)
	
	save_const_vect(ml, cl, savecons, savevects, s_clusters, clusters_train, new_centers)
	# pdb.set_trace()
	return s_clusters, clusters_train


def save_const_vect(ml, cl, savecons, savevects, clusters, clusters_train, centers_id):
	clusters_vect = []
	for i in range(len(clusters)):
		clusters_vect.append(set())
	uf_ids = set()

	num_data = len(ml)
	const = np.zeros([num_data, num_data])

	newml, newcl = [],[]
	for i in range(num_data):
		for j in ml[i]:
			const[i][j] = 1
			const[j][i] = 1
		for j in cl[i]:
			const[i][j] = -1
			const[j][i] = -1
	for i in range(num_data):
		for j in range(num_data):
			if i==j:
				continue
			constraint = (int(i),int(j))
			if const[i][j] == 1:
				newml.append(constraint)
			if const[i][j] == -1:
				newcl.append(constraint)
	ml, cl = transitive_closure(newml, newcl, num_data)

	for i in range(num_data):
		if i in ml[i]:
			ml[i].remove(i)
		if i in cl[i]:
			cl[i].remove(i)

	def add_both(oneset, a, b):
		oneset.add(a)
		oneset.add(b)

	const = np.zeros([num_data, num_data])
	for i_center in centers_id:
		label = clusters_train[i_center]
		clusters_vect[i_center].add(label)
		for j in ml[i_center]:
			clusters_vect[j].add(label)
			add_both(uf_ids, i_center, j)
			const[i_center][j] = 1
			const[j][i_center] = 1
		for k in cl[i_center]:
			clusters_vect[k].add(-label)
			add_both(uf_ids, i_center, k)
			const[i_center][k] = -1
			const[k][i_center] = -1
	for i in range(num_data):
		const[i][i]=0

	if not check_label(clusters_vect):
		print "wrong label vectors"
		pdb.set_trace()
	with open(os.path.join(data_root, savevects), 'w') as voutf:
		count = 0
		for k in sorted(uf_ids):
			voutf.write(','.join(map(str, clusters_vect[k])) + '\n')
	consts = dict()
	consts['const'] = const
	consts['ufids'] = sorted(uf_ids)
	savecons_file = os.path.join(data_root, savecons)
	sio.savemat(savecons_file, consts)

def check_label(c_vectors):
	flag = False
	for vect in c_vectors:
		idx  = -1
		for id, v in enumerate(vect):
			if v>0:
				idx  = id
		if idx<0:
			continue
		if -list(vect)[idx] in vect:
			return False
	return True

def update_const_label(clusters_train, centers_id, gt_array, dataset, ml, cl):
	cluster_array = np.array(clusters_train)
	ids = list(set(clusters_train))
	gt_labels = np.ndarray.tolist(gt_array[centers_id,0])
	center_dict = dict()
	for i in range(len(gt_labels)):
		if not gt_labels[i] in center_dict:
			center_dict[gt_labels[i]] = centers_id[i]
	new_centers = []
	for i in range(len(centers_id)):
		new_centers.append(centers_id[i])

	for label, i_center in enumerate(centers_id):
		i_list = np.ndarray.tolist(np.where(cluster_array == label+1)[0])
		i_distance = [euclidean(dataset[j], dataset[i_center]) for j in i_list]
		i_sort = sorted(range(len(i_list)), key=lambda x: i_distance[x])
		idx_after_sort = [ i_list[i_sort[k]] for k in range(len(i_sort))]
		if idx_after_sort[0]!=i_center:
			raise Exception("center %d's medoid is not correct" % i)

		K = 10
		i_gt = gt_array[i_center][0]
		# whether there are more than 1 center from the same category
		if gt_labels.count(i_gt)>1:
			if (i_center != center_dict[i_gt]):
				new_centers.remove(i_center)
			add_link(ml, i_center, center_dict[i_gt])
			delete_link(cl, i_center, center_dict[i_gt])
			label_ = clusters_train[center_dict[i_gt]]
			clusters_train[i_center] = label_
			for i in ml[i_center]:
				clusters_train[i] = label_
		for j in idx_after_sort[1:K+1]:
			j_gt = gt_array[j][0]
			if (j_gt == i_gt):
				clusters_train[j] = clusters_train[i_center]
				add_link(ml, i_center, j)
				delete_link(cl, i_center, j)
				for k in ml[i_center]:
					if k != j:
						add_link(ml, j, k)
						delete_link(cl, j, k)
				for k in ml[j]:
					if k != i_center:
						clusters_train[k] = clusters_train[i_center]
						add_link(ml, k, i_center)
						delete_link(cl, k, i_center)
				for k1 in ml[j]:
					for k2 in ml[j]:
						if k1 == k2:
							continue
						if not k2 in ml[k1]:
							add_link(ml, k1, k2)
							delete_link(cl, k1, k2)
				for k in cl[i_center]:
					add_link(cl, j, k)
					delete_link(ml, j, k)
				for k in cl[j]:
					add_link(cl, k, i_center)
					delete_link(ml, k, i_center)
				for k1 in cl[i_center]:
					for k2 in ml[i_center]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)

			if (j_gt != i_gt) and (gt_labels.count(j_gt) > 0):
				new_center = center_dict[j_gt]
				new_label = clusters_train[new_center]
				add_link(ml, new_center, j)
				delete_link(cl, new_center, j)
				add_link(cl, new_center, i_center)
				delete_link(ml, new_center, i_center)
				add_link(cl, j, i_center)
				delete_link(ml, j, i_center)
				for k in ml[new_center]:
					if k != j:
						add_link(ml, j, k)
						delete_link(cl, j, k)
				for k in ml[j]:
					if k != new_center:
						add_link(ml, new_center, k)
						delete_link(cl, new_center, k)
				for k1 in ml[j]:
					for k2 in ml[j]:
						if k1 == k2:
							continue
						if not k2 in ml[k1]:
							add_link(ml, k1, k2)
							delete_link(cl, k1, k2)
				for k in ml[i_center]:
					add_link(cl, k, new_center)
					delete_link(ml, k, new_center)
				for k in cl[new_center]:
					add_link(cl, j, k)
					delete_link(ml, j, k)
				for k in cl[j]:
					add_link(cl, k, new_center)
					delete_link(ml, k, new_center)
				for k1 in cl[new_center]:
					for k2 in ml[new_center]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)
			if not j_gt in center_dict.keys(): # no center belong to the same category with datapoint j
				new_label = max(center_dict.keys())+1
				clusters_train[j] = new_label
				for dif_center in new_centers:
					add_link(cl, dif_center, j)
					delete_link(ml, dif_center, j)
					for k in ml[dif_center]:
						add_link(cl, k, j)
						delete_link(ml, k, j)
				for k1 in ml[j]:
					clusters_train[k1] = new_label
					for k2 in cl[j]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)
				center_dict[j_gt] = j
				new_centers.append(j)
		for j in idx_after_sort[-K:]:
			j_gt = gt_array[j][0]
			if (j_gt == i_gt):
				clusters_train[j] = clusters_train[i_center]
				add_link(ml, i_center, j)
				delete_link(cl, i_center, j)
				for k in ml[i_center]:
					if k != j:
						add_link(ml, j, k)
						delete_link(cl, j, k)
				for k in ml[j]:
					if k != i_center:
						clusters_train[k] = clusters_train[i_center]
						add_link(ml, k, i_center)
						delete_link(cl, k, i_center)
				for k1 in ml[j]:
					for k2 in ml[j]:
						if k1 == k2:
							continue
						if not k2 in ml[k1]:
							add_link(ml, k1, k2)
							delete_link(cl, k1, k2)
				for k in cl[i_center]:
					add_link(cl, j, k)
					delete_link(ml, j, k)
				for k in cl[j]:
					add_link(cl, k, i_center)
					delete_link(ml, k, i_center)
				for k1 in cl[i_center]:
					for k2 in ml[i_center]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)

			if (j_gt != i_gt) and (gt_labels.count(j_gt) > 0):
				new_center = center_dict[j_gt]
				new_label = clusters_train[new_center]
				add_link(ml, new_center, j)
				delete_link(cl, new_center, j)
				add_link(cl, new_center, i_center)
				delete_link(ml, new_center, i_center)
				add_link(cl, j, i_center)
				delete_link(ml, j, i_center)
				for k in ml[new_center]:
					if k != j:
						add_link(ml, j, k)
						delete_link(cl, j, k)
				for k in ml[j]:
					if k != new_center:
						add_link(ml, new_center, k)
						delete_link(cl, new_center, k)
				for k1 in ml[j]:
					for k2 in ml[j]:
						if k1 == k2:
							continue
						if not k2 in ml[k1]:
							add_link(ml, k1, k2)
							delete_link(cl, k1, k2)
				for k in ml[i_center]:
					add_link(cl, k, new_center)
					delete_link(ml, k, new_center)
				for k in cl[new_center]:
					add_link(cl, j, k)
					delete_link(ml, j, k)
				for k in cl[j]:
					add_link(cl, k, new_center)
					delete_link(ml, k, new_center)
				for k1 in cl[new_center]:
					for k2 in ml[new_center]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)
			if not j_gt in center_dict.keys(): # no center belong to the same category with datapoint j
				new_label = max(center_dict.keys())+1
				clusters_train[j] = new_label
				for dif_center in new_centers:
					add_link(cl, dif_center, j)
					delete_link(ml, dif_center, j)
					for k in ml[dif_center]:
						add_link(cl, k, j)
						delete_link(ml, k, j)
				for k1 in ml[j]:
					clusters_train[k1] = new_label
					for k2 in cl[j]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)
				center_dict[j_gt] = j
				new_centers.append(j)
	print "cluster update finishes."
	return clusters_train, ml, cl, new_centers
			

def update_const(clusters_train, centers_id, gt_array, dataset, ml, cl):
	cluster_array = np.array(clusters_train)
	ids = list(set(clusters_train))
	for label,i_center in enumerate(centers_id):
		i_list = np.ndarray.tolist(np.where( cluster_array == label+1)[0]) # the index of datapoints belong to cluster i
		i_distance = [euclidean(dataset[j], dataset[i_center]) for j in i_list] # the distance between datapoint and center
		i_sort = sorted(range(len(i_list)), key=lambda x: i_distance[x])
		idx_after_sort = [ i_list[i_sort[k]] for k in range(len(i_sort)) ]
		if idx_after_sort[0]!=i_center:
			print ("center %d's medoid is not correct" % i)
			pdb.set_trace()

		# user-feedback
		K = 10
		i_gt = gt_array[ i_center,0 ]
		for j in idx_after_sort[1:K+1]:
			if gt_array[j]!=i_gt: # ground truth negative
				clusters_train[j] = -(label+1)
				add_link(cl, j, i_center)
				delete_link(ml, j, i_center)
				for k in ml[i_center]:
					add_link(cl, j, k)
					delete_link(ml, j, k)
				for k in ml[j]:
					clusters_train[k] = -(label+1)
					add_link(cl, i_center, k)
					delete_link(ml, i_center, k)
				for k1 in ml[j]:
					for k2 in cl[j]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)
				for k1 in ml[i_center]:
					for k2 in cl[i_center]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)
			else: # ground truth positive
				add_link(ml, i_center, j)
				delete_link(cl, i_center, j)
				for k in ml[i_center]:
					if k != j:
						add_link(ml, j, k)
						delete_link(cl, j, k)
				for k in ml[j]:
					if k != i_center:
						add_link(ml, i_center, k)
						delete_link(cl, i_center, k)
				for k in cl[i_center]:
					add_link(cl, j, k)
					delete_link(ml, j, k)
				for k in cl[j]:
					add_link(cl, i_center, k)
					delete_link(ml, i_center, k)
				for k1 in ml[j]:
					for k2 in ml[j]:
						if k1 == k2:
							continue
						if not k2 in ml[k1]:
							add_link(ml, k1, k2)
							delete_link(cl, k1, k2)
				for k1 in ml[j]:
					for k2 in cl[j]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)

		for j in idx_after_sort[-K:]:
			if gt_array[j]!=i_gt: # ground truth negative
				clusters_train[j] = -(label+1)
				add_link(cl, j, i_center)
				delete_link(ml, j, i_center)
				for k in ml[i_center]:
					add_link(cl, j, k)
					delete_link(ml, j, k)
				for k in ml[j]:
					clusters_train[k] = -(label+1)
					add_link(cl, i_center, k)
					delete_link(ml, i_center, k)
				for k1 in ml[j]:
					for k2 in cl[j]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)
				for k1 in ml[i_center]:
					for k2 in cl[i_center]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)
			else: # ground truth positive
				add_link(ml, i_center, j)
				delete_link(cl, i_center, j)
				for k in ml[i_center]:
					if k != j:
						add_link(ml, j, k)
						delete_link(cl, j, k)
				for k in ml[j]:
					if k != i_center:
						add_link(ml, i_center, k)
						delete_link(cl, i_center, k)
				for k in cl[i_center]:
					add_link(cl, j, k)
					delete_link(ml, j, k)
				for k in cl[j]:
					add_link(cl, i_center, k)
					delete_link(ml, i_center, k)
				for k1 in ml[j]:
					for k2 in ml[j]:
						if k1 == k2:
							continue
						if not k2 in ml[k1]:
							add_link(ml, k1, k2)
							delete_link(cl, k1, k2)
				for k1 in ml[j]:
					for k2 in cl[j]:
						add_link(cl, k1, k2)
						delete_link(ml, k1, k2)
	print "cluster update finishes."
	return clusters_train, ml, cl, centers_id

def add_link(l_graph, idx, idy):
	if not idy in l_graph[idx]:
		l_graph[idx].add(idy)
	if not idx in l_graph[idy]:
		l_graph[idy].add(idx)

def delete_link(l_graph, idx, idy):
	if idy in l_graph[idx]:
		l_graph[idx].remove(idy)
	if idx in l_graph[idy]:
		l_graph[idy].remove(idx)

def check_initialization(centers_id, ml):
	if not any(x != set([]) for x in ml.itervalues()):
		return True
	for i_id in centers_id:
		for j_id in centers_id:
			if j_id in ml[i_id]:
				return False
	return True

def closet_clusters(centers, datapoint):
	distances = [euclidean( np.array([center]), np.array([datapoint])) for center in centers]
	return sorted(range(len(distances)), key=lambda x: distances[x]) # i-th element is the index of i-th smallest entry

def initialize_centers(dataset, k):
	# ids = range(len(dataset))
	# random.shuffle(ids)
	# return [dataset[id] for id in ids[:k]], [id for id in ids[:k]]
	centers = []
	centers_id = [0, 145, 259, 404, 536, 644, 799, 949, 1083, 1214]
	for id in centers_id:
		centers.append(dataset[id])
	return centers, centers_id

def violate_constraints(data_index, cluster_index, label, clusters, ml, cl):
	if data_index in cl[cluster_index]:
		return True
	if data_index in ml[cluster_index]:
		return False
	for i in ml[data_index]:
		if clusters[i] != label+1 and clusters[i] != 0:
			return True
	for i in cl[data_index]:
		if clusters[i] == label+1:
			return True
	return False

def compute_centers(clusters, dataset):
	# type: dataset-2d list
	# type: clusters-1d list
	ids = list(set(clusters)) # unique cluster ids
	k = len(ids)
	dim = len(dataset[0])

	c_to_id = dict()
	for j,c in enumerate(ids):
		c_to_id[c] = j
	for j,c in enumerate(clusters):
		clusters[j] = c_to_id[c]
	ids = list(set(clusters))
	centers = [ [0]*dim for i in range(k) ]
	centers_id = [0] * k
	sumds = [0]*k
	for id in ids:
		data_ids = np.ndarray.tolist( np.where( np.array(clusters) == id )[0] )
		datas = np.array( dataset )[data_ids,:]
		center = np.mean(datas, axis=0, keepdims=True)
		distance_list = sorted(range(len(data_ids)), key=lambda x: euclidean(center, datas[[x],:]))
		centers_id[id] = data_ids[distance_list[0]]
		centers[id] = dataset[centers_id[id]]
		sumds[id] = sum([euclidean(dataset[centers_id[id]], i) for i in datas])
	for i in range(len(clusters)):
		clusters[i] += 1
	return clusters, centers, centers_id, sum(sumds)

def transitive_closure(ml, cl, n): # n: the number of data points
	ml_graph = dict()
	cl_graph = dict()
	for i in range(n):
		ml_graph[i] = set()
		cl_graph[i] = set()

	def add_both(d, i ,j):
		d[i].add(j)
		d[j].add(i)
	for (i, j) in ml:
		add_both(ml_graph, i, j)

	def dfs(i, graph, visited, component):
		visited[i] = True
		for j in graph[i]:
			if not visited[j]:
				dfs(j, graph, visited, component)
		component.append(i)

	visited = [False] * n
	for i in range(n):
		if not visited[i]:
			component = []
			dfs(i, ml_graph, visited, component) # component consists of all unvisited nodes MUST linked with i
			for x1 in component:
				for x2 in component:
					if x1 != x2:
						ml_graph[x1].add(x2) # transivily link all nodes logically MUST linked with node i

	for (i,j) in cl:
		add_both(cl_graph, i, j)
		for y in ml_graph[j]:
			add_both(cl_graph, i, y)
		for x in ml_graph[i]:
			add_both(cl_graph, j, x)
			for y in ml_graph[j]:
				add_both(cl_graph, x, y)

	for i in ml_graph:
		for j in ml_graph[i]:
			if j!=i and (j in cl_graph[i]):
				raise Exception("inconsistent constraints between %d and %d" % (i,j))

	return ml_graph, cl_graph

def read_data(datafile):
	data = []
	inputdict = sio.loadmat(os.path.join(data_root, datafile))
	data = np.ndarray.tolist(inputdict['data'])
	return data

def read_constrains(consfile):
	ml, cl = [], []
	constdict = sio.loadmat(os.path.join(data_root, consfile)) # load constraint matrix, 1:MUST LINK -1:CANNOT LINK
	cons = constdict['const']
	for i in range(cons.shape[0]):
		for j in range(cons.shape[1]):
			constraint = (int(i), int(j))
			if cons[i,j] == 1:
				ml.append(constraint)
			if cons[i,j] == -1:
				cl.append(constraint)

	return ml, cl

def run(datafile, consfile, k, outfile, repeat, savecons, savevects):

	data = read_data(datafile) # 2D list
	if not consfile:
		ml, cl = [], []
	else:
		ml, cl = read_constrains(consfile) # two 2D lists
	labels, train_labels=cop_kmeans(data, k, repeat, savecons, savevects, ml, cl)
	if not labels:
		pdb.set_trace()
	data_save = dict()
	data_save['pdlabels'] = labels
	data_save['train_labels'] = train_labels
	sio.savemat(os.path.join(data_root, 'cop-kmeans-fix-veri-8ft.mat'), data_save)

if __name__ == '__main__':
	run(opts.datafile, opts.consfile, opts.ncluster, opts.outfile, opts.repeat, opts.savecons, opts.savevects)
