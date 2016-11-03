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
optparser.add_option("--rp", "--repeat",   dest ="repeat", default=10, help="Input repeat number of clustering")
optparser.add_option("--sc", "--savecons", dest ="savecons", default='const1.mat', help="Input name of save const file")
optparser.add_option("--sv", "--savevects",dest ="savevects", default=None, help="Input name of save labelvector file")
(opts, _) = optparser.parse_args()

data_root = '/local-scratch/xla193/cluster_video_/output/UCF-101'

def cop_kmeans(dataset, k, repeat, savecons, savevects, ml=[], cl=[]):
	ml, cl = transitive_closure(ml, cl, len(dataset)) # return two dictionaries
	# pdb.set_trace()
	finding_count   = 0
	converged_count = 0
	update_count    = 0
	init_count      = 0
	break_num       = 3e4
	init_num		= 2 * repeat

	sumd = float("inf")
	centers, centers_id = initialize_centers(dataset, k)
	clusters = [0] * len(dataset)
	while(converged_count < repeat):
		converged = False
		centers, centers_id = initialize_centers(dataset, k)
		init_count          += 1
		print "initial times: ", init_count, " total allowed: ", init_num, "\r"
		if not check_initialization(centers_id, ml):
			continue
		print "initialization check passed"
		clusters= [0] * len(dataset)
		vialate_count   = 0
		while (not converged):
			# print ("-%d-th finding clustering - %f " % (finding_count, sumd))
			clusters_ = [0] * len(dataset)
			for i, d in enumerate(dataset): # d:datapoint i:index in dataset
				indices = closet_clusters(centers, d)
				counter = 0
				found_cluster = False
				while (not found_cluster) and (counter < len(indices)):
					index = indices[counter]
					if not violate_constraints(i, centers_id[index], index, clusters_, ml, cl):
						found_cluster = True 
						clusters_[i]  = index+1
					else:
						vialate_count += 1
						print "violate, try again, number: ", vialate_count, " total allowed: ", break_num, "\r",
						if vialate_count == break_num:
							break
					counter += 1
				if vialate_count >= break_num:
					break
				if not found_cluster and (converged_count == 0):
					return None
			if vialate_count == break_num:
				if init_count == init_num:
					repeat = converged_count
				break
			clusters_, centers, centers_id, sumd_ = compute_centers(clusters_, dataset)

			finding_count += 1

			converged = True
			i = 0
			while converged and (i < len(dataset)):
				if clusters[i] != clusters_[i]:
					converged = False
				i += 1
			clusters = clusters_
			if converged == True:
				converged_count += 1
				print ("--%d-th clustering - smallest: %f - now: %f " % (converged_count, sumd, sumd_))
				print time.ctime()
				if sumd_ < sumd:
					s_clusters= map(lambda x: x+1, clusters)
					s_centers = centers_id
					sumd = sumd_
					update_count += 1
	gt_dict = sio.loadmat(os.path.join(data_root, 'UCF-101-gtlabel-10.mat'))
	gt_array = gt_dict['label']
	clusters_train = []
	for i in range(len(s_clusters)):
		clusters_train.append(s_clusters[i])
	clusters_train, ml, cl = update_const(clusters_train, s_centers, gt_array, dataset, ml, cl)
	save_const_vect(ml, cl, savecons, savevects, s_clusters, clusters_train)
	pdb.set_trace()
	return s_clusters, clusters_train

def save_const_vect(ml, cl, savecons, savevects, clusters, clusters_train):
	clusters_vect = []
	for i in range(len(clusters)):
		clusters_vect.append(set())
	uf_ids = set()

	num_data = len(ml)
	const = np.zeros([num_data, num_data])

	for i in range(num_data):
		if i in ml[i]:
			ml[i].remove(i)
		if i in cl[i]:
			cl[i].remove(i)
	def add_both(oneset, a, b):
		oneset.add(a)
		oneset.add(b)

	for i in range(num_data):
		label = clusters_train[i]
		if label<0 or (not cl[i]):
			continue

		clusters_vect[i].add(label)
		for j in range(num_data):
			if (j in ml[i]):
				if -label in clusters_vect[j]:
					pdb.set_trace()
				if (clusters_train[j] != label) and (clusters_train[j] > 0):
					raise Exception("ml data points have different assignments with center")
				clusters_vect[j].add(label)
				add_both(uf_ids, i, j)
				const[i][j] = 1
				const[j][i] = 1
				
			if (j in cl[i]):
				if label in clusters_vect[j]:
					pdb.set_trace()
				clusters_vect[j].add(-label)
				add_both(uf_ids, i, j)
				const[i][j] = -1
				const[j][i] = -1
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

def update_const(clusters_train, centers_id, gt_array, dataset, ml, cl):
	cluster_array = np.array(clusters_train)
	ids = list(set(clusters_train))
	for i,label in enumerate(ids):
		i_list = np.ndarray.tolist(np.where( cluster_array == label)[0]) # the index of datapoints belong to cluster i
		i_distance = [euclidean(dataset[j], dataset[centers_id[i]]) for j in i_list] # the distance between datapoint and center
		i_sort = sorted(range(len(i_list)), key=lambda x: i_distance[x])
		idx_after_sort = [ i_list[i_sort[k]] for k in range(len(i_sort)) ]
		if idx_after_sort[0]!=centers_id[i]:
			print ("center %d's medoid is not correct" % i)
			pdb.set_trace()

		# user-feedback
		K = 10
		i_gt = gt_array[ centers_id[i],0 ]
		for j in idx_after_sort[1:K+1]:
			if gt_array[j]!=i_gt: # ground truth negative
				clusters_train[j] = -label
				delete_link(ml, centers_id[i], j)
				add_link(cl, centers_id[i], j)
				for k in ml[j]:
					clusters_train[k] = -label
					delete_link(ml, centers_id[i], k)
					add_link(cl, centers_id[i], k)
			else: # ground truth positive
				add_link(ml, centers_id[i], j)
				delete_link(cl, centers_id[i], j)
				for k in ml[j]:
					clusters_train[k] = label
					add_link(ml, centers_id[i], k)
					delete_link(cl, centers_id[i], k)
				for k in cl[j]:
					clusters_train[k] = -label
					add_link(cl, centers_id[i], k)
					delete_link(ml, centers_id[i], k)
		for j in idx_after_sort[-K:]:
			if gt_array[j]!=i_gt:
				clusters_train[j] = -label
				delete_link(ml, centers_id[i], j)
				add_link(cl, centers_id[i], j)
				for k in ml[j]:
					clusters_train[k] = -label
					delete_link(ml, centers_id[i], k)
					add_link(cl, centers_id[i], k)
			else:
				add_link(ml, centers_id[i], j)
				delete_link(cl, centers_id[i], j)
				for k in ml[j]:
					clusters_train[k] = label
					add_link(ml, centers_id[i], k)
					delete_link(cl, centers_id[i], k)
	print "cluster update finishes."
	return clusters_train, ml, cl

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
	ids = range(len(dataset))
	random.shuffle(ids)
	return [dataset[id] for id in ids[:k]], [id for id in ids[:k]]

def violate_constraints(data_index, cluster_index, label, clusters, ml, cl):
	if data_index in ml[cluster_index] and clusters[data_index] == 0:
		return False
	if data_index in cl[cluster_index] or clusters[data_index] != 0:
		return True
	for i in ml[data_index]:
		if clusters[i] != 0 and clusters[i] != label+1:
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
	pdb.set_trace()
	labels, train_labels=cop_kmeans(data, k, repeat, savecons, savevects, ml, cl)
	if not labels:
		pdb.set_trace()
	data_save = dict()
	data_save['pdlabels'] = labels
	data_save['train_labels'] = train_labels
	sio.savemat(os.path.join(data_root, 'cop-kmeans-result2.mat'), data_save)

if __name__ == '__main__':
	run(opts.datafile, opts.consfile, opts.ncluster, opts.outfile, opts.repeat, opts.savecons, opts.savevects)
