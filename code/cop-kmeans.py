import random
import numpy as np 
import math
import pdb
import scipy.spatial.distance as ssd
import scipy.io as sio
import optparse
import os

optparser = optparse.OptionParser()
optparser.add_option("--df", "--datafile", dest="datafile", default='UCF-101-gtlabel-10.mat', help="Input data file name")
optparser.add_option("--cf", "--consfile", dest="consfile", default=None, help="Input const file name")
optparser.add_option("-k", "--ncluster",   dest="ncluster", default=10, help="Input the number of clusters")
optparser.add_option("--of", "--outfile",  dest="outfile",  default=None, help="Input output file name")
optparser.add_option("--rp", "--repeat",   dest="repeat", default=5, help="Input repeat number of clustering")

(opts, _) = optparser.parse_args()

data_root = '/cs/vml2/xla193/cluster_video/output/UCF-101'

def cop_kmeans(dataset, k, ml=[], cl=[], repeat):
	ml, cl = transitive_closure(ml, cl, len(dataset)) # return two dictionaries

	centers = initialize_centers(dataset, k)
	clusters= [-1] * len(dataset)
	count = 1
	converged = False
	while (not converged) or (count != repeat):
		print ("Doing the %d-th clustering" % count)
		clusters_ = [-1] * len(dataset)
		for i, d in enumerate(dataset): # d:datapoint i:index in dataset
			indices = closet_clusters(centers, d)
			counter = 0
			found_cluster = False
			while (not found_cluster) and (counter < len(indices)):
				index = indices[counter]
				if not violate_constraints(i, index, clusters_, ml, cl):
					found_cluster = True 
					clusters_[i]  = index
				counter += 1

			if not found_cluster:
				return None
		clusters_, centers = compute_centers(clusters_, dataset)

		converged = True
		i = 0
		while converged and (i < len(dataset)):
			if clusters[i] != clusters_[i]:
				converged = False
			i += 1
		clusters = clusters_
		if converged==True:
			sumd = calculate_within_distance(dataset, clusters, centers)
			s_centers = centers
			count += 1

	pdb.set_trace()
	return clusters, centers

def calculate_within_distance(dataset, clusters, centers):
	

def closet_clusters(centers, datapoint):
	# pdb.set_trace()
	distances = [ssd.cdist(np.array(center).reshape(1,len(center)), 
				 np.array(datapoint).reshape(1,len(datapoint)), 'euclidean')[0][0] for center in centers]
	return sorted(range(len(distances)), key=lambda x: distances[x]) # i-th element is the index of i-th smallest entry

def initialize_centers(dataset, k):
	ids = range(len(dataset))
	random.shuffle(ids)
	return [dataset[id] for id in ids[:k]]

def violate_constraints(data_index, cluster_index, clusters, ml, cl):
	for i in ml[data_index]:
		if clusters[i] != -1 and clusters[i] != cluster_index:
			return True
	for i in cl[data_index]:
		if clusters[i] == cluster_index:
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

	for id in ids:
		data_ids = np.ndarray.tolist( np.where( np.array(clusters) == id )[0] )
		datas = np.array( dataset )[data_ids,:]
		center = np.mean(datas, axis=0, keepdims=True)
		distance_list = sorted(range(len(data_ids)), key=lambda x: ssd.cdist(center, datas[[x],:], 'euclidean')[0][0])
		centers[id] = np.ndarray.tolist(datas[distance_list[0]])

	return clusters, centers

	# ids = list(set(clusters))
	# c_to_id = dict()
	# for j, c in enumerate(ids):
	# 	c_to_id[c] = j
	# for j, c in enumerate(clusters):
	# 	clusters[j] = c_to_id[c]

	# k = len(ids)
	# dim = len(dataset[0])
	# centers = [[0.0] * dim for i in range(k)]
	# counts = [0] * k
	# for j,c in enumerate(clusters):
	# 	for i in range(dim):
	# 		centers[c][i] += dataset[j][i]
	# 	counts[c] += 1
	# for j in range(k):
	# 	for i in range(dim):
	# 		centers[j][i] = centers[j][i]/float(counts[j])

	# return clusters, centers

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
	constdict = sio.loadmat(consfile) # load constraint matrix, 1:MUST LINK -1:CANNOT LINK
	cons = constdict['data']
	for i in range(cons.shape[0]):
		for j in range(cons.shape[1]):
			constraint = (int(i), int(j))
			if cons[i,j] == 1:
				ml.append(constraint)
			if cons[i,j] == -1:
				cl.append(constraint)

	return ml, cl

def run(datafile, consfile, k, outfile):

	data = read_data(datafile) # 2D list
	# ml, cl = read_constrains(consfile) # two 2D lists
	ml, cl = [], []
	cop_kmeans(data, k, ml, cl, repeat)

if __name__ == '__main__':
	run(opts.datafile, opts.consfile, opts.ncluster, opts.outfile, opts.repeat)