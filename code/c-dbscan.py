import numpy as np 
import os
import pdb
import scipy.io as sio
from sklearn.neighbors import KDTree

def c_dbscan(data, eps, minpts):
	N = data.shape[0]
	labels = np.zeros([N, 1]) # 0: noise point, 1: border point, 2: core point
	clusters = []
	tree = KDTree(data, minpts)
	pdb.set_trace()

	cluster_index = 0

	for dataID in range(N):
		if labels[dataID] != 0:
			continue
		stack = [dataID]
		cluster = []
		while stack:
			point = stack.pop()
			indices = tree.query_radius(data[point:point+1], eps, return_distance=False, \
				count_only=False)[0]
			if len(indices) >= minpts:
				labels[point] = 2
				cluster.append(point)
				neighbor_points = [index for index in indices if not labels[index]]
				stack.extend(neighbor_points)
				labels[neighbor_points] = 1
			elif cluster:
				cluster.append(point)

		if cluster:
			clusters.append(cluster)

	return clusters, labels

if __name__ == '__main__':
	data_dir = '/cs/vml2/xla193/cluster_video/output/UCF-101'
	file     = 'output-UCF-101-10-0ft.mat'
	mat      = sio.loadmat(os.path.join(data_dir, file))
	data     = mat['data']
	eps      = 35
	minpts   = 10

	cluster  = []
	label    = []

	cluster, label = c_dbscan(data, eps, minpts)