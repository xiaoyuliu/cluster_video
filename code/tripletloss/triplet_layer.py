import caffe
import numpy as np
import random


class TripletLayer(caffe.Layer):
	def setup(self, bottom, top):
		if len(bottom)!=2:
			raise Exception("Need two inputs to compute triplets")

	def reshape(self, bottom, top):
		feats = bottom[0].data
		labels= bottom[1].data
		n_neg = int(bottom[2].data)

		num_s = feats.shape[0]
		unilables, counts = np.unique( labels, return_counts=True)
		nclusters = unilables.size

		if nclusters == 1:
			return
		if nclusters <= n_neg:
			n_neg = nclusters - 1

		n_trip = int(0)
		for i in range(nclusters):
			if counts[i] > 1:
				n_trip += counts[i] * (counts[i]-1) * n_neg / 2
				
		top[0].reshape(n_trip, feats.shape[1])
		top[1].reshape(n_trip, feats.shape[1])
		top[2].reshape(n_trip, feats.shape[1])

	def forward(self, bottom, top):
		# feats = bottom[0].data
		# labels= bottom[1].data
		# n_neg = int(bottom[2].data)

		# num_s = feats.shape[0]
		# unilables, counts = np.unique( labels, return_counts=True)
		# nclusters = unilables.size

		# if nclusters == 1:
		# 	return
		# if nclusters <= n_neg:
		# 	n_neg = nclusters - 1

		# n_trip = int(0)
		# for i in range(nclusters):
		# 	if counts[i] > 1:
		# 		n_trip += counts[i] * (counts[i]-1) * n_neg / 2

		if n_trip==0:
			return

		A = np.zeros([n_trip, feats.shape[1]])
		B = np.zeros([n_trip, feats.shape[1]])
		C = np.zeros([n_trip, feats.shape[1]])
		A_ind = np.zeros([n_trip,1])
		C_ind = np.zeros([n_trip,1])
		B_ind = np.zeros([n_trip,1])
		id_triplet = int(1)

		for i in range(nclusters):
			if counts[i]>1:
				for m in range(counts[i]):
					index_list = np.where( labels == unilables[i] )[0]
					for n in range(m+1, counts[i]):
						if m!=n:
							is_choosed = np.zeros([num_s,1])
							while 1:
								rdn = random.uniform(0,1)
								id_s = np.ceil(rdn*num_s)
								if is_choosed[id_s]==0 and labels[id_s]!=labels[index_list[m]]:
									A_ind[id_triplet] = index_list[m]
									B_ind[id_triplet] = index_list[n]
									C_ind[id_triplet] = id_s
									is_choosed[id_s]  = 1
									id_triplet += 1
								if (id_triplet) % n_neg == 1:
									break

		A = feats[A_ind,:] #anchors
		B = feats[B_ind,:] #positives
		C = feats[C_ind,:] #negatives

		top[0].data[...] = A
		top[1].data[...] = B
		top[2].data[...] = C

	def backward(self, top, propagate_down, bottom):
		pass
