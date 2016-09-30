import caffe
import numpy as np
import random
import time


class TripletLSTMLayer(caffe.Layer):

	def setup(self, bottom, top):
		if len(bottom)!=2:
			raise Exception("Need two inputs to compute triplets")
	def reshape(self, bottom, top):
		# print time.ctime(), " starting tripletLSTM layer"
		# start_time = time.time()
		frame_feat = bottom[0].data # 16x24x256
		frame_label= bottom[1].data # 16x24

		num_video = frame_feat.shape[1]
		len_frame = frame_feat.shape[0]
		self.video_feat = np.zeros([num_video, frame_feat.shape[-1]])
		self.video_label= np.zeros(num_video)

		for i in range(num_video):
			self.video_feat[i,:] = np.mean( frame_feat[:,i,:], axis = 0, keepdims=True  )
			self.video_label[i]= frame_label[0,i]
		feats = self.video_feat
		labels= self.video_label
		# from IPython.core.debugger import Tracer
		# Tracer()()
		negative_sample_number = int(5)
		n_neg = negative_sample_number

		num_s = num_video
		unilabels, counts = np.unique( labels, return_counts=True)
		nclusters = unilabels.size

		if nclusters == 1:
			return
		if nclusters <= n_neg:
			n_neg = nclusters - 1

		n_trip = int(0)
		for i in range(nclusters):
			if counts[i] > 1:
				n_trip += counts[i] * (counts[i]-1) * n_neg / 2
		# print "n_trip:", n_trip
		if n_trip==0:
			return
		top[0].reshape(n_trip, feats.shape[1])
		top[1].reshape(n_trip, feats.shape[1])
		top[2].reshape(n_trip, feats.shape[1])
		# print("--- %s seconds for reshape in LSTM layer ---" % (time.time() - start_time))
	def forward(self, bottom, top):
		# print time.ctime(), " starting forward in tripletLSTM layer"
		# start_time = time.time()
		# from IPython.core.debugger import Tracer
		# Tracer()()
		feats = self.video_feat
		labels= self.video_label
		# param =json.loads( self.param_str )
		negative_sample_number = int(5)
		n_neg = negative_sample_number

		num_s = feats.shape[0]
		unilabels, counts = np.unique( labels, return_counts=True)
		nclusters = unilabels.size
		if nclusters == 1:
			return
		if nclusters <= n_neg:
			n_neg = nclusters - 1

		n_trip = int(0)
		for i in range(nclusters):
			if counts[i] > 1:
				n_trip += counts[i] * (counts[i]-1) * n_neg / 2
		if n_trip==0:
			return
		# print '---------number of triplets: ', n_trip

		A = np.zeros([n_trip, feats.shape[1]])
		B = np.zeros([n_trip, feats.shape[1]])
		C = np.zeros([n_trip, feats.shape[1]])
		A_ind = np.zeros([n_trip,1])
		C_ind = np.zeros([n_trip,1])
		B_ind = np.zeros([n_trip,1])
		id_triplet = int(0)

		for i in range(nclusters):
			if counts[i]>1:
				for m in range(counts[i]):
					index_list = np.where( labels == unilabels[i] )[0]
					for n in range(m+1, counts[i]):
						# from IPython.core.debugger import Tracer
						# Tracer()()
						if m!=n:
							is_choosed = np.zeros([num_s,1])
							while 1:
								rdn = random.uniform(0,1)
								id_s = np.ceil(rdn*(num_s-1))
								# from IPython.core.debugger import Tracer
								# Tracer()()
								if is_choosed[id_s]==0 and labels[id_s]!=labels[index_list[m]]:
									A_ind[id_triplet] = int(index_list[m])
									B_ind[id_triplet] = int(index_list[n])
									C_ind[id_triplet] = int(id_s)
									is_choosed[id_s]  = 1
									id_triplet += 1
								if (id_triplet+1) % n_neg == 1:
									break
		A_ind_list = []
		B_ind_list = []
		C_ind_list = []
		# from IPython.core.debugger import Tracer
		# Tracer()()
		for i in range(A_ind.shape[0]):
			A_ind_list.append(A_ind[i][0])
			B_ind_list.append(B_ind[i][0])
			C_ind_list.append(C_ind[i][0])

		A = feats[A_ind_list,:] #anchors
		B = feats[B_ind_list,:] #positives
		C = feats[C_ind_list,:] #negatives
		# from IPython.core.debugger import Tracer
		# Tracer()()
		top[0].data[...] = A
		top[1].data[...] = B
		top[2].data[...] = C
		# print time.ctime(), " ending forward in tripletLSTM layer"

		# print("--- %s seconds for forward in LSTM layer ---" % (time.time() - start_time))
		
	def backward(self, top, propagate_down, bottom):
		pass
