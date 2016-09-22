import caffe
import numpy as np
import random
import time
import pdb


class PurityLayer(caffe.Layer):

	def setup(self, bottom, top):
		if len(bottom)!=2:
			raise Exception("Need two inputs to compute triplets")
	def reshape(self, bottom, top):
		if bottom[0].data.shape[0] != bottom[1].data.shape[0]:
			raise Exception("Need two inputs have same number of entries")
		top[0].reshape(1)

	def forward(self, bottom, top):
		probs = bottom[0].data
		# pdb.set_trace()
		pdlabels = probs.argmax( axis=1 )+1
		gtlabels = bottom[1].data
		unigt    = np.unique( gtlabels )
		n_unigt  = unigt.size

		unipd    = np.unique( pdlabels )
		n_unipd  = unipd.size

		count_int= np.zeros([n_unigt, n_unipd])
		for i in range(n_unigt-1):
			for j in range(n_unipd-1):
				count_int[i,j] = np.sum( (gtlabels == unigt[i]) & (pdlabels == unipd[j]) )
		count_pd = np.sum( count_int, axis=0 )
		n_maxgt  = np.max( count_int, axis=0 ).astype( np.float32 )
		pa       = np.sum( n_maxgt ) / np.sum( count_pd )

		top[0].data[...] = pa
		
	def backward(self, top, propagate_down, bottom):
		pass
