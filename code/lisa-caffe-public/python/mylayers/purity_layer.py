import caffe
import numpy as np
import random
import time


class PurityLayer(caffe.Layer):

	def setup(self, bottom, top):
		if len(bottom)!=2:
			raise Exception("Need two inputs to compute triplets")
	def reshape(self, bottom, top):
		if len(bottom[0]) != len(bottom[1]):
			raise Exception("Need two inputs have same dimensions")
		top[0].reshape(1, dtype = float32)

	def forward(self, bottom, top):
		gtlabels = bottom[0].data
        pdlabels = bottom[1].data
		unigt = np.unique( gtlabels )
        nugt  = unigt.size

        unipd = np.unique( pdlabels )
        #nupd(unipd == 0) =[]
        nupd  = unipd.size

        count_int = np.zeros([nugt, nupd])
        for i in range(nugt-1):
            for j in range(nupd-1):
                count_int[i, j] = np.sum((gtlabels == unigt[i]) & (pdlabels == unipd[j]))

        count_pd = np.sum( count_int, axis = 0)
        nmaxgt = np.max(count_int, axis = 0).astype( np.float32 );
        pa = np.sum(nmaxgt) / np.sum(count_pd)

        top[0].data[...] = pa
		
	def backward(self, top, propagate_down, bottom):
		pass
