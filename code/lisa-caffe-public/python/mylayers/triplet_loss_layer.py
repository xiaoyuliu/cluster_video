#In the name of God
import caffe
import numpy as np
import math
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing
import time
import scipy.spatial.distance as ssd


class TripletLossLayer(caffe.Layer):
    """
    Compute the hing loss of the dot product of the 2 input layers
    """
    # loss = sum( max( |a-p| - |a-n|, 0 ) )
    global no_residual_list

    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute triplet loss.")
    
    def reshape(self, bottom, top):

        top[0].reshape(1)

    def forward(self, bottom, top):
        print time.ctime(), " starting forward in loss layer" 
        self.margin = 1
        anchor = bottom[0].data
        positive = bottom[1].data
        negative = bottom[2].data

        loss = float(0)
        self.no_residual_list = []
        for i in range(anchor.shape[0]):
            # start_time = time.time()
            # from IPython.core.debugger import Tracer
            # Tracer()()
            ap = ssd.cdist( np.array([anchor[i]]), np.array([positive[i]]), 'sqeuclidean' )[0][0]
            an = ssd.cdist( np.array([anchor[i]]), np.array([negative[i]]), 'sqeuclidean' )[0][0]
            
            dist = (self.margin + ap - an) # safety margin
            _loss = max(dist, 0.0) # hinge
            if i == 0:
                # print ('loss:'+str(_loss)+' ap:'+str(ap)+' an:'+str(an)+'------------------------------')
                pass
            if _loss == 0:
                self.no_residual_list.append(i)
            loss += _loss
        loss = loss / ( 2*anchor.shape[0] )
        top[0].data[...] = loss
        # print("--- %s seconds for each forward in loss layer ---" % (time.time() - start_time))
    def backward(self, top, propagate_down, bottom):
        print time.ctime(), " starting backward in loss layer" 
        # start_time = time.time()
        anchor = bottom[0].data
        positive = bottom[1].data
        negative = bottom[2].data

        self.a = 1
        count = 0
        if propagate_down[0]:
            for i in range(anchor.shape[0]):
                # from IPython.core.debugger import Tracer
                # Tracer()()
                if not i in self.no_residual_list:
                    # from IPython.core.debugger import Tracer
                    # Tracer()()
                    x_a = anchor[i]
                    x_p = positive[i]
                    x_n = negative[i]
                    # from IPython.core.debugger import Tracer
                    # Tracer()()
                    bottom[0].diff[i][...] = self.a*2*(x_n-x_p)
                    bottom[1].diff[i][...] = self.a*2*(x_p-x_a)
                    bottom[2].diff[i][...] = self.a*2*(x_a-x_n)
                    count += 1
                else:
                    bottom[0].diff[i][...] = np.zeros([1,anchor[[0],:].shape[1]])
                    bottom[1].diff[i][...] = np.zeros([1,anchor[[0],:].shape[1]])
                    bottom[2].diff[i][...] = np.zeros([1,anchor[[0],:].shape[1]])

        for i in range(len(bottom)):
            bottom[i].diff[...] /= bottom[i].num