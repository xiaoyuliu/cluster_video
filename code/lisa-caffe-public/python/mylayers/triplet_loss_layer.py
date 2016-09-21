#In the name of God
import caffe
import numpy as np
import math
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing


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
        self.margin = 1
        anchor = bottom[0].data
        positive = bottom[1].data
        negative = bottom[2].data
        # from IPython.core.debugger import Tracer
        # Tracer()()

        loss = float(0)
        self.no_residual_list = []
        for i in range(anchor.shape[0]):
            a = anchor[i]
            p = positive[i]
            n = negative[i]
            ap2 = [(aa - bb)**2 for aa, bb in zip(a, p)]
            an2 = [(aa - bb)**2 for aa, bb in zip(a, n)]
            ap = math.sqrt(sum(ap2))
            an = math.sqrt(sum(an2))
            
            dist = (self.margin + ap - an)
            _loss = max(dist, 0.0)
            if i == 0:
                print ('loss:'+str(_loss)+' ap:'+str(ap)+' an:'+str(an))
            if _loss == 0:
                self.no_residual_list.append(i)
            loss += _loss

        loss = loss/(2*anchor.shape[0])
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
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
                    apdist = [(a - b)**2 for a, b in zip(x_a, x_p)]
                    andist = [(a - b)**2 for a, b in zip(x_a, x_n)]
                    dist_ap = math.sqrt(sum(apdist))
                    dist_an = math.sqrt(sum(andist))
                    if dist_ap == 0:
                        dist_ap = 0.00000000001
                    if dist_an == 0:
                        dist_an = 0.00000000001
                    # from IPython.core.debugger import Tracer
                    # Tracer()()
                    # print 'dist_ap:',dist_ap, ' dist_an:',dist_an
                    bottom[0].diff[i][...] = self.a*((x_a-x_p)/dist_ap - (x_a-x_n)/dist_an)
                    bottom[1].diff[i][...] = self.a*(x_p-x_a)/dist_ap
                    bottom[2].diff[i][...] = self.a*(x_a-x_n)/dist_an
                    count += 1
                else:
                    bottom[0].diff[i][...] = np.zeros([1,anchor[[0],:].shape[1]])
                    bottom[1].diff[i][...] = np.zeros([1,anchor[[0],:].shape[1]])
                    bottom[2].diff[i][...] = np.zeros([1,anchor[[0],:].shape[1]])
