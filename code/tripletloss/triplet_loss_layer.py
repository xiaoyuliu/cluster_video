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
    # loss = sum(|a-p| - |a-n|)
    global no_residual_list, margin

    def distance(vector1, vector2):
        '''calculate the euclidean distance, no numpy
        input: numpy.arrays or lists
        return: euclidean distance
        '''
        dist = [(a - b)**2 for a, b in zip(vector1, vector2)]
        dist = math.sqrt(sum(dist))
        return dist

    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute triplet loss.")
        assert bottom[0].shape == bottom[1].shape
        assert bottom[0].shape == bottom[2].shape

        layer_params = yaml.load(self.param_str_)
        self.margin = layer_params['margin']

        self.a = 1
    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        # anchor = []
        # positive = []
        # negative = []

        # for i in range(len(bottom[0])):
        #     anchor.append( bottom[0].data[i] )
        #     positive.append( bottom[1].data[i] )
        #     negative.append( bottom[2].data[i] )
        anchor = bottom[0].data
        positive = bottom[1].data
        negative = bottom[2].data

        loss = float(0)
        self.no_residual_list = []
        for i in range(len(bottom[0])):
            a = np.array(anchor[i])
            p = np.array(positive[i])
            n = np.array(negative[i])
            ap = distance(a,p)
            an = distance(a,n)
            dist = (self.margin + ap - an)
            _loss = max(dist, 0.0)
            if i == 0:
                print ('loss:'+str(_loss)+' ap:'+str(ap)+' an:'+str(an))
            if _loss == 0:
                self.no_residual_list.append(i)
            loss += _loss

        loss = loss/(2*len(bottom[0]))
        top[0].data[...] = loss

    def backward(self, top, propagate_down, bottom):
        count = 0
        if propagate_down[0]:
            for i in range(len(bottom[0])):
                if not i in self.no_residual_list:
                    x_a = float(anchor[i])
                    x_p = float(positive[i])
                    x_n = float(negative[i])

                    bottom[0].diff[i] = self.a*((x_a-x_p)/distance(x_a,x_p) - (x_a-x_n)/distance(x_a,x_n))
                    bottom[1].diff[i] = self.a*(x_p-x_a)/distance(x_a,x_p)
                    bottom[2].diff[i] = self.a*(x_a-x_n)/distance(x_a,x_n)

                    count += 1
                else:
                    bottom[0].diff[i] = np.zeros(1,anchor[0].shape[1])
                    bottom[1].diff[i] = np.zeros(1,anchor[0].shape[1])
                    bottom[2].diff[i] = np.zeros(1,anchor[0].shape[1])
