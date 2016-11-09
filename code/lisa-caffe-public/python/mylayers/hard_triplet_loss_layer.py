import caffe, random, pdb
import numpy as np 
import scipy.spatial.distance as ssd
from scipy.spatial.distance import euclidean
import time
import scipy.io as sio

class HardTripletLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom)!=3:
            raise Exception("Need two inputs to compute triplet loss")

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        # print "forward starts:  ", time.ctime()
        self.feats = bottom[0].data
        labels= bottom[1].data
        num_s = self.feats.shape[0]

        # building pair-wise distance array
        distances = np.zeros([num_s, num_s])
        for raw in range(num_s):
            for col in range(num_s):
                if (raw != col):
                    distances[raw, col] = euclidean(self.feats[raw,:], self.feats[col,:])
                    distances[col, raw] = distances[raw, col]

        unilabels, counts = np.unique( labels, return_counts=True )
        nclusters = unilabels.size
        total_list = range(num_s)

        n_trip = int(0)
        self.A_ind = []
        self.B_ind = []
        self.C_ind = []
        for i in total_list:
            label_vec = labels[i]
            flag = False
            idx  = 0
            for id,l in enumerate(label_vec):
                if l>0:
                    flag = True
                    idx  = id
                    break
            if not flag:
                continue
            i_label = label_vec[idx]
            ids_p = []
            ids_n = []
            for j in total_list:
                if j==i:
                    continue
                if i_label in labels[j]:
                # if i_label == labels[j]:
                    ids_p.append(j)
                    continue
                if (-i_label in labels[j]):
                # if i_label != labels[j]:
                    ids_n.append(j)
            for p in ids_p:
                for n in ids_n:
                    self.A_ind.append(i)
                    self.B_ind.append(p)
                    self.C_ind.append(n)
                    n_trip += 1
        if n_trip==0:
            print "No triplet found"
            return
        self.anchor   = self.feats[self.A_ind,:]
        self.positive = self.feats[self.B_ind,:]
        self.negative = self.feats[self.C_ind,:]

        margin = 10000
        loss   = float(0)
        self.no_residual_list = []
        for i in range(n_trip):
            ap = euclidean(self.anchor[i,:], self.positive[i,:])**2
            an = euclidean(self.anchor[i,:], self.negative[i,:])**2
            if ap>margin or an>margin:
                if ap>an:
                    an = an*margin*10/ap
                    ap = margin*10
                if an>ap:
                    ap = ap*margin*10/an
                    an = margin*10
            _loss = max( margin+ap-an, 0)
            if i == 0:
                # print ('loss:' + str(_loss) + ' ap:' + str(ap) + ' an:' + str(an))
                pass
            if _loss == 0:
                self.no_residual_list.append(i)
            loss += _loss
        loss = loss / ( 2 * n_trip )
        top[0].data[...] = loss

        # print("--- %s seconds for forward ---" % (time.time() - start_time))
    def backward(self, top, propagate_down, bottom):
        # print "backward starts: ", time.ctime()
        feats = self.feats
        anchor = self.anchor
        positive = self.positive
        negative = self.negative
        aids = self.A_ind
        pids = self.B_ind
        nids = self.C_ind
        a = 1
        diffs = np.zeros([ feats.shape[0], anchor.shape[0], feats.shape[1] ])
        if propagate_down[0]:
            for i in range(anchor.shape[0]):
                if not i in self.no_residual_list:
                    x_a = anchor[i]
                    x_p = positive[i]
                    x_n = negative[i]
                    aid = aids[i]
                    pid = pids[i]
                    nid = nids[i]

                    diffs[aid, i, :] = a*2*(x_n-x_p)
                    diffs[pid, i, :] = a*2*(x_p-x_a)
                    diffs[nid, i, :] = a*2*(x_a-x_n)
        # start_time = time.time()
        bottom[0].diff[:][...] = diffs.sum(axis=1)
    	# print("--- %s seconds for calculating mean ---" % (time.time() - start_time))
        bottom[0].diff[...] /= bottom[0].num
        # print("--- %s seconds for backward ---" % (time.time() - start_time))
