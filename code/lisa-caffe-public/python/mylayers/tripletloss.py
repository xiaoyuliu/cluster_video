import caffe, random, pdb
import numpy as np 
import scipy.spatial.distance as ssd

class TripletLossLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom)!=2:
            raise Exception("Need two inputs to compute triplet loss")

    def reshape(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        feats = bottom[0].data
        labels= bottom[1].data
        n_neg = int(5)
        num_s = feats.shape[0]

        unilabels, counts = np.unique( labels, return_counts=True)
        nclusters = unilabels.size

        if nclusters == 1:
            return
        if nclusters < n_neg:
            n_neg = nclusters - 1

        n_trip = int(0)
        id_trip= int(0)
        self.A_ind = []
        self.B_ind = []
        self.C_ind = []
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
                                    print "number of triplets:", id_triplet
                                if (id_triplet+1) % n_neg == 1:
                                    break
        if n_trip == 0:
            return

        self.anchor   = feats[self.A_ind,:]
        self.positive = feats[self.B_ind,:]
        self.negative = feats[self.C_ind,:]

        margin = 1
        loss   = float(0)
        self.no_residual_list = []
        for i in range(n_trip):
            pdb.set_trace()
            ap = ssd.cdist( self.anchor[i,:], self.positive[i,:], 'sqeuclidean' )[0][0]
            an = ssd.cdist( self.anchor[i,:], self.negative[i,:], 'sqeuclidean' )[0][0]
            _loss = max( margin+ap-an, 0)
            if i == 0:
                # print ('loss:' + str(_loss) + ' ap:' + str(ap) + ' an:' + str(an))
                pass
            if _loss == 0:
                self.no_residual_list.append(i)
            loss += _loss
        loss = loss / ( 2 * n_trip )
        top[0].data[...] = loss

    def backward(self, bottom, top, propagate_down):
        anchor = self.anchor
        positive = self.positive
        negative = self.negative
        aids = self.A_ind
        pids = self.B_ind
        nids = self.C_ind
        a = 1
        diffs = np.zeros([ bottom[0].shape[0], anchor.shape[0], bottom[0].shape[1] ])
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

        for i in range(bottom[0].shape):
            bottom[i].diff[...] = np.mean(diffs[i,:,:], dims=0, keep_dim=True)