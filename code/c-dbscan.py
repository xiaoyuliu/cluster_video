import numpy as np 
import os
import scipy.io as sio
import scipy.spatial.KDTree as KDTree

data_dir = '/cs/vml2/xla193/cluster_video/output/UCF-101'
mat_file = 'output-UCF-101-10-0ft.mat'

mat = sio.loadmat(os.path.join(data_dir, mat_file))
kdtree = KDTree(mat, 20)