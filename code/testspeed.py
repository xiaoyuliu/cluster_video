import numpy as np
import time
import scipy.io as sio

a = sio.loadmat('/cs/vml2/xla193/cluster_video/output/UCF-101/snapshots_singleFrame_RGB/3darray.mat')
A = np.array(a['data']).astype(np.float32)
start_time = time.time()
print(A.shape)
B = np.zeros([A.shape[0], A.shape[2]])
start_time = time.time()
B = A.mean(axis=1)
print("--- %s seconds for calculating mean ---" % (time.time() - start_time))
