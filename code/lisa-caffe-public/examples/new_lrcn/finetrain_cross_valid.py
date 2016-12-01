import os, sys, optparse
import scipy.io
import pdb
import numpy as np
from caffe.proto import caffe_pb2
import caffe

optparser = optparse.OptionParser()
optparser.add_option("-i", "--countinput", dest="countinput", default="input-UCF-101-20-fmcount.txt", help="fms number of each video")
optparser.add_option("-o", "--output", dest="output", default="all-4_pred.mat", help="Output data name")
optparser.add_option("-m", "--mode", dest="mode", type=int, default=int(0), help="Mode for extracting feature. [ 0-GPU, 1- CPU]")
optparser.add_option("-s", "--batch_size", dest="batch_size", type=int, default=int(100), help="Batch size.")
optparser.add_option("-c", "--device_id", dest="device_id", type=int, default=int(0), help="Device id.")
optparser.add_option("-n", "--num", dest="num", type=int, default=int(197941), help="Number of instances")
optparser.add_option("-f", "--feat", dest="feat", type=int, default=int(4096), help="Dimensionality of feature")
(opts, _)= optparser.parse_args()

data_root     = '/local-scratch/xla193/cluster_video_'
code_root    = '/local-scratch/xla193/cluster_video'
model_root    = os.path.join( data_root, 'output/UCF-101' )
net_root      = os.path.join( code_root, 'code/lisa-caffe-public/examples/new_lrcn' )

def convert_feature(incount_file, out_filename, N, F, device_id, batch_size, mode=0):
  assert os.path.isdir(model_root),    "Model path not exists."
  assert os.path.isdir(net_root),      "Net path not exists."
  with open(os.path.join(model_root, incount_file), 'r') as countf:
    counts = countf.readlines()[:1374]

  data  = np.zeros((len(counts), F),  dtype=np.float32  )

  sys.stdout.write( 'Start extract feature...' )
  if mode   == 0:   caffe.set_mode_gpu()
  elif mode == 1:   caffe.set_mode_cpu()
    
  caffe.set_device(device_id)
  net = caffe.Net(os.path.join( net_root, 'train_test_singleFrame_RGB.prototxt' ),
                  os.path.join( model_root, 'snapshots_singleFrame_RGB/3_fix_margin10e4_10e-9lr_iter_6165.caffemodel' ),
                  caffe.TEST)
  frame_id = 0
  video_feats = np.zeros((N, F))
  temp_idx = 0
  start_idx = 0
  for batch_num in range( int(np.ceil( float(N)/batch_size )) ):
    net.forward()
    frame_feat = net.blobs['fc7'].data # 50x4096
    # pdb.set_trace()
    if (batch_num * batch_size) % 1000 == 0:
      # output a progress indicator every 1000 samples
      sys.stderr.write( 'processed {0} frames.'.format(batch_num * batch_size) )
      sys.stderr.write( " in total: {0}\n". format(N) )
      sys.stderr.flush()
    if start_idx+batch_size < N:
      video_feats[start_idx:start_idx+batch_size, :] = frame_feat
      start_idx += batch_size
      # pdb.set_trace()
      continue
    else:
      this_left = N - start_idx
      video_feats[start_idx:, :] = frame_feat[:this_left, :]

  start_idx = 0
  for video_id in range( len(counts) ):
    data[video_id,:] = np.mean( video_feats[start_idx:start_idx+int(counts[video_id]), :], axis = 0, keepdims=True  )
    start_idx += int(counts[video_id])

  sys.stdout.write('Done.\n')
  # pdb.set_trace()
  out_mat = dict()
  out_mat['data']   = data

  out_path = os.path.join( data_root, 'output/UCF-101', out_filename )

  sys.stdout.write( 'Dump mat file: {0}.\n'.format(out_path) )
  scipy.io.savemat( out_path, out_mat )
  sys.stdout.write( 'Done.\n' )




if __name__ == '__main__':
  convert_feature(opts.countinput, \
    opts.output, opts.num, opts.feat, opts.device_id, opts.batch_size, opts.mode )
