import os, sys, optparse
import scipy.io
import pdb
import numpy as np
from caffe.proto import caffe_pb2
import caffe

optparser = optparse.OptionParser()
optparser.add_option("-o", "--output", dest="output", default="outputlstm-UCF-101-20-1ft.mat", help="Output data name")
optparser.add_option("-m", "--mode", dest="mode", type=int, default=int(0), help="Mode for extracting feature. [ 0-GPU, 1- CPU]")
optparser.add_option("-s", "--batch_size", dest="batch_size", type=int, default=int(10), help="Batch size.")
optparser.add_option("-c", "--device_id", dest="device_id", type=int, default=int(2), help="Device id.")
optparser.add_option("-n", "--num", dest="num", type=int, default=int(2694), help="Number of instances")
optparser.add_option("-f", "--feat", dest="feat", type=int, default=int(256), help="Dimensionality of feature")
(opts, _)= optparser.parse_args()

data_root     = '/cs/vml2/xla193/cluster_video'
model_root    = os.path.join( data_root, 'output/UCF-101' )
net_root      = os.path.join( data_root, 'code/lisa-caffe-public/examples/new_lrcn' )

def convert_feature(out_filename, N, F, device_id, batch_size, mode=0):
  assert os.path.isdir(model_root),    "Model path not exists."
  assert os.path.isdir(net_root),      "Net path not exists."

  data  = np.zeros((N, F),  dtype=np.float32  )

  sys.stdout.write( 'Start extract feature...' )
  if mode   == 0:   caffe.set_mode_gpu()
  elif mode == 1:   caffe.set_mode_cpu()
    
  caffe.set_device(device_id)
  net = caffe.Net( os.path.join( net_root, 'train_test_lstm_RGB.prototxt'  ),
  # net = caffe.Net( os.path.join( net_root, 'train_test_singleFrame_RGB.prototxt' ),
                 os.path.join( model_root, 'snapshots_lstm_RGB/_iter_1800.caffemodel'),
                 # os.path.join( model_root, 'caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000' )
                 caffe.TEST)
  for batch_num in range( int(np.ceil( float(N)/batch_size )) ):
    net.forward()
    # pdb.set_trace()
    frame_feat = net.blobs['lstm1-drop'].data # 16x10x256
    num_video = frame_feat.shape[1]
    len_frame = frame_feat.shape[0]
    assert num_video == batch_size, "input sizes not match."
    video_feat = np.zeros([num_video, frame_feat.shape[-1]])

    for i in range(num_video):
      video_feat[i,:] = np.mean( frame_feat[:,i,:], axis = 0, keepdims=True  )
    feats = video_feat
    batch_start = batch_num*batch_size
    batch_end   = (batch_num + 1)*batch_size
    if batch_end<=N:
      data[batch_start:batch_end, :] = feats
    else:
      data[batch_start:,:] = feats[N-batch_start,:]
    # pdb.set_trace()
    if (batch_end) % 1000 == 0:
      # output a progress indicator every 1000 samples
      sys.stdout.write( 'processed {0} queries.\n'.format(batch_end + 1) )
      sys.stdout.flush()

  sys.stdout.write('Done.\n')
  pdb.set_trace()
  out_mat = dict()
  # out_mat['mcf_label']  = data2
  out_mat['data']   = data

  out_path = os.path.join( data_root, 'output/UCF-101', out_filename )

  sys.stdout.write( 'Dump mat file: {0}.\n'.format(out_path) )
  scipy.io.savemat( out_path, out_mat )
  sys.stdout.write( 'Done.\n' )




if __name__ == '__main__':
  convert_feature(opts.output, opts.num, opts.feat, opts.device_id, opts.batch_size, opts.mode )
