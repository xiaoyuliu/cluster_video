import numpy as np
import scipy.io as io
import cv2 as cv
import sys, os
import caffe
import pdb

model_path = os.path.join( '/cs/vml2/xla193/cluster_video/output/UCF-101/snapshots_lstm_RGB/_iter_2500.caffemodel' )
print "Start loading network.."
# load net
net = caffe.Net('/cs/vml2/xla193/cluster_video/code/lisa-caffe-public/examples/new_lrcn/train_test_lstm_RGB.prototxt', 
				model_path, 
				caffe.TEST)

net.forward()

out = net.blobs['lstm1-drop'].data
lab = net.blobs['label'].data
pdb.set_trace()

print "Output shape: {}".format( str(out.shape) )
print "Output maximum: {}".format( str(out.max()) )
print "Output minimum: {}".format( str(out.min()) )


# result = Image.fromarray( (visual * 255).astype(np.uint8))

io.savemat('out.mat', {"data": visual} )

 

# result.save('out.png') 