#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_RGB.prototxt -weights /cs/vml2/xla193/cluster_video/output/UCF-101/caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000
# -weights single_frame_all_layers_hyb_RGB_iter_5000.caffemodel !!! NEVER USE: Pre-trained on UCF-101 !!!
echo "Done."
