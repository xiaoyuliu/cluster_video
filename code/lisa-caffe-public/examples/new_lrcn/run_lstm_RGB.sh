#!/bin/bash

TOOLS=../../build/tools

export HDF5_DISABLE_VERSION_CHECK=1
export PYTHONPATH=.

GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_RGB.prototxt -weights /cs/vml2/xla193/cluster_video/output/UCF-101/snapshots_singleFrame_RGB/_iter_4500.caffemodel
# -weights single_frame_all_layers_hyb_RGB_iter_5000.caffemodel !!! NEVER USE: Pre-trained on UCF-101 !!!
echo "Done."
