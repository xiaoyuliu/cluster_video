#!/bin/sh
TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe train -solver singleFrame_solver_RGB.prototxt \
-weights /local-scratch/xla193/cluster_video_/output/UCF-101/caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000
echo 'Done.'
