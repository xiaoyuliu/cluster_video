#!/bin/sh
TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe train -solver singleFrame_solver_RGB.prototxt \
-weights /local-scratch/xla193/cluster_video_/output/UCF-101/snapshots_singleFrame_RGB/1_iter_1300.caffemodel
echo 'Done.'
