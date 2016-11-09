#!/bin/sh
TOOLS=../../build/tools

GLOG_logtostderr=0 GLOG_log_dir=/local-scratch/xla193/cluster_video_/output/UCF-101/snapshots_singleFrame_RGB/ \
$TOOLS/caffe train -solver singleFrame_solver_RGB.prototxt \
-weights /local-scratch/xla193/cluster_video_/output/UCF-101/snapshots_singleFrame_RGB/2w_iter_3855.caffemodel
echo 'Done.'
