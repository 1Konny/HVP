#! /bin/bash

cd structure_generator

DATASET=$1
if [ -z "$DATASET" ]; then
    exit 125
elif [ "$DATASET" == "KITTI" ]; then
    bash scripts/kitti/test.sh 1 1 0.0005 KITTI_64 dp '' 'load_ds_ckpt'
elif [ "$DATASET" == "Cityscapes" ]; then
    bash scripts/cityscapes/test.sh 1 1 0.0005 Cityscapes_128x256 dp '' 'load_ds_ckpt'
elif [ "$DATASET" == "Pose" ]; then
    bash scripts/pose/test.sh 1 1 0.0005 Pose_64 dp '' 'load_ds_ckpt'
fi
