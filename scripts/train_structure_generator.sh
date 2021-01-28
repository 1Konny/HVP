#! /bin/bash

cd structure_generator

DATASET=$1
if [ -z "$DATASET" ]; then
    exit 125
elif [ "$DATASET" == "KITTI" ]; then
    bash scripts/kitti/train.sh 1 1 0.0005 KITTI_64 ds
elif [ "$DATASET" == "Cityscapes" ]; then
    bash scripts/cityscapes/train.sh 1 1 0.0005 Cityscapes_128x256 ds
elif [ "$DATASET" == "Pose" ]; then
    bash scripts/pose/train.sh 1 1 0.0005 Pose_64 ds
fi
