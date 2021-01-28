#! /bin/bash

cd image_generator 

DATASET=$1
if [ -z "$DATASET" ]; then
    exit 125
elif [ "$DATASET" == "KITTI" ]; then
    bash scripts/kitti/train_g1_64.sh
elif [ "$DATASET" == "Cityscapes" ]; then
    bash scripts/cityscapes/train_g4_128x256.sh
elif [ "$DATASET" == "Pose" ]; then
    bash scripts/pose/train_g1_64.sh
fi
