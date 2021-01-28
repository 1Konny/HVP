#! /bin/bash

cd image_generator 

DATASET=$1
if [ -z "$DATASET" ]; then
    exit 125
elif [ "$DATASET" == "KITTI" ]; then
    bash scripts/kitti/test_g1_64.sh ../structure_generator/logs/K1_M1_KITTI_64_b0.0005/preddump/100000/ 
elif [ "$DATASET" == "Cityscapes" ]; then
    bash scripts/cityscapes/test_g1_128x256.sh #../structure_generator/logs/K1_M1_Cityscapes_128x256_b0.0005/preddump/100000/ 
elif [ "$DATASET" == "Pose" ]; then
    bash scripts/pose/test_g1_64.sh ../structure_generator/logs/K1_M1_Pose_64_b0.0005/preddump/100000/ 
fi
