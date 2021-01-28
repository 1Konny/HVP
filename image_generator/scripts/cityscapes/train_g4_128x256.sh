#! /bin/bash

CONTINUE=$1
if [ -z "$CONTINUE" ]; then
    CONTINUE=""
elif [ "$CONTINUE" == "True" ]; then
    CONTINUE="--continue_train"
else
    CONTINUE=""
fi

python train.py \
    --dataroot 'datasets/Cityscapes_256x512' \
    --label_nc 19 \
    --loadSize 256 \
    --gpu_ids 0,1,2,3 --n_gpus_gen 3 \
    --max_frames_per_gpu 2 \
    --n_frames_total 6 \
    --fg --fg_labels '13' \
    --n_downsample_G 2 --num_D 1 \
    --name cityscapes_128x256_fg \
    $CONTINUE
