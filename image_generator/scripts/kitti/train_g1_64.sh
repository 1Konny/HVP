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
    --dataroot 'datasets/KITTI_vid2vid_90' \
    --save_epoch_freq 20 \
    --label_nc 19 \
    --resize_or_crop scaleWidth_and_crop --loadSize 72 --fineSize 64 \
    --n_frames_G 6 \
    --max_frames_per_gpu 10 \
    --n_frames_total 10 \
    --n_scales_spatial 1 \
    --n_downsample_G 2 \
    --num_D 1 \
    --n_layers_D 3 \
    --niter 20 --niter_decay 400 --niter_step 40 \
    --fg --fg_labels '13' \
    --name kitti_64_g1_fg \
    $CONTINUE
