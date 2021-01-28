#! /bin/bash

svg_prediction_root=$1

if [ -z "$svg_prediction_root" ]; then
    svg_prediction_root=''
else
    svg_prediction_root="--custom_data_root $svg_prediction_root"
fi

python test.py \
    --dataroot 'datasets/KITTI_vid2vid_90' \
    --label_nc 19  \
    --loadSize 64 --fineSize 64 \
    --n_frames_G 6 \
    --n_scales_spatial 1 \
    --n_downsample_G 2 \
    --use_real_img \
    --fg --fg_labels '13' \
    $svg_prediction_root \
    --name kitti_64_g1_fg
