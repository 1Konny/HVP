#! /bin/bash

svg_prediction_root=$1

if [ -z "$svg_prediction_root" ]; then
    svg_prediction_root=''
else
    svg_prediction_root="--custom_data_root $svg_prediction_root"
fi

python test.py \
    --dataroot 'datasets/Cityscapes_256x512' \
    --label_nc 19  \
    --loadSize 256 --fineSize 256 \
    --n_downsample_G 2 \
    --use_real_img \
    --fg --fg_labels '13' \
    $svg_prediction_root \
    --name cityscapes_128x256_fg 
