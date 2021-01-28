#! /bin/bash

svg_prediction_root=$1

if [ -z "$svg_prediction_root" ]; then
    svg_prediction_root=''
else
    svg_prediction_root="--custom_data_root $svg_prediction_root"
fi

DATAROOT='datasets/KITTI_vid2vid_280'
RESIZE_OR_CROP='scaleWidth'
LOADSIZE='256'
FINESIZE='256'
BATCHSIZE='1'

LABEL_NC="19"
FG_LABELS='13'
FG='--fg'
NAME="kitti_fg_car_ndsg2_256"                                                                                                                                                                                                                                 #NAME="kitti_fg_car_ndsg2_256_v2"

N_DOWNSAMPLE_G=2

python test.py --name $NAME \
--dataroot $DATAROOT \
--label_nc $LABEL_NC \
--resize_or_crop $RESIZE_OR_CROP --loadSize $LOADSIZE --fineSize $FINESIZE \
--batchSize $BATCHSIZE \
--n_downsample_G $N_DOWNSAMPLE_G \
$FG --fg_labels $FG_LABELS $svg_prediction_root \
--use_real_img
