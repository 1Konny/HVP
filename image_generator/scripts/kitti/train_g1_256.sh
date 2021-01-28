#! /bin/bash

GPU='0,1,2,3'

DATAROOT='datasets/KITTI_vid2vid_280'
RESIZE_OR_CROP='scaleWidth'
LOADSIZE='256'
FINESIZE='256'
BATCHSIZE='1'
MAX_FRAMES_PER_GPU=1                                                                                                                                                                                                                                          N_FRAMES_TOTAL=6
N_GPUS_GEN=3

LABEL_NC="19"
FG_LABELS='13'
N_DOWNSAMPLE_G=2
FG='--fg'

NITER='100' # epoch before linearly decaying learning rate
NITER_DECAY='500' # during which learning rate is linearly decayed to zero


NAME="kitti_fg_car_ndsg2_256"

N_LAYERS_D=3
NUM_D=1 

python train.py --name $NAME \
--dataroot $DATAROOT \
--label_nc $LABEL_NC --nThreads 8 \
--resize_or_crop $RESIZE_OR_CROP --loadSize $LOADSIZE --fineSize $FINESIZE \
--batchSize $BATCHSIZE \
--niter $NITER --niter_decay  $NITER_DECAY --n_downsample_G $N_DOWNSAMPLE_G --num_D $NUM_D \
--max_frames_per_gpu $MAX_FRAMES_PER_GPU --n_frames_total $N_FRAMES_TOTAL --n_layers_D $N_LAYERS_D \
$FG --fg_labels $FG_LABELS \
--gpu_ids $GPU --n_gpus_gen $N_GPUS_GEN
