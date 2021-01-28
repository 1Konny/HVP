#! /bin/bash

#1: M (lstm multiplier): 1
#2: K (vgg multiplier): 1
#3: BETA: 0.0005
#4: DATASET: Pose_64 | Pose_128 
#5: MODE: ds/dp
#6: TAG
#7: LOAD_CKPT_TYPE

M=$1
K=$2
BETA=$3
DATASET=$4
MODE=$5
TAG=$6
if [ ! -z "$TAG" ]; then
    TAG="_$TAG"
fi
LOAD_CKPT_TYPE=$7
if [ -z "$LOAD_CKPT_TYPE" ]; then
    LOAD_CKPT_TYPE=""
elif [ "$LOAD_CKPT_TYPE" == "load_dp_ckpt" ]; then
    LOAD_CKPT_TYPE="--load_dp_ckpt"
elif [ "$LOAD_CKPT_TYPE" == "load_ds_ckpt" ]; then
    LOAD_CKPT_TYPE="--load_ds_ckpt"
else
    echo "${LOAD_CKPT_TYPE} is not supported. options: '' | 'load_dp_ckpt' | 'load_ds_ckpt'"
    exit 125
fi

NAME="K${K}_M${M}_${DATASET}_b${BETA}${TAG}"

if [ "$DATASET" == "Pose_64" ]; then
    CHANNELS=25
    DATA_ROOT='datasets/Pose_64'
elif [ "$DATASET" == "Pose_128" ]; then
    CHANNELS=25
    DATA_ROOT='datasets/Pose_128'
fi

GPU_FLAG="localhost:${CUDA_VISIBLE_DEVICES}"

Z_DIM=128
G_DIM=128
RNN_SIZE=256
AE_SIZE=64

LR=0.0001
BATCH_SIZE=16

N_PAST=5
N_FUTURE=40
N_EVAL=40
FRAME_SAMPLING_RATE=2
N_PREDICTION=0

MAX_ITER=100000
PRINT_ITER=5
LOG_LINE_ITER=100
LOG_IMG_ITER=2000
LOG_CKPT_ITER=2000
VALIDATE_ITER=2000
LOG_CKPT_SEC=600

if [ "$MODE" == "ds" ]; then
    CMD="deepspeed --include=$GPU_FLAG train.py --deepspeed --deepspeed_config deepspeed_util/ds_config.json"
elif [ "$MODE" == "dp" ]; then
    CMD="python train.py"
fi

CMD="${CMD}\
    --dataset $DATASET --data_root $DATA_ROOT --channels $CHANNELS \
    --n_past $N_PAST --n_future $N_FUTURE --n_eval $N_EVAL --frame_sampling_rate $FRAME_SAMPLING_RATE \
    --g_dim $G_DIM --z_dim $Z_DIM --rnn_size $RNN_SIZE --ae_size $AE_SIZE --M $M --K $K --beta $BETA \
    --batch_size $BATCH_SIZE --lr $LR \
    --max_iter $MAX_ITER --print_iter $PRINT_ITER --validate_iter $VALIDATE_ITER --log_img_iter $LOG_IMG_ITER --log_line_iter $LOG_LINE_ITER --log_ckpt_iter $LOG_CKPT_ITER --log_ckpt_sec $LOG_CKPT_SEC \
    --name $NAME $LOAD_CKPT_TYPE"
echo $CMD
eval $CMD
