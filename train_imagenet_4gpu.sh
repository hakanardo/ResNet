#!/usr/bin/env sh
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:4 --mem-per-cpu=3100 -tasks-per-node=20
#SBATCH --exclusive

cp /lunarc/nobackup/projects/lu-haar/imagenet/{val_256_q90,train_480_q90}.rec $SNIC_TMP

export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

time python -u train_resnet.py --data-dir $SNIC_TMP --data-type imagenet --depth 50 --batch-size 256 --epochs 120 --gpus=0,1,2,3