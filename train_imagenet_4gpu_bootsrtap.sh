#!/usr/bin/env bash
#SBATCH -t 100:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:4
#SBATCH --exclusive

cp /lunarc/nobackup/projects/lu-haar/imagenet/{val_256_q90,train_480_q90_ids}.* $SNIC_TMP

export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

time python -u train_resnet_bootstrap.py --data-dir $SNIC_TMP --data-type imagenet --depth 50 --batch-size 256 --epochs 120 --gpus=0,1,2,3