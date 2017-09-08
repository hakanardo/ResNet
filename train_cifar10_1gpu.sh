#!/usr/bin/env sh
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1 --mem-per-cpu=3100 -tasks-per-node=5
#SBATCH --exclusive

cp /lunarc/nobackup/projects/lu-haar/cifar10/* $SNIC_TMP

export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

python -u train_resnet.py --data-dir $SNIC_TMP --data-type cifar10 --depth 164 \
       --batch-size 128 --num-classes 10 --num-examples 50000 --epochs 20 --gpus=0
