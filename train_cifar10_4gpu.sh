#!/usr/bin/env sh
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:4 --mem-per-cpu=3100 -tasks-per-node=20
#SBATCH --exclusive

cp /lunarc/nobackup/projects/lu-haar/cifar10/* $SNIC_TMP

export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
#export GOMP_CPU_AFFINITY="0-19:2"

python -u train_resnet.py --data-dir $SNIC_TMP --data-type cifar10 --depth 164 \
       --batch-size 512 --num-classes 10 --num-examples 50000 --kv-store local --epochs 20 --gpus=0,1,2,3
