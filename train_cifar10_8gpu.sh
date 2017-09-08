#!/usr/bin/env sh
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:8 --mem-per-cpu=24800 -tasks-per-node=16

export MXNET_CUDNN_AUTOTUNE_DEFAULT=1

python -u train_resnet.py --data-dir /lunarc/nobackup/projects/lu-haar/cifar10 --data-type cifar10 --depth 164 \
       --batch-size 128 --num-classes 10 --num-examples 50000 --gpus=0,1,2,3,4,5,6,7
