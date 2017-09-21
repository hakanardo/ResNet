#!/usr/bin/env sh
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:1 --exclusive

cd /lunarc/nobackup/projects/lu-haar/imagenet/
python -u /lunarc/nobackup/users/hakanardo/miniconda2/lib/python2.7/site-packages/mxnet/tools/im2rec.py \
    --resize 480 --quality  90 train_ids ILSVRC2012_img_train/ --num-thread 8
