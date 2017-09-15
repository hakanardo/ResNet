#!/usr/bin/env sh
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH -p gpu --gres=gpu:2 --mem-per-cpu=3100 -tasks-per-node=10

cd /lunarc/nobackup/projects/lu-haar/imagenet/
python /lunarc/nobackup/users/hakanardo/miniconda2/lib/python2.7/site-packages/mxnet/tools/im2rec.py \
    --resize 480 --quality  90 train ILSVRC2012_img_train/ --num-thread 10
