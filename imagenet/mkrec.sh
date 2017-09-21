#!/usr/bin/env bash
python /usr/local/src/mxnet/tools/im2rec.py --list 1 --recursive 1 train train/
head -100 train.lst > train_small.lst
python /usr/local/src/mxnet/tools/im2rec.py --resize 480 --quality  90 --num-thread 4  train_small train/
mv train_small.idx train_480_q90.idx
mv train_small.rec train_480_q90.rec
python /usr/local/src/mxnet/tools/im2rec.py --resize 256 --quality  90 --num-thread 4  train_small train/
mv train_small.idx train_256_q90.idx
mv train_small.rec train_256_q90.rec
ln -sf train_256_q90.idx val_256_q90.idx
ln -sf train_256_q90.rec val_256_q90.rec

python addid.py
python /usr/local/src/mxnet/tools/im2rec.py --resize 480 --quality  90 --num-thread 4  --pack-label 1 train_ids train/
mv train_ids.idx train_480_q90_ids.idx
mv train_ids.rec train_480_q90_ids.rec
python /usr/local/src/mxnet/tools/im2rec.py --resize 256 --quality  90 --num-thread 4 --pack-label 1  train_ids train/
mv train_ids.idx train_256_q90_ids.idx
mv train_ids.rec train_256_q90_ids.rec
ln -sf train_256_q90_ids.idx val_256_q90_ids.idx
ln -sf train_256_q90_ids.rec val_256_q90_ids.rec
