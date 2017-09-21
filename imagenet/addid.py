import sys

lines = open("train_small.lst").readlines()
lines = [l.split('\t') for l in lines]
lines = [l[:2] + [str(i)] + l[2:] for i, l in enumerate(lines)]
open("train_ids.lst", "w").write(''.join('\t'.join(l) for l in lines))