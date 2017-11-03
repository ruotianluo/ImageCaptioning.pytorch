"""
This file is provided for those who has already extracted features in the old mode (one file for each image).
This can merge all the features into one file.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import json
import io

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', default='data', help='output h5 file')
args = parser.parse_args()

def merge(output_dir, _type):
    folder = output_dir+'_'+_type
    f = open(os.path.join(folder, 'feats.np'), 'wb')
    toc = {}
    cnt = 0

    offset = 0
    length = 0
    for root, dirs, files in os.walk(folder, topdown=False):
      for name in files:
        if os.path.splitext(os.path.basename(name))[1] not in ['.npy', '.npz']:
          continue
        img_id = os.path.splitext(os.path.basename(name))[0]
        print(cnt, img_id)
        cnt += 1

        offset = f.tell()
        feat = np.load(os.path.join(root, name))
        if _type == 'att':
            feat = feat['feat']
        np.savez_compressed(f, feat=feat)
        length = f.tell() - offset
        toc[img_id] = (offset, length)

    f.close()
    json.dump(toc, open(os.path.join(folder, 'toc.json'), 'w'))

from dataloader import Reader

def test(output_dir, _type):
    folder = output_dir+'_'+_type

    reader = Reader(folder)

    def old_load(ix):
        if _type == 'att':
            return np.load(os.path.join(folder, ix+'.npz'))['feat']
        else:
            return np.load(os.path.join(folder, ix+'.npy'))

    for ix in reader.toc.keys()[:10]:
        assert np.all(reader.load(ix) == old_load(ix))

merge(args.output_dir, 'att')
merge(args.output_dir, 'fc')

# test if they are identical
test(args.output_dir, 'att')
test(args.output_dir, 'fc')
print('test passed')