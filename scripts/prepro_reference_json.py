# coding: utf-8
"""
Create a reference json file used for evaluation with `coco-caption` repo.
Used when reference json is not provided, (e.g., flickr30k, or you have your own split of train/val/test)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
import sys
import hashlib
from random import shuffle, seed


def main(params):

    imgs = json.load(open(params['input_json'][0], 'r'))['images']
    # tmp = []
    # for k in imgs.keys():
    #     for img in imgs[k]:
    #         img['filename'] = img['image_id']  # k+'/'+img['image_id']
    #         img['image_id'] = int(
    #             int(hashlib.sha256(img['image_id']).hexdigest(), 16) % sys.maxint)
    #         tmp.append(img)
    # imgs = tmp

    # create output json file
    out = {'info': {'description': 'This is stable 1.0 version of the 2014 MS COCO dataset.', 'url': 'http://mscoco.org', 'version': '1.0', 'year': 2014, 'contributor': 'Microsoft COCO group', 'date_created': '2015-01-27 09:11:52.357475'}, 'licenses': [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}, {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}, {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}, {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'}, {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}, {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}], 'type': 'captions'}
    out.update({'images': [], 'annotations': []})

    cnt = 0
    empty_cnt = 0
    for i, img in enumerate(imgs):
        if img['split'] == 'train':
            continue
        out['images'].append(
            {'id': img.get('cocoid', img['imgid'])})
        for j, s in enumerate(img['sentences']):
            if len(s) == 0:
                continue
            s = ' '.join(s['tokens'])
            out['annotations'].append(
                {'image_id': out['images'][-1]['id'], 'caption': s, 'id': cnt})
            cnt += 1

    json.dump(out, open(params['output_json'], 'w'))
    print('wrote ', params['output_json'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', nargs='+', required=True,
                        help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data.json',
                        help='output json file')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)

