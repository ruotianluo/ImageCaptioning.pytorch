from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import skimage
import skimage.io
import scipy.misc

class DataLoaderRaw():
    
    def __init__(self, opt):
        self.opt = opt
        self.coco_json = opt.get('coco_json', '')
        self.folder_path = opt.get('folder_path', '')

        self.batch_size = opt.get('batch_size', 1)

        # load the json file which contains additional information about the dataset
        print('DataLoaderRaw loading images from folder: ', self.folder_path)

        self.files = []
        self.ids = []

        print(len(self.coco_json))
        if len(self.coco_json) > 0:
            print('reading from ' + opt.coco_json)
            # read in filenames from the coco-style json file
            self.coco_annotation = json.load(open(self.coco_json))
            for k,v in enumerate(self.coco_annotation['images']):
                fullpath = os.path.join(self.folder_path, v['file_name'])
                self.files.append(fullpath)
                self.ids.append(v['id'])
        else:
            # read in all the filenames from the folder
            print('listing all images in directory ' + self.folder_path)
            def isImage(f):
                supportedExt = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.ppm','.PPM']
                for ext in supportedExt:
                    start_idx = f.rfind(ext)
                    if start_idx >= 0 and start_idx + len(ext) == len(f):
                        return True
                return False

            n = 1
            for root, dirs, files in os.walk(self.folder_path, topdown=False):
                for file in files:
                    fullpath = os.path.join(self.folder_path, file)
                    if isImage(fullpath):
                        self.files.append(fullpath)
                        self.ids.append(str(n)) # just order them sequentially
                        n = n + 1

        self.N = len(self.files)
        print('DataLoaderRaw found ', self.N, ' images')

        self.iterator = 0

    def get_batch(self, split, batch_size=None):
        batch_size = batch_size or self.batch_size

        # pick an index of the datapoint to load next
        img_batch = np.ndarray([batch_size, 224,224,3], dtype = 'float32')
        max_index = self.N
        wrapped = False
        infos = []

        for i in range(batch_size):
            ri = self.iterator
            ri_next = ri + 1
            if ri_next >= max_index:
                ri_next = 0
                wrapped = True
                # wrap back around
            self.iterator = ri_next

            img = skimage.io.imread(self.files[ri])

            if len(img.shape) == 2:
                img = img[:,:,np.newaxis]
                img = img.concatenate((img, img, img), axis=2)

            img_batch[i] = img[16:240, 16:240, :].astype('float32')/255.0

            info_struct = {}
            info_struct['id'] = self.ids[ri]
            info_struct['file_path'] = self.files[ri]
            infos.append(info_struct)

        data = {}
        data['images'] = img_batch
        data['bounds'] = {'it_pos_now': self.iterator, 'it_max': self.N, 'wrapped': wrapped}
        data['infos'] = infos

        return data

    def reset_iterator(self, split):
        self.iterator = 0

    def get_vocab_size(self):
        return len(self.ix_to_word)

    def get_vocab(self):
        return self.ix_to_word
        