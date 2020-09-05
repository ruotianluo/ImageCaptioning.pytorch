# copy from https://github.com/Lyken17/Efficient-PyTorch/tools

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import os, sys
import os.path as osp
from PIL import Image
import six
import string

from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import pickle
import tqdm
import numpy as np
import argparse
import json

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import csv
csv.field_size_limit(sys.maxsize)
FIELDNAMES = ['image_id', 'status']

class FolderLMDB(data.Dataset):
    def __init__(self, db_path, fn_list=None):
        self.db_path = db_path
        self.lmdb = lmdbdict(db_path, unsafe=True)
        self.lmdb._key_dumps = DUMPS_FUNC['ascii']
        self.lmdb._value_loads = LOADS_FUNC['identity']
        if fn_list is not None:
            self.length = len(fn_list)
            self.keys = fn_list
        else:
            raise Error

    def __getitem__(self, index):
        byteflow = self.lmdb[self.keys[index]]

        # load image
        imgbuf = byteflow
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            if args.extension == '.npz':
                feat = np.load(buf)['feat']
            else:
                feat = np.load(buf)
        except Exception as e:
            print(self.keys[index], e)
            return None

        return feat

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def make_dataset(dir, extension):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, [extension]):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data


def raw_npz_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    try:
        npz_data = np.load(six.BytesIO(bin_data))['feat']
    except Exception as e:
        print(path)
        npz_data = None
        print(e)
    return bin_data, npz_data


def raw_npy_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    try:
        npy_data = np.load(six.BytesIO(bin_data))
    except Exception as e:
        print(path)
        npy_data = None
        print(e)
    return bin_data, npy_data


class Folder(data.Dataset):

    def __init__(self, root, loader, extension, fn_list=None):
        super(Folder, self).__init__()
        self.root = root
        if fn_list:
            samples = [os.path.join(root, str(_)+extension) for _ in fn_list]
        else:
            samples = make_dataset(self.root, extension)

        self.loader = loader
        self.extension = extension
        self.samples = samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = self.samples[index]
        sample = self.loader(path)

        return (path.split('/')[-1].split('.')[0],) + sample

    def __len__(self):
        return len(self.samples)


def folder2lmdb(dpath, fn_list, write_frequency=5000):
    directory = osp.expanduser(osp.join(dpath))
    print("Loading dataset from %s" % directory)
    if args.extension == '.npz':
        dataset = Folder(directory, loader=raw_npz_reader, extension='.npz',
                         fn_list=fn_list)
    else:
        dataset = Folder(directory, loader=raw_npy_reader, extension='.npy',
                         fn_list=fn_list)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

    # lmdb_path = osp.join(dpath, "%s.lmdb" % (directory.split('/')[-1]))
    lmdb_path = osp.join("%s.lmdb" % (directory))
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdbdict(lmdb_path, mode='w', key_method='ascii', value_method='identity')

    tsvfile = open(args.output_file, 'a')
    writer = csv.DictWriter(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
    names = []
    all_keys = []
    for idx, data in enumerate(tqdm.tqdm(data_loader)):
        # print(type(data), data)
        name, byte, npz = data[0]
        if npz is not None:
            db[name] = byte
            all_keys.append(name)
        names.append({'image_id': name, 'status': str(npz is not None)})
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            print('writing')
            db.flush()
            # write in tsv
            for name in names:
                writer.writerow(name)
            names = []
            tsvfile.flush()
            print('writing finished')
    # write all keys
    # txn.put("keys".encode(), pickle.dumps(all_keys))
    # # finish iterating through dataset
    # txn.commit()
    for name in names:
        writer.writerow(name)
    tsvfile.flush()
    tsvfile.close()

    print("Flushing database ...")
    db.flush()
    del db

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    # parser.add_argument('--json)
    parser.add_argument('--input_json', default='./data/dataset_coco.json', type=str)
    parser.add_argument('--output_file', default='.dump_cache.tsv', type=str)
    parser.add_argument('--folder', default='./data/cocobu_att', type=str)
    parser.add_argument('--extension', default='.npz', type=str)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    global args
    args = parse_args()

    args.output_file += args.folder.split('/')[-1]
    if args.folder.find('/') > 0:
        args.output_file = args.folder[:args.folder.rfind('/')+1]+args.output_file
    print(args.output_file)

    img_list = json.load(open(args.input_json, 'r'))['images']
    fn_list = [str(_['cocoid']) for _ in img_list]
    found_ids = set()
    try:
        with open(args.output_file, 'r') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
            for item in reader:
                if item['status'] == 'True':
                    found_ids.add(item['image_id'])
    except:
        pass
    fn_list = [_ for _ in fn_list if _ not in found_ids]
    folder2lmdb(args.folder, fn_list)

    # Test existing.
    found_ids = set()
    with open(args.output_file, 'r') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            if item['status'] == 'True':
                found_ids.add(item['image_id'])

    folder_dataset = FolderLMDB(args.folder+'.lmdb', list(found_ids))
    data_loader = DataLoader(folder_dataset, num_workers=16, collate_fn=lambda x: x)
    for data in tqdm.tqdm(data_loader):
        assert data[0] is not None