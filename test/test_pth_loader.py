import torch
from captioning.data.pth_loader import CaptionDataset
from captioning.utils.misc import pickle_load

def test_folder():
    x = pickle_load(open('log_trans/infos_trans.pkl', 'rb'))
    dataset = CaptionDataset(x['opt'])
    ds = torch.utils.data.Subset(dataset, dataset.split_ix['train'])
    ds[0]

def test_lmdb():
    x = pickle_load(open('log_trans/infos_trans.pkl', 'rb'))
    x['opt'].input_att_dir = 'data/vilbert_att.lmdb'
    dataset = CaptionDataset(x['opt'])
    ds = torch.utils.data.Subset(dataset, dataset.split_ix['train'])
    ds[0]