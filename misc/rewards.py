from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import misc.utils as utils
from collections import OrderedDict
import torch
from torch.autograd import Variable

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD

CiderD_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_cider_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()

def get_self_critical_reward(model, fc_feats, att_feats, data, gen_result):
    batch_size = gen_result.size(0)# batch_size = sample_size * seq_per_img
    seq_per_img = batch_size // len(data['gts'])
    
    # get greedy decoding baseline
    greedy_res, _ = model.sample(Variable(fc_feats.data, volatile=True), Variable(att_feats.data, volatile=True))

    res = OrderedDict()
    
    gen_result = gen_result.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data['gts'])):
        gts[i] = [array_to_str(data['gts'][i][j]) for j in range(len(data['gts'][i]))]

    #_, scores = Bleu(4).compute_score(gts, res)
    #scores = np.array(scores[3])
    res = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size // seq_per_img] for i in range(2 * batch_size)}
    _, scores = CiderD_scorer.compute_score(gts, res)
    print('Cider scores:', _)

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)

    return rewards
