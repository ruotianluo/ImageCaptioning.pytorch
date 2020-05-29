from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
from . import misc as utils
from eval_utils import getCOCO

from .div_utils import compute_div_n, compute_global_div_n

import sys
try:
    sys.path.append("coco-caption")
    annFile = 'coco-caption/annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    from pycocoevalcap.eval_spice import COCOEvalCapSpice
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    sys.path.append("cider")
    from pyciderevalcap.cider.cider import Cider
except:
    print('Warning: requirements for eval_multi not satisfied')


def eval_allspice(dataset, preds_n, model_id, split):
    coco = getCOCO(dataset)
    valids = coco.getImgIds()
    
    capsById = {}
    for d in preds_n:
        capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt_n = [p for p in preds_n if p['image_id'] in valids]
    print('using %d/%d predictions_n' % (len(preds_filt_n), len(preds_n)))
    cache_path_n = os.path.join('eval_results/', model_id + '_' + split + '_n.json')
    json.dump(preds_filt_n, open(cache_path_n, 'w')) # serialize to temporary json file. Sigh, COCO API...

    # Eval AllSPICE
    cocoRes_n = coco.loadRes(cache_path_n)
    cocoEvalAllSPICE = COCOEvalCapSpice(coco, cocoRes_n)
    cocoEvalAllSPICE.params['image_id'] = cocoRes_n.getImgIds()
    cocoEvalAllSPICE.evaluate()

    out = {}
    for metric, score in cocoEvalAllSPICE.eval.items():
        out['All'+metric] = score

    imgToEvalAllSPICE = cocoEvalAllSPICE.imgToEval
    # collect SPICE_sub_score
    for k in list(imgToEvalAllSPICE.values())[0]['SPICE'].keys():
        if k != 'All':
            out['AllSPICE_'+k] = np.array([v['SPICE'][k]['f'] for v in imgToEvalAllSPICE.values()])
            out['AllSPICE_'+k] = (out['AllSPICE_'+k][out['AllSPICE_'+k]==out['AllSPICE_'+k]]).mean()
    for p in preds_filt_n:
        image_id, caption = p['image_id'], p['caption']
        imgToEvalAllSPICE[image_id]['caption'] = capsById[image_id]
    return {'overall': out, 'imgToEvalAllSPICE': imgToEvalAllSPICE}

def eval_oracle(dataset, preds_n, model_id, split):
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '_n.json')

    coco = getCOCO(dataset)
    valids = coco.getImgIds()

    capsById = {}
    for d in preds_n:
        capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]
    
    sample_n = capsById[list(capsById.keys())[0]]
    for i in range(len(capsById[list(capsById.keys())[0]])):
        preds = [_[i] for _ in capsById.values()]

        json.dump(preds, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

        cocoRes = coco.loadRes(cache_path)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()

        imgToEval = cocoEval.imgToEval
        for img_id in capsById.keys():
            tmp = imgToEval[img_id]
            for k in tmp['SPICE'].keys():
                if k != 'All':
                    tmp['SPICE_'+k] = tmp['SPICE'][k]['f']
                    if tmp['SPICE_'+k] != tmp['SPICE_'+k]: # nan
                        tmp['SPICE_'+k] = -100
            tmp['SPICE'] = tmp['SPICE']['All']['f']
            if tmp['SPICE'] != tmp['SPICE']: tmp['SPICE'] = -100
            capsById[img_id][i]['scores'] = imgToEval[img_id]

    out = {'overall': {}, 'ImgToEval': {}}
    for img_id in capsById.keys():
        out['ImgToEval'][img_id] = {}
        for metric in capsById[img_id][0]['scores'].keys():
            if metric == 'image_id': continue
            out['ImgToEval'][img_id]['oracle_'+metric] = max([_['scores'][metric] for _ in capsById[img_id]])
            out['ImgToEval'][img_id]['avg_'+metric] = sum([_['scores'][metric] for _ in capsById[img_id]]) / len(capsById[img_id])
        out['ImgToEval'][img_id]['captions'] = capsById[img_id]
    for metric in list(out['ImgToEval'].values())[0].keys():
        if metric == 'captions':
            continue
        tmp = np.array([_[metric] for _ in out['ImgToEval'].values()])
        tmp = tmp[tmp!=-100]
        out['overall'][metric] = tmp.mean()
        
    return out

def eval_div_stats(dataset, preds_n, model_id, split):
    tokenizer = PTBTokenizer()

    capsById = {}
    for i, d in enumerate(preds_n):
        d['id'] = i
        capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]

    n_caps_perimg = len(capsById[list(capsById.keys())[0]])
    print(n_caps_perimg)
    _capsById = capsById # save the untokenized version
    capsById = tokenizer.tokenize(capsById)

    div_1, adiv_1 = compute_div_n(capsById,1)
    div_2, adiv_2 = compute_div_n(capsById,2)

    globdiv_1, _= compute_global_div_n(capsById,1)

    print('Diversity Statistics are as follows: \n Div1: %.2f, Div2: %.2f, gDiv1: %d\n'%(div_1,div_2, globdiv_1))

    # compute mbleu
    scorer = Bleu(4)
    all_scrs = []
    scrperimg = np.zeros((n_caps_perimg, len(capsById)))

    for i in range(n_caps_perimg):
        tempRefsById = {}
        candsById = {}
        for k in capsById:
            tempRefsById[k] = capsById[k][:i] + capsById[k][i+1:]
            candsById[k] = [capsById[k][i]]

        score, scores = scorer.compute_score(tempRefsById, candsById)
        all_scrs.append(score)
        scrperimg[i,:] = scores[1]

    all_scrs = np.array(all_scrs)
    
    out = {}
    out['overall'] = {'Div1': div_1, 'Div2': div_2, 'gDiv1': globdiv_1}
    for k, score in zip(range(4), all_scrs.mean(axis=0).tolist()):
        out['overall'].update({'mBLeu_%d'%(k+1): score})
    imgToEval = {}
    for i,imgid in enumerate(capsById.keys()):
        imgToEval[imgid] = {'mBleu_2' : scrperimg[:,i].mean()}
        imgToEval[imgid]['individuals'] = []
        for j, d in enumerate(_capsById[imgid]):
            imgToEval[imgid]['individuals'].append(preds_n[d['id']])
            imgToEval[imgid]['individuals'][-1]['mBleu_2'] = scrperimg[j,i]
    out['ImgToEval'] = imgToEval

    print('Mean mutual Bleu scores on this set is:\nmBLeu_1, mBLeu_2, mBLeu_3, mBLeu_4')
    print(all_scrs.mean(axis=0))

    return out

def eval_self_cider(dataset, preds_n, model_id, split):
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '_n.json')

    coco = getCOCO(dataset)
    valids = coco.getImgIds()
    
    # Get Cider_scorer
    Cider_scorer = Cider(df='corpus')

    tokenizer = PTBTokenizer()
    gts = {}
    for imgId in valids:
        gts[imgId] = coco.imgToAnns[imgId]
    gts  = tokenizer.tokenize(gts)

    for imgId in valids:
        Cider_scorer.cider_scorer += (None, gts[imgId])
    Cider_scorer.cider_scorer.compute_doc_freq()
    Cider_scorer.cider_scorer.ref_len = np.log(float(len(Cider_scorer.cider_scorer.crefs)))

    # Prepare captions
    capsById = {}
    for d in preds_n:
        capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]

    capsById = tokenizer.tokenize(capsById)
    imgIds = list(capsById.keys())
    scores = Cider_scorer.my_self_cider([capsById[_] for _ in imgIds])

    def get_div(eigvals):
        eigvals = np.clip(eigvals, 0, None)
        return -np.log(np.sqrt(eigvals[-1]) / (np.sqrt(eigvals).sum())) / np.log(len(eigvals))
    sc_scores = [get_div(np.linalg.eigvalsh(_/10)) for _ in scores]
    score = np.mean(np.array(sc_scores))
    
    imgToEval = {}
    for i, image_id in enumerate(imgIds):
        imgToEval[image_id] = {'self_cider': sc_scores[i], 'self_cider_mat': scores[i].tolist()}
    return {'overall': {'self_cider': score}, 'imgToEval': imgToEval}


    return score
