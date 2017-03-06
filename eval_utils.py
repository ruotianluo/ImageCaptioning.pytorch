from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

def language_eval(dataset, preds):
    import sys
    if 'coco' in dataset:
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/captions_val2014.json'
    else:
        sys.path.append("f30k-caption")
        annFile = 'f30k-caption/annotations/dataset_flickr30k.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    random.seed(time.time())
    tmp_name = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(tmp_name+'.json', 'w')) # serialize to temporary json file. Sigh, COCO API...

    resFile = tmp_name+'.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # delete the temp file
    os.system('rm '+tmp_name+'.json')

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    return out

def eval_split(cnn_model, model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    cnn_model.eval()
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 0
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        # forward the model to get loss
        tmp = [data['images'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        images, labels, masks = tmp

        att_feats = _att_feats = cnn_model(images)
        fc_feats = _fc_feats = att_feats.mean(2).mean(3).squeeze(2).squeeze(2)

        att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), loader.seq_per_img,) + att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * loader.seq_per_img,) + att_feats.size()[1:]))
        fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), loader.seq_per_img,) + fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * loader.seq_per_img,) + fc_feats.size()[1:]))
        
        loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:]).data[0]

        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats, att_feats = _fc_feats, _att_feats

        seq, _ = model.sample(fc_feats, att_feats, {'beam_size': beam_size})

        #set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)
            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))
        
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(n - ix1):
            predictions.pop()
        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if n>= val_images_use:
            break

    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


# Evaluation fun(ction)
def eval_eval(cnn_model, model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', -1)
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    batch_size = eval_kwargs.get('batch_size', 1)

    # Make sure in the evaluation mode
    cnn_model.eval()
    model.eval()
    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []

    while True:
        # fetch a batch of data
        data = loader.get_batch(split, batch_size)
        n = n + batch_size

        #evaluate loss if we have the labels
        loss = 0

        # Get the image features first
        tmp = [data['images'], data.get('labels', np.zeros(1)), data.get('masks', np.zeros(1))]
        tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
        images, labels, masks = tmp

        att_feats = _att_feats = cnn_model(images)
        fc_feats = _fc_feats = att_feats.mean(2).mean(3).squeeze(2).squeeze(2)

        # forward the model to get loss
        if data.get('labels', None) is not None:

            att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), loader.seq_per_img,) + att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * loader.seq_per_img,) + att_feats.size()[1:]))
            fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), loader.seq_per_img,) + fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * loader.seq_per_img,) + fc_feats.size()[1:]))

            loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:]).data[0]
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        fc_feats, att_feats = _fc_feats, _att_feats
        # forward the model to also get generated samples for each image
        seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)
        
        #set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            predictions.append(entry)
            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs['dump_path'] == 1:
                entry['file_name'] = data['infos'][k]['file_path']
                table.insert(predictions, entry)
            if eval_kwargs['dump_images'] == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats