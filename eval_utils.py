from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

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

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open('tmp.json', 'w')) # serialize to temporary json file. Sigh, COCO API...

    resFile = 'tmp.json'
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    return out

def eval_split(model, crit, loader, eval_kwargs):
    verbose = eval_kwargs.get('verbose', True)
    val_images_use = eval_kwargs.get('val_images_use', -1)
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 1)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss_sum = 0
    loss_evals = 0
    predictions = []
    while True:
        if beam_size > 1:
            data = loader.get_batch(split, 1)
            n = n + 1
        else:
            data = loader.get_batch(split)
            n = n + loader.batch_size

        # forward the model to get loss
        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
        tmp = [Variable(torch.from_numpy(_)).cuda() for _ in tmp]
        fc_feats, att_feats, labels, masks = tmp

        loss = crit(model(fc_feats, att_feats, labels)[:, 1:], labels[:,1:], masks[:,1:])

        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        if beam_size == 1:
            # forward the model to also get generated samples for each image
            tmp = [data['fc_feats'], data['att_feats']]
            tmp = [Variable(torch.from_numpy(_)).cuda() for _ in tmp]
            fc_feats, att_feats = tmp

            seq, _ = model.sample(fc_feats, att_feats, {})

            #set_trace()
            sents = utils.decode_sequence(loader.get_vocab(), seq)

            for k, sent in enumerate(sents):
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                predictions.append(entry)
                if verbose:
                    print('image %s: %s' %(entry['image_id'], entry['caption']))
        else:
            # seq = model.decode(data['images'], beam_size, sess)
            # sents = [' '.join([loader.ix_to_word.get(str(ix), '') for ix in sent]).strip() for sent in seq]
            # entry = {'image_id': data['infos'][0]['id'], 'caption': sents[0]}
            predictions.append(entry)
            if verbose:
                for sent in sents:
                    print('image %s: %s' %(entry['image_id'], sent))
        
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if val_images_use != -1:
            ix1 = min(ix1, val_images_use)
        for i in range(n - ix1):
            predictions.pop()
        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss.data[0]))

        if data['bounds']['wrapped']:
            break
        if n>= val_images_use:
            break

    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions)

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
