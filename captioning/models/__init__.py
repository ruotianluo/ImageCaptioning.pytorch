from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import torch

from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .AttModel import *
from .TransformerModel import TransformerModel
from .cachedTransformer import TransformerModel as cachedTransformer
from .BertCapModel import BertCapModel
from .M2Transformer import M2TransformerModel
from .AoAModel import AoAModel

def setup(opt):
    if opt.caption_model in ['fc', 'show_tell']:
        print('Warning: %s model is mostly deprecated; many new features are not supported.' %opt.caption_model)
        if opt.caption_model == 'fc':
            print('Use newfc instead of fc')
    if opt.caption_model == 'fc':
        model = FCModel(opt)
    elif opt.caption_model == 'language_model':
        model = LMModel(opt)
    elif opt.caption_model == 'newfc':
        model = NewFCModel(opt)
    elif opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opt.caption_model == 'att2in2':
        model = Att2in2Model(opt)
    elif opt.caption_model == 'att2all2':
        print('Warning: this is not a correct implementation of the att2all model in the original paper.')
        model = Att2all2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model in ['topdown', 'updown']:
        model = UpDownModel(opt)
    # StackAtt
    elif opt.caption_model == 'stackatt':
        model = StackAttModel(opt)
    # DenseAtt
    elif opt.caption_model == 'denseatt':
        model = DenseAttModel(opt)
    # Transformer
    elif opt.caption_model == 'transformer':
        if getattr(opt, 'cached_transformer', False):
            model = cachedTransformer(opt)
        else:
            model = TransformerModel(opt)
    # AoANet
    elif opt.caption_model == 'aoa':
        model = AoAModel(opt)
    elif opt.caption_model == 'bert':
        model = BertCapModel(opt)
    elif opt.caption_model == 'm2transformer':
        model = M2TransformerModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model
