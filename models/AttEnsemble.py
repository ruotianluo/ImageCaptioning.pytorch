# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel
from .AttModel import pack_wrapper, AttModel

class AttEnsemble(AttModel):
    def __init__(self, models, weights=None):
        CaptionModel.__init__(self)
        # super(AttEnsemble, self).__init__()

        self.models = nn.ModuleList(models)
        self.vocab_size = models[0].vocab_size
        self.seq_length = models[0].seq_length
        self.bad_endings_ix = models[0].bad_endings_ix
        self.ss_prob = 0
        weights = weights or [1.0] * len(self.models)
        self.register_buffer('weights', torch.tensor(weights))

    def init_hidden(self, batch_size):
        state = [m.init_hidden(batch_size) for m in self.models]
        return self.pack_state(state)

    def pack_state(self, state):
        self.state_lengths = [len(_) for _ in state]
        return sum([list(_) for _ in state], [])

    def unpack_state(self, state):
        out = []
        for l in self.state_lengths:
            out.append(state[:l])
            state = state[l:]
        return out

    def embed(self, it):
        return [m.embed(it) for m in self.models]

    def core(self, *args):
        return zip(*[m.core(*_) for m, _ in zip(self.models, zip(*args))])

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state):
        # 'it' contains a word index
        xt = self.embed(it)

        state = self.unpack_state(state)
        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, state, tmp_att_masks)
        logprobs = torch.stack([F.softmax(m.logit(output[i]), dim=1) for i,m in enumerate(self.models)], 2).mul(self.weights).div(self.weights.sum()).sum(-1).log()

        return logprobs, self.pack_state(state)

    def _prepare_feature(self, *args):
        return tuple(zip(*[m._prepare_feature(*args) for m in self.models]))

    # def _prepare_feature(self, fc_feats, att_feats, att_masks):

    #     att_feats, att_masks = self.clip_att(att_feats, att_masks)

    #     # embed fc and att feats
    #     fc_feats = [m.fc_embed(fc_feats) for m in self.models]
    #     att_feats = [pack_wrapper(m.att_embed, att_feats[...,:m.att_feat_size], att_masks) for m in self.models]

    #     # Project the attention feats first to reduce memory and computation comsumptions.
    #     p_att_feats = [m.ctx2att(att_feats[i]) for i,m in enumerate(self.models)]

    #     return fc_feats, att_feats, p_att_feats, [att_masks] * len(self.models)

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        fc_feats, att_feats, p_att_feats, att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = [fc_feats[i][k:k+1].expand(beam_size, fc_feats[i].size(1)) for i,m in enumerate(self.models)]
            tmp_att_feats = [att_feats[i][k:k+1].expand(*((beam_size,)+att_feats[i].size()[1:])).contiguous() for i,m in enumerate(self.models)]
            tmp_p_att_feats = [p_att_feats[i][k:k+1].expand(*((beam_size,)+p_att_feats[i].size()[1:])).contiguous() for i,m in enumerate(self.models)]
            tmp_att_masks = [att_masks[i][k:k+1].expand(*((beam_size,)+att_masks[i].size()[1:])).contiguous() if att_masks[i] is not None else att_masks[i] for i,m in enumerate(self.models)]

            it = fc_feats[0].data.new(beam_size).long().zero_()
            logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)
        # return the samples and their log likelihoods