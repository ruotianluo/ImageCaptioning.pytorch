"""
Instruction to use meshed_memory_transformer (https://arxiv.org/abs/1912.08226)

pip install git+https://github.com/ruotianluo/meshed-memory-transformer.git

Note:
Currently m2transformer is not performing as well as original transformer. Not sure why? Still investigating.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

try:
    from m2transformer.models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
except:
    print('meshed-memory-transformer not installed; please run `pip install git+https://github.com/ruotianluo/meshed-memory-transformer.git`')
from .TransformerModel import subsequent_mask, TransformerModel


class M2TransformerModel(TransformerModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        encoder = MemoryAugmentedEncoder(N_enc, 0, attention_module=ScaledDotProductAttentionMemory,
                                        attention_module_kwargs={'m': 40})
        # Another implementation is to use MultiLevelEncoder + att_embed
        decoder = MeshedDecoder(tgt_vocab, 54, N_dec, -1) # -1 is padding;
        model = Transformer(0, encoder, decoder) # 0 is bos
        return model

    def __init__(self, opt):
        super(M2TransformerModel, self).__init__(opt)
        delattr(self, 'att_embed')
        self.att_embed = lambda x: x # The visual embed is in the MAEncoder
        # Notes: The dropout in MAEncoder is different from my att_embed, mine is 0.5?
        # Also the attention mask seems wrong in MAEncoder too...intersting
        
    def logit(self, x): # unsafe way
        return x # M2transformer always output logsoftmax

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory, att_masks = self.model.encoder(att_feats)

        return fc_feats[...,:0], att_feats[...,:0], memory, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        seq = seq.clone()
        seq[~seq_mask.any(-2)] = -1 # Make padding to be -1 (my dataloader uses 0 as padding)
        outputs = self.model(att_feats, seq)

        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decoder(ys, memory, mask)
        return out[:, -1], [ys.unsqueeze(0)]

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        
        att_feats, _, __, ___ = self._prepare_feature_forward(att_feats, att_masks)
        seq, logprobs, seqLogprobs = self.model.beam_search(att_feats, self.seq_length, 0,
                                                     beam_size, return_probs=True, out_size=beam_size)
        seq = seq.reshape(-1, *seq.shape[2:])
        seqLogprobs = seqLogprobs.reshape(-1, *seqLogprobs.shape[2:])

        # if not (seqLogprobs.gather(-1, seq.unsqueeze(-1)).squeeze(-1) == logprobs.reshape(-1, logprobs.shape[-1])).all():
        #     import pudb;pu.db
        # seqLogprobs = logprobs.reshape(-1, logprobs.shape[-1]).unsqueeze(-1).expand(-1,-1,seqLogprobs.shape[-1])
        return seq, seqLogprobs