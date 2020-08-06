"""
BartCapModel is using huggingface transformer bart model.
This has the same structure as my TransformerModel.
(Currently, the weights are from scratch.)
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
    from transformers import BartModel, BartConfig
except:
    print('Hugginface transformers not installed; please visit https://github.com/huggingface/transformers')
from .TransformerModel import subsequent_mask, TransformerModel, Generator

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, bart, generator):
        super(EncoderDecoder, self).__init__()
        self.bart = bart
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.bart.encoder(input_ids=src,
                                 attention_mask=src_mask[:, 0])[0]
    
    def decode(self, memory, src_mask, tgt, tgt_mask, cached_states=None, use_cache=None):
        if use_cache:
            out = self.bart(input_ids=None,
                            decoder_input_ids=tgt+1,  # +1 because we reserve 0 for padding
                            encoder_outputs=(memory,),
                            attention_mask=src_mask[:, 0],
                            decoder_cached_states=cached_states,
                            use_cache=use_cache)[:2]
            return out[0], out[1][1]
        else:
            return self.bart(input_ids=None,
                             decoder_input_ids=tgt+1,  # +1 because we reserve 0 for padding
                             decoder_attention_mask=tgt_mask[:, -1],
                             encoder_outputs=(memory,),
                             attention_mask=src_mask[:, 0],
                             use_cache=False)[0]
        # return self.bart.decoder(
        #                  input_ids=tgt+1,  # +1 because we reserve 0 for padding
        #                  encoder_hidden_states=memory,
        #                  encoder_padding_mask=src_mask[:, 0],
        #                  decoder_padding_mask=tgt_mask[:,-1],
        #                  decoder_causal_mask=tgt_mask)[0]


class BartCapModel(TransformerModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        # bart_config = BartConfig(vocab_size=tgt_vocab+1,
        #                          d_model=d_model,
        #                          encoder_ffn_dim=d_ff,
        #                          encoder_layers=N_enc,
        #                          encoder_attention_heads=h,
        #                          decoder_ffn_dim=d_ff,
        #                          decoder_layers=N_dec,
        #                          decoder_attention_heads=h,
        #                          dropout=dropout,  # bart defautl not use dropout on attention
        #                          max_position_embeddings=40,
        #                          # make sure they wont do bad things
        #                          pad_token_id=0, # With multiptle attempts, finally, we just choose to abandon token idx 0.
        #                          bos_token_id=float('-inf'),
        #                          eos_token_id=float('-inf'),
        #                          )
        # Trying to be identical to my transformermodel.
        bart_config = BartConfig(vocab_size=tgt_vocab+1,
                                 d_model=d_model,
                                 encoder_ffn_dim=d_ff,
                                 encoder_layers=N_enc,
                                 encoder_attention_heads=h,
                                 decoder_ffn_dim=d_ff,
                                 decoder_layers=N_dec,
                                 decoder_attention_heads=h,
                                 activation_function='relu',
                                 attention_dropout=dropout,
                                 activation_dropout=dropout,
                                 dropout=dropout,  # bart defautl not use dropout on attention
                                 max_position_embeddings=5000, # to match transformer; important!!!!! it seems
                                 static_position_embeddings=True,
                                 pad_token_id=0, # With multiptle attempts, finally, we just choose to abandon token idx 0.
                                 bos_token_id=float('-inf'),
                                 eos_token_id=float('-inf'),
                                 scale_embedding=True,
                                 normalize_embedding=False,
                                 add_final_layer_norm=True,
                                 normalize_before=True,
                                 )
        bart = BartModel(bart_config)
        del bart.encoder.embed_tokens; bart.encoder.embed_tokens = lambda x: x
        del bart.encoder.embed_positions; bart.encoder.embed_positions = lambda x: 0
        # Although add_final_layer_norm True, we have to manually add for bartencoder because of a bug
        if not bart.encoder.layer_norm:
            bart.encoder.layer_norm = nn.LayerNorm(d_model, eps=1e-5, elementwise_affine=True)
        # The image feature should not be scaled
        bart.encoder.embed_scale = 1.0
        model = EncoderDecoder(
            bart,
            Generator(d_model, tgt_vocab))

        # A new attempt: it seems bart is not using the same init.
        for p in model.parameters():
            if p.dim() > 1 and p.requires_grad:  # this excludes embed_positions
                nn.init.xavier_uniform_(p)

        # Add static_postion_embddings, match at the first fiew checkpoints, then worse.
        # Add xavier, worse
        # Add scale layer norm, and layernorm in encoder
        # Fix scale embedding and normalize embedding: worse
        # Add final lyaer norm.
        # fix scale embedding in encoder: still bad
        # Add normalize before. # Important!!!
        return model

    def __init__(self, opt):
        super(BartCapModel, self).__init__(opt)

    def core_(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, 
                               ys, 
                               subsequent_mask(ys.size(1)).expand(ys.shape[0], -1, -1)
                                        .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]

    
    def _flatten_cached_states(self, cached_states):
        _states = []
        for l in range(len(cached_states)):
            for k1 in ['self', 'encoder_decoder']:
                for k2 in ['prev_key', 'prev_value', 'prev_key_padding_mask']:
                    if cached_states[l][k1][k2] is not None:
                        batch_size = cached_states[l][k1][k2].shape[0]
                        _states.append(cached_states[l][k1][k2].unsqueeze(0))  # The bs has to be at dim 1
                    else:
                        _states.append(torch.empty(0, batch_size))  # use it to replace None
        return _states

    def _unflatten_cached_states(self, _states):
        cached_states = []
        while len(_states) > 0:
            cached_states.append({})
            for k1 in ['self', 'encoder_decoder']:
                cached_states[-1][k1] = {}
                for k2 in ['prev_key', 'prev_value', 'prev_key_padding_mask']:
                    tmp = _states.pop(0)
                    if tmp.shape[0] == 0:
                        cached_states[-1][k1][k2] = None
                    else:
                        cached_states[-1][k1][k2] = tmp[0].contiguous()  # noncontigous when shuffle states during beam search
        return cached_states

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
            cached_states = None
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            cached_states = self._unflatten_cached_states(state[1:])
        out, cached_states = self.model.decode(memory, mask, 
                               ys, 
                               subsequent_mask(ys.size(1)).expand(ys.shape[0], -1, -1)
                                        .to(memory.device),
                               cached_states=cached_states,
                               use_cache=True)

        return out[:, -1], [ys.unsqueeze(0)] + self._flatten_cached_states(cached_states)

