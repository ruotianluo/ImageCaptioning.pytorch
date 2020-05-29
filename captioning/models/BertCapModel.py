"""
BertCapModel is using huggingface transformer bert model as seq2seq model.

The result is not as goog as original transformer.
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
    from transformers import BertModel, BertConfig
except:
    print('Hugginface transformers not installed; please visit https://github.com/huggingface/transformers')
from .TransformerModel import subsequent_mask, TransformerModel, Generator

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(inputs_embeds=src,
                            attention_mask=src_mask)[0]
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(input_ids=tgt,
                            attention_mask=tgt_mask,
                            encoder_hidden_states=memory,
                            encoder_attention_mask=src_mask)[0]


class BertCapModel(TransformerModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        enc_config = BertConfig(vocab_size=1,
                                hidden_size=d_model,
                                num_hidden_layers=N_enc,
                                num_attention_heads=h,
                                intermediate_size=d_ff,
                                hidden_dropout_prob=dropout,
                                attention_probs_dropout_prob=dropout,
                                max_position_embeddings=1,
                                type_vocab_size=1)
        dec_config = BertConfig(vocab_size=tgt_vocab,
                                hidden_size=d_model,
                                num_hidden_layers=N_dec,
                                num_attention_heads=h,
                                intermediate_size=d_ff,
                                hidden_dropout_prob=dropout,
                                attention_probs_dropout_prob=dropout,
                                max_position_embeddings=17,
                                type_vocab_size=1,
                                is_decoder=True)
        encoder = BertModel(enc_config)
        def return_embeds(*args, **kwargs):
            return kwargs['inputs_embeds']
        del encoder.embeddings; encoder.embeddings = return_embeds
        decoder = BertModel(dec_config)
        model = EncoderDecoder(
            encoder,
            decoder,
            Generator(d_model, tgt_vocab))
        return model

    def __init__(self, opt):
        super(BertCapModel, self).__init__(opt)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state = [ys.unsqueeze(0)]
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decode(memory, mask, 
                               ys, 
                               subsequent_mask(ys.size(1))
                                        .to(memory.device))
        return out[:, -1], [ys.unsqueeze(0)]
