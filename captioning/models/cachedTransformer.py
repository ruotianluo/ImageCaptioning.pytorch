# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask, past=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, past=past)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        _x = sublayer(self.norm(x))
        if type(_x) is tuple: # for multi-head attention that returns past
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask,
                      layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        "Follow Figure 1 (right) for connections."
        m = memory
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None, layer_past=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # The past works differently here. For self attn, the query and key be updated incrementailly
        # For src_attn the past is fixed.

        # For src_attn, when the layer past is ready
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1: # suppose memory size always greater than 1 
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            # 1) Do all the linear projections in batch from d_model => h x d_k 
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
        
        # self attn + past OR the first time step of src attn
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for x in [query, key, value]]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N_enc=6, N_dec=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N_enc),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                                 c(ff), dropout), N_dec),
            lambda x:x, # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))
        
        # This was important from their code. 
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TransformerModel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        
        self.N_enc = getattr(opt, 'N_enc', opt.num_layers)
        self.N_dec = getattr(opt, 'N_dec', opt.num_layers)
        self.d_model = getattr(opt, 'd_model', opt.input_encoding_size)
        self.d_ff = getattr(opt, 'd_ff', opt.rnn_size)
        self.h = getattr(opt, 'num_att_heads', 8)
        self.dropout = getattr(opt, 'dropout', 0.1)

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.d_model),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.d_model),) if self.use_bn==2 else ())))
        
        delattr(self, 'embed')
        self.embed = lambda x : x
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1


        self.model = self.make_model(0, tgt_vocab,
            N_enc=self.N_enc,
            N_dec=self.N_dec,
            d_model=self.d_model,
            d_ff=self.d_ff,
            h=self.h,
            dropout=self.dropout)

    def logit(self, x): # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[...,:0], att_feats[...,:0], memory, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks]
                )
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        out = self.model(att_feats, seq, att_masks, seq_mask)

        outputs = self.model.generator(out)
        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        """
        state is the precomputed key/value. N_dec x seq_len x d_model
        Note: due to the layer norm, it's not equivalant to stateless,
        but it seems behaving similar
        """
        # state is tokens + past
        if len(state) == 0:
            ys = it.unsqueeze(1)
            # basically empty state, just to let it know to return past
            # The second dim has to be batch_size, for beam search purpose
            past = [fc_feats_ph.new_zeros(self.N_dec * 2, fc_feats_ph.shape[0], 0, self.d_model),  # self
                    fc_feats_ph.new_zeros(self.N_dec * 2, fc_feats_ph.shape[0], 0, self.d_model)]  # src
            # 2 for self attn, 2 for src attn
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        out, past = self.model.decode(memory, mask, 
                               ys, # We still feed the full past words, because we need it for position embedding to know the position id
                               subsequent_mask(ys.size(1))
                                        .to(memory.device),
                               past=past)
        return out[:, -1], [ys.unsqueeze(0)] + past
