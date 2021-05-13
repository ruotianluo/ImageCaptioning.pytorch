# This file contains Att2in2, AdaAtt, AdaAttMO, UpDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# UpDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttModel(CaptionModel):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length # maximum sample length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.bos_idx = getattr(opt, 'bos_idx', 0)
        self.eos_idx = getattr(opt, 'eos_idx', 0)
        self.pad_idx = getattr(opt, 'pad_idx', 0)
        self.unk_idx = getattr(opt, 'unk_idx', None)

        self.use_bn = getattr(opt, 'use_bn', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU(),
                                nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                    (nn.Linear(self.att_feat_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))+
                                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in range(opt.logit_layers - 1)]
            self.logit = nn.Sequential(*(reduce(lambda x,y:x+y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        # For remove bad endding
        self.vocab = opt.vocab
        self.bad_endings_ix = [int(k) for k,v in self.vocab.items() if v in bad_endings]

    def init_hidden(self, bsz):
        weight = self.logit.weight \
                 if hasattr(self.logit, "weight") \
                 else self.logit[0].weight
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        batch_size = fc_feats.size(0)
        if seq.ndim == 3:  # B * seq_per_img * seq_len
            seq = seq.reshape(-1, seq.shape[2])
        seq_per_img = seq.shape[0] // batch_size
        state = self.init_hidden(batch_size*seq_per_img)

        outputs = fc_feats.new_zeros(batch_size*seq_per_img, seq.size(1), self.vocab_size+1)

        # Prepare the features
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # pp_att_feats is used for attention, we cache it in advance to reduce computation cost

        if seq_per_img > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(seq_per_img,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )

        for i in range(seq.size(1)):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size*seq_per_img).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
            outputs[:, i] = output

        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it)

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    def _old_sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks = utils.repeat_tensors(beam_size,
                [p_fc_feats[k:k+1], p_att_feats[k:k+1], pp_att_feats[k:k+1], p_att_masks[k:k+1] if att_masks is not None else None]
            )

            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.new_full([beam_size], self.bos_idx, dtype=torch.long)

                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.old_beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq[k*sample_n+_n, :] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k*sample_n+_n, :] = self.done_beams[k][_n]['logps']
            else:
                seq[k, :] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
                seqLogprobs[k, :] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs


    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # when sample_n == beam_size then each beam is a sample.
        assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        
        state = self.init_hidden(batch_size)

        # first step, feed bos
        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
            [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
        )
        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:
                for _n in range(sample_n):
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k*sample_n+_n, :seq_len] = self.done_beams[k][_n]['logps']
            else:
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size*sample_n)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                [p_fc_feats, p_att_feats, pp_att_feats, p_att_masks]
            )

        trigrams = [] # will be a list of batch_size dictionaries
        
        seq = fc_feats.new_full((batch_size*sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size*sample_n, self.seq_length, self.vocab_size + 1)
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_full([batch_size*sample_n], self.bos_idx, dtype=torch.long)

            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state, output_logsoftmax=output_logsoftmax)
            
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device) # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length: # skip if we achieve maximum length
                break
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                it[~unfinished] = self.pad_idx # This allows eos_idx not being overwritten to 0
                logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
                unfinished = unfinished & (it != self.eos_idx)
            seq[:,t] = it
            seqLogprobs[:,t] = logprobs
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs

    def _diverse_sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        trigrams_table = [[] for _ in range(group_size)] # will be a list of batch_size dictionaries

        seq_table = [fc_feats.new_full((batch_size, self.seq_length), self.pad_idx, dtype=torch.long) for _ in range(group_size)]
        seqLogprobs_table = [fc_feats.new_zeros(batch_size, self.seq_length) for _ in range(group_size)]
        state_table = [self.init_hidden(batch_size) for _ in range(group_size)]

        for tt in range(self.seq_length + group_size):
            for divm in range(group_size):
                t = tt - divm
                seq = seq_table[divm]
                seqLogprobs = seqLogprobs_table[divm]
                trigrams = trigrams_table[divm]
                if t >= 0 and t <= self.seq_length-1:
                    if t == 0: # input <bos>
                        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
                    else:
                        it = seq[:, t-1] # changed

                    logprobs, state_table[divm] = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state_table[divm]) # changed
                    logprobs = F.log_softmax(logprobs / temperature, dim=-1)

                    # Add diversity
                    if divm > 0:
                        unaug_logprobs = logprobs.clone()
                        for prev_choice in range(divm):
                            prev_decisions = seq_table[prev_choice][:, t]
                            logprobs[:, prev_decisions] = logprobs[:, prev_decisions] - diversity_lambda
                    
                    if decoding_constraint and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                        logprobs = logprobs + tmp

                    if remove_bad_endings and t > 0:
                        tmp = logprobs.new_zeros(logprobs.size())
                        prev_bad = np.isin(seq[:,t-1].data.cpu().numpy(), self.bad_endings_ix)
                        # Impossible to generate remove_bad_endings
                        tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                        logprobs = logprobs + tmp

                    # Mess with trigrams
                    if block_trigrams and t >= 3:
                        # Store trigram generated at last step
                        prev_two_batch = seq[:,t-3:t-1]
                        for i in range(batch_size): # = seq.size(0)
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            current  = seq[i][t-1]
                            if t == 3: # initialize
                                trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                            elif t > 3:
                                if prev_two in trigrams[i]: # add to list
                                    trigrams[i][prev_two].append(current)
                                else: # create list
                                    trigrams[i][prev_two] = [current]
                        # Block used trigrams at next step
                        prev_two_batch = seq[:,t-2:t]
                        mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                        for i in range(batch_size):
                            prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                            if prev_two in trigrams[i]:
                                for j in trigrams[i][prev_two]:
                                    mask[i,j] += 1
                        # Apply mask to log probs
                        #logprobs = logprobs - (mask * 1e9)
                        alpha = 2.0 # = 4
                        logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)

                    it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, 1)

                    # stop when all finished
                    if t == 0:
                        unfinished = it != self.eos_idx
                    else:
                        unfinished = (seq[:,t-1] != self.pad_idx) & (seq[:,t-1] != self.eos_idx)
                        it[~unfinished] = self.pad_idx
                        unfinished = unfinished & (it != self.eos_idx) # changed
                    seq[:,t] = it
                    seqLogprobs[:,t] = sampleLogprobs.view(-1)

        return torch.stack(seq_table, 1).reshape(batch_size * group_size, -1), torch.stack(seqLogprobs_table, 1).reshape(batch_size * group_size, -1)

class AdaAtt_lstm(nn.Module):
    def __init__(self, opt, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size

        self.use_maxout = use_maxout

        # Build a LSTM
        self.w2h = nn.Linear(self.input_encoding_size, (4+(use_maxout==True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size)

        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers - 1)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(self.num_layers)])

        # Layers for getting the fake region
        if self.num_layers == 1:
            self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
            self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        else:
            self.r_i2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)


    def forward(self, xt, img_fc, state):

        hs = []
        cs = []
        for L in range(self.num_layers):
            # c,h from previous timesteps
            prev_h = state[0][L]
            prev_c = state[1][L]
            # the input to this layer
            if L == 0:
                x = xt
                i2h = self.w2h(x) + self.v2h(img_fc)
            else:
                x = hs[-1]
                x = F.dropout(x, self.drop_prob_lm, self.training)
                i2h = self.i2h[L-1](x)

            all_input_sums = i2h+self.h2h[L](prev_h)

            sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
            sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
            # decode the gates
            in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
            forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
            out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
            # decode the write inputs
            if not self.use_maxout:
                in_transform = torch.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
            else:
                in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
                in_transform = torch.max(\
                    in_transform.narrow(1, 0, self.rnn_size),
                    in_transform.narrow(1, self.rnn_size, self.rnn_size))
            # perform the LSTM update
            next_c = forget_gate * prev_c + in_gate * in_transform
            # gated cells form the output
            tanh_nex_c = torch.tanh(next_c)
            next_h = out_gate * tanh_nex_c
            if L == self.num_layers-1:
                if L == 0:
                    i2h = self.r_w2h(x) + self.r_v2h(img_fc)
                else:
                    i2h = self.r_i2h(x)
                n5 = i2h+self.r_h2h(prev_h)
                fake_region = torch.sigmoid(n5) * tanh_nex_c

            cs.append(next_c)
            hs.append(next_h)

        # set up the decoder
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)

        state = (torch.cat([_.unsqueeze(0) for _ in hs], 0), 
                torch.cat([_.unsqueeze(0) for _ in cs], 0))
        return top_h, fake_region, state

class AdaAtt_attention(nn.Module):
    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size

        # fake region embed
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.ReLU(), 
            nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        # h out embed
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.input_encoding_size),
            nn.Tanh(), 
            nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.input_encoding_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed, att_masks=None):

        # View into three dimensions
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)

        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)

        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))

        img_all = torch.cat([fake_region.view(-1,1,self.input_encoding_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.input_encoding_size), conv_feat_embed], 1)

        hA = torch.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA,self.drop_prob_lm, self.training)
        
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1), dim=1)

        if att_masks is not None:
            att_masks = att_masks.view(-1, att_size)
            PI = PI * torch.cat([att_masks[:,:1], att_masks], 1) # assume one one at the first time step.
            PI = PI / PI.sum(1, keepdim=True)

        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)

        atten_out = visAttdim + h_out_linear

        h = torch.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h

class AdaAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(AdaAttCore, self).__init__()
        self.lstm = AdaAtt_lstm(opt, use_maxout)
        self.attention = AdaAtt_attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        h_out, p_out, state = self.lstm(xt, fc_feats, state)
        atten_out = self.attention(h_out, p_out, att_feats, p_att_feats, att_masks)
        return atten_out, state

class UpDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(UpDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


############################################################################
# Notice:
# StackAtt and DenseAtt are models that I randomly designed.
# They are not related to any paper.
############################################################################

from .FCModel import LSTMCore
class StackAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(StackAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([h_1,att_res_2],1), [state[0][2:3], state[1][2:3]])

        return h_2, [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

class DenseAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(DenseAttCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        # self.att0 = Attention(opt)
        self.att1 = Attention(opt)
        self.att2 = Attention(opt)

        opt_input_encoding_size = opt.input_encoding_size
        opt.input_encoding_size = opt.input_encoding_size + opt.rnn_size
        self.lstm0 = LSTMCore(opt) # att_feat + word_embedding
        opt.input_encoding_size = opt.rnn_size * 2
        self.lstm1 = LSTMCore(opt)
        self.lstm2 = LSTMCore(opt)
        opt.input_encoding_size = opt_input_encoding_size

        # self.emb1 = nn.Linear(opt.rnn_size, opt.rnn_size)
        self.emb2 = nn.Linear(opt.rnn_size, opt.rnn_size)

        # fuse h_0 and h_1
        self.fusion1 = nn.Sequential(nn.Linear(opt.rnn_size*2, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))
        # fuse h_0, h_1 and h_2
        self.fusion2 = nn.Sequential(nn.Linear(opt.rnn_size*3, opt.rnn_size),
                                     nn.ReLU(),
                                     nn.Dropout(opt.drop_prob_lm))

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        # att_res_0 = self.att0(state[0][-1], att_feats, p_att_feats, att_masks)
        h_0, state_0 = self.lstm0(torch.cat([xt,fc_feats],1), [state[0][0:1], state[1][0:1]])
        att_res_1 = self.att1(h_0, att_feats, p_att_feats, att_masks)
        h_1, state_1 = self.lstm1(torch.cat([h_0,att_res_1],1), [state[0][1:2], state[1][1:2]])
        att_res_2 = self.att2(h_1 + self.emb2(att_res_1), att_feats, p_att_feats, att_masks)
        h_2, state_2 = self.lstm2(torch.cat([self.fusion1(torch.cat([h_0, h_1], 1)),att_res_2],1), [state[0][2:3], state[1][2:3]])

        return self.fusion2(torch.cat([h_0, h_1, h_2], 1)), [torch.cat(_, 0) for _ in zip(state_0, state_1, state_2)]

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = torch.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).to(weight)
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1)) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class Att2in2Core(nn.Module):
    def __init__(self, opt):
        super(Att2in2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2c = nn.Linear(self.rnn_size, 2 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1])
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size) + \
            self.a2c(att_res)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class Att2inCore(Att2in2Core):
    def __init__(self, opt):
        super(Att2inCore, self).__init__(opt)
        del self.a2c
        self.a2c = nn.Linear(self.att_feat_size, 2 * self.rnn_size)

"""
Note this is my attempt to replicate att2all model in self-critical paper.
However, this is not a correct replication actually. Will fix it.
"""
class Att2all2Core(nn.Module):
    def __init__(self, opt):
        super(Att2all2Core, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        #self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # Build a LSTM
        self.a2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.i2h = nn.Linear(self.input_encoding_size, 5 * self.rnn_size)
        self.h2h = nn.Linear(self.rnn_size, 5 * self.rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks=None):
        att_res = self.attention(state[0][-1], att_feats, p_att_feats, att_masks)

        all_input_sums = self.i2h(xt) + self.h2h(state[0][-1]) + self.a2h(att_res)
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = torch.sigmoid(sigmoid_chunk)
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)

        in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
        in_transform = torch.max(\
            in_transform.narrow(1, 0, self.rnn_size),
            in_transform.narrow(1, self.rnn_size, self.rnn_size))
        next_c = forget_gate * state[1][-1] + in_gate * in_transform
        next_h = out_gate * torch.tanh(next_c)

        output = self.dropout(next_h)
        state = (next_h.unsqueeze(0), next_c.unsqueeze(0))
        return output, state

class AdaAttModel(AttModel):
    def __init__(self, opt):
        super(AdaAttModel, self).__init__(opt)
        self.core = AdaAttCore(opt)

# AdaAtt with maxout lstm
class AdaAttMOModel(AttModel):
    def __init__(self, opt):
        super(AdaAttMOModel, self).__init__(opt)
        self.core = AdaAttCore(opt, True)

class Att2in2Model(AttModel):
    def __init__(self, opt):
        super(Att2in2Model, self).__init__(opt)
        self.core = Att2in2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x

class Att2all2Model(AttModel):
    def __init__(self, opt):
        super(Att2all2Model, self).__init__(opt)
        self.core = Att2all2Core(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x : x

class UpDownModel(AttModel):
    def __init__(self, opt):
        super(UpDownModel, self).__init__(opt)
        self.num_layers = 2
        self.core = UpDownCore(opt)

class StackAttModel(AttModel):
    def __init__(self, opt):
        super(StackAttModel, self).__init__(opt)
        self.num_layers = 3
        self.core = StackAttCore(opt)

class DenseAttModel(AttModel):
    def __init__(self, opt):
        super(DenseAttModel, self).__init__(opt)
        self.num_layers = 3
        self.core = DenseAttCore(opt)

class Att2inModel(AttModel):
    def __init__(self, opt):
        super(Att2inModel, self).__init__(opt)
        del self.embed, self.fc_embed, self.att_embed
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.fc_embed = self.att_embed = lambda x: x
        del self.ctx2att
        self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.core = Att2inCore(opt)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)


class NewFCModel(AttModel):
    def __init__(self, opt):
        super(NewFCModel, self).__init__(opt)
        self.fc_embed = nn.Linear(self.fc_feat_size, self.input_encoding_size)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self._core = LSTMCore(opt)
        delattr(self, 'att_embed')
        self.att_embed = lambda x : x
        delattr(self, 'ctx2att')
        self.ctx2att = lambda x: x
    
    def core(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks):
        # Step 0, feed the input image
        # if (self.training and state[0].is_leaf) or \
        #     (not self.training and state[0].sum() == 0):
        #     _, state = self._core(fc_feats, state)
        # three cases
        # normal mle training
        # Sample
        # beam search (diverse beam search)
        # fixed captioning module.
        is_first_step = (state[0]==0).all(2).all(0) # size: B
        if is_first_step.all():
            _, state = self._core(fc_feats, state)
        elif is_first_step.any():
            # This is mostly for diverse beam search I think
            new_state = [torch.zeros_like(_) for _ in state]
            new_state[0][:, ~is_first_step] = state[0][:, ~is_first_step]
            new_state[1][:, ~is_first_step] = state[1][:, ~is_first_step]
            _, state = self._core(fc_feats, state)
            new_state[0][:, is_first_step] = state[0][:, is_first_step]
            new_state[1][:, is_first_step] = state[1][:, is_first_step]
            state = new_state
        # if (state[0]==0).all():
        #     # Let's forget about diverse beam search first
        #     _, state = self._core(fc_feats, state)
        return self._core(xt, state)
    
    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        fc_feats = self.fc_embed(fc_feats)

        return fc_feats, att_feats, att_feats, att_masks


class LMModel(AttModel):
    def __init__(self, opt):
        super(LMModel, self).__init__(opt)
        delattr(self, 'fc_embed')
        self.fc_embed = lambda x: x.new_zeros(x.shape[0], self.input_encoding_size)
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self._core = LSTMCore(opt)
        delattr(self, 'att_embed')
        self.att_embed = lambda x : x
        delattr(self, 'ctx2att')
        self.ctx2att = lambda x: x
    
    def core(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks):
        if (state[0]==0).all():
            # Let's forget about diverse beam search first
            _, state = self._core(fc_feats, state)
        return self._core(xt, state)
    
    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        fc_feats = self.fc_embed(fc_feats)

        return fc_feats, None, None, None