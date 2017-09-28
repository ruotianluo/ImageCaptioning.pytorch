# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    # implements beam search
    # calls beam_step and returns the final set of beams
    # augments log-probabilities with diversity terms when number of groups > 1

    def beam_search(self, init_state, init_logprobs, *args, **kwargs):

        # function computes the similarity score to be augmented
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.clone()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf

        # does one step of classical beam search

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            #INPUTS:
            #logprobsf: probabilities augmented after diversity
            #beam_size: obvious
            #t        : time instant
            #beam_seq : tensor contanining the beams
            #beam_seq_logprobs: tensor contanining the beam logprobs
            #beam_logprobs_sum: tensor contanining joint logprobs
            #OUPUTS:
            #beam_seq : tensor containing the word indices of the decoded captions
            #beam_seq_logprobs : log-probability of each decision made, same size as beam_seq
            #beam_logprobs_sum : joint log-probability of each beam

            ys,ix = torch.sort(logprobsf,1,True)
            candidates = []
            cols = min(beam_size, ys.size(1))
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): # for each column (word, essentially)
                for q in range(rows): # for each beam expansion
                    #compute logprob of expanding beam q with word in (sorted) position c
                    local_logprob = ys[q,c]
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_unaug_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.clone() for _ in state]
            #beam_seq_prev, beam_seq_logprobs_prev
            if t >= 1:
            #we''ll need these as reference when we fork beams around
                beam_seq_prev = beam_seq[:t].clone()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].clone()
            for vix in range(beam_size):
                v = candidates[vix]
                #fork beam index q into index vix
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                #rearrange recurrent states
                for state_ix in range(len(new_state)):
                #  copy over state in previous beam q to new beam at vix
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] # dimension one is time step
                #append new end terminal at the end of this beam
                beam_seq[t, vix] = v['c'] # c'th word is the continuation
                beam_seq_logprobs[t, vix] = v['r'] # the raw logprob here
                beam_logprobs_sum[vix] = v['p'] # the new (sum) logprob along this beam
            state = new_state
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,candidates

        # Start diverse_beam_search
        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        bdash = beam_size // group_size # beam per group

        # INITIALIZATIONS
        beam_seq_table = [torch.LongTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_seq_logprobs_table = [torch.FloatTensor(self.seq_length, bdash).zero_() for _ in range(group_size)]
        beam_logprobs_sum_table = [torch.zeros(bdash) for _ in range(group_size)]

        # logprobs # logprobs predicted in last time step, shape (beam_size, vocab_size+1)
        done_beams_table = [[] for _ in range(group_size)]
        state_table = [list(torch.unbind(_)) for _ in torch.stack(init_state).chunk(group_size, 2)]
        logprobs_table = list(init_logprobs.chunk(group_size, 0))
        # END INIT

        # Chunk elements in the args
        args = list(args)
        args = [_.chunk(group_size) for _ in args]
        args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= self.seq_length + divm - 1:
                    # add diversity
                    logprobsf = logprobs_table[divm].data.float()
                    # suppress UNK tokens in the decoding
                    logprobsf[:,logprobsf.size(1)-1] = logprobsf[:, logprobsf.size(1)-1] - 1000  
                    # diversity is added here
                    # the function directly modifies the logprobsf values and hence, we need to return
                    # the unaugmented ones for sorting the candidates in the end. # for historical
                    # reasons :-)
                    unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)

                    # infer new beams
                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    state_table[divm],\
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    # if time's up... or if end token is reached then copy beams
                    for vix in range(bdash):
                        if beam_seq_table[divm][t-divm,vix] == 0 or t == self.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].clone(), 
                                'logps': beam_seq_logprobs_table[divm][:, vix].clone(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum(),
                                'p': beam_logprobs_sum_table[divm][vix]
                            }
                            done_beams_table[divm].append(final_beam)
                            # don't continue beams from finished sequences
                            beam_logprobs_sum_table[divm][vix] = -1000

                    # move the current group one step forward in time
                    
                    it = beam_seq_table[divm][t-divm]
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(Variable(it.cuda()), *(args[divm] + [state_table[divm]]))

        # all beams are sorted by their log-probabilities
        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = reduce(lambda a,b:a+b, done_beams_table)
        return done_beams