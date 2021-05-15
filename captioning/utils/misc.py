from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os

import torch.nn.functional as F

import six
from six.moves import cPickle

bad_endings = ['with','in','on','of','a','at','to','for','an','this','his','her','that']
bad_endings += ['the']


def pickle_load(f):
    """ Load a pickle.
    Parameters
    ----------
    f: file-like object
    """
    if six.PY3:
        return cPickle.load(f, encoding='latin-1')
    else:
        return cPickle.load(f)


def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)


# modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/comm.py
def serialize_to_tensor(data):
    device = torch.device("cpu")

    buffer = cPickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def deserialize(tensor):
    buffer = tensor.cpu().numpy().tobytes()
    return cPickle.loads(buffer)


# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        if int(os.getenv('REMOVE_BAD_ENDINGS', '0')):
            flag = 0
            words = txt.split(' ')
            for j in range(len(words)):
                if words[-j-1] not in bad_endings:
                    flag = -j
                    break
            txt = ' '.join(words[0:len(words)+flag])
        out.append(txt.replace('@@ ', ''))
    return out


def save_checkpoint(opt, model, infos, optimizer, histories=None, append=''):
    if len(append) > 0:
        append = '-' + append
    # if checkpoint_path doesn't exist
    if not os.path.isdir(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)
    checkpoint_path = os.path.join(opt.checkpoint_path, 'model%s.pth' %(append))
    torch.save(model.state_dict(), checkpoint_path)
    print("model saved to {}".format(checkpoint_path))
    optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer%s.pth' %(append))
    torch.save(optimizer.state_dict(), optimizer_path)
    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
        pickle_dump(infos, f)
    if histories:
        with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'%s.pkl' %(append)), 'wb') as f:
            pickle_dump(histories, f)


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']


def build_optimizer(params, opt):
    if opt.optim == 'rmsprop':
        return optim.RMSprop(params, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adagrad':
        return optim.Adagrad(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        return optim.SGD(params, opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdm':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay)
    elif opt.optim == 'sgdmom':
        return optim.SGD(params, opt.learning_rate, opt.optim_alpha, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optim == 'adam':
        return optim.Adam(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    elif opt.optim == 'adamw':
        return optim.AdamW(params, opt.learning_rate, (opt.optim_alpha, opt.optim_beta), opt.optim_epsilon, weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))
    

def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x,y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x,y: length_wu(x,y,alpha)
    if pen_type == 'avg':
        return lambda x,y: length_average(x,y,alpha)

def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return (logprobs / modifier)

def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


class NoamOpt(torch.optim.Optimizer):
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self, *args, **kwargs):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step(*args, **kwargs)
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)

    def state_dict(self):
        state_dict = self.optimizer.state_dict()
        state_dict['_step'] = self._step
        return state_dict

    def load_state_dict(self, state_dict):
        if '_step' in state_dict:
            self._step = state_dict['_step']
            del state_dict['_step']
        self.optimizer.load_state_dict(state_dict)

class ReduceLROnPlateau(torch.optim.Optimizer):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode, factor, patience, verbose, threshold, threshold_mode, cooldown, min_lr, eps)
        self.optimizer = optimizer
        self.current_lr = get_lr(optimizer)
        
    def step(self, *args, **kwargs):
        "Update parameters and rate"
        self.optimizer.step(*args, **kwargs)

    def scheduler_step(self, val):
        self.scheduler.step(val)
        self.current_lr = get_lr(self.optimizer)

    def state_dict(self):
        return {'current_lr':self.current_lr,
                'scheduler_state_dict': self.scheduler.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        if 'current_lr' not in state_dict:
            # it's normal optimizer
            self.optimizer.load_state_dict(state_dict)
            set_lr(self.optimizer, self.current_lr) # use the lr fromt the option
        else:
            # it's a schduler
            self.current_lr = state_dict['current_lr']
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
            self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
            # current_lr is actually useless in this case
    
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def __getattr__(self, name):
        return getattr(self.optimizer, name)
        
def get_std_opt(model, optim_func='adam', factor=1, warmup=2000):
    # return NoamOpt(model.tgt_embed[0].d_model, 2, 4000,
    #         torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    optim_func = dict(adam=torch.optim.Adam,
                      adamw=torch.optim.AdamW)[optim_func]
    return NoamOpt(model.d_model, factor, warmup,
            optim_func(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))