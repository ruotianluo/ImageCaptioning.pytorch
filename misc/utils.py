from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import misc.resnet as resnet
import os
import random

def build_cnn(opt):
    net = getattr(resnet, opt.cnn_model)()
    if vars(opt).get('start_from', None) is None and vars(opt).get('cnn_weight', '') != '':
        net.load_state_dict(torch.load(opt.cnn_weight))
    net = nn.Sequential(\
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool,
        net.layer1,
        net.layer2,
        net.layer3,
        net.layer4)
    if vars(opt).get('start_from', None) is not None:
        net.load_state_dict(torch.load(os.path.join(opt.start_from, 'model-cnn.pth')))
    return net

def prepro_images(imgs, data_augment=False):
    # crop the image
    h,w = imgs.shape[2], imgs.shape[3]
    cnn_input_size = 224

    # cropping data augmentation, if needed
    if h > cnn_input_size or w > cnn_input_size:
        if data_augment:
          xoff, yoff = random.randint(0, w-cnn_input_size), random.randint(0, h-cnn_input_size)
        else:
          # sample the center
          xoff, yoff = (w-cnn_input_size)//2, (h-cnn_input_size)//2
    # crop.
    imgs = imgs[:,:, yoff:yoff+cnn_input_size, xoff:xoff+cnn_input_size]

    return imgs

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
                txt = txt + ix_to_word[str(ix)]
            else:
                break
        out.append(txt)
    return out

def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]
        input = to_contiguous(input).view(-1, input.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)
        output = - input.gather(1, target) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)