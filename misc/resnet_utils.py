import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import math

# SpatialAdaptiveAveragePooling
def start_end(a,b,c):
    return a*c//b, int(math.ceil(float((a+1)*c)/b))

def spatialAdaAvgPool(x, oH, oW):
    B, C, iH, iW  = x.size()
    output = Variable(x.data.new(B, C, oH, oW))
    for oh in xrange(oH):
        for ow in xrange(oW):
            i1, i2 = start_end(oh, oH, iH)

            j1, j2 = start_end(ow, oW, iW)

            output[:, :, oh, ow] = x[:,:,i1:i2,j1:j2].mean(2).mean(3).squeeze(3).squeeze(2)
    return output

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(2).mean(3).squeeze()
        att = spatialAdaAvgPool(x,att_size,att_size).squeeze().permute(1, 2, 0)

        return fc, att

