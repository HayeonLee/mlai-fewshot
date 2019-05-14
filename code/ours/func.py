import torch
import torch.nn as nn
import torch.distributions.kl as kl
import torch.distributions as dist
import torch.nn.functional as F
Normal = dist.Normal
from collections import OrderedDict

import pdb

#-----------------------------------------------------------------------#
#                               Sub Network                             #    
#-----------------------------------------------------------------------#
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, args):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch,  3, padding=1)
        nn.init.xavier_uniform_(self.conv.weight)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.01)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x) # output.size: [B, 32, w, h]
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x


class Convnet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=32, z_dim=32, args=None):
        super(Convnet, self).__init__()
        
        self.block1 = Block(x_dim, hid_dim, args)
        self.block2 = Block(hid_dim, hid_dim, args)
        self.block3 = Block(hid_dim, hid_dim, args)
        self.block4 = Block(hid_dim, z_dim, args)
        self.out_channels = 800
        if args.fc or (args.dropout_rate > 0):
            self.fc = nn.Linear(args.dim, args.fc_dim)
        else:
            self.fc = None
        if args.dropout_rate > 0:
            self.dropout = nn.Dropout(args.dropout_rate)
        else:
            self.dropout = None

    def forward(self, x, is_tr=True):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)

        if (self.dropout is not None) and is_tr:
            x = self.dropout(x)
        if self.fc is not None:
            x = self.fc(x)
        return x # [b, 1600]


#-----------------------------------------------------------------------#
#                               Loss and Metric                         #    
#-----------------------------------------------------------------------#
def dot_metric(a, b, l2norm=False):
    if l2norm:
        a = L2norm(a)
        b = L2norm(b)
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    '''
    for 5way 5shot 15query
    Input size: a [75, 1600], b [5,1600]
    Output size: [75, 5]
    '''
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2) # [75, 5]
    return logits


def proto_loss(shot, query, args):
    spt = gen_set(shot, args.shot, args.way)
    logits = euclidean_metric(query, spt)
    loss = F.cross_entropy(logits, args.label)

    output ={'loss': loss,
             'logits': logits,
             'shot': shot,
             'spt': spt,
             'query': query,
             'log': None}
    return output

#-----------------------------------------------------------------------#
#                               MISC                                    #    
#-----------------------------------------------------------------------#
def L2norm(x):
    # [batch, dim]
    return F.normalize(x, p=2, dim=1)


def gen_set(x, ns, nw):
    return x.reshape(ns, nw, -1).mean(0)
