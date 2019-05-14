import torch
import torch.nn as nn
import torch.nn.functional as F

from func import *

import numpy as np
import time

import pdb


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        label = torch.arange(args.way).repeat(args.query)
        args.label = label.type(torch.cuda.LongTensor)
        self.args = args
        # task type
        self.ns = args.shot
        self.nw = args.way
        self.nq = args.query

        # network
        self.encoder = Convnet(x_dim=args.xdim, hid_dim=args.hdim, z_dim=args.zdim, args=args)
        self.net = [self.encoder]

        self.set_loss(args)

    #-----------------------------------------------------------------------#
    #                            Forward functions                          #
    #-----------------------------------------------------------------------#
    def forward(self, shot, query):
        shot = self.encoder(shot)
        query = self.encoder(query)

        output = self.loss(shot, query, self.args)

        return output
    #-----------------------------------------------------------------------#
    #                               Set functions                           #
    #-----------------------------------------------------------------------#
    def set_loss(self, args):
        if args.model == 'proto':
            self.loss = proto_loss

    def set_cuda(self):
        for net in self.net:
            net = net.cuda()

    def set_train(self):
        for net in self.net:
            net = net.train()

    def set_eval(self):
        for net in self.net:
            net = net.eval()

    def get_parameters(self):
        params = []
        for net in self.net:
            params.append(net.parameters())
        return params 