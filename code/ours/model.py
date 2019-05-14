import torch
import torch.nn as nn
import torch.nn.functional as F

from func import *
from utils import *

import numpy as np
import time

import pdb
from copy import deepcopy


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        label = torch.arange(args.way).repeat(args.query)
        self.label = label.type(torch.cuda.LongTensor)
        self.args = args
        self.model = args.model
        self.loss = None
        # task type
        self.ns = args.shot
        self.nw = args.way
        self.nq = args.query

        # maml + proto
        self.update_step = args.update_step
        self.update_lr = args.update_lr

        self.xdim = args.xdim
        self.size = args.size

        self.test_bn = args.test_bn
        # self.en_num = args.en_num

        # network
        self.encoder = Convnet(x_dim=args.xdim, hid_dim=args.hdim, z_dim=args.zdim, args=args)
        self.test_encoder = Convnet(x_dim=args.xdim, hid_dim=args.hdim, z_dim=args.zdim, args=args)
        self.net = [self.encoder]

        self.set_loss(args)


    def sub_sampling(self, data):
        with torch.no_grad():
            data = data.view(self.ns, self.nw, self.xdim, self.size, self.size)
            subn = torch.arange(self.ns)[1:-1]
            subn = subn[torch.randperm(len(subn))[0]]
            rand = torch.randperm(self.ns)
            data_query = data[rand[:subn]].view(-1, self.xdim, self.size, self.size)
            data_shot = data[rand[subn:]].view(-1, self.xdim, self.size, self.size)
            label = torch.arange(self.nw).repeat(subn)
            label = label.type(torch.cuda.LongTensor)
            # print('subnum', subn)
            # print('sub data shot size:', data_shot.size())
            # print('sub data query size:', data_query.size())
        return data_shot, data_query, label


    def ensemble_sampling(self, data):
        with torch.no_grad():
            data = data.view(self.ns, self.nw, self.xdim, self.size, self.size)
            subn = torch.arange(self.ns)[1:-1]
            query_list = []
            shot_list = []
            label_list = []

            for i in range(self.en_num):
                n = subn[torch.randperm(len(subn))[0]] # random sub shot/sub query split
                rand = torch.randperm(self.ns) # shuffle shot 

                data_query = data[rand[:n]].view(-1, self.xdim, self.size, self.size)
                data_shot = data[rand[n:]].view(-1, self.xdim, self.size, self.size)

                label = torch.arange(self.nw).repeat(n)
                label = label.type(torch.cuda.LongTensor)
                query_list.append(data_query)
                shot_list.append(data_shot)
                label_list.append(label)

        return shot_list, query_list, label_list
                
    #-----------------------------------------------------------------------#
    #                            Forward functions                          #
    #-----------------------------------------------------------------------#
    def forward(self, shot, query, shot_inner, query_inner, is_tr=True):
        if is_tr:
            corrects = [0 for _ in range(self.update_step + 1)]  
            losses = [0 for _ in range(self.update_step + 1)]  

            weights = None

            # =================================================================================
            # Inner Loop
            # =================================================================================
            for i in range(self.update_step):
                # 1. run the i-th task and compute loss for k=0
                # sub_shot, sub_query, sub_label = self.sub_sampling(shot)
                spt_enc, qry_enc = self.forward_disc(shot_inner, query_inner, weights, self.encoder, is_tr)
                output = self.loss(spt_enc, qry_enc, self.label, self.encoder, last=False)
                weights = output['weights']
                corrects[i] = output['correct']
                losses[i] = output['loss'].item()
            shot_inner = None; query_inner = None; spt_enc = None; qry_enc = None; output = None

            # ================================================================================
            # Outer Loop: Meta-update
            # ================================================================================
            spt_enc, qry_enc = self.forward_disc(shot, query, weights, self.encoder, is_tr)
            output = self.loss(spt_enc, qry_enc, self.label, self.encoder, last=True)
            corrects[-1] = output['correct']
            losses[-1] = output['loss']
            output['correct'] = corrects
            output['loss'] = losses
        #elif self.ensemble:
        else:

            self.test_encoder.load_state_dict(self.encoder.state_dict())
            self.test_encoder = self.test_encoder.cuda()
            if self.test_bn:
                self.test_encoder.train()
            else:
                self.test_encoder.eval()
            # net = deepcopy(self.encoder)
            corrects = [0 for _ in range(self.update_step + 1)]  
            losses = [0 for _ in range(self.update_step + 1)]  
            weights = None

            # =========================================================================
            # Inner Loop
            # =========================================================================
            for i in range(self.update_step):
                # 1. run the i-th task and compute loss for k=0
                # sub_shots, sub_querys, sub_labels = self.ensemble_sampling(shot)
                spt_enc, qry_enc = self.forward_disc(shot_inner, query_inner, weights, self.test_encoder, is_tr)
                output = self.loss(spt_enc, qry_enc, self.label, self.test_encoder, last=False)
                weights = output['weights']
                corrects[i] = output['correct']
                losses[i] = output['loss'].item()
            shot_inner = None; query_inner = None; spt_enc = None; qry_enc = None; output = None

            # =======================================================================
            # Outer Loop: Meta-update
            # =======================================================================
            spt_enc, qry_enc = self.forward_disc(shot, query, weights, self.test_encoder, is_tr)
            output = self.loss(spt_enc, qry_enc, self.label, self.test_encoder, last=True)
            corrects[-1] = output['correct']
            losses[-1] = output['loss'].item()
            output['correct'] = corrects
            output['loss'] = losses

        return output


    def forward_disc(self, x_spt, x_qry, fast_weights, encoder, is_tr):
        if fast_weights is not None:
            self.update_weights(fast_weights, encoder)

        spt_enc = encoder(x_spt, is_tr)
        qry_enc = encoder(x_qry, is_tr)

        return spt_enc, qry_enc


    def update_weights(self, weights_dict, encoder):
        state_dict = encoder.state_dict()
        state_dict.update(weights_dict)
        encoder.load_state_dict(state_dict)


    def compute_weights(self, loss, encoder):
        named_weights = list(encoder.named_parameters())
        weights = [named_weights[i][1] for i in range(len(named_weights))]
        grad = torch.autograd.grad(loss, weights, allow_unused=True)

        weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, weights)))
        
        new_weights_dict = OrderedDict({})

        for i in range(len(named_weights)):
            new_weights_dict[named_weights[i][0]] = weights[i]
        named_weights = None; weights = None; grad = None 
        return new_weights_dict


    def our_loss(self, shot, query, label, encoder, last=False):
        spt = gen_set(shot, int(shot.size()[0]/self.nw), self.nw)
        logits = euclidean_metric(query, spt)

        loss = F.cross_entropy(logits, label)

        if not last:
            weights = self.compute_weights(loss, encoder)
        else:
            weights = None

        correct = count_acc(logits, label)

        # pred_q = F.softmax(logits, dim=1).argmax(dim=1)
        # correct = torch.eq(pred_q, label).sum().item()

        out = {'loss': loss,
               'correct': correct,
               'logits': logits, 
               'weights': weights,
               'shot': shot,
               'spt': spt,
               'query': query,
               }
        return out
    #-----------------------------------------------------------------------#
    #                               Set functions                           #
    #-----------------------------------------------------------------------#
    def set_loss(self, args):
        if args.model == 'proto':
            self.loss = proto_loss
        elif args.model =='ours':
            self.loss = self.our_loss
        else:
            assert False, 'loss error'

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
