import argparse
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader

from data.mini_imagenet import MiniImageNet
#from loader import CategoriesSampler
from model import Model
from utils import *
from data import *
from collections import OrderedDict

from pmodel import Model as PModel
import tensorboard_logger as tb_logger
import logging
import logging.handlers
from train import test
import json
import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='tiered-5w5s-1600-uplr0.01-upst1-st60-dr0-wd0.5-G5-1')
    parser.add_argument('--ckpt', type=str, default='epoch-last.pth')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--test_batch', type=int, default=600)
    # parser.add_argument('--test_bn', type=str2bool, default='False')
    parser.add_argument('--fname', default='db')
    # parser.add_argument('--way', type=int, default=10)
    # parser.add_argument('--shot', type=int, default=5)
    # parser.add_argument('--query', type=int, default=20)


    test_args = parser.parse_args()
    with open('../../save/{}/args.txt'.format(test_args.load), 'r') as f:
        adict = json.load(f)
        args = Bunch(adict)
        args.update(test_args)

    # pprint(vars(args))
    
    set_gpu(args.gpu)
    ars = set_path(args)
    pprint(vars(args))

    dataset = get_dataset(args.data, args.root_path, ['test'])
    print('load dataset')

    label = torch.arange(args.way).repeat(args.query)
    label = label.type(torch.cuda.LongTensor)

# ======================================================================
# Load our model
# ======================================================================
    our_model = Model(args)
    our_model.set_cuda()

    args.load = osp.join('../../save', args.load, args.ckpt)
    print('load: {}'.format(args.load))
    # our_model.load_state_dict(torch.load(args.load))
    test(our_model, dataset['test'], args, 'last', None, 'test')


