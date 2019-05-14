import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import numpy as np
import torch
import pickle
import time


#ROOT_PATH = '../materials/'

# tr: [64, 600, 84, 84, 3]
# val: [16, 600, 84, 84, 3]
# test: [20, 600, 84, 84, 3]
def TieredImageNet(setname='train', ROOT_PATH='../../materials'):
    st = time.time()
    if setname == 'train':
        path = osp.join(ROOT_PATH,'tiered-imagenet',  'imtiered_' + setname + '-1.pt')
        data = pickle.load(open(path, 'rb'))
        path = osp.join(ROOT_PATH, 'tiered-imagenet', 'imtiered_' +  setname + '-2.pt')
        data += pickle.load(open(path, 'rb'))
    else:
        path = osp.join(ROOT_PATH, 'tiered-imagenet', 'imtiered_' +  setname + '.pt')
        print(path)
        data = pickle.load(open(path, 'rb'))
    dt = time.time() - st

    print('=> [{:.3f}min] {} data size: {}'.format(dt/60, setname, len(data)))
    return data

