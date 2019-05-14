import os.path as osp
from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision import transforms
import pdb


#ROOT_PATH = '../materials/'

# tr: [1200, 20, 28, 28, 1]
# te: [423, 20, 28, 28, 1]

def Omniglot(setname='train', ROOT_PATH='../../materials', num=None):
    if setname == 'val':
        setname = 'test'
    path = osp.join(ROOT_PATH, 'omniglot', 'omni_{}.npy'.format(setname))
    data = np.load(path)
    data = np.reshape(data, [-1, 20, 28, 28, 1])
    data = torch.from_numpy(data)

    rand = torch.randperm(len(data))
    if num is not None:
        rand = rand[:num]
    data = data[rand]
    data = data.permute(0, 1, 4, 2, 3)
    print('=> {} data size: {}'.format(setname, data.size()))
    return data

