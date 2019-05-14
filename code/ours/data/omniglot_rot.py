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

def OmniglotRot(setname='train', ROOT_PATH='../../materials', num=None):
    path = osp.join(ROOT_PATH, 'rotated-omniglot', 'omni_' + setname + '_rot.npy')
    print(path)
    data = np.load(path)
    #data = np.reshape(data, [-1, 20, 28, 28, 1])
    data = torch.from_numpy(data)

    if num is not None:
        rand = torch.randperm(len(data))
        rand = rand[:num]
        data = data[rand]
    data = data.permute(0, 1, 4, 2, 3)
    print('=> {} data size: {}'.format(setname, data.size()))
    return data

