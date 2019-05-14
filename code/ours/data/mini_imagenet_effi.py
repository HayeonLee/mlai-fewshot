import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import numpy as np
import torch


#ROOT_PATH = '../materials/'

# tr: [64, 600, 84, 84, 3]
# val: [16, 600, 84, 84, 3]
# test: [20, 600, 84, 84, 3]
def MiniImageNet(setname='train', ROOT_PATH='../../materials'):
    path = osp.join(ROOT_PATH, 'mini-imagenet', setname + '.npy')
    data = np.load(path)
    # data = data.reshape(-1, 600, 84, 84, 3)
    # data = data.transpose(0, 1, 4, 2, 3)
    # print('=> {} data size: {}'.format(setname, np.shape(data)))
    data = torch.from_numpy(data).float() # [38400, 84, 84, 3]
    data = data.view(-1, 600, 84, 84, 3).contiguous().permute(0, 1, 4, 2, 3).cuda() # [64, 600, 3, 84, 84]
    print('=> {} data size: {}'.format(setname, data.size()))
    return data

