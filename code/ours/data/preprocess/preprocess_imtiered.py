import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import numpy as np
import torch
import pickle



#ROOT_PATH = '../materials/'

# tr: [64, 600, 84, 84, 3]
# val: [16, 600, 84, 84, 3]
# test: [20, 600, 84, 84, 3]
def TieredImageNet(setname='train', ROOT_PATH='/st1/hayeon/materials'):
    path = osp.join(ROOT_PATH, 'tiered-imagenet', 'imtiered_' + setname + '.npy')
    data = np.load(path, encoding='latin1') # [160, 1300, 84, 84, 3]
    new_data = []
    for c in range(len(data)):
        images = data[c]
        for i in range(len(images)):
            image = images[i] # np.array (84, 84, 3)
            #image = torch.from_numpy(image).permute(2, 0, 1) / 255.0 # [3, 84, 84]
            image = torch.from_numpy(image).float() # [3, 84, 84]
            image = image.permute(2, 0, 1) / 255.0 # [3, 84, 84]
            if i == 0:
                processed = image.unsqueeze(0)
            else:
                processed = torch.cat((processed, image.unsqueeze(0)))
            if i % 100 == 0:
                print('{} {} class {}th image processed..'.format(setname, c, i))
            # processed size: [1300, 3, 84, 84]
        new_data.append(processed)
        if c == 160:
            with open('/st1/hayeon/materials/tiered-imagenet/imtiered_train-1.pt'.format(setname), 'wb') as f:
                pickle.dump(new_data, f)
            print('==> train-1 saved')
            new_data = []
            
        if c % 10 == 0:
            print('=> {} {} class done'.format(setname, c))
    #torch.save(new_data, '/st1/hayeon/materials/tiered-imagenet/imtiered_{}.pt'.format(setname))
    with open('/st1/hayeon/materials/tiered-imagenet/imtiered_{}-2.pt'.format(setname), 'wb') as f:
        pickle.dump(new_data, f)
    return new_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='train')
args = parser.parse_args()

TieredImageNet(args.name)

