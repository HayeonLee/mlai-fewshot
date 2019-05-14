import os.path as osp
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import time

#ROOT_PATH = '../materials/'

# tr: [64, 600, 84, 84, 3]
# val: [16, 600, 84, 84, 3]
# test: [20, 600, 84, 84, 3]

def preprocessing(setname='train', ROOT_PATH='../../../materials', data_aug=False):
    csv_path = osp.join(ROOT_PATH, setname + '.csv')
    lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

    data = []
    label = []
    lb = -1

    wnids = []

    for l in lines:
        name, wnid = l.split(',')
        path = osp.join(ROOT_PATH, 'images', name)
        if wnid not in wnids:
            wnids.append(wnid)
            lb += 1
        data.append(path)
        label.append(lb)

    trans_list = []
    print('data_aug', data_aug, 'setname', setname)
    if data_aug and setname == 'train':
        trans_list.append(transforms.RandomResizedCrop(84))
        trans_list.append(transforms.RandomHorizontalFlip(0.5))
        trans_list.append(transforms.RandomRotation(20))
    else:
        trans_list.append(transforms.Resize((84, 84)))
    trans_list.append(transforms.ToTensor())
    trans_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]))
   
    transform = transforms.Compose(trans_list)

    st = time.time()
    for i in range(lb+1):
        for j in range(600):
            image = Image.open(data[i * 600 + j]).convert('RGB')
            for k in range(1):
                if k == 0:
                    trans_img = transform(image).unsqueeze(0)
                else:
                    trans2 =  transform(image).unsqueeze(0) 
                    trans_img = torch.cat((trans_img, trans2)) # [50, 3, 84, 84]
            if j == 0:
                cls_img = trans_img.unsqueeze(0)
            else:
                cls2 = trans_img.unsqueeze(0)
                cls_img = torch.cat((cls_img, cls2)) # [600, 50, 3, 84, 84]
        dt = ( time.time() - st) / 3600
        total = (dt / (i+1)) * 64
        print('Test {}/{} class done: {}, time: {:.2f}/{:.2f}'.format(i+1, lb+1, cls_img.size(), dt, total))
        if i == 0:
            data_ten = cls_img.unsqueeze(0)
        else:
            data2 = cls_img.unsqueeze(0)
            data_ten = torch.cat((data_ten, data2)) # [64, 600, 50, 3, 84, 84]
    torch.save(data_ten, 'test.pt')


preprocessing(setname='test', ROOT_PATH='../../../materials', data_aug=False)
