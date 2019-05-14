import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import pdb

#ROOT_PATH = '../materials/'

# tr: [64, 600, 84, 84, 3]
# val: [16, 600, 84, 84, 3]
# test: [20, 600, 84, 84, 3]

class MiniImageNet(Dataset):

    def __init__(self, setname, ROOT_PATH, data_aug=False):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        trans_list = []
        trans_list.append(transforms.Resize((84, 84)))
        print('data_aug', data_aug, 'setname', setname)
        if data_aug and setname == 'train':
            # trans_list.append(transforms.ToPILImage())
            trans_list.append(transforms.RandomCrop(84, 4))
            trans_list.append(transforms.RandomHorizontalFlip(0.5))
            trans_list.append(transforms.RandomRotation(20))
        # trans_list.append(transforms.Resize(84))
        # trans_list.append(transforms.CenterCrop(84))
        trans_list.append(transforms.ToTensor())
        trans_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]))
       
        self.transform = transforms.Compose(trans_list)
        # self.transform = transforms.Compose([
        #     transforms.Resize(84),
        #     transforms.CenterCrop(84),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

