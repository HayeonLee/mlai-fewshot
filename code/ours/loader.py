import torch
import numpy as np
from collections import OrderedDict

from torch.utils.data import DataLoader
from data.mini_imagenet import MiniImageNet
import pdb
#from samplers import CategoriesSampler

def get_loaders(args, debug=False):
    loaders = OrderedDict()
    batch = {'train': args.task_num, 'val': args.val_batch, 'test': args.test_batch}
    for mode in ['train', 'val', 'test']:
        dataset = MiniImageNet(mode, args.root_path)
        # n_batch=100, n_cls=way, n_per=shot+query
        sampler = CategoriesSampler(dataset.label, batch[mode],
                                    args.way * 2, args.shot + args.query, debug)
        loader = DataLoader(dataset=dataset, batch_sampler=sampler,
                                  num_workers=8, pin_memory=True)
        loaders[mode] = loader
    return loaders


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, debug=False):

        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.debug = debug

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            # batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            batch = self.sampling(classes[:(self.n_cls//2)])
            batch2 = self.sampling(classes[(self.n_cls//2):])
            batch = torch.cat((batch, batch2))
            # for c in classes:
            #     l = self.m_ind[c]
            #     pos = torch.randperm(len(l))[:self.n_per]
            #     batch.append(l[pos])
            # batch = torch.stack(batch).t().reshape(-1)

            yield batch

    def sampling(self, classes_per_epi):
        batch = []
        for c in classes_per_epi:
            l = self.m_ind[c]
            pos = torch.randperm(len(l))[:self.n_per]
            batch.append(l[pos])
        batch = torch.stack(batch).t().reshape(-1)

        return batch
