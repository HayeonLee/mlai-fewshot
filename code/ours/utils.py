import os
import os.path as osp
import shutil
import time
import pprint
import logging
import logging.handlers

import torch
import torch.nn as nn
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

import pdb
from loader import get_loaders
from data.omniglot_effi import Omniglot
from data.omniglot_rot import OmniglotRot
from data.mini_imagenet_effi import MiniImageNet
from data.tiered_imagenet import TieredImageNet
import argparse

from torchvision import transforms


def get_args(argv=None):
    parser = argparse.ArgumentParser()
    # Configures
      # train/ test
    parser.add_argument('--data', default='mini')
    parser.add_argument('--max_episode', type=int, default=60000)
    parser.add_argument('--save_episode', type=int, default=10000)
    parser.add_argument('--test_epi', type=int, default=100)
    # parser.add_argument('--test_epi', type=str2list, default="20000, 60000")
    parser.add_argument('--test_batch', type=int, default=600)
    parser.add_argument('--val_batch', default=40)
      # task type
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--way', type=int, default=5)
      # algorithms
    parser.add_argument('--test_bn', type=str2bool, default=False)
      # model
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--hdim', type=int, default=32)
    parser.add_argument('--zdim', type=int, default=32)
    parser.add_argument('--xdim', type=int, default=3)
    parser.add_argument('--size', type=int, default=84)
    parser.add_argument('--fc', type=str2bool, default=False)
    parser.add_argument('--fc_dim', type=int, default=800)
      # loss
    parser.add_argument('--model', default='proto')

      # misc
    parser.add_argument('--fname', default=None)
    parser.add_argument('--load', default=None)
    parser.add_argument('--path', default='../..')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--speed', type=str2bool, default=False)
       # maml + proto
    parser.add_argument('--task_num', type=int, default=5)
    # parser.add_argument('--en_num', type=int, default=1)
    parser.add_argument('--update_lr', type=float, default=0.02)
    parser.add_argument('--update_step', type=int, default=1)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--lr_decay', type=int, default=0.5)
       # handling overfitting
    #parser.add_argument('--dropout', type=str2bool, default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    #parser.add_argument('--data_aug', type=str2bool, default=False)
    parser.add_argument('--w_decay', type=float, default=0)

    args = parser.parse_args(argv)
    return args


def get_dataset(data_name, root_path, mode_list=['train', 'test']):
    dataset = {}
    for mode in mode_list:
        if data_name == 'omni':
            dataset[mode] = Omniglot(mode, root_path)
        elif data_name == 'omni_rot':
            dataset[mode] = OmniglotRot(mode, root_path)
        elif data_name == 'mini':
            dataset[mode] = MiniImageNet(mode, root_path) # [64, 600, 3, 84, 84]
        elif data_name == 'tiered':
            dataset[mode] = TieredImageNet(mode, root_path) # [64, 600, 3, 84, 84]

    return dataset


def to_shot_cls(data, dim, size):
    data = data.permute(1, 0, 2, 3, 4).contiguous().view(-1, dim, size, size) # [nshot * way, 3, 84, 84]
    return data


def sample_instances(cls, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rand = torch.randperm(len(cls)).to(device)
    data_shot = cls[rand[:args.shot]].unsqueeze(0)
    data_query = cls[rand[args.shot:args.shot + args.query]].unsqueeze(0) # [nquery, 3, 84, 84]
    return data_shot, data_query


def randsamp_tiered(data, args, shuf_cls=True, shuf_cls2=True):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if shuf_cls:
            rand = torch.randperm(len(data)).to(device)                                                                                    
            for i in range(args.way):
                cls = data[rand[i]].to(device)
                _shot, _query = sample_instances(cls, args)
                if i == 0:
                    shot = _shot # [1, 5, 3, 84, 84]
                    query = _query # [1, 15, 3, 84, 84]
                else:
                    shot = torch.cat((shot, _shot))
                    query = torch.cat((query, _query))
            shot = to_shot_cls(shot, args.xdim, args.size)
            query = to_shot_cls(query, args.xdim, args.size)

        if shuf_cls2:
            for i in range(args.way, 2 * args.way):
                cls = data[rand[i]].to(device)
                _shot, _query = sample_instances(cls, args)
                if i == args.way:
                    shot2 = _shot
                    query2 = _query
                else:
                    shot2 = torch.cat((shot2, _shot))
                    query2 = torch.cat((query2, _query))
            shot2 = to_shot_cls(shot2, args.xdim, args.size)
            query2 = to_shot_cls(query2, args.xdim, args.size)
        else:
            shot2 = None; query2 = None
    return shot, query, shot2, query2


def rand_sampling(data, args, shuf_cls=True, shuf_cls2=False):
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        picked_cls2 = None              
        if shuf_cls:
            rand = torch.randperm(len(data)).to(device)                                                                                    
            # pick random N way cls
            picked_cls = data[rand[0]].to(device).unsqueeze(0)

            for i in range(1, args.way):
                picked_cls = torch.cat((picked_cls, data[rand[i]].to(device).unsqueeze(0)))

            if shuf_cls2:
                picked_cls2 = data[rand[args.way]].to(device).unsqueeze(0)

                for i in range(args.way+1, 2 * args.way):
                    picked_cls2 = torch.cat((picked_cls2, data[rand[i]].to(device).unsqueeze(0)))  

            data = None
            rand = None
            
        else:
            picked_cls = data

        ndata = picked_cls.size()[1]

        cls = picked_cls[0]
        rand = torch.randperm(ndata).to(device)
        data_shot = cls[rand[:args.shot]].unsqueeze(0)
        data_query = cls[rand[args.shot:args.shot + args.query]].unsqueeze(0) # [nquery, 3, 84, 84]
        
        for cls in picked_cls[1:]:
            rand = torch.randperm(ndata).to(device)
            data_shot_ = cls[rand[:args.shot]].unsqueeze(0)
            data_query_ = cls[rand[args.shot:args.shot + args.query]].unsqueeze(0) # [nquery, 3, 84, 84]
            data_shot = torch.cat((data_shot, data_shot_))
            data_query = torch.cat((data_query, data_query_))

        rand = None
        picked_cls = None

        data_shot = to_shot_cls(data_shot, args.xdim, args.size)
        data_query = to_shot_cls(data_query, args.xdim, args.size)
        # data_shot = data_shot.permute(1, 0, 2, 3, 4).contiguous().view(-1, args.xdim, args.size, args.size) # [nshot * way, 3, 84, 84]
        # data_query = data_query.permute(1, 0, 2, 3, 4).contiguous().view(-1, args.xdim, args.size, args.size) # [nquery * way, 3, 84, 84]
    return data_shot, data_query, picked_cls2


# def rand_sampling(data, args, shuf_cls=True, shuf_cls2=False):
#     with torch.no_grad():
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         picked_cls2 = None              
#         if shuf_cls:
#             rand = torch.randperm(len(data))                                                                                    
#             # pick random N way cls
#             picked_cls = data[rand[0]].unsqueeze(0)

#             for i in range(1, args.way):
#                 picked_cls = torch.cat((picked_cls, data[rand[i]].unsqueeze(0)))

#             if shuf_cls2:
#                 picked_cls2 = data[rand[args.way]].unsqueeze(0)

#                 for i in range(args.way+1, 2 * args.way):
#                     picked_cls2 = torch.cat((picked_cls2, data[rand[i]].unsqueeze(0)))  

#             data = None
#             rand = None
            
#         else:
#             picked_cls = data

#         ndata = picked_cls.size()[1]

#         cls = picked_cls[0]
#         rand = torch.randperm(ndata).to(device)
#         data_shot = cls[rand[:args.shot]].unsqueeze(0)
#         data_query = cls[rand[args.shot:args.shot + args.query]].unsqueeze(0) # [nquery, 3, 84, 84]
        
#         for cls in picked_cls[1:]:
#             rand = torch.randperm(ndata).to(device)
#             data_shot_ = cls[rand[:args.shot]].unsqueeze(0)
#             data_query_ = cls[rand[args.shot:args.shot + args.query]].unsqueeze(0) # [nquery, 3, 84, 84]
#             data_shot = torch.cat((data_shot, data_shot_))
#             data_query = torch.cat((data_query, data_query_))

#         rand = None
#         picked_cls = None

#         data_shot = to_shot_cls(data_shot, args.xdim, args.size)
#         data_query = to_shot_cls(data_query, args.xdim, args.size)
#         # data_shot = data_shot.permute(1, 0, 2, 3, 4).contiguous().view(-1, args.xdim, args.size, args.size) # [nshot * way, 3, 84, 84]
#         # data_query = data_query.permute(1, 0, 2, 3, 4).contiguous().view(-1, args.xdim, args.size, args.size) # [nquery * way, 3, 84, 84]
#     return data_shot, data_query, picked_cls2


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

  def update(self, args):
    adict = args.__dict__
    self.__dict__.update(adict)


def set_path(args):
    path = args.path
    fname = args.fname
    args.save_path = osp.join(path, 'save', fname)
    args.tb_path = osp.join(path, 'runs', fname)
    args.log_path = osp.join(path, 'log', fname + '.log')
    args.test_log_path = osp.join(path, 'log', 'test', fname + '.log')
    args.root_path = os.path.join(args.path, 'materials')

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.tb_path):
        os.mkdir(args.tb_path)
    return args


def save_model(model, args, name):
    torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))


def str2list(x):
    my_list = [int(item) for item in x.split(',')]
    return my_list


def str2bool(x):
    if x in ['True', 'true', True]:
        return True
    elif x in ['False', 'false', False]:
        return False
    assert False, 'str2bool error'


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def set_logger(args, name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fileH = logging.FileHandler(path)
    streamH = logging.StreamHandler()
    logger.addHandler(streamH)
    logger.addHandler(fileH)

    return logger

def ensure_test_log(args):
    path = '../../log/test/{}w{}s_{}_{}_MAXE_{}'.format(
                                            args.train_way,
                                            args.shot,
                                            args.norm,
                                            args.net,
                                            args.max_epoch)
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            f = open(path, 'w')
            return f
    else:
        assert False, 'log path wrong'


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

#### Visualization ####
csample = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525', 'red', 'black', 'green', 'blue', 'yellow']
ssample = ['o', 'v', 's', 'd', '>']


def _preprocess(x, way=None, n=None, clu_idx=None, is_spt=False, is_shot=False):
    '''
    Convert input (size: [instances * way, ...], type: torch, GPU) 
    into   output (size: [way * instances, ...], type: numpy, CPU)
    '''
    if way is not None:
        x = x.contiguous().view(n, way, -1).permute(1, 0, 2)
        x = x.contiguous().view(n*way, -1)
    x = x.view(x.size(0), -1).cpu().detach().numpy()

    if clu_idx is not None:
        color = _marker(x, way, clu_idx, csample) 
        shape = _marker(x, way, None, ssample) 
    else:
        color = _marker(x, way, None, csample, n=n) 
        if is_spt:
            s = (5,1)
        elif is_shot:
            s = 'd'
        else:
            s = 'o'
        shape = _marker(x, way, None, [s]*way, n=n)  
    return x, color, shape

def _tsne(q, spt, shot=None, w=5, n=15):
    '''
    Reduce dimension sizes to 2 for the visualization
    '''
    if spt is not None:
        q = np.concatenate((q, spt))
    if shot is not None:
        q = np.concatenate((q, shot))

    tsne = TSNE(n_components=2, early_exaggeration=30, random_state=0)
    q = tsne.fit_transform(q)
    if (spt is not None) and (shot is not None):
        return q[:n * w], q[n * w:(n+1)*w], q[(n+1)*w:]
    elif (spt is None) and (shot is None):
        return q, None, None
    else:
        return q[:n * w], q[n * w:], None

def _marker(x, w, idx, sample, n=15):
    '''
    Set colors or shapes of query instances depending on the cluster set
    '''
    marker = [None] * x.shape[0]
    if idx is None:
        for i in range(w):
            for j in range(n):
                marker[i*n + j] = sample[i]
    else:
        idx = idx.contiguous().view(n, w).permute(1, 0).contiguous().view(-1)
        for i in range(w):
            for j in torch.nonzero(idx == i):
                marker[j] = sample[i]
    return marker

def _scatter(x, w, n, c, s, m):
    for i in range(w):
        for j in range(n):
            plt.scatter(x[n*i+j, 0], x[n*i+j,1], c=c[n*i+j], s=s, marker=m[n*i+j])


def _debug_sample(loader, p, n=4):
    sample = []
    for i, batch in enumerate(loader):
        if i >= 4:
            break
        data, _ = [_.cuda() for _ in batch]
        sample.append([data[:p], data[p:]])
    return sample
    

def debug_sample(args):
    db_loaders = get_loaders(args, debug=True)
    sample = {'train': _debug_sample(db_loaders['train'], args.way * args.shot),
              'val': _debug_sample(db_loaders['test'], args.way *args.shot)}
    return sample


def scatter_points(qry, spt, shot=None, way=5, qry_n=15, shot_n=5, tsne=False, clu_idx=None):
    qry, qcolor, qshape = _preprocess(qry, way=way, n=qry_n, clu_idx=None) 
    spt, scolor, sshape = _preprocess(spt, way=way, n=1, clu_idx=None, is_spt=True) 
    if shot is not None:
        shot, shot_color, shot_shape = _preprocess(shot, way=way, n=shot_n, clu_idx=None, is_spt=False, is_shot=True) 
    if tsne:
        qry, spt, shot = _tsne(qry, spt, shot, way, qry_n)
    _scatter(qry, way, qry_n, qcolor, 250, qshape)
    _scatter(spt, way, 1, scolor, 1000, sshape)
    if shot is not None:
        _scatter(shot, way, shot_n, shot_color, 500, shot_shape)


def draw_debug_image(model, sample, args, epoch, logger):
    label = torch.arange(args.way).repeat(args.query)
    label = label.type(torch.cuda.LongTensor)
    plt.clf()
    fig = plt.figure(figsize=(20,10))
    #model.set_eval()
    result = []
    n = len(sample['train'])
    for i, mode in enumerate(['train', 'val']):
        for j, (shot, query) in enumerate(sample[mode]):
            o = model(shot, query)

            acc = count_acc(o['logits'], label)
            if logger is not None:
                logger.debug('draw loss={:.4f}, acc={:.4f}'.format(o['loss'].item(), acc))

            ax = plt.subplot(2, n, i*n + j+1)
            ax.set_xticks([])
            ax.set_yticks([])
            #plt.tight_layout()
            ax.set_title('{}_{}th_acc_{:.4f}'.format(mode, j+1, acc))
            #ax.axis('off')
            scatter_points(o['query'], o['spt'], o['shot'], way=args.way, qry_n=15, shot_n=5, tsne=args.tsne, clu_idx=None)
    spath = '{}/db_{}_ep{}.png'.format('/st1/hayeon/save/images', args.fname, epoch)
    if logger is not None:
        logger.debug(spath)
    plt.savefig(spath)
    plt.close()


def draw_debug_image_one(o, args, epoch, logger):
    label = torch.arange(args.way).repeat(args.query)
    label = label.type(torch.cuda.LongTensor)
    plt.clf()
    fig = plt.figure(figsize=(5,5))
    #model.set_eval()

    acc = count_acc(o['logits'], label)
    if logger is not None:
        logger.debug('draw loss={:.4f}, acc={:.4f}'.format(o['loss'].item(), acc))

    ax = plt.subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
            #plt.tight_layout()
    ax.set_title('{}th_acc_{:.4f}'.format(epoch, acc))
            #ax.axis('off')
    scatter_points(o['query'], o['spt'], o['shot'], way=args.way, qry_n=15, shot_n=5, tsne=args.tsne, clu_idx=None)
    
    spath = '{}/test_{}_ep{}.png'.format('/st1/hayeon/save/images', args.fname, epoch)
    if logger is not None:
        logger.debug(spath)


def draw_debug_image_one(o, args, epoch, logger):
    label = torch.arange(args.way).repeat(args.query)
    label = label.type(torch.cuda.LongTensor)
    plt.clf()
    fig = plt.figure(figsize=(5,5))
    #model.set_eval()

    acc = count_acc(o['logits'], label)
    if logger is not None:
        logger.debug('draw loss={:.4f}, acc={:.4f}'.format(o['loss'].item(), acc))

    ax = plt.subplot(1, 1, 1)
    ax.set_xticks([])
    ax.set_yticks([])
            #plt.tight_layout()
    ax.set_title('{}th_acc_{:.4f}'.format(epoch, acc))
            #ax.axis('off')
    scatter_points(o['query'], o['spt'], o['shot'], way=args.way, qry_n=15, shot_n=5, tsne=args.tsne, clu_idx=None)
    
    spath = '{}/test_{}_ep{}.png'.format('/st1/hayeon/save/images', args.fname, epoch)
    if logger is not None:
        logger.debug(spath)
    plt.savefig(spath)
    plt.close()


class Debug(object):
    def __init__(self, args, logger, num=3):
        self.args = args
        self.logger = logger
        self.loaders = get_loaders(args, debug=True)
        self.model = None
        self.epoch = None
        self.num = 3
        self.fname = args.fname
        label = torch.arange(args.way).repeat(args.query)
        self.label = label.type(torch.cuda.LongTensor)
        self.p = args.shot * args.way


    def draw_images(self, model, epoch):
        self.epoch = epoch

        model.set_train()
        self.forward(model, self.loaders['train'], 'train')

        model.set_eval()
        self.forward(model, self.loaders['test'], 'test')

    def forward(self, model, loader, mode):
        plt.clf()
        fig = plt.figure(figsize=(20,10))
        for i, batch in enumerate(loader):
            data, _ = [_.cuda() for _ in batch]
            data_shot, data_query = data[:self.p], data[self.p:]
            
            o = model(data_shot, data_query) # logits.size: [75, 5]
            loss = o['loss']
            # loss, logits,  _, _, _, trip_log = model(data_shot, data_query) # logits.size: [75, 5]

            acc = count_acc(o['logits'], self.label)            
            if self.logger is not None:
                self.logger.debug('draw loss={:.4f}, acc={:.4f}'.format(o['loss'].item(), acc))

            ax = plt.subplot(2, 4, (i % 8 + 1))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('{}_{}th_acc_{:.4f}'.format(mode, (i % 8 + 1), acc))
            scatter_points(o['query'], o['spt'], o['shot'], way=5, qry_n=15, shot_n=5, tsne=True, clu_idx=None)

            if ((i + 1) % 8) == 0:
                spath = '{}/{}_e{}_{}_{}.png'.format('/st1/hayeon/save/images', self.fname, self.epoch, mode, int((i+1)/8))
                if self.logger is not None:
                    self.logger.debug(spath)
                else:
                    print(spath)
                plt.savefig(spath)
                plt.clf()
            if i > self.num * 8:
                plt.close()
                break
