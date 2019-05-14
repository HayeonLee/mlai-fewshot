import argparse
import os
import json
from collections import OrderedDict

import torch
import tensorboard_logger as tb_logger
import logging
import logging.handlers
import pdb

from model import Model
#from loader import get_loaders
from utils import *
from data import *
import sys


# def adjust_learning_rate(args, optimizer, epoch):
#     if (epoch * args.task_num < 62000):
#         if (epoch + 1) // args.step_size == 0:
#             learning_rate = optimizer.param_groups[0]['lr'] * args.lr_decay
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = learning_rate
#     else:
#         if (epoch + 1) // args.step_size == 0:
#             learning_rate = optimizer.param_groups[0]['lr'] * args.lr_decay2
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = learning_rate


def main():
    args = get_args(sys.argv[1:])

    # Set GPU and paths
    set_gpu(args.gpu)
    args = set_path(args)
   
    # Save configures
    with open('{}/args.txt'.format(args.save_path), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    pprint(vars(args))
    
    # Set loggers
    tb_logger.configure(args.tb_path, flush_secs=5)

    logger = set_logger(args, 'train', args.log_path)
    args.logger = logger
    logger.debug(vars(args))


    # Get data loaders
    # loaders = get_loaders(args)
    dataset = get_dataset(args.data, args.root_path)

    # Set model and optimizer 
    model = Model(args)
    model.set_cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.w_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    max_acc = 0.0
    min_loss = 1000000.0

    max_epoch = int(args.max_episode / args.task_num ) + 1

    # Start training
    timer = Timer()
    for epoch in range(1, max_epoch+1): # max_epoch: 200 or 600
        # adjust_learning_rate(args, optimizer, epoch)
        lr_scheduler.step()

        tl, ta = train(model, dataset['train'], args, epoch * args.task_num, optimizer, logger)
        # tl, ta = train(model, loaders['train'], args, epoch, optimizer, logger)

        # Save checkpoints
        save_model(model, args, 'epoch-last')
        if (epoch * args.task_num) % args.save_episode == 0:
            save_model(model, args, 'epi-{}'.format(epoch * args.task_num))

        if epoch % 10 == 0:
            logger.debug('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / max_epoch)))

        # Test (off when speed up mode)
        #if (not args.speed) and (epoch * 100 in args.test_epi):
        if (epoch * args.task_num) % args.test_epi == 0: # per 1000 episode
            timer_te = Timer()
            tel, tea = test(model, dataset['test'], args, epoch * args.task_num, logger)
            # test(model, loaders['test'], args, epoch * args.task_num, logger)
            if tea > max_acc:
                max_acc = tea
                save_model(model, args, 'max_acc')
            if tel < min_loss:
                min_loss = tel
                save_model(model, args, 'min_loss')
            logger.debug('Test time:{}'.format(timer_te.measure()))

    # if not args.speed:
    #     test(model, loaders['test'], args, 'max_acc', logger, sample)

def train(model, dataset, args, episode, optimizer, logger):

    model.set_train()

    tl = Averager()
    ta = Averager()
    tl_inner = Averager()
    ta_inner = Averager()

    label = torch.arange(args.way).repeat(args.query)
    label = label.type(torch.cuda.LongTensor)

    corrects = [0 for _ in range(args.update_step + 1)]  # self.update_step: 5
    loss = 0

    optimizer.zero_grad()
    for i in range(1, args.task_num + 1):
        if args.data == 'tiered':
            data_shot, data_query, data_shot2, data_query2 = randsamp_tiered(dataset, args)
        else:
            data_shot, data_query, picked_cls2 = rand_sampling(dataset, args, shuf_cls=True, shuf_cls2=True)
            data_shot2, data_query2, _ = rand_sampling(picked_cls2, args, shuf_cls=False, shuf_cls2=False)
            picked_cls2 = None

        o = model(data_shot, data_query, data_shot2, data_query2) # logits.size: [75, 5]

        tl.add(o['loss'][-1].item())
        ta.add(o['correct'][-1])
        tl_inner.add(o['loss'][-2])
        ta_inner.add(o['correct'][-2])

        loss = o['loss'][-1]
        loss.backward()

    # Update loss
    optimizer.step()

    # Print logs
    plog = 'episode {}, acc:{:.2f}, inner acc:{:.2f}, loss:{:.2f}, inner loss:{:.2f}'.format(
                                                                            episode,
                                                                            ta.item() * 100,
                                                                            ta_inner.item() * 100,
                                                                            tl.item(),
                                                                            tl_inner.item())

    logger.debug(plog)

    tb_logger.log_value('train_acc', ta.item(), step=(episode))
    tb_logger.log_value('train_loss', tl.item(), step=(episode))    
    tb_logger.log_value('train_inner_acc', ta_inner.item(), step=(episode))
    tb_logger.log_value('train_inner_loss', tl_inner.item(), step=(episode))

    return tl, ta


def test(model, dataset, args, episode, logger, is_test=None):
    # Set test logger
    # logger_te = set_logger(args, 'test', args.test_log_path)
    if args.load is not None:
        model.load_state_dict(torch.load(args.load))        
        
    if args.test_bn:
        model.set_train()
    else:
        model.set_eval()

    ave_acc = Averager()
    ave_loss = Averager()

    ave_acc_inner = Averager()
    ave_loss_inner = Averager()

    label = torch.arange(args.way).repeat(args.query)
    label = label.type(torch.cuda.LongTensor)

    for i in range(args.test_batch):
        if args.data == 'tiered':
            data_shot, data_query, data_shot2, data_query2 = randsamp_tiered(dataset, args)
        else:
            data_shot, data_query, picked_cls2 = rand_sampling(dataset, args, shuf_cls=True, shuf_cls2=True)
            data_shot2, data_query2, _ = rand_sampling(picked_cls2, args, shuf_cls=False, shuf_cls2=False)
            picked_cls2 = None

        o = model(data_shot, data_query, data_shot2, data_query2, is_tr=False) # logits.size: [75, 5]

        with torch.no_grad():
            loss = o['loss']

            ave_loss.add(loss[-1])
            ave_loss_inner.add(loss[-2])

            # acc = count_acc(o['logits'], label)
            ave_acc.add(o['correct'][-1])        
            ave_acc_inner.add(o['correct'][-2])        
        if i % 10 == 0 and is_test is not None:
            plog = '{}th acc:{:.2f} inner acc:{:.2f} loss:{:.2f} inner loss:{:.2f}'.format(
                                                                            i,
                                                                            ave_acc.item() * 100,
                                                                            ave_acc_inner.item() * 100,
                                                                            ave_loss.item(),
                                                                            ave_loss_inner.item())
            print(plog)



        
            # logits = None

    plog = 'Test result of episode {}\nacc:{:.2f}\ninner acc:{:.2f}\nloss:{:.2f}\ninner loss:{:.2f}'.format(
                                                                            episode,
                                                                            ave_acc.item() * 100,
                                                                            ave_acc_inner.item() * 100,
                                                                            ave_loss.item(),
                                                                            ave_loss_inner.item())

    if logger is not None:
        logger.debug(plog)
    else:
        print(plog)
    if is_test is None:
        tb_logger.log_value('test_acc', ave_acc.item(), step=(episode))
        tb_logger.log_value('test_loss', ave_loss.item(), step=(episode))
        tb_logger.log_value('test_inner_acc', ave_acc_inner.item(), step=(episode))
        tb_logger.log_value('test_inner_loss', ave_loss_inner.item(), step=(episode))
    return ave_loss.item(), ave_acc.item()


if __name__ == '__main__':
    main()
