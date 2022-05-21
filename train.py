'''
The source codes are heavily relied on "https://github.com/wyharveychen/CloserLookFewShot".

Before running the code, first prepare datasets. 
In the filelists folder, there are 'download_miniImagenet.sh' and 'download_CUB.sh'. Execute both.
To train TCMAML/TCProtoNet you should run this file, 'train.py'.
The following command lines will reproduce the results. (see 'command.txt')

miniImagenet 5-way 5-shot TCMAML
--------------------------------
python train.py --dataset miniImagenet --temp 1

To evaluate and check the calibration results after training is done, run 'test.py' with the same arguments.
For example,

python test.py --dataset miniImagenet --temp 1
'''


import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import time
import os
import glob
import random

import path_configs
import backbone
from dataloader.datamgr import SimpleDataManager, SetDataManager
from methods.tcmaml import TCMAML
from methods.protonet import ProtoNet
from utils import *  


def train(train_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    # set random seed
    set_seeds(params.seed)

    max_acc = 0       
    log_dir = os.path.join(params.checkpoint_dir, 'log')
    logfile = open(log_dir, 'w')
    logfile.write(str(params)+'\n')

    for epoch in range(start_epoch+1, stop_epoch+1):
        # train
        model.train()
        model.train_loop(epoch, train_loader, optimizer, logfile, exclude_task=params.exclude_task)
        model.eval()
        
        # validation
        if epoch % params.valid_freq == 0:
            acc, avg_uncertainty = model.test_loop(val_loader, return_std=False)
            logfile.write('Epoch {}, Valid-Acc {:.4f}, Valid-Uncertainty {:.4f} \n'.format(epoch, acc/100, avg_uncertainty))
        
            # save the model with best validation acc
            if acc > max_acc : 
                print("best model! save...")
                max_acc = acc
                outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

            if (epoch % params.save_freq == 0) or (epoch == stop_epoch):
                outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
                torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        
        if params.lr_schedule and epoch % 1000 == 0:
            logfile.write('LR decay!')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    logfile.close()
    return model


if __name__=='__main__':
    params = parse_args('train')
    set_seeds(params.seed)

    if params.dataset == 'cross':
        train_file = path_configs.data_dir['miniImagenet'] + 'all.json' 
        val_file   = path_configs.data_dir['CUB'] + 'val.json' 
    else:
        train_file = path_configs.data_dir[params.dataset] + 'base.json' 
        val_file   = path_configs.data_dir[params.dataset] + 'val.json' 
    
    if params.model in ['Conv4', 'Conv6']:
        image_size = 84
    else: # ResNet backbone for domain shift
        image_size = 224
        
    optimization = 'Adam'
    
    if params.stop_epoch == -1: 
        if params.n_shot == 1:
            params.stop_epoch = 2400
        elif params.n_shot == 5:
            params.stop_epoch = 1600
        else:
            params.stop_epoch = 2400
     
    if params.method in ['tcmaml', 'tcmaml_approx', 'maml_approx', 'proto', 'tcproto']:
        assert params.train_n_way == params.test_n_way

        n_query = 15 
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot, train_lr = params.train_lr) 
        train_datamgr            = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)  # default number of episodes (tasks) is 100 per epoch
        train_loader             = train_datamgr.get_data_loader(train_file, params.train_aug, params.outlier_task, params.mixed_task, params.corrupted_task)
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot, train_lr=params.train_lr)
        val_datamgr             = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader(val_file, aug=False)        
        
        if params.method in ['tcmaml', 'tcmaml_approx', 'maml_approx']: 
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.ResNet.maml = True
        
            if params.method == 'maml_approx':
                model = TCMAML(model_dict[params.model], approx=True, maml=True, **train_few_shot_params)
            else:
                model = TCMAML(model_dict[params.model], approx=(params.method == 'tcmaml_approx'), maml=False, **train_few_shot_params)
            
        elif params.method in ['proto', 'tcproto']:
            train_few_shot_params = dict(n_way = params.train_n_way, n_support = params.n_shot)
            model = ProtoNet(model_dict[params.model], **train_few_shot_params, task_calibration=(params.method == 'tcproto'))

        if params.linear_scaling:
            model.linear_scaling = True

        if params.method in ['maml_approx', 'proto']:
            params.temp = 1
        if params.method == 'proto':
            params.n_task = 1

    else:
        raise ValueError('Unknown method')

    model.T = params.temp
    model.n_task = params.n_task
    
    model = model.cuda()

    if params.checkpoint_dir is None:
        params.checkpoint_dir = '%s/%s/%s_%s' %(path_configs.save_dir, params.dataset, params.model, params.method)
        if params.train_aug:
            params.checkpoint_dir += '_aug'
        params.checkpoint_dir += '_%dway_%dshot' %(params.train_n_way, params.n_shot)
        #params.checkpoint_dir += '_temp%2.1f' %(params.temp)
        if params.linear_scaling:
            params.checkpoint_dir += '_ls'

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch 

    if params.resume:
        resume_file = get_assigned_file(params.checkpoint_dir, params.start_epoch)
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            model.load_state_dict(tmp['state'])
    
    print(params)
    model = train(train_loader, val_loader, model, optimization, start_epoch, stop_epoch, params)
