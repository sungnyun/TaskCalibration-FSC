import json
import os
import glob
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data.sampler
from torch.autograd import Variable

import path_configs
import backbone
from dataloader.datamgr import SetDataManager
from methods.tcmaml import TCMAML
from methods.protonet import ProtoNet
from platt_scaling import platt_scale
from utils import * 


if __name__ == '__main__':
    params = parse_args('test')

    # number of test tasks: 1000
    iter_num = 1000
    assert params.train_n_way == params.test_n_way
    few_shot_params = dict(n_way = params.test_n_way, n_support = params.n_shot, train_lr = params.train_lr) 

    if params.method in ['tcmaml' , 'tcmaml_approx', 'maml_approx']:
        backbone.ConvBlock.maml = True
        backbone.SimpleBlock.maml = True
        backbone.ResNet.maml = True
        if params.method == 'maml_approx':
            model = TCMAML(model_dict[params.model], approx=True, maml=True, **few_shot_params)
        else:
            model = TCMAML(model_dict[params.model], approx=(params.method == 'tcmaml_approx'), maml=False, **few_shot_params)
        model.task_update_num = params.task_update_num

    elif params.method in ['proto', 'tcproto']:
        few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
        model = ProtoNet(model_dict[params.model], **few_shot_params, task_calibration=(params.method=='tcproto'))

    else:
        raise ValueError('Unknown method')
    
    model = model.cuda()

    if params.checkpoint_dir is None:
        checkpoint_dir = '%s/%s/%s_%s' %(path_configs.save_dir, params.dataset, params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'
        checkpoint_dir += '_%dway_%dshot' %(params.train_n_way, params.n_shot)
        #checkpoint_dir += '_temp%2.1f' %(params.temp)
        if params.linear_scaling:
            checkpoint_dir += '_ls'
    else:
        checkpoint_dir = params.checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        raise ValueError('Checkpoint not exists')
    
    if params.save_iter != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
    elif params.best_iter:
        modelfile = get_best_file(checkpoint_dir)
    elif params.last_iter:
        modelfile = get_resume_file(checkpoint_dir)
    else:
        raise NotImplementedError
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" + str(params.save_iter)
    else:
        split_str = split

    if 'Conv' in params.model:
        image_size = 84
    else:  # ResNet backbone for domain shift
        image_size = 224
    
    # number of query examples per class in testset is 15
    datamgr = SetDataManager(image_size, n_episode=iter_num, n_query=15, **few_shot_params)
    if params.dataset == 'cross':
        if split == 'base':
            loadfile = path_configs.data_dir['miniImagenet'] + 'all.json' 
        else:
            loadfile   = path_configs.data_dir['CUB'] + split + '.json'
    else: 
        loadfile    = path_configs.data_dir[params.dataset] + split + '.json'
    test_loader     = datamgr.get_data_loader(loadfile, aug=False)
    
    if params.only_low_uncertainty:
        if params.threshold is None:
            raise ValueError('Denote the threshold for uncertainty')

    print(params)
    model.eval()

    if params.platt_scaling:
        if params.dataset == 'cross':
            val_file = path_configs.data_dir['miniImagenet'] + 'val.json'
        else:
            val_file = path_configs.data_dir[params.dataset] + 'val.json'
        val_datamgr = SetDataManager(image_size, n_episode=100, n_query=15, **few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)
        eval_temp = platt_scale(model, val_loader, params)
        print('calibrated temperature: {:.4f}'.format(eval_temp))
    else:
        eval_temp = 1.0

    ### TEST
    set_seeds(params.seed)
    acc_mean, acc_std = model.test_loop(test_loader, return_std=True, threshold=params.threshold, eval_temp=eval_temp)
        
    # with open('./record/results.txt', 'a') as f:
    #     timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime()) 
    #     aug_str = '-aug' if params.train_aug else ''
    #     exp_setting = '%s-%s-%s-%s%s %sshot %sway_train %sway_test' %(params.dataset, split_str, params.model, params.method, aug_str, params.n_shot, params.train_n_way, params.test_n_way)
    #     acc_str = '%d Test Acc = %4.2f%% +/- %4.2f%%' %(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num))
    #     f.write('Time: %s, Setting: %s, Acc: %s \n' %(timestamp, exp_setting, acc_str))
