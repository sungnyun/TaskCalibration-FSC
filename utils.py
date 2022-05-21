import os
import random
import glob
import argparse
import backbone

import torch
import numpy as np


model_dict = dict(Conv4 = backbone.Conv4,
                  Conv6 = backbone.Conv6,
                  ResNet10 = backbone.ResNet10,
                  ResNet18 = backbone.ResNet18) 

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--seed'        , default=42, type=int,  help='random seed')
    parser.add_argument('--dataset'     , default='miniImagenet',        help='miniImagenet/CUB/cross')  # cross: miniImagenet -> CUB
    parser.add_argument('--model'       , default='Conv4',      help='model: Conv{4|6} / ResNet{10|18}')  # backbone CNN model
    parser.add_argument('--method'      , default='tcmaml_approx',   help='tcmaml{_approx}, proto, tcproto') 
    parser.add_argument('--train-n-way' , default=5, type=int,  help='class num to classify for training') 
    parser.add_argument('--test-n-way'  , default=5, type=int,  help='class num to classify for testing (validation)') 
    parser.add_argument('--n-shot'      , default=5, type=int,  help='number of shots (support examples) in each class') 
    parser.add_argument('--train-aug'   , default=True, type=str2bool, help='perform data augmentation or not during training')  # We used data augmentation in the paper.
    parser.add_argument('--temp'        , default=1, type=float, help='temperature scaling factor')
    parser.add_argument('--lr'          , default=5e-4, type=float, help='learning rate for meta-update')
    parser.add_argument('--train-lr'    , default=0.01, type=float, help='learning rate for inner-update')
    parser.add_argument('--lr-schedule' , action='store_true', help='learning rate decay')
    parser.add_argument('--n-task'      , default=4, type=int, help='number of tasks per batch when computing total loss')
    parser.add_argument('--task-update-num', default=5, type=int, help='number of inner loops')
    parser.add_argument('--weight-decay', default=0, type=float, help='Adam l2 regularizer')
    parser.add_argument('--linear-scaling', action='store_true', help='linear scaling normalization')
    parser.add_argument('--checkpoint-dir', default=None, type=str, help='checkpoint directory name')
    
    if script == 'train':
        parser.add_argument('--save-freq'   , default=200, type=int, help='Save frequency')
        parser.add_argument('--valid-freq'  , default=100, type=int, help='Validation frequency')
        parser.add_argument('--start-epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop-epoch'  , default=-1, type=int, help ='Stopping epoch')  # Each epoch contains 100 episodes. The default epoch number is dependent on number of shots. See train.py for details.
        parser.add_argument('--resume'      , action='store_true', help='continue from previous trained model with largest epoch')
        parser.add_argument('--exclude-task', action='store_true', help='train without the most uncertain task in the task batch') 
        parser.add_argument('--outlier-task', action='store_true', help='insert outlier tasks during training') 
        parser.add_argument('--mixed-task'  , action='store_true', help='insert mixed tasks during training') 
        parser.add_argument('--corrupted-task', action='store_true', help='insert corrupted tasks during training') 
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel')  # base: train, val: validation, novel: test
        parser.add_argument('--save-iter'   , default=-1, type=int, help ='saved model in x epoch, use the best model if x is -1')
        parser.add_argument('--last-iter'   , default=True, type=str2bool, help ='saved model in last epoch')
        parser.add_argument('--best-iter'   , action='store_true', help ='saved model in best epoch')
        parser.add_argument('--only-low-uncertainty', action='store_true', help='evaluate only the tasks that have low uncertainty value')
        parser.add_argument('--threshold'   , default=None, type=float, help='uncertainty threshold over which are excluded in evaluation')
        parser.add_argument('--platt-scaling', action='store_true', help='obtain temperature value by platt scaling before evaluation')
        
    else:
        raise ValueError('Unknown script')
        
    return parser.parse_args()


def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        print(checkpoint_dir)
        return best_file
    else:
        return get_resume_file(checkpoint_dir)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
