### This code is modified from https://github.com/cnguyen10/few_shot_meta_learning ###

'''
python abml.py --dataset=miniImagenet --n_way=5 --k_shot=1 --minibatch=4 --Lt=5 --Lv=5 --kl_reweight=0.1 --num_epochs=50 (--task_calibration)
python abml.py --dataset=miniImagenet --n_way=5 --k_shot=1 --minibatch=1 --Lt=5 --Lv=10 --kl_reweight=0.1 --test --resume_epoch=50
python abml.py --dataset=miniImagenet --n_way=5 --k_shot=5 --minibatch=4 --Lt=5 --Lv=5 --kl_reweight=0.1 --num_epochs=50 (--task_calibration)
python abml.py --dataset=miniImagenet --n_way=5 --k_shot=5 --minibatch=1 --Lt=5 --Lv=10 --kl_reweight=0.1 --test --resume_epoch=50
'''

import torch
import torch.nn as nn

import numpy as np
import random
import itertools

import os
import sys

import csv

import argparse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataloader.datamgr import SimpleDataManager, SetDataManager
import path_configs

# -------------------------------------------------------------------------------------------------
# Setup input parser
# -------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables for ABML.')

parser.add_argument('--seed', default=42, type=int,  help='random seed')
parser.add_argument('--dataset', type=str, default='miniImagenet', help='dataset: miniImageNet, CUB')
parser.add_argument('--dst_folder', type=str, default=None, help='checkpoint directory')
parser.add_argument('--n_way', type=int, default=5, help='Number of classes per task')
parser.add_argument('--k_shot', type=int, default=1, help='Number of training samples per class or k-shot')
parser.add_argument('--num_val_shots', type=int, default=15, help='Number of validation samples per class')

parser.add_argument('--inner_lr', type=float, default=1e-2, help='Learning rate for task adaptation')
parser.add_argument('--num_inner_updates', type=int, default=5, help='Number of gradient updates for task adaptation')
parser.add_argument('--meta_lr', type=float, default=1e-3, help='Learning rate of meta-parameters')
parser.add_argument('--lr_decay', type=float, default=1, help='Decay factor of meta-learning rate (<=1), 1 = no decay')
parser.add_argument('--minibatch', type=int, default=4, help='Number of tasks per minibatch to update meta-parameters')

parser.add_argument('--num_epochs', type=int, default=100, help='How many 10,000 tasks are used to train?')
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch id to resume learning or perform testing')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--num_val_tasks', type=int, default=1000, help='Number of validation tasks')

parser.add_argument('--kl_reweight', type=float, default=0.1, help='Reweight factor of the KL divergence between variational posterior q and prior p')
parser.add_argument('--Lt', type=int, default=5, help='Number of ensemble networks to train task specific parameters')
parser.add_argument('--Lv', type=int, default=5, help='Number of ensemble networks to validate meta-parameters')

parser.add_argument('--uncertainty', dest='uncertainty_flag', action='store_true')
parser.add_argument('--no_uncertainty', dest='uncertainty_flag', action='store_false')
parser.set_defaults(uncertainty_flag=True)

parser.add_argument('--task_calibration', dest='TC_flag', action='store_true')
parser.add_argument('--platt-scaling', dest='platt_scaling', action='store_true')
parser.set_defaults(TC_flag=False)
parser.add_argument('--temp', type=float, default=1.0)

args = parser.parse_args()

# -------------------------------------------------------------------------------------------------
# Setup CPU or GPU
# -------------------------------------------------------------------------------------------------
gpu_id = 0
device = torch.device('cuda:{0:d}'.format(gpu_id) if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------------------------------------------------------------------------
# Parse dataset and related variables
# -------------------------------------------------------------------------------------------------
dataset = args.dataset
print('Dataset = {0:s}'.format(dataset))

train_flag = args.train_flag
print('Learning mode = {0}'.format(train_flag))

num_classes_per_task = args.n_way
print('Number of ways = {0:d}'.format(num_classes_per_task))

num_training_samples_per_class = args.k_shot
print('Number of shots = {0:d}'.format(num_training_samples_per_class))

num_val_samples_per_class = args.num_val_shots
print('Number of validation samples per class = {0:d}'.format(num_val_samples_per_class))

num_samples_per_class = num_training_samples_per_class + num_val_samples_per_class

TC_flag = args.TC_flag
if TC_flag:
    print('Task Calibration ABML!')
temp = args.temp
platt_scaling = args.platt_scaling

# -------------------------------------------------------------------------------------------------
#   Setup based model/network
# -------------------------------------------------------------------------------------------------
if dataset == 'sine_line':
    from DataGeneratorT import DataGenerator
    from FCNet import FCNet
    from utils import get_task_sine_line_data

    # loss function as mean-squared error
    loss_fn = torch.nn.MSELoss()

    # Bernoulli probability for sine and line
    # 0.5 = uniform
    p_sine = 0.5

    noise_flag = True

    # based network
    net = FCNet(
        dim_input=1,
        dim_output=1,
        # num_hidden_units=(40, 40),
        num_hidden_units=(100, 100, 100),
        device=device
    )
else:
    train_set = 'train'
    val_set = 'val'
    test_set = 'test'

    loss_fn = torch.nn.CrossEntropyLoss()
    sm = torch.nn.Softmax(dim=-1)

    if dataset in ['omniglot', 'miniImagenet', 'CUB']:
        from ConvNet import ConvNet

        DIM_INPUT = {
            'omniglot': (1, 28, 28),
            'miniImagenet': (3, 84, 84),
            'CUB': (3, 84, 84)
        }

        net = ConvNet(
            dim_input=DIM_INPUT[dataset],
            dim_output=num_classes_per_task,
            num_filters=(32, 32, 32, 32),
            filter_size=(3, 3),
            device=device
        )

    elif dataset in ['miniImagenet_640', 'tieredImageNet_640']:
        import pickle
        from FC640 import FC640
        net = FC640(
            dim_output=num_classes_per_task,
            num_hidden_units=(128, 32),
            device=device
        )
    else:
        sys.exit('Unknown dataset!')

weight_shape = net.get_weight_shape()

# -------------------------------------------------------------------------------------------------
# Parse training parameters
# -------------------------------------------------------------------------------------------------
inner_lr = args.inner_lr
print('Inner learning rate = {0}'.format(inner_lr))

num_inner_updates = args.num_inner_updates
print('Number of inner updates = {0:d}'.format(num_inner_updates))

meta_lr = args.meta_lr
print('Meta learning rate = {0}'.format(meta_lr))

num_tasks_per_minibatch = args.minibatch
print('Minibatch = {0:d}'.format(num_tasks_per_minibatch))

num_meta_updates_print = int(1000 / num_tasks_per_minibatch) 
print('Mini batch size = {0:d}'.format(num_tasks_per_minibatch))

num_epochs_save = 10

num_epochs = args.num_epochs

expected_total_tasks_per_epoch = 10000
num_tasks_per_epoch = int(expected_total_tasks_per_epoch / num_tasks_per_minibatch)*num_tasks_per_minibatch

expected_tasks_save_loss = 10000
num_tasks_save_loss = int(expected_tasks_save_loss / num_tasks_per_minibatch)*num_tasks_per_minibatch

num_val_tasks = args.num_val_tasks
uncertainty_flag = args.uncertainty_flag

# Number of emsembling models/networks
Lt = args.Lt
Lv = args.Lv
print('Lt = {0:d}, Lv = {1:d}'.format(Lt, Lv))

KL_reweight = args.kl_reweight
print('KL reweight = {0}'.format(KL_reweight))

# -------------------------------------------------------------------------------------------------
# Setup destination folder
# -------------------------------------------------------------------------------------------------
dst_folder_root = 'abml/checkpoints'
if args.dst_folder is None:
    dst_folder = '{0:s}/{1:s}_{2:d}way_{3:d}shot'.format(
        dst_folder_root,
        dataset,
        num_classes_per_task,
        num_training_samples_per_class
    )
else:
    dst_folder = '{0:s}/{1:s}'.format(
            dst_folder_root,
            args.dst_folder)
if not os.path.exists(dst_folder):
    os.makedirs(dst_folder)
    print('No folder for storage found')
    print('Make folder to store meta-parameters at')
else:
    print('Found existing folder. Meta-parameters will be stored at')
print(dst_folder)

# -------------------------------------------------------------------------------------------------
# Initialize/Load meta-parameters
# -------------------------------------------------------------------------------------------------
resume_epoch = args.resume_epoch
if resume_epoch == 0:
    # initialise meta-parameters
    theta = {}
    theta['mean'] = {}
    theta['logSigma'] = {}
    for key in weight_shape.keys():
        if 'b' in key:
            theta['mean'][key] = torch.zeros(weight_shape[key], device=device, requires_grad=True)
        else:
            theta['mean'][key] = torch.empty(weight_shape[key], device=device)
            torch.nn.init.xavier_normal_(tensor=theta['mean'][key], gain=np.sqrt(2))
            theta['mean'][key].requires_grad_()
        theta['logSigma'][key] = torch.rand(weight_shape[key], device=device) - 4
        theta['logSigma'][key].requires_grad_()
else:
    print('Restore previous theta...')
    print('Resume epoch {0:d}'.format(resume_epoch))
    checkpoint_filename = 'Epoch_{0:d}.pth'.format(resume_epoch)
    checkpoint_file = os.path.join(dst_folder, checkpoint_filename)
    print('Start to load weights from')
    print('{0:s}'.format(checkpoint_file))
    if torch.cuda.is_available():
        saved_checkpoint = torch.load(
            checkpoint_file,
            map_location=lambda storage,
            loc: storage.cuda(gpu_id)
        )
    else:
        saved_checkpoint = torch.load(
            checkpoint_file,
            map_location=lambda storage,
            loc: storage
        )

    theta = saved_checkpoint['theta']

op_theta = torch.optim.Adam(
    [
        {
            'params': theta['mean'].values(),
            'weight_decay': 0
        },
        {
            'params': theta['logSigma'].values(),
            'weight_decay': 0
        }
    ],
    lr=meta_lr
)

if resume_epoch > 0:
    op_theta.load_state_dict(saved_checkpoint['op_theta'])
    # op_theta.param_groups[0]['lr'] = meta_lr
    # op_theta.param_groups[1]['lr'] = meta_lr
    del saved_checkpoint

# decay the learning rate
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=op_theta,
    gamma=args.lr_decay
)

print(op_theta)
print()

# -------------------------------------------------------------------------------------------------
# MAIN program
# -------------------------------------------------------------------------------------------------
def main():


    if train_flag:
        meta_train()
    else: # validation
        assert resume_epoch > 0

        if dataset == 'sine_line':
            raise NotImplementedError
        else:
            val_file = path_configs.data_dir[dataset] + 'novel.json'
            val_few_shot_params = dict(n_way = num_classes_per_task, n_support = num_training_samples_per_class, train_lr = inner_lr) 
            valid_datamgr = SetDataManager(84, n_query=num_val_samples_per_class, n_episode=num_val_tasks, **val_few_shot_params) 
            valid_loader = valid_datamgr.get_data_loader(val_file, aug=False)

            net.eval()
            validate_classification(
                valid_loader,
                num_val_tasks=num_val_tasks,
                rand_flag=False,
                uncertainty=uncertainty_flag,
                csv_flag=True,
                evaluate=True
            )

def meta_train(train_subset=train_set):
    log_dir = os.path.join(dst_folder, 'log')
    logfile = open(log_dir, 'w')

    train_file = path_configs.data_dir[dataset] + 'base.json'
    train_few_shot_params = dict(n_way = num_classes_per_task, n_support = num_training_samples_per_class, train_lr = inner_lr) 
    train_datamgr = SetDataManager(84, n_query=num_val_samples_per_class, **train_few_shot_params) 
    train_loader = train_datamgr.get_data_loader(train_file, aug=True)
    
    val_file = path_configs.data_dir[dataset] + 'val.json'
    val_few_shot_params = dict(n_way = num_classes_per_task, n_support = num_training_samples_per_class, train_lr = inner_lr) 
    valid_datamgr = SetDataManager(84, n_query=num_val_samples_per_class, **val_few_shot_params) 
    valid_loader = valid_datamgr.get_data_loader(val_file, aug=False)

    for epoch in range(resume_epoch, resume_epoch + num_epochs):
        # variables used to store information of each epoch for monitoring purpose
        meta_loss_saved = [] # meta loss to save
        val_accuracies = []
        train_accuracies = []
        task_uncertainty_all = []

        meta_loss = 0 # accumulate the loss of many ensambling networks to descent gradient for meta update
        num_meta_updates_count = 0

        meta_loss_avg_print = 0 # compute loss average to print

        meta_loss_avg_save = [] # meta loss to save
        meta_loss_all = []

        task_count = 0 # a counter to decide when a minibatch of task is completed to perform meta update
        std_list = []

        while (task_count < num_tasks_per_epoch):
            for x,_ in train_loader:
                x = x.to(device)
                x_t = x[:,:num_training_samples_per_class,:,:,:].contiguous().view(num_classes_per_task*num_training_samples_per_class, *x.size()[2:])  # support data 
                x_v = x[:,num_training_samples_per_class:,:,:,:].contiguous().view(num_classes_per_task*num_val_samples_per_class, *x.size()[2:])  # query data 
                y_t = torch.from_numpy(np.repeat(range(num_classes_per_task), num_training_samples_per_class)).to(device)  # label for support data
                y_v = torch.from_numpy(np.repeat(range(num_classes_per_task), num_val_samples_per_class)).to(device)  # label for query data

                q = adapt_to_task(x=x_t, y=y_t, theta0=theta, approx=True)
                ### TODO : track std values
                # if (task_count+1) % num_tasks_per_minibatch == 0:
                #     std_ = []
                #     for key in weight_shape.keys():
                #         std_.append(torch.exp(q['logSigma'][key]).flatten())
                #     std_list.append(torch.cat(std_, -1).mean().item())
                # if (task_count+1) % 100 == 0:
                #     max_std, min_std = np.array(std_list).max(), np.array(std_list).min()
                #     avg_std, std_std = np.array(std_list).mean(), np.array(std_list).std()
                #     print('Epoch {}, Tasks {}, Max Std {:.4f}, Min Std {:.4f}, Avg Std {:.4f}, Std Std {:.4f}\n'.format(epoch, task_count+1, max_std, min_std, avg_std, std_std))
                #     logfile.write('Epoch {}, Tasks {}, Max Std {:.4f}, Min Std {:.4f}, Avg Std {:.4f}, Std Std {:.4f}\n'.format(epoch, task_count+1, max_std, min_std, avg_std, std_std))
                #     std_list = []
                ###
                y_pred, feats = predict(x=x_v, q=q, num_models=Lv)
                mean_feats = [feats[v].contiguous().view(num_classes_per_task, num_val_samples_per_class, -1).mean(1) for v in range(len(feats))]
                task_uncertainty_v = [measure_uncertainty(mean_feats[v]) for v in range(len(mean_feats))]
                task_uncertainty = torch.stack(task_uncertainty_v).max()
                task_uncertainty_all.append(task_uncertainty)
                loss_NLL = 0
                for lv in range(Lv):
                    loss_NLL_temp = loss_fn(input=y_pred[lv], target=y_v)
                    loss_NLL = loss_NLL + loss_NLL_temp
                loss_NLL = loss_NLL / Lv

                if torch.isnan(loss_NLL).item():
                    sys.exit('NaN error')

                # accumulate meta loss
                # meta_loss = meta_loss + loss_NLL
                meta_loss_all.append(loss_NLL)

                task_count = task_count + 1
                if task_count % num_tasks_per_minibatch == 0:
                    if TC_flag:
                        task_uncertainty_all = nn.functional.softmax(- torch.stack(task_uncertainty_all) / temp, dim=-1)
                    else:
                        task_uncertainty_all = torch.ones(num_tasks_per_minibatch).to(device) / num_tasks_per_minibatch
                    task_uncertainty_all = task_uncertainty_all.detach()

                    # meta_loss = meta_loss / num_tasks_per_minibatch
                    meta_loss = torch.mul(task_uncertainty_all, torch.stack(meta_loss_all)).sum(0)

                    # accumulate into different variables for printing purpose
                    meta_loss_avg_print = meta_loss_avg_print + meta_loss.item()

                    op_theta.zero_grad()
                    meta_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters=theta['mean'].values(),
                        max_norm=10
                    )
                    torch.nn.utils.clip_grad_norm_(
                        parameters=theta['logSigma'].values(),
                        max_norm=10
                    )
                    op_theta.step()

                    # Printing losses
                    num_meta_updates_count += 1
                    if (num_meta_updates_count % num_meta_updates_print == 0):
                        meta_loss_avg_save.append(meta_loss_avg_print/num_meta_updates_count)
                        print('{0:d}, {1:2.4f}'.format(
                            task_count,
                            meta_loss_avg_save[-1]
                        ))

                        num_meta_updates_count = 0
                        meta_loss_avg_print = 0
                    
                    if (task_count % num_tasks_save_loss == 0):
                        meta_loss_saved.append(np.mean(meta_loss_avg_save))

                        meta_loss_avg_save = []

                        if dataset != 'sine_line':
                            net.eval()
                            val_accs = validate_classification(
                                valid_loader,
                                num_val_tasks=100,
                                rand_flag=True,
                                uncertainty=True,
                                csv_flag=False
                            )
                            val_acc = np.mean(val_accs)
                            val_ci95 = 1.96*np.std(val_accs)/np.sqrt(num_val_tasks)
                            print('Validation accuracy = {0:2.4f} +/- {1:2.4f}'.format(val_acc, val_ci95))
                            val_accuracies.append(val_acc)
                            net.train()
                   
                    # reset meta loss
                    meta_loss = 0
                    meta_loss_all = []
                    task_uncertainty_all = []

                if (task_count >= num_tasks_per_epoch):
                    break
        if ((epoch + 1)% num_epochs_save == 0):
            checkpoint = {
                'theta': theta,
                'meta_loss': meta_loss_saved,
                'val_accuracy': val_accuracies,
                'train_accuracy': train_accuracies,
                'op_theta': op_theta.state_dict()
            }
            print('SAVING WEIGHTS...')
            checkpoint_filename = 'Epoch_{0:d}.pth'.format(epoch + 1)
            print(checkpoint_filename)
            torch.save(checkpoint, os.path.join(dst_folder, checkpoint_filename))
        print()
        scheduler.step()
    logfile.close()

@torch.no_grad()
def measure_uncertainty(mean_features):
    '''
    Compute task uncertainty with class-wise average features.
    Each element of the similarity matrix is cosine similarity.
    '''
    dot_product = torch.mm(mean_features, mean_features.t())
    norm = mean_features.pow(2).sum(1).sqrt().unsqueeze(1)
    normsq = torch.mm(norm, norm.t())
    similarity_matrix = dot_product / normsq

    task_uncertainty = torch.norm(similarity_matrix, p='fro')
    
    return task_uncertainty

def adapt_to_task(x, y, theta0, approx=True):
    q = initialize_q(key_list=weight_shape.keys())

    # 1st gradient update - KL divergence is zero
    loss_NLL = 0
    for _ in range(Lt):
        w = sample_nn_weight(meta_params=theta0)
        y_pred = net.forward(x=x, w=w)
        loss_NLL_temp = loss_fn(input=y_pred, target=y)
        loss_NLL = loss_NLL + loss_NLL_temp
    loss_NLL = loss_NLL / Lt

    loss_NLL_grads_mean = torch.autograd.grad(
        outputs=loss_NLL,
        inputs=theta0['mean'].values(),
        create_graph=True
    )
    loss_NLL_grads_logSigma = torch.autograd.grad(
        outputs=loss_NLL,
        inputs=theta0['logSigma'].values(),
        create_graph=True
    )
    if approx:
        loss_NLL_grads_mean = [g.detach() for g in loss_NLL_grads_mean]
        loss_NLL_grads_logSigma = [g.detach() for g in loss_NLL_grads_logSigma]
    gradients_mean = dict(zip(theta0['mean'].keys(), loss_NLL_grads_mean))
    gradients_logSigma = dict(zip(theta0['logSigma'].keys(), loss_NLL_grads_logSigma))
    for key in weight_shape.keys():
        q['mean'][key] = theta0['mean'][key] - inner_lr * gradients_mean[key]
        q['logSigma'][key] = theta0['logSigma'][key] - inner_lr * gradients_logSigma[key]
    
    # 2nd updates
    for _ in range(num_inner_updates - 1):
        loss_NLL = 0
        for _ in range(Lt):
            w = sample_nn_weight(meta_params=q)
            y_pred = net.forward(x=x, w=w)
            loss_NLL_temp = loss_fn(input=y_pred, target=y)
            loss_NLL = loss_NLL + loss_NLL_temp
        loss_NLL = loss_NLL / Lt

        loss_NLL_grads_mean = torch.autograd.grad(
            outputs=loss_NLL,
            inputs=q['mean'].values(),
            retain_graph=True
        )
        loss_NLL_grads_logSigma = torch.autograd.grad(
            outputs=loss_NLL,
            inputs=q['logSigma'].values(),
            retain_graph=True
        )
        if approx:
            loss_NLL_grads_mean = [g.detach() for g in loss_NLL_grads_mean]
            loss_NLL_grads_logSigma = [g.detach() for g in loss_NLL_grads_logSigma]
        loss_NLL_gradients_mean = dict(zip(q['mean'].keys(), loss_NLL_grads_mean))
        loss_NLL_gradients_logSigma = dict(zip(q['logSigma'].keys(), loss_NLL_grads_logSigma))

        for key in w.keys():
            loss_KL_grad_mean = -torch.exp(-2 * theta0['logSigma'][key]) * (theta0['mean'][key] - q['mean'][key])
            loss_KL_grad_logSigma = torch.exp(2 * (q['logSigma'][key] - theta0['logSigma'][key])) - 1
            
            q['mean'][key] = q['mean'][key] - inner_lr * (loss_NLL_gradients_mean[key] + KL_reweight * loss_KL_grad_mean)
            q['logSigma'][key] = q['logSigma'][key] - inner_lr * (loss_NLL_gradients_logSigma[key] \
                + KL_reweight * loss_KL_grad_logSigma)
    return q

def predict(x, q, num_models=1):
    y_pred = torch.empty(
        size=(num_models, x.shape[0], num_classes_per_task),
        device=device
    )
    feats = []
    for lv in range(num_models):
        w = sample_nn_weight(meta_params=q)
        y_pred[lv, :, :], feat = net.forward(x=x, w=w, feat=True)
        feats.append(feat)
    return y_pred, feats

# def temperature_scale(logits, temperature):
#     temperature_ = temperature.unsqueeze(1).unsqueeze(1).expand(*logits.size())
#     return torch.log(nn.functional.softmax(logits / temperature_, dim=-1).mean(0))
    

def validate_classification(
    valid_loader,
    num_val_tasks,
    rand_flag=False,
    uncertainty=False,
    csv_flag=False,
    evaluate=False
):
    if csv_flag:
        filename = 'ABML_{0:s}_{1:d}way_{2:d}shot_{3:s}_{4:d}.csv'.format(
            dataset,
            num_classes_per_task,
            num_training_samples_per_class,
            'uncertainty' if uncertainty else 'accuracy',
            resume_epoch
        )
        outfile = open(file=os.path.join('csv', filename), mode='w')
        wr = csv.writer(outfile, quoting=csv.QUOTE_NONE)
        
    accuracies = []
    
    total_val_samples_per_task = num_val_samples_per_class * num_classes_per_task
    # all_class_names = [class_name for class_name in sorted(all_classes.keys())]
    # all_task_names = itertools.combinations(all_class_names, r=num_classes_per_task)

    bin_count = np.zeros(10)
    acc_count = np.zeros(10)
    conf_count = np.zeros(10)

    if platt_scaling:
        if dataset == 'cross':
            val_file = path_configs.data_dir['miniImagenet'] + 'val.json'
        else:
            val_file = path_configs.data_dir[dataset] + 'val.json'
        val_few_shot_params = dict(n_way = num_classes_per_task, n_support = num_training_samples_per_class, train_lr = inner_lr) 
        val_datamgr = SetDataManager(84, n_episode=100, n_query=15, **val_few_shot_params)
        val_loader = val_datamgr.get_data_loader(val_file, aug=False)

        eval_temp = platt_scale(theta, val_loader)
        print('calibrated temperature: {:.4f}'.format(eval_temp))
    else:
        eval_temp = 1.0

    task_count = 0
    for x,_ in valid_loader:
        if rand_flag:
            skip_task = np.random.binomial(n=1, p=0.5) # sample from an uniform Bernoulli distribution
            if skip_task == 1:
                continue

        x = x.to(device)
        x_t = x[:,:num_training_samples_per_class,:,:,:].contiguous().view(num_classes_per_task*num_training_samples_per_class, *x.size()[2:])  # support data 
        x_v = x[:,num_training_samples_per_class:,:,:,:].contiguous().view(num_classes_per_task*num_val_samples_per_class, *x.size()[2:])  # query data 
        y_t = torch.from_numpy(np.repeat(range(num_classes_per_task), num_training_samples_per_class)).to(device)  # label for support data
        y_v = torch.from_numpy(np.repeat(range(num_classes_per_task), num_val_samples_per_class)).to(device)  # label for query data

        q = adapt_to_task(x=x_t, y=y_t, theta0=theta, approx=True)
        raw_scores, _ = predict(x=x_v, q=q, num_models=Lv) # Lv x num_samples x num_classes
        sm_scores = sm(input=raw_scores/eval_temp)
        sm_scores_avg = torch.mean(sm_scores, dim=0)
        
        prob, y_pred = torch.max(input=sm_scores_avg, dim=1)
        correct = [1 if y_pred[i] == y_v[i] else 0 for i in range(total_val_samples_per_task)]

        accuracy = np.mean(a=correct, axis=0)
        
        if evaluate:
            topk_scores, topk_labels = sm_scores_avg.data.topk(1, 1, True, True)
            topk_conf = topk_scores.cpu().numpy()
            topk_ind = topk_labels.cpu().numpy()
                    
            for i, (ind, conf) in enumerate(zip(topk_ind, topk_conf)):
                correctness = int(topk_ind[i,0] == y_v[i])
                bin_id = int(conf*10)
                if bin_id == 10:
                    bin_id = 9
                bin_count[bin_id] += 1
                acc_count[bin_id] += correctness  # for acc(B_m)
                conf_count[bin_id] += conf    # for conf(B_m)

        if csv_flag:
            if not uncertainty:
                # outline = [class_label for class_label in class_labels]
                outline = []
                outline.append(accuracy)
                wr.writerow(outline)
            else:
                for correct_, prob_ in zip(correct, prob):
                    outline = [correct_, prob_.item()]
                    wr.writerow(outline)

        accuracies.append(accuracy)

        task_count = task_count + 1
        if not train_flag:
            sys.stdout.write('\033[F')
            print(task_count)
        if task_count >= num_val_tasks:
            break
    if evaluate:
        acc_B = np.array([acc_count[i] / bin_count[i] if bin_count[i] > 0 else 0 for i in range(10)])
        conf_B = np.array([conf_count[i] / bin_count[i] if bin_count[i] > 0 else 0 for i in range(10)])
        ECE = (bin_count / bin_count.sum() *abs(acc_B - conf_B)).sum()
        MCE = abs(acc_B - conf_B).max()
    
        acc_mean = np.mean(100*np.array(accuracies))
        acc_std = np.std(100*np.array(accuracies))
        print('Test Acc = %4.2f%% +/- %4.2f%%' %(acc_mean, 1.96* acc_std/np.sqrt(len(valid_loader))))
        print('Correct/Total: {}/{}'.format(acc_count.sum(), bin_count.sum()))
        print('Bin Accuracy: ' + str(acc_B))
        print('ECE: {:.5f}, MCE: {:.5f}'.format(ECE, MCE))

    if csv_flag:
        outfile.close()
        return None
    else:
        return accuracies

def sample_nn_weight(meta_params):
    w = {}
    for key in meta_params['mean'].keys():
        eps_sampled = torch.randn_like(input=meta_params['mean'][key], device=device)
        w[key] = meta_params['mean'][key] + eps_sampled * torch.exp(meta_params['logSigma'][key])

    return w

def initialize_q(key_list):
    q = dict.fromkeys(['mean', 'logSigma'])
    for para in q.keys():
        q[para] = {}
        for key in key_list:
            q[para][key] = 0
    return q

def platt_scale(theta, loader):
    model_with_temperature = ModelWithTemperature(theta, loader)
    temp = model_with_temperature.set_temperature()
    return temp

class ModelWithTemperature(nn.Module):
    def __init__(self, theta, loader):
        super(ModelWithTemperature, self).__init__()
        self.theta = theta
        self.loader = loader
        self.temperature = nn.Parameter(torch.ones(1))
    
    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).unsqueeze(1).expand(*logits.size())
        return torch.log(nn.functional.softmax(logits / temperature, dim=-1).mean(0))

    def set_temperature(self):
        self.to(device)
        nll_criterion = nn.NLLLoss().to(device)
        optimizer  = torch.optim.LBFGS([self.temperature], lr=0.001, max_iter=50)

        for x,_ in self.loader:
            x = x.to(device)
            x_t = x[:,:num_training_samples_per_class,:,:,:].contiguous().view(num_classes_per_task*num_training_samples_per_class, *x.size()[2:])  # support data 
            x_v = x[:,num_training_samples_per_class:,:,:,:].contiguous().view(num_classes_per_task*num_val_samples_per_class, *x.size()[2:])  # query data 
            y_t = torch.from_numpy(np.repeat(range(num_classes_per_task), num_training_samples_per_class)).to(device)  # label for support data
            y_v = torch.from_numpy(np.repeat(range(num_classes_per_task), num_val_samples_per_class)).to(device)  # label for query data

            q = adapt_to_task(x=x_t, y=y_t, theta0=self.theta, approx=True)
            raw_scores, _ = predict(x=x_v, q=q, num_models=Lv) # Lv x num_samples x num_classes

            def eval():
                loss = nll_criterion(self.temperature_scale(raw_scores.detach()), y_v)
                loss.backward()
                return loss
            optimizer.zero_grad()
            optimizer.step(eval)

        return self.temperature.item()

if __name__ == "__main__":
    main()
