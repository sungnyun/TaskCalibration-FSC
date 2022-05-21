# This code is modified from https://github.com/dragen1860/MAML-Pytorch and https://github.com/katerakelly/pytorch-maml 

import random
import numpy as np
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import backbone
from methods.meta_template import MetaTemplate


class TCMAML(MetaTemplate):
    def __init__(self, model_func, n_way, n_support, train_lr, approx=True, maml=False):
        super(TCMAML, self).__init__(model_func, n_way, n_support, change_way=False)

        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = backbone.Linear_fw(self.feat_dim, n_way)
        self.classifier.bias.data.fill_(0)
        
        self.linear_scaling = False
        self.maml = maml

        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = train_lr
        self.approx = approx  # first order approx. default: True   
        self.T = 1  # temperature scaling factor

    def forward(self, x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def set_forward(self, x):
        x = x.cuda()
        x_var = Variable(x)
        x_a_i = x_var[:,:self.n_support,:,:,:].contiguous().view( self.n_way* self.n_support, *x.size()[2:])  # support data 
        x_b_i = x_var[:,self.n_support:,:,:,:].contiguous().view( self.n_way* self.n_query,   *x.size()[2:])  # query data
        y_a_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_support))).cuda()  # label for support data

        fast_parameters = list(self.parameters())
        for weight in self.parameters():
            weight.fast = None
        self.zero_grad()

        for task_step in range(self.task_update_num):
            out = self.feature.forward(x_a_i)
            scores = self.classifier.forward(out)
            
            set_loss = self.loss_fn(scores, y_a_i)
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)  # build full graph support gradient of gradient
            if self.approx:
                grad = [g.detach()  for g in grad]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py 
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]  # create weight.fast 
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k]
                fast_parameters.append(weight.fast)

        out = self.feature.forward(x_b_i)  # average features with query examples
        scores = self.classifier.forward(out)
        out = out.contiguous().view(self.n_way, self.n_query, -1)
        mean_features = out.mean(dim=1)
        # with torch.no_grad():
        #     out_spt = self.feature.forward(x_a_i)
        #     scores_spt = self.classifier.forward(out_spt)
        #     out_spt = out_spt.contiguous().view(self.n_way, self.n_support, -1)
        #     mean_features_spt = out_spt.mean(dim=1)
        
        return scores, mean_features
        
    def set_forward_loss(self, x):
        scores, mean_features = self.set_forward(x)
        y_b_i = Variable(torch.from_numpy(np.repeat(range(self.n_way), self.n_query))).cuda()
        loss = self.loss_fn(scores, y_b_i)
        task_uncertainty = self.measure_uncertainty(mean_features.detach())
        # task_uncertainty = self.measure_uncertainty_cross(mean_features_spt, mean_features.detach())
        # acc_qry = (torch.max(scores, dim=1)[1] == y_b_i).float().mean()
        # acc_spt = (torch.max(scores_spt.cpu().data, dim=1)[1] == torch.arange(self.n_way).repeat_interleave(self.n_support)).float().mean()

        return loss, task_uncertainty
        
    def train_loop(self, epoch, train_loader, optimizer, logfile, exclude_task=False): 
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        task_uncertainty_all = []
        uncertainty = []
        optimizer.zero_grad()

        for i, (x,_) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0)

            loss, task_uncertainty = self.set_forward_loss(x)
            avg_loss = avg_loss + loss.item()
            loss_all.append(loss)
            uncertainty.append(task_uncertainty.item())
            task_uncertainty_all.append(task_uncertainty)

            task_count += 1
            # TCMAML updates several aggregated tasks at one time
            if task_count == self.n_task:
                # vanilla MAML
                if self.maml:
                    task_uncertainty_all = torch.ones(self.n_task).cuda() / self.n_task
                # linear scaling (LS)
                elif self.linear_scaling:
                    task_uncertainty_all = (1/torch.stack(task_uncertainty_all)) / torch.sum(1/torch.stack(task_uncertainty_all))
                # exponential scaling (ES) - tcmaml default
                else:
                    task_uncertainty_all = nn.functional.softmax(- torch.stack(task_uncertainty_all) / self.T, dim=-1)

                # exclude most uncertain task
                if exclude_task:
                    _, index = torch.min(task_uncertainty_all, dim=-1)
                    loss_all = loss_all[:index]+loss_all[index+1:]
                    task_uncertainty_all = torch.ones(self.n_task-1).cuda() / (self.n_task-1)

                task_uncertainty_all = task_uncertainty_all.detach()

                loss_q = torch.mul(task_uncertainty_all, torch.stack(loss_all)).sum(0)
                optimizer.zero_grad()
                loss_q.backward()
                optimizer.step()
                
                task_count = 0
                loss_all = []
                task_uncertainty_all = []
                
            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))
                
        avg_uncertainty, std_uncertainty = np.array(uncertainty).mean(), np.array(uncertainty).std()
        logfile.write('Epoch {}, Loss {:.4f}, Train-Uncertainty {:.4f} \n'.format(epoch+1, avg_loss/len(train_loader), avg_uncertainty))
        # max_sim, min_sim = np.array(uncertainty).max(), np.array(uncertainty).min()
        # avg_uncertainty_spt, std_uncertainty_spt = np.array(uncertainty_spt).mean(), np.array(uncertainty_spt).std()
        # avg_acc_spt, std_acc_spt = np.array(acc['spt']).mean(), np.array(acc['spt']).std()
        # avg_acc_qry, std_acc_qry = np.array(acc['qry']).mean(), np.array(acc['qry']).std()
        # logfile.write('Epoch {}, Loss {:.4f}, Avg-Sim {:.4f}, Max-Sim {:4f}, Min-Sim {:4f} \n'.format(
        #     epoch+1, avg_loss/len(train_loader), avg_uncertainty, max_sim, min_sim))
        # logfile.write('Epoch {}, Avg_Acc_spt {:.4f}, Avg_Acc_qry {:.4f}, Std_Acc_spt {:.4f}, Std_Acc_qry {:.4f} \n'.format(
        #             epoch, avg_acc_spt, avg_acc_qry, std_acc_spt, std_acc_qry))
                      
    def test_loop(self, test_loader, return_std=False, threshold=None, eval_temp=1.0): 
        acc_all = []
        task_uncertainty_all = []
        not_evaluated = 0
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way  ==  x.size(0)
            correct_this, count_this, task_uncertainty = self.correct(x, eval_temp=eval_temp)
            if threshold is not None:
                if task_uncertainty.item() > threshold:
                    not_evaluated += 1
                    continue
            task_uncertainty_all.append(task_uncertainty.item())
            acc_all.append(correct_this/count_this *100)

        avg_uncertainty = np.array(task_uncertainty_all).mean()
        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        if threshold is not None:
            print('{} / {} evaluated, {} tasks held out due to higher uncertainty than {:.3f}'.format(iter_num - not_evaluated, iter_num, not_evaluated, threshold)) 
        print('{} Test Acc = {:4.2f}% +/- {:4.2f}%, Valid-Uncertainty {:4.4f}' .format(iter_num, acc_mean, 1.96* acc_std/np.sqrt(iter_num), avg_uncertainty))
        
        if return_std:
            # calibration result
            acc_B = np.array([self.acc_count[i] / self.bin_count[i] if self.bin_count[i] > 0 else 0 for i in range(10)])
            conf_B = np.array([self.conf_count[i] / self.bin_count[i] if self.bin_count[i] > 0 else 0 for i in range(10)])
            ECE = (self.bin_count / self.bin_count.sum() *abs(acc_B - conf_B)).sum()
            MCE = abs(acc_B - conf_B).max()
        
            print('Correct/Total: {}/{}'.format(self.acc_count.sum(), self.bin_count.sum()))
            print('Bin Accuracy: ' + str(acc_B))
            print('ECE: {:.5f}, MCE: {:.5f}'.format(ECE, MCE))
            
            return acc_mean, acc_std
        
        else:
            return acc_mean, avg_uncertainty

