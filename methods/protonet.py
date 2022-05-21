# This code is modified from https://github.com/jakesnell/prototypical-networks 

import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate

class ProtoNet(MetaTemplate):
    def __init__(self, model_func,  n_way, n_support, task_calibration):
        super(ProtoNet, self).__init__(model_func, n_way, n_support, change_way=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.task_calibration = task_calibration
        self.linear_scaling = False

        self.n_task = 4
        self.T = 1
        if not self.task_calibration:
            self.n_task = 1

    def set_forward(self, x, is_feature = False):
        z_support, z_query  = self.parse_feature(x, is_feature)

        z_support   = z_support.contiguous()
        z_proto     = z_support.view(self.n_way, self.n_support, -1).mean(1) #the shape of z is [n_data, n_dim]
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

        dists = euclidean_dist(z_query, z_proto) #[n_way*n_qry, n_way]
        scores = -dists

        mean_features = dists.view(self.n_way, self.n_query, -1).mean(1)
        assert mean_features.size(1) == self.n_way

        return scores, mean_features


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())

        scores, mean_features = self.set_forward(x)
        task_uncertainty = self.measure_uncertainty(mean_features)

        return self.loss_fn(scores, y_query), task_uncertainty

    def train_loop(self, epoch, train_loader, optimizer, logfile, exclude_task=False):
        print_freq = 10 
        avg_loss = 0
        task_count = 0 
        loss_all = []
        task_uncertainty_all = []
        uncertainty = []

        for i, (x,_) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support
            assert self.n_way == x.size(0)
            
            loss, task_uncertainty  = self.set_forward_loss(x)
            avg_loss = avg_loss + loss.item()
            loss_all.append(loss)
            uncertainty.append(task_uncertainty.item())
            task_uncertainty_all.append(task_uncertainty)

            task_count += 1
            if task_count == self.n_task:
                # vanilla ProtoNet
                if not self.task_calibration:
                    task_uncertainty_all = torch.ones(self.n_task).cuda() / self.n_task
                # TCProtoNet
                else:
                    task_uncertainty_all = nn.functional.softmax(-torch.stack(task_uncertainty_all)/self.T, dim=-1)
                
                task_uncertainty_all = task_uncertainty_all.detach()

                loss_q = torch.mul(task_uncertainty_all, torch.stack(loss_all)).sum(0)
                optimizer.zero_grad()
                loss_q.backward()
                optimizer.step()

                task_count = 0
                loss_all = []
                task_uncertainty_all = []

            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))

        avg_uncertainty = np.array(uncertainty).mean()
        logfile.write('Epoch {}, Loss {:.4f}, Train-Uncertainty {:.4f} \n'.format(epoch+1, avg_loss/len(train_loader), avg_uncertainty))

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


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
