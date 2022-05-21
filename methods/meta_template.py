import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod


class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, change_way = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input) 
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way 
        self.bin_count = np.zeros(10)
        self.acc_count = np.zeros(10)
        self.conf_count = np.zeros(10)

    @abstractmethod
    def set_forward(self, x):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass
    
    @torch.no_grad()
    def measure_uncertainty(self, mean_features, no_eye=False, unbiased=False):
        '''
        Compute task uncertainty with class-wise average features.
        Each element of the similarity matrix is cosine similarity.
        '''
        if unbiased:
            bias = mean_features.mean(0)
            mean_features = mean_features - bias
        dot_product = torch.mm(mean_features, mean_features.t())
        norm = mean_features.pow(2).sum(1).sqrt().unsqueeze(1)
        normsq = torch.mm(norm, norm.t())
        similarity_matrix = dot_product / normsq

        if no_eye:
            similarity_matrix = similarity_matrix - torch.diag(torch.diag(similarity_matrix, 0))

        task_uncertainty = torch.norm(similarity_matrix, p='fro')
        
        return task_uncertainty
    
    # @torch.no_grad()
    # def measure_uncertainty_cross(self, mean_features_spt, mean_features_qry, unbiased=False):
    #     '''
    #     Compute task uncertainty with class-wise average features.
    #     Each element of the similarity matrix is cosine similarity.
    #     '''
    #     if unbiased:
    #         bias_spt = mean_features_spt.mean(0)
    #         bias_qry = mean_features_qry.mean(0)
    #         mean_features_spt = mean_features_spt - bias_spt
    #         mean_features_qry = mean_features_qry - bias_qry

    #     dot_product = torch.mm(mean_features_spt, mean_features_qry.t())
    #     norm_spt = mean_features_spt.pow(2).sum(1).sqrt().unsqueeze(1)
    #     norm_qry = mean_features_qry.pow(2).sum(1).sqrt().unsqueeze(1)
    #     normsq = torch.mm(norm_spt, norm_qry.t())
    #     similarity_matrix = dot_product / normsq

    #     diag = torch.diag(torch.diag(similarity_matrix, 0))
    #     task_uncertainty = (similarity_matrix - diag).mean() - diag.mean() 
        
    #     return task_uncertainty

    def forward(self, x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query
        
    def first_feature(self,x,is_feature):
        x    = Variable(x.cuda())
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:]) 
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:,:1]
        z_query     = z_all[:,1:]

        return z_support, z_query
        
    def correct(self, x, eval_temp=1.0, Brier=False, NLL=False):       
        scores, mean_features = self.set_forward(x)
        task_uncertainty = self.measure_uncertainty(mean_features.detach())
        y_query = np.repeat(range(self.n_way), self.n_query)
        scores_prob = nn.functional.softmax(scores / eval_temp, 1)

        topk_scores, topk_labels = scores_prob.data.topk(1, 1, True, True)
        topk_conf = topk_scores.cpu().numpy()
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
                
        for i, (ind, conf) in enumerate(zip(topk_ind, topk_conf)):
            correct = int(topk_ind[i,0] == y_query[i])
            bin_id = int(conf*10)
            if bin_id == 10:
                bin_id = 9
            self.bin_count[bin_id] += 1
            self.acc_count[bin_id] += correct  # for acc(B_m)
            self.conf_count[bin_id] += conf    # for conf(B_m)
        # if Brier:
        #     target = F.one_hot(torch.from_numpy(y_query), num_classes=self.n_way).float()
        #     self.brier.append(torch.sum((scores_prob.cpu() - target)**2, 1).detach().numpy())
        # if NLL:
        #     scores_prob = scores_prob.contiguous().cpu().detach()
        #     self.nll.append(-torch.log(torch.stack([scores_prob[i, y_query[i]] for i in range(len(y_query))])).mean().item())

        return float(top1_correct), len(y_query), task_uncertainty

    def train_loop(self, epoch, train_loader, optimizer, logfile):
        print_freq = 10
        avg_loss=0
        for i, (x,_ ) in enumerate(train_loader):
            self.n_query = x.size(1) - self.n_support           
            if self.change_way:
                self.n_way  = x.size(0)
                
            optimizer.zero_grad()
            loss = self.set_forward_loss(x)
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss+loss.item()

            if i % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i, len(train_loader), avg_loss/float(i+1)))

    def test_loop(self, test_loader, return_std=False, threshold=None):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            if self.change_way:
                self.n_way  = x.size(0)
            correct_this, count_this, _ = self.correct(x)
            acc_all.append(correct_this/count_this *100)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +/- %4.2f%%' %(iter_num,  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean
