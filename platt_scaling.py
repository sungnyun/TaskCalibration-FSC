'''
code modified from
https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
'''

import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

def platt_scale(model, loader, params):
    model_with_temperature = ModelWithTemperature(model, loader, params)
    temp = model_with_temperature.set_temperature()
    return temp


class ModelWithTemperature(nn.Module):
    def __init__(self, model, loader, params):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.loader = loader
        self.n_way = params.test_n_way
        self.n_query = 15
        self.model.n_query = self.n_query
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self):
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        optimizer  = optim.LBFGS([self.temperature], lr=0.001, max_iter=50)

        for input, label in self.loader:
            input = input.cuda()
            logits,_ = self.model.set_forward(input)
            y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query)).cuda()
            def eval():
                loss = nll_criterion(self.temperature_scale(logits.detach()), y_query)
                loss.backward()
                return loss
            optimizer.zero_grad()
            optimizer.step(eval)

        return self.temperature.item()
