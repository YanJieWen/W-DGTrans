'''
@File: norm.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 23, 2025
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn as nn




class LayerNorm(nn.Module):
    def __init__(self,feats,eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(feats))
        self.b_2 = nn.Parameter(torch.zeros(feats))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2


def batch_norm(is_training,x,gamma,beta,moving_mean,moving_var,eps,momentum):
    if not is_training:
        x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        mean = x.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        var = ((x-mean)**2).mean(dim=(0,1,2),keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1 - momentum) * mean#更新
        moving_var = momentum * moving_var + (1 - momentum) * var
    Y = gamma * x_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self,feats):
        shape = (1,feats,1,1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self,x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        Y, self.moving_mean, self.moving_var = batch_norm(self.training,
                                                          x, self.gamma, self.beta, self.moving_mean,
                                                          self.moving_var, eps=1e-5, momentum=0.9)
        return Y

