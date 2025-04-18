'''
@File: drop_path.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 23, 2025
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn as nn

def drop_path_fn(x,drop_prob,training):
    if drop_prob==0. or not training:
        return x
    keep_prob = 1-drop_prob
    shape = (x.shape[0],)+(1,)*(x.ndim-1)
    random_tensor = keep_prob+torch.rand(shape,dtype=x.dtype,device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob)*random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self,drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self,x):
        return drop_path_fn(x,self.drop_prob,self.training)