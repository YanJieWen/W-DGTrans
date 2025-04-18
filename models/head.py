'''
@File: head.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 22, 2025
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn as nn


class DistributionHead(nn.Module):
    def __init__(self,dim,out_feats,use_mdn=False):
        super().__init__()
        self.outlayer = nn.Linear(dim,out_feats,bias=True)
        self.prrelu  = nn.PReLU()
        self.use_mdn = use_mdn

    def forward(self,x):
        x = self.outlayer(x)
        if not self.use_mdn:
            out = self.prrelu(x)
        else:
            out = x

        return out