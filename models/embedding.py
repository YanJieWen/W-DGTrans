'''
@File: embedding.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 22, 2025
@HomePage: https://github.com/YanJieWen
'''

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from einops import rearrange

#出来之后需要乘上mask_gt
# class Se_ResNet(nn.Module):
#     def __init__(self,fan_in,fan_out,reduce=16):
#         super().__init__()
#         self.residual = basicblock(fan_in,fan_out)
#         self.shortcut = nn.Sequential(nn.Conv2d(fan_in,fan_out,kernel_size=1),
#                                       nn.BatchNorm2d(fan_out))
#         self.se = nn.Sequential(nn.Linear(fan_out,fan_out//reduce),
#                                 nn.ReLU(inplace=True),
#                                 nn.Linear(fan_out//reduce,fan_out),
#                                 nn.Sigmoid())
#
#     def forward(self,x):
#         x = rearrange(x,'b t n d->b d t n')
#         identity = x
#         x = self.residual(x)
#         b,d,_,_ = x.shape
#         y = nn.AvgPool2d((x.size()[-2:]))(x)
#         y = y.view(b,-1)
#         y = self.se(y).view(b, d, 1, 1)
#         y = x*y
#         out = y+self.shortcut(identity)
#         return rearrange(out,'b d t n -> b t n d',d=d)


class Se_ResNet(nn.Module):
    def __init__(self,fan_in,fan_out,reduce=16):
        super().__init__()
        self.out_dim = fan_out
        self.w_0 = nn.Linear(fan_in,fan_out)
        self.w_1 = nn.Linear(fan_out,fan_out//4)
        self.w_2 = nn.Linear(fan_out//4,fan_out)
        self.se = nn.Sequential(nn.Linear(fan_out, fan_out // reduce),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(fan_out//reduce,fan_out),
                                        nn.Sigmoid())
        self.act = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.w_0(x)
        identity = x
        x = self.w_1(x)
        x = self.w_2(x)
        x+=identity
        x = rearrange(x, 'b t n d->b d t n',d=self.out_dim)
        b, d, _, _ = x.shape
        y = nn.AvgPool2d((x.size()[-2:]))(x)
        y = y.view(b,-1)
        y = self.se(y).view(b, d, 1, 1)
        y = x*y#bdtn
        out = rearrange(y,'b d t n -> b t n d',d=d)
        out = out+identity
        return self.act(out)




class basicblock(nn.Module):
    '''
    Residual basic block: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test5_resnet/model.py
    '''
    def __init__(self,fan_in,fan_out):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=fan_in,out_channels=fan_out//4,kernel_size=1,stride=1,bias=False)
        self.bn1 = nn.BatchNorm2d(fan_out//4)
        self.conv2 = nn.Conv2d(in_channels=fan_out//4,out_channels=fan_out,kernel_size=1,stride=1,bias=False)
        self.bn2 = nn.BatchNorm2d(fan_out)
        self.shortcut = nn.Sequential(nn.Conv2d(fan_in,fan_out,kernel_size=1),nn.BatchNorm2d(fan_out))
    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if x.size(1)!=identity.size(1):
            identity = self.shortcut(identity)
        out = x+identity
        return nn.ReLU(inplace=True)(out)


class PositionEmbedding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = np.array([[l/np.power(10000,(d-d%2)/d_model) for d in range(d_model)] for l in range(max_len)])
        pe[:,0::2] = np.sin(pe[:,0::2])
        pe[:,1::2] = np.cos(pe[:,1::2])
        pe = torch.from_numpy(pe)#1xtx1xd
        self.register_buffer('pe',pe)
    def forward(self,x):
        x = x+Variable(self.pe[None,:,None,:], requires_grad=False)
        return self.dropout(x.to(torch.float))





# if __name__ == '__main__':
#     x = torch.randn((32,8,42,2))
#     model = Se_ResNet(2,64)
#     print(model(x).shape)
#     pmodel = PositionEmbedding(512,max_len=100)
#     print(pmodel.pe.shape)
#     import matplotlib.pyplot as plt
#     cax = plt.matshow(pmodel.pe)
#     plt.gcf().colorbar(cax)
#     plt.show()
