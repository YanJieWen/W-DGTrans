'''
@File: txpcnn.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 30, 2025
@HomePage: https://github.com/YanJieWen
'''


import torch.nn as nn


class TXPCnns(nn.Module):
    def __init__(self,n_layers,kernel_size,obs_len,pred_len):
        super().__init__()
        self.number_layers = n_layers
        pad = (kernel_size-1)//2
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(obs_len,pred_len,kernel_size,padding=pad))
        for j in range(1,self.number_layers):
            self.tpcnns.append(nn.Conv2d(pred_len,pred_len,kernel_size,padding=pad))
        self.tpcnn_out = nn.Conv2d(pred_len,pred_len,kernel_size,padding=pad)

        self.preuls = nn.ModuleList()
        for j in range(self.number_layers):
            self.preuls.append(nn.PReLU())


    def forward(self,x):
        '''

        :param x: bxtxnx5
        :return: bxtpxnx5
        '''
        x = self.preuls[0](self.tpcnns[0](x))
        for k in range(1,self.number_layers):
            x = self.preuls[k](self.tpcnns[k](x))+x
        out = self.tpcnn_out(x)
        return out


# if __name__ == '__main__':
#     import torch
#     x = torch.rand(32,8,42,5)
#     model = TXPCnns(5,3,8,12)
#     print(model(x).shape)