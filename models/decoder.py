'''
@File: decoder.py
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

from .norm import LayerNorm,BatchNorm
from .attentions import TemporalAttention,SpatialAttention,FFN
from .drop_path import DropPath

NORM_LAYER = {
    'ln': LayerNorm,
    'bn':BatchNorm
}

class Decoder(nn.Module):
    def __init__(self,layer,n,dim,norm_name='ln',drop_path_rate=0.1,**kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = NORM_LAYER.get(norm_name)(dim)
        assert self.norm is not None, f'{norm_name} is not supported'
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,n)]
        for i in range(n):
            self.layers.append(layer(dim=dim,drop_path=dpr[i],**kwargs))

    def create_mask(self,x_0,x_1):
        '''
        创建蒙版组合蒙版/未来空间/过去时间
        :param x_0:bxnxt_obs
        :param x_1:bxnxt_pred
        :return:bxnxt_predxt_pred&bxnxt_obsxt_obs
        '''
        b_1,n_1,t_k = x_0.size()#过去的蒙版
        b_2,n_2,t_q = x_1.size()#未来的蒙版
        assert n_1==n_2,f'{int(n_1)} is not equal to {int(n_2)}'
        src_mask = x_0[:,:,None,:]#bxnx1xt_obs
        tgt_mask = x_1[:,:,None,:]#bxnx1xt_pred
        casual_mask = torch.triu(torch.ones(t_q,t_q))-torch.eye(t_q)
        cmask = casual_mask==0#t_predxt_pred
        cmask = cmask[None,None,:,:]#1x1xt_predxt_pred
        cmask = cmask.expand((b_2,n_2,t_q,t_q))
        tgt_mask = tgt_mask.expand((b_2,n_2,t_q,t_q))
        combined_mask = 1-(tgt_mask==0&cmask.to(tgt_mask.device)).to(torch.float)#为0的地方不被填充，不为0的填充-inf
        t_mask = src_mask.expand((b_1,n_1,t_q,t_k))
        smask = x_1.transpose(1,2)##bxtxn
        smask = smask[:,:,None,:]#bxtx1xn
        s_mask = smask.expand((b_2, t_q, n_2, n_2))
        return combined_mask,s_mask,t_mask

    def forward(self,x,memory,mask_o,mask_p,adjm):
        combined_mask,s_mask,t_mask = self.create_mask(mask_o,mask_p)
        for layer in self.layers:
            x = layer(x,memory,combined_mask,t_mask,s_mask,adjm)
        return self.norm(x)



class DecoderLayer(nn.Module):
    def __init__(self,dim,num_heads,ffn_ratio=4,qkv_bias=True,act_layer=nn.GELU,
                 norm_layer=LayerNorm,attn_drop=0.,drop=0.,drop_path=0.,if_ta=True,if_sa=True,if_ita=True,**kwargs):

        super().__init__()
        self.tmad = TemporalAttention(dim,num_heads=num_heads,qkv_bias=qkv_bias,
                                      attn_drop=attn_drop,proj_drop=drop) if if_ta else nn.Identity()
        self.smad = SpatialAttention(dim,num_heads=num_heads,qkv_bias=qkv_bias,
                                      attn_drop=attn_drop,proj_drop=drop) if if_sa else nn.Identity()
        self.itmad = TemporalAttention(dim,num_heads=num_heads,qkv_bias=qkv_bias,
                                      attn_drop=attn_drop,proj_drop=drop) if if_ita else nn.Identity()
        self.ffn = FFN(fan_in=dim,hidden_dim=ffn_ratio*dim,act_layer=act_layer,drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

    def forward(self,x,memory,combined_mask,t_mask,s_mask,adjm):
        shortcut = x
        if not isinstance(self.tmad,nn.Identity):
            x = self.norm1(x)
            x = shortcut+self.drop_path(self.tmad(x,x,x,combined_mask))
            shortcut = x
        else:
            x = self.tmad(x)
        if not isinstance(self.smad,nn.Identity):
            x = self.norm2(x)
            x = shortcut+self.drop_path(self.smad(x,x,x,s_mask,adjm))
            shortcut = x
        else:
            x = self.smae(x)
        if not isinstance(self.itmad,nn.Identity):
            x = self.norm3(x)
            x = shortcut+self.drop_path(self.itmad(x,memory,memory,t_mask))
            shortcut = x
        else:
            x = self.itmad(x)
        x = shortcut+self.drop_path(self.ffn(self.norm4(x)))
        return x


# if __name__ == '__main__':
#     from thop import profile
#     from thop import clever_format
#     model = Decoder(DecoderLayer, 6, norm_name='ln', drop_path_rate=0.1, dim=128, num_heads=4, ffn_ratio=4,
#                     qkv_bias=True, act_layer=nn.GELU, )
#     tgt = torch.rand(32,12,42,128)
#     memory = torch.rand(32,8,42,128)
#     adjm = torch.rand(32, 12, 42, 42)
#     #create mask
#     mask_o = torch.zeros(32,42,8)
#     index = torch.asarray(np.array([[[0, 2, 2], [3, 4, 4]]]), dtype=torch.long)
#     mask_o.scatter_(-1, index, torch.ones_like(index, dtype=torch.float))
#     mask_p = torch.zeros(32, 42, 12)
#     index = torch.asarray(np.array([[[0, 1, 4], [5, 8, 9]]]), dtype=torch.long)
#     mask_p.scatter_(-1, index, torch.ones_like(index, dtype=torch.float))
#     flops, parms = profile(model, inputs=(tgt, memory,mask_o,mask_p,adjm,))
#     flops, params = clever_format([flops, parms], "%.3f")
#     print(params, flops)