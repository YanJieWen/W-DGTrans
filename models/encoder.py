'''
@File: encoder.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 22, 2025
@HomePage: https://github.com/YanJieWen
'''
import numpy as np
import torch
import torch.nn as nn
from .norm import LayerNorm,BatchNorm
from .attentions import TemporalAttention,SpatialAttention,FFN
from .drop_path import DropPath

NORM_LAYER = {
    'ln': nn.LayerNorm,
    'bn':BatchNorm
}

class Encoder(nn.Module):
    def __init__(self,layer,n,dim,norm_name='ln',drop_path_rate=0.1,**kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = NORM_LAYER.get(norm_name)(dim)
        assert self.norm is not None, f'{norm_name} is not supported'
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,n)]
        for i in range(n):
            self.layers.append(layer(dim=dim,drop_path=dpr[i],**kwargs))

    def create_mask(self,x):
        '''
        在encoder中需要创建基于时间/空间的键蒙版
        :param x: bxnxt
        :return: bxnxtxt&bxtxnxn
        '''
        b,n,t = x.shape
        t_mask= x[:,:,None,:]#bxnx1xt
        t_mask = t_mask.expand((b,n,t,t))
        s_mask = x.transpose(1,2)#bxtxn
        s_mask = s_mask[:,:,None,:]#bxtx1xn
        s_mask = s_mask.expand((b,t,n,n))
        return t_mask,s_mask

    # def _create_mask(self,v_obs):
    #     '''
    #     利用节点创建蒙版
    #     :param v_obs:bxtxnx2
    #     :return:
    #     '''
    #     v_obs = v_obs.transpose(1,2)
    #     b,n,t,_ = v_obs.shape
    #     mask = torch.sum(v_obs,dim=-1)==0 #需要掩码的地方为1，不需要掩码的为0
    #     mask = mask.to(torch.float)
    #     t_mask = mask[:,:,None,:]
    #     t_mask = t_mask.expand((b, n, t, t))
    #     s_mask = mask.transpose(1, 2)  # bxtxn
    #     s_mask = s_mask[:, :, None, :]  # bxtx1xn
    #     s_mask = s_mask.expand((b, t, n, n))
    #     return t_mask,s_mask


    def forward(self,x,mask_o,adjm):
        t_mask, s_mask = self.create_mask(mask_o)
        # t_mask, s_mask = self._create_mask(v_obs)
        for layer in self.layers:
            x = layer(x,t_mask, s_mask,adjm)
        return self.norm(x)


#搭建encoder layer
class EncoderLayer(nn.Module):
    def __init__(self,dim,num_heads,ffn_ratio=4,qkv_bias=True,act_layer=nn.ReLU,
                 norm_layer=LayerNorm,attn_drop=0.,drop=0.,drop_path=0.,if_ta=True,if_sa=True,**kwargs):

        super().__init__()
        self.tmae = TemporalAttention(dim,num_heads=num_heads,qkv_bias=qkv_bias,
                                      attn_drop=attn_drop,proj_drop=drop) if if_ta else nn.Identity()
        self.smae = SpatialAttention(dim,num_heads=num_heads,qkv_bias=qkv_bias,
                                      attn_drop=attn_drop,proj_drop=drop) if if_sa else nn.Identity()
        self.ffn = FFN(fan_in=dim,hidden_dim=ffn_ratio*dim,act_layer=act_layer,drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path>0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)

    def forward(self,x,t_mask,s_mask,adjm):
        shortcut = x
        if not isinstance(self.tmae,nn.Identity):
            x = self.norm1(x)
            x = shortcut+self.drop_path(self.tmae(x,x,x,t_mask))
            shortcut = x
        else:
            x = self.tmae(x)
        if not isinstance(self.smae,nn.Identity):
            x = self.norm2(x)
            x = shortcut+self.drop_path(self.smae(x,x,x,s_mask,adjm))
            shortcut = x
        else:
            x = self.smae(x)

        x = shortcut+self.drop_path(self.ffn(self.norm3(x)))
        return x


# if __name__ == '__main__':
#     from thop import profile
#     from thop import clever_format
#     model = Encoder(EncoderLayer,6,norm_name='ln',drop_path_rate=0.1,dim=128,num_heads=4,ffn_ratio=4,
#                     qkv_bias=True,act_layer=nn.GELU,)
#     x = torch.rand(32,8,42,128)
#     mask = torch.zeros(32,42,8)
#     index = torch.asarray(np.array([[[0,2,2],[3,4,4]]]),dtype=torch.long)
#     mask.scatter_(-1,index,torch.ones_like(index,dtype=torch.float))
#     adjm = torch.rand(32,8,42,42)
#     flops,parms = profile(model,inputs=(x,mask,adjm,))
#     flops, params = clever_format([flops, parms], "%.3f")
#     print(params,flops)
#     # print(model(x,mask,adjm).shape)
