'''
@File: attentions.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 23, 2025
@HomePage: https://github.com/YanJieWen
'''
import numpy as np
import torch
import torch.nn as nn

from einops import rearrange
import math

class TemporalAttention(nn.Module):
    def __init__(self,dim,num_heads=4,qkv_bias=True,attn_drop=0.,proj_drop=0.):
        super().__init__()
        assert dim%num_heads==0,f'{dim} can not be divided by {num_heads}'
        self.d_k = dim//num_heads
        self.heads = num_heads
        self.qkv = nn.ModuleList([nn.Linear(dim,dim,bias=qkv_bias) for _ in range(3)])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.attn = None

    def forward(self,query,key,value,mask=None):
        '''
        执行时间多头注意力
        :param query: bxtxn_mxd
        :param key: bxtxn_mx2
        :param value: bxtxn_mxd
        :param mask: bxnxtxt from bxnxt
        :return: bxtxn_mxd
        '''
        b,t_q,n,d = query.shape
        _,t_k,_,_ = key.shape
        query,key,value = [layer(x).view(b,-1,n,self.heads,self.d_k) for layer,x in zip(self.qkv,(query,key,value))]
        k,v = map(lambda x: rearrange(x,'b t n h d -> b h n t d',t=t_k),(key,value))
        q = torch.permute(query,(0,3,2,1,4)).contiguous()#bxhxnxtxd
        pw_scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.d_k)#bxhxnxtxt
        if mask is not None:
            mask = mask[:,None,:,:,:].expand_as(pw_scores)
            pw_scores.masked_fill_(mask!=0,-1e9)
        attn = self.softmax(pw_scores)
        # print('attn:',attn[0,0,0])
        # print('mask:',mask[0,0,0])
        attn = self.attn_drop(attn)
        self.attn = attn#bxhxnxtxt
        x = torch.matmul(attn,v)#bxhxnxtxd
        x = rearrange(x,'b h n t d -> b t n (h d)', h=self.heads)
        out = self.proj_drop(self.proj(x))
        return out



class SpatialAttention(nn.Module):
    def __init__(self,dim,num_heads=4,qkv_bias=True,attn_drop=0.,proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f'{dim} can not be divided by {num_heads}'
        self.d_k = dim // num_heads
        self.heads = num_heads
        self.qkv = nn.ModuleList([nn.Linear(dim, dim, bias=qkv_bias) for _ in range(3)])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.attn = None
    def forward(self,query,key,value,mask=None,adjm=None):
        '''
        执行时间多头注意力
        :param query: bxtxnxd
        :param key: bxtxnxd
        :param value: bxtxnxd
        :param mask: bxtxnxn from bxnxt
        :param adjm: bxtxnxn
        :return: bxtxnxd
        '''
        b,t,n,d = query.shape
        query, key, value = [layer(x).view(b, -1, n, self.heads, self.d_k) for layer, x in
                             zip(self.qkv, (query, key, value))]
        q,k,v = map(lambda x: rearrange(x,'b t n h d -> b h t n d',n=n),(query, key, value))
        pw_scores = torch.matmul(q,k.transpose(-2,-1)/math.sqrt(self.d_k))#bxhxtxnxn
        if mask is not None:
            mask = mask[:,None,:,:,:]
            pw_scores.masked_fill_(mask!=0,-1e9)
        if adjm is not None:
            adjm = adjm[:,None,:,:,:]
            attn = self.softmax(pw_scores)*adjm
        else:
            attn = self.softmax(pw_scores)
        attn = self.attn_drop(attn)#bxhxtxnxn
        self.attn = attn
        x = torch.matmul(attn,v)
        x = rearrange(x, 'b h t n d -> b t n (h d)', h=self.heads)
        out = self.proj_drop(self.proj(x))
        return out


class FFN(nn.Module):
    def __init__(self,fan_in,hidden_dim,act_layer,drop):
        super().__init__()
        self.w_1 = nn.Linear(fan_in,hidden_dim)
        self.w_2 = nn.Linear(hidden_dim,fan_in)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
    def forward(self,x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.w_2(x)
        x = self.drop2(x)
        return x



# if __name__ == '__main__':
    # model = TemporalAttention(128,4)
    # mask = torch.zeros(32,31,8,8)
    # index = torch.as_tensor(np.array([[[[0,2,3],[1,3,4]]]]),dtype=torch.long)
    # value = torch.ones_like(index,dtype=torch.float)
    # mask.scatter_(-1,index,value)
    # x = torch.rand(32,8,31,128)
    # print(model(x,x,x,mask).shape)

    # model = SpatialAttention(128,4)
    # mask = torch.zeros(32, 8, 31, 31)
    # index = torch.as_tensor(np.array([[[[0, 2, 3], [1, 3, 4]]]]), dtype=torch.long)
    # value = torch.ones_like(index, dtype=torch.float)
    # mask.scatter_(-1, index, value)
    # x = torch.rand(32, 8, 31, 128)
    # adjm = torch.rand(32,8,31,31)
    # print(model(x, x, x, mask,adjm).shape)




