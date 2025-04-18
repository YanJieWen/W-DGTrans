'''
@File: preprocess.py
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


class Preprocesser(nn.Module):
    def __init__(self,device='cuda:0'):
        super().__init__()
        self.device = device
        self.counts = 0

    def trmask(self,tr):
        _,_,t = tr[0].shape
        num_batch = len(tr)
        out = torch.zeros(num_batch,self.counts,2,t,device=self.device)
        for j in range(num_batch):
            n = tr[j].size(0)
            out[j,:n] = tr[j]
        return out
    def mmask(self,mask):#未缺失的轨迹点为0，缺失的点>1,令不足nmax的轨迹设置为-1
        _,t = mask[0].shape
        num_batch = len(mask)
        out = torch.full((num_batch,self.counts,t),-1,device=self.device)
        for j in range(num_batch):
            n = mask[j].size(0)
            out[j,:n] = mask[j]
        return out
    def vmask(self,vnode):
        t,_,_ = vnode[0].shape
        num_batch = len(vnode)
        out = torch.zeros(num_batch,t,self.counts,2,device=self.device,dtype=torch.float)
        for j in range(num_batch):
            n = vnode[j].size(1)
            out[j,:,:n] = vnode[j]
        return out
    def amask(self,am):
        t,_,_ = am[0].shape
        num_batch = len(am)
        out = torch.zeros(num_batch,t,self.counts,self.counts,device=self.device,dtype=torch.float)
        for j in range(num_batch):
            n = am[j].size(1)
            out[j,:,:n,:n] = am[j]
        return out


    def forward(self,dataset):
        '''
        根据YOLOv8中的最大实例数目将每个样本组合为Batch:preprocess()
        https://github.com/YanJieWen/MSCMR-Pytorch-Implementation/blob/master/ultralytics/utils/loss.py
        :param dataset: List[Tensor]-->w/o batch_size
        :return: Tuple(Tensor)
        '''
        tr_o,tr_p,tr_ro,tr_rp,m_o,m_p,nl_m,inv_o,inv_p,v_o,a_o,v_p,a_p = dataset
        counts = torch.from_numpy(np.array([x.size(0) for x in tr_o])).amax()
        #将蒙版为0的地方转为1非0的地方转为

        self.counts = counts
        _tro,_trp = list(map(self.trmask,[tr_o,tr_p]))
        _mo,_mp = list(map(self.mmask,[m_o,m_p]))
        _vo,_vp = list(map(self.vmask,[v_o,v_p]))
        _ao,_ap = list(map(self.amask,[a_o,a_p]))
        _nlm = torch.zeros(_tro.shape[0],self.counts)
        for i in range(_nlm.shape[0]):
            n = len(nl_m[i])
            _nlm[i,:n] = nl_m[i]
        _invo = torch.stack(inv_o,dim=0)
        _invp = torch.stack(inv_p,dim=0)
        # _mask = torch.concatenate((_mo,_mp),dim=-1)
        # _mask = _mask.sum(dim=2,keepdim=True)
        # _mask.ge_(0)
        return _tro,_trp,_mo,_mp,_vo,_vp,_ao,_ap,_invo,_invp

