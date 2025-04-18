'''
@File: model.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 24, 2025
@HomePage: https://github.com/YanJieWen
'''
import numpy as np
import torch
import torch.nn as nn

from .preprocess import Preprocesser
from .embedding import Se_ResNet,PositionEmbedding
from .encoder import Encoder,EncoderLayer
from .decoder import Decoder,DecoderLayer
from .head import DistributionHead

class WDGTrans(nn.Module):
    def __init__(self,in_dim,dim,obs_len,pred_len,num_layer,drop_path_rate,out_dim,device,**kwargs):
        super().__init__()
        self.preprocess = Preprocesser(device)
        #encoding-part
        self.embedding_enc = Se_ResNet(in_dim,dim)
        self.pe_enc = PositionEmbedding(dim,max_len=obs_len)
        self.encoder = Encoder(EncoderLayer,num_layer,dim,drop_path_rate=drop_path_rate,**kwargs)
        #decoding_part
        self.embedding_dec = Se_ResNet(in_dim,dim)
        self.pe_dec = PositionEmbedding(dim,max_len=pred_len)
        self.decoder = Decoder(DecoderLayer,num_layer,dim,drop_path_rate=drop_path_rate,**kwargs)
        #head
        self.header = DistributionHead(dim,out_dim,use_mdn=True)

    def forward(self,dataset):
        '''
        :param dataset:List[Tuple()]
        :return: bxtxnx5,bxn(bool),bxnx2xt(后续要被用于恢复绝对位置轨迹),bxt后续用于恢复绝对位置轨迹
        '''
        tr_o,tr_p,mask_o,mask_p,v_obs,v_pred,a_obs,a_pred,inv_o,inv_p,mask_gt = self.preprocess(dataset)
        mask_p_0 = torch.concatenate((mask_o[:,:,-1:],mask_p[:,:,:-1]),dim=-1)#需要输出计算损失的掩码不能改变mask_p
        v_pred = torch.concatenate((v_obs[:,-1:,:,:],v_pred[:,:-1,:,:]),dim=1)
        a_pred = torch.concatenate((a_obs[:,-1:,:,:],a_pred[:,:-1,:,:]),dim=1)
        mem = self.encode(v_obs,mask_o,a_obs,mask_gt)
        x = self.decode(v_pred,mem,mask_o,mask_p_0,a_pred,mask_gt)
        out = self.header(x)
        out_mask = torch.sum(torch.concatenate((mask_o,mask_p),dim=-1),dim=-1)==0
        traj_out = torch.concatenate((tr_o,tr_p),dim=-1)
        inv_out = torch.concatenate((inv_o,inv_p),dim=-1)
        return out,out_mask,traj_out,inv_out

    def encode(self,x,mask_o,adjm,mask_gt):
        x = self.embedding_enc(x)
        x = self.pe_enc(x)#bxtxnxd
        x = x*mask_gt[:,None,:,:].to(torch.float)
        x = self.encoder(x,mask_o,adjm)
        return x


    def decode(self,x,memory,mask_o,mask_p,adjm,mask_gt):
        x = self.embedding_dec(x)
        x = self.pe_dec(x)
        x = x*mask_gt[:,None,:,:].to(torch.float)
        x = self.decoder(x,memory,mask_o,mask_p,adjm)
        return x

#
# if __name__ == '__main__':
#     from thop import profile
#     from thop import clever_format
#     tr_o = [torch.rand(42,2,8)]
#     tr_p = [torch.rand(42,2,12)]
#     tr_ro = [torch.rand(42,2,8)]
#     tr_rp = [torch.rand(42,2,12)]
#     m_o = torch.zeros(42,8)
#     m_p = torch.zeros(42,12)
#     index = torch.asarray(np.array([[0,2,3],[1,7,5],[0,2,4]]),dtype=torch.long)
#     m_o.scatter_(-1,index,torch.ones_like(index,dtype=torch.float))
#     m_p.scatter_(-1,index,torch.ones_like(index,dtype=torch.float))
#     m_o = [m_o]
#     m_p = [m_p]
#     nl_m = [torch.ones(42,)]
#     inv_o = [torch.zeros(8,)]
#     inv_p = [torch.zeros(12,)]
#     v_o = [torch.rand(8,42,2)]
#     a_o = [torch.rand(8,42,42)]
#     v_p = [torch.rand(12,42,2)]
#     a_p = [torch.rand(12, 42, 42)]
#     dataset = (tr_o,tr_p,tr_ro,tr_rp,m_o,m_p,nl_m,inv_o,inv_p,v_o,a_o,v_p,a_p)
#     model = WDGTrans(in_dim=2,dim=128,obs_len=8,pred_len=12,num_layer=6,drop_path_rate=0.1,out_dim=5,device='cpu',
#                      num_heads=4)
#     # flops, parms = profile(model, inputs=(dataset,))
#     # flops, params = clever_format([flops, parms], "%.3f")
#     # print(params, flops)
#     [print(x.shape) for x in model(dataset)]