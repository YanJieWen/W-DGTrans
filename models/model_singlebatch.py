'''
@File: model_singlebatch.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 4月 01, 2025
@HomePage: https://github.com/YanJieWen
'''



import torch
import torch.nn as nn

from .preprocess import Preprocesser
from .embedding import Se_ResNet,PositionEmbedding
from .encoder import Encoder,EncoderLayer
from .decoder import Decoder,DecoderLayer
from .head import DistributionHead
from .txpcnn import TXPCnns

class WDGTrans_sb(nn.Module):
    def __init__(self,in_dim,dim,obs_len,pred_len,num_layer,drop_path_rate,out_dim,device,**kwargs):
        super().__init__()
        # self.preprocess = Preprocesser(device)
        self.device = device
        #encoding-part
        self.embedding_enc = Se_ResNet(in_dim,dim)
        self.pe_enc = PositionEmbedding(dim,max_len=obs_len)
        self.encoder = Encoder(EncoderLayer,num_layer,dim,drop_path_rate=drop_path_rate,**kwargs)
        # decoding_part
        self.embedding_dec = Se_ResNet(in_dim, dim)
        self.pe_dec = PositionEmbedding(dim, max_len=pred_len)
        self.decoder = Decoder(DecoderLayer, num_layer, dim, drop_path_rate=drop_path_rate, **kwargs)
        # self.txpcnn = TXPCnns(n_layers=5,kernel_size=3,obs_len=obs_len,pred_len=pred_len)
        # head
        self.header = DistributionHead(dim, out_dim, use_mdn=False)

    def forward(self,dataset):
        '''
        :param dataset:List[Tuple()]
        :return: bxtxnx2,bxn(bool)
        '''
        _,_,_,_,mask_o,mask_p,_,_,_,v_obs,a_obs,v_pred,a_pred = dataset
        mask_o,mask_p,v_obs,a_obs,v_pred,a_pred = map(lambda x:x[0].to(self.device).unsqueeze(0),
                                        (mask_o,mask_p,v_obs,a_obs,v_pred,a_pred))
        # tr_o,tr_p,mask_o,mask_p,v_obs,v_pred,a_obs,a_pred,inv_o,inv_p = self.preprocess(dataset)
        #encoder
        mask_obs = mask_o.transpose(1,2)[:,:,:,None]==0
        mem = self.encode(v_obs,mask_o,a_obs,mask_obs)
        #decoder
        v_pred = torch.concatenate((v_obs[:, -1:, :, :], v_pred[:, :-1, :, :]), dim=1)
        a_pred = torch.concatenate((a_obs[:, -1:, :, :], a_pred[:, :-1, :, :]), dim=1)
        mask_pred = torch.concatenate((mask_o[:, :, -1:], mask_p[:, :, :-1]), dim=-1)
        mask_tgt = mask_pred.transpose(1, 2)[:, :, :, None]==0
        x = self.decode(v_pred, mem, mask_o, mask_pred, a_pred, mask_tgt)
        # out head
        out = self.header(x)

        # x = self.txpcnn(x)
        # #out head
        # out = self.header(x)
        out_mask = torch.sum(torch.concatenate((mask_o,mask_p),dim=-1),dim=-1)==0
        return out,out_mask

    def encode(self,x,mask_o,adjm,mask_gt):
        x = self.embedding_enc(x)
        x = self.pe_enc(x)#bxtxnxd
        x = x * mask_gt.to(torch.float)
        x = self.encoder(x,mask_o,adjm)
        return x
    def decode(self,x,memory,mask_o,mask_p,adjm,mask_gt):
        x = self.embedding_dec(x)
        x = self.pe_dec(x)
        x = x*mask_gt.to(torch.float)
        x = self.decoder(x,memory,mask_o,mask_p,adjm)
        return x
