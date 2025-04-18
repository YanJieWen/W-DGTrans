'''
@File: evaluator.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 27, 2025
@HomePage: https://github.com/YanJieWen
'''

import torch

from collections import OrderedDict
import copy


from .evaluator_wodist import BaseEvaluator,anorm
from tools import ade,fde,rel2abs

class Evaluator_determin(BaseEvaluator):
    def __init__(self,model,device,k=20,obs_len=8,pred_len=12,**kwargs):
        super().__init__(model=model,k=k,**kwargs)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.device = device
        self.dist_trajs = OrderedDict()

    def greedy_decode(self,v_obs,a_obs,mask_o,mask_p,tr_rp):
        '''
        针对单个场景进行贪婪解码
        :param v_obs: txnx2
        :param a_obs: txnxn
        :param mask_o: nxto
        :param mask_p: nxtp
        :param tr_rp: nx2xt-->用于评估最佳的预测点,以预测的相对位置进行评估
        :return:bxtxnx2
        '''

        #init input tensors
        v_obs, a_obs, mask_o, mask_p = map(lambda x: x.to(self.device), (v_obs, a_obs, mask_o, mask_p))
        t_o,n,_ = v_obs.shape
        v_obs,a_obs,mask_o,mask_p = map(lambda x:x.unsqueeze(0),(v_obs,a_obs,mask_o,mask_p))
        #执行贪婪解码
        outv_pred = torch.zeros(1, self.pred_len, n, 2, dtype=torch.float, device=v_obs.device)  # 1xtxnx2
        outa_pred = torch.zeros(1, self.pred_len, n, n, dtype=torch.float, device=a_obs.device)  # 1xtxnxn
        outm_pred = torch.ones(1,n,self.pred_len,dtype=torch.float,device=mask_p.device) #1xnxt
        src_mask = mask_o.transpose(1,2)[:,:,:,None]==0
        mem = self.model.encode(v_obs, mask_o, a_obs, src_mask)
        # mask_pred = torch.concatenate((mask_o[:, :, -1:], mask_p[:, :, :-1]), dim=-1)
        # mask_tgt = mask_pred.transpose(1,2)[:, :, :, None] == 0
        mask_out = torch.sum(torch.concatenate((mask_o,mask_p),dim=-1),dim=-1)==0
        #逐步解码
        for t in range(self.pred_len):
            v_pred = torch.concatenate((v_obs[:,-1:,:,:],outv_pred[:,:-1,:,:]),dim=1)
            a_pred = torch.concatenate((a_obs[:, -1:, :, :], outa_pred[:, :-1, :, :]), dim=1)
            m_pred = torch.concatenate((mask_o[:,:,-1:],outm_pred[:,:,:-1]),dim=-1)
            mask_tgt = m_pred.transpose(1, 2)[:, :, :, None] == 0
            dec_out = self.model.decode(v_pred, mem, mask_o, m_pred, a_pred, mask_tgt)

            out_dist = self.model.header(dec_out)#bxtxnx5
            deter_traj = out_dist[0,t,:,:2].clone()
            #1.利用out_dist来更新outv_pred
            outv_pred[0,t] = deter_traj
            #3.更新a_pred
            outa_pred[0,t] = self.update_graph(deter_traj,mask_p[0,:,t])
            # 4.更新mask
            outm_pred[..., t] = mask_p[..., t]
        return outv_pred,mask_out


    def decodedisp(self,ppoints):
        pass


    def update_graph(self, pred, mask):
        '''
        根据预测结果更新图
        :param pred: nx2
        :param mask: n
        :return: nxn
        '''
        n, _ = pred.size()
        a = torch.zeros(n, n, dtype=torch.float, device=pred.device)
        for h in range(a.shape[0]):
            a[h, h] = 1
            if mask[h] != 0:
                continue
            for k in range(a.shape[1]):
                if k == h:
                    a[h, k] = 1
                else:
                    if mask[k] != 0:
                        continue
                    else:
                        l2_norm = anorm(pred[h], pred[k])
                        a[h, k] = l2_norm
        g = (a.transpose(0, 1) * torch.pow(torch.sum(a, dim=1), -1)).transpose(0, 1)
        return g

    def eval(self,dataloder):
        ades = []
        fdes = []
        scount = 0
        raw_data_dict = {}
        with torch.no_grad():
            for dataset in dataloder:
                tr_os, tr_ps, _, tr_rps, m_os, m_ps, _, _, inv_ps, v_os, a_os, v_ps,a_ps = dataset
                #遍历每一个场景
                for (tr_o, tr_p, tr_rp, mask_o, mask_p, v_obs, a_obs, inv_p) in zip(tr_os, tr_ps, tr_rps, m_os, m_ps,
                                                                                    v_os, a_os, inv_ps):
                    scount+=1
                    tr_o, tr_p, inv_p = map(lambda x: x.to(self.device), (tr_o, tr_p, inv_p))
                    tr_pred, mask = self.greedy_decode(v_obs, a_obs, mask_o, mask_p, tr_rp)
                    tr_pred, mask = tr_pred.squeeze(0), mask.squeeze(0)  # txnxd,n
                    tr_gt = torch.permute(tr_p[mask],(2,0,1)).contiguous().cpu().numpy()#txnx2
                    tr_obs = torch.permute(tr_o[mask],(2,0,1)).contiguous().cpu().numpy()#txnx2
                    init_node = tr_o[mask].cpu().numpy()[...,-1]#nxd
                    inv_p = inv_p.cpu().numpy()
                    raw_data_dict[scount] = {}
                    raw_data_dict[scount]['obs'] = copy.deepcopy(tr_obs)
                    raw_data_dict[scount]['pred'] = copy.deepcopy(tr_gt)
                    raw_data_dict[scount]['pred'] = []
                    _, num_objs, _ = tr_gt.shape
                    tr_pred = rel2abs(tr_pred[:, mask].cpu().numpy(), init_node, inv_p)#txnxd
                    raw_data_dict[scount]['pred']= copy.deepcopy(tr_pred)
                    _ade = ade(tr_pred, tr_gt)#选择的最优的trpred
                    _fde = fde(tr_pred, tr_gt)
                    ades.append(_ade)
                    fdes.append(_fde)
            ade_ = sum(ades)/len(ades)
            fde_ = sum(fdes)/len(fdes)
            return ade_,fde_,raw_data_dict


