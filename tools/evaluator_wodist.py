'''
@File: evaluator_wodist.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 30, 2025
@HomePage: https://github.com/YanJieWen
'''
import torch
import torch.distributions.multivariate_normal as torchdist

from collections import OrderedDict
import copy
import math
from tools import ade,fde,rel2abs

from tqdm import tqdm
import numpy as np

from abc import ABC,abstractmethod

def anorm(p1,p2):
    norm = torch.sqrt(torch.pow(p1[0]-p2[0],2)+torch.pow(p1[1]-p2[1],2))
    if norm==0:
        return 0
    else:
        return 1/norm

#贪心搜索，分布解码，多采样评估（如果不进行20-of-best）直接采用均值，保存采样的20条轨迹
class BaseEvaluator(ABC):
    def __init__(self,model,k,**kwargs):
        super().__init__()
        self.model = model
        self.k = k
        self.model.eval()
    @abstractmethod
    def greedy_decode(self,v_obs,a_obs,mask_o,mask_p):
        '''
        对输入进行自回归解码
        :param v_obs: bxtxnx2
        :param a_obs: bxtxnxn
        :param mask_o: bxnxt
        :param mask_p: bxnxt
        :param tr_rp: txnx2-->用于评估最佳的预测点
        :return: bxtxnx2
        '''
        pass

    @abstractmethod
    def decodedisp(self,ppoints):
        '''

        :param ppoints: nx5
        :return:
        '''
        pass


    @abstractmethod
    def eval(self,dataloader):
        '''

        :param k: bast-of-k
        :return:
        '''
        pass


class Evaluator_wodist(BaseEvaluator):
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
            mask_tgt = m_pred.transpose(1,2)[:,:,:,None]==0
            dec_out = self.model.decode(v_pred, mem, mask_o, m_pred, a_pred, mask_tgt)
            out_dist = self.model.header(dec_out)#bxtxnx5
            ut,covt = self.decodedisp(out_dist[0,t])
            #1.采样单个t时刻最佳的相对位置
            best_ade,best_fde = self.get_best_rel(ut,covt,tr_rp[...,t],t)
            #2.更新v_pred
            outv_pred[0,t] = best_ade
            #3.更新a_pred
            outa_pred[0,t] = self.update_graph(best_ade,mask_p[0,:,t])
            #4.更新mask
            outm_pred[...,t] = mask_p[...,t]
        return outv_pred,mask_out

    def get_best_rel(self,mean,cov,trt,t):
        '''
        根据t时刻所有人相对位置的最小值选取
        :param mean: nx2
        :param cov: nx2x2
        :param trt: nx2
        :param t: int
        :return: nx2
        '''
        device = mean.device
        n,_ = trt.size()
        best_of_ade_tr = []
        best_of_fde_tr = []
        mvnormal = torchdist.MultivariateNormal(mean,cov)
        ade_ls = {i:[] for i in range(n)}#每个行人t时刻k次采样的ade
        fde_ls = {i:[] for i in range(n)}
        dist_traj = {i:[] for i in range(n)}#每个行人k次采样的轨迹
        trt = trt.data.cpu().numpy()
        for k in range(self.k):
            _pred = mvnormal.sample()#nx2
            _pred = _pred.data.cpu().numpy()
            #遍历每一个行人
            for i in range(n):
                rl_ade = ade(_pred[None,i:i+1],trt[None,i:i+1])
                rl_fde = fde(_pred[None,i:i+1],trt[None,i:i+1])
                ade_ls[i].append(rl_ade)
                fde_ls[i].append(rl_fde)
                dist_traj[i].append(_pred[i])
        #获取最优的行人轨迹
        for i in range(n):
            best_of_ade_tr.append(dist_traj[i][np.argmin(ade_ls[i])])
            best_of_fde_tr.append(dist_traj[i][np.argmin(fde_ls[i])])
        best_of_fde_tr = np.stack(best_of_ade_tr,axis=0)#nx2
        best_of_fde_tr = np.stack(best_of_fde_tr,axis=0)#nx2
        best_of_ade_tr = torch.as_tensor(best_of_ade_tr).to(device)
        best_of_fde_tr = torch.as_tensor(best_of_fde_tr).to(device)
        dist_traj = np.stack([v for k,v in dist_traj.items()],axis=0)#nxkx2
        self.dist_trajs.setdefault(t,[]).append(dist_traj)
        return best_of_ade_tr,best_of_fde_tr




    def decodedisp(self,ppoints):
        '''
        单个场景进行解码
        :param ppoints: nx5
        :return: mean,cov
        '''
        sx = torch.exp(ppoints[:, 2])
        sy = torch.exp(ppoints[:, 3])
        corr = torch.tanh(ppoints[:, 4])
        cov = torch.zeros(ppoints.shape[0], 2, 2).to(ppoints.device)
        cov[:, 0, 0] = sx * sx
        cov[:, 0, 1] = corr * sx * sy
        cov[:, 1, 0] = corr * sx * sy
        cov[:, 1, 1] = sy * sy
        mean = ppoints[:, :2]
        return mean, cov


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
                    # mask = torch.sum(torch.concatenate((mask_o,mask_p),dim=-1),dim=-1)==0#N
                    tr_gt = torch.permute(tr_p[mask],(2,0,1)).contiguous().cpu().numpy()#txnx2
                    tr_obs = torch.permute(tr_o[mask],(2,0,1)).contiguous().cpu().numpy()#txnx2
                    init_node = tr_o[mask].cpu().numpy()[...,-1]#nxd
                    inv_p = inv_p.cpu().numpy()
                    raw_data_dict[scount] = {}
                    raw_data_dict[scount]['obs'] = copy.deepcopy(tr_obs)
                    raw_data_dict[scount]['gt'] = copy.deepcopy(tr_gt)
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

def decodedisp(ppoints):
    '''
    单个时间节点将预测结果解码为mean和conv
    :param ppoints: txnx5(ux,uy,sx,sy,corr)
    :return:mean:txnx2,cov:txnx2x2
    '''
    t,n,_ = ppoints.shape
    sx = torch.exp(ppoints[...,2])
    sy = torch.exp(ppoints[...,3])
    corr = torch.tanh(ppoints[...,4])
    cov = torch.zeros(ppoints.shape[0],ppoints.shape[1],2,2).to(ppoints.device)
    cov[:,:,0,0] = sx * sx
    cov[:,:,0,1] = corr*sx*sy
    cov[:,:,1,0] = corr*sx*sy
    cov[:,:,1,1] = sy*sy
    mean = ppoints[..., :2]
    return mean,cov


def greedy_decode(model,v_obs, a_obs, mask_o,device):
    '''
    针对一个场景的预测
    :param v_obs: txnx2
    :param a_obs: txnxn
    :param mask_o: nxt
    :return: mean-->txnx2,conv-->txnx2x2
    '''

    # init input tensors
    t_o, n, _ = v_obs.shape
    v_obs, a_obs, mask_o = map(lambda x: x.to(device).unsqueeze(0), (v_obs, a_obs, mask_o))
    # 增加batch维度
    src_mask = mask_o.transpose(1,2)[:,:,:,None]
    x = model.encode(v_obs, mask_o, a_obs, src_mask)
    x = model.header(x)
    out = model.txpcnn(x)
    out = out.squeeze(0)  # txnx5
    mean, conv = decodedisp(out)
    return mean, conv




def evaluate(model,dataloder,device,k=20):
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    model.eval()
    sccount = 0
    with torch.no_grad():
        for dataset in dataloder:
            tr_os, tr_ps, _, tr_rps, m_os, m_ps, _, _, inv_ps, v_os, a_os, v_ps, a_ps = dataset
            # 遍历每一个场景
            for (tr_o, tr_p, tr_rp, mask_o, mask_p, v_obs, a_obs, inv_p) in zip(tr_os, tr_ps, tr_rps, m_os, m_ps,
                                                                                v_os, a_os, inv_ps):
                sccount += 1  # 一个场景即一个样本
                tr_o, tr_p, inv_p = map(lambda x: x.to(device), (tr_o, tr_p, inv_p))
                mask = torch.sum(torch.concatenate((mask_o, mask_p), dim=-1), dim=-1) == 0  # N
                tr_gt = torch.permute(tr_p[mask], (2, 0, 1)).contiguous().cpu().numpy()  # txnx2
                tr_obs = torch.permute(tr_o[mask], (2, 0, 1)).contiguous().cpu().numpy()  # txnx2
                init_node = tr_o[mask].cpu().numpy()[..., -1]  # nxd
                inv_p = inv_p.cpu().numpy()
                mean, cov = greedy_decode(model,v_obs, a_obs, mask_o,device)# txnx5
                mvnormal = torchdist.MultivariateNormal(mean[:,mask], cov[:,mask])
                raw_data_dict[sccount] = {}
                raw_data_dict[sccount]['obs'] = copy.deepcopy(tr_obs)
                raw_data_dict[sccount]['pred'] = copy.deepcopy(tr_gt)
                raw_data_dict[sccount]['pred'] = []
                _, num_objs, _ = tr_gt.shape
                ade_ls = {}
                fde_ls = {}
                for n in range(num_objs):
                    ade_ls[n] = []
                    fde_ls[n] = []
                for i in range(k):
                    tr_pred = mvnormal.sample()
                    # print('GT:',tr_rp.to(tr_o.device)[mask][0].transpose(0,1))
                    # print('Pred:',tr_pred[:,0,:2])
                    # print('init_node:',init_node[0,:])
                    tr_pred = rel2abs(tr_pred.cpu().numpy(), init_node, inv_p)  # txnxd
                    # print('prediction:',tr_pred[:,0,0])
                    # print('gt:',tr_gt[:,0,0])
                    # print(tr_pred[:,n:n+1,:][:,0,:])
                    # print(tr_gt[:,n:n+1,:][:,0,:])
                    raw_data_dict[sccount]['pred'].append(copy.deepcopy(tr_pred))
                    for n in range(num_objs):  # 针对每个行人轨迹选择对应最优的k下的ade和fde
                        _ade = ade(tr_pred[:, n:n + 1, :], tr_gt[:, n:n + 1, :])
                        _fde = fde(tr_pred[:, n:n + 1, :], tr_gt[:, n:n + 1, :])
                        # print(f'{i}-th/{n}-th: {_ade,_fde}')
                        ade_ls[n].append(_ade)  # 存储每个k下第n个行人ade
                        fde_ls[n].append(_fde)
                for n in range(num_objs):
                    ade_bigls.append(min(ade_ls[n]))
                    fde_bigls.append(min(fde_ls[n]))
                raw_data_dict[sccount]['pred'] = np.stack(raw_data_dict[sccount]['pred'], axis=0)  # kxtxnx2
        # print(ade_bigls)
        ade_ = sum(ade_bigls) / len(ade_bigls)
        fde_ = sum(fde_bigls) / len(fde_bigls)
        return ade_, fde_, raw_data_dict


