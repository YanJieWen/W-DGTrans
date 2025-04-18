'''
@File: dataset.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 17, 2025
@HomePage: https://github.com/YanJieWen
'''

import math
import sys

import torch
import numpy as np

from torch.utils.data import Dataset
import networkx as nx
from tqdm import tqdm

def anorm(p1,p2):
    norm = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    if norm==0:
        return 0
    else:
        return 1/norm

def seq_to_graph(_obs_traj,_obs_traj_rel,_obs_mask):
    seq_ = _obs_traj.squeeze()
    seq_rel = _obs_traj_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]
    v = np.zeros((seq_len, max_nodes, 2))
    a = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]  # nxd
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            v[s, h, :] = step_rel[h]
            a[s, h, h] = 1
            if _obs_mask[h, s] != 0:
                continue
            # for k in range(h + 1, len(step_)):
            for k in range(len(step_)):
                if k==h:
                    a[s,h,k] = 1
                else:
                    if _obs_mask[k, s] != 0:
                        continue
                    else:
                        l2_norm = anorm(step_rel[h], step_rel[k])
                        a[s, h, k] = l2_norm
                        a[s, k, h] = l2_norm
        # g = nx.from_numpy_array(a[s, :, :])
        g = a[s, :, :].copy()
        a[s, :, :] = (g.T * np.sum(g, axis=1) ** (-1)).T
        # a[s, :, :] = nx.normalized_laplacian_matrix(g).toarray()
    return torch.from_numpy(v).type(torch.float),torch.from_numpy(a).type(torch.float)


class TrajectoryDateset(Dataset):
    def __init__(self,dataset,norm_lap_matr=True,obs_len=8,pred_len=12,**kwargs):
        super().__init__()
        self.norm_lap_matr = norm_lap_matr
        self.obs_len = obs_len
        self.pred_len = pred_len

        (_non_linear_ped, _num_peds_in_seq, _seq_list_mask,
         _seq_gan, _seq_list, _seq_list_rel,s_e_frame) = dataset
        self.obs_traj = torch.from_numpy(_seq_list[:, :, :obs_len]).type(torch.float)#mxdxt_o
        self.pred_traj = torch.from_numpy(_seq_list[:, :, obs_len:]).type(torch.float)#mxdxt_d
        self.obs_traj_rel = torch.from_numpy(_seq_list_rel[:, :, :obs_len]).type(torch.float)#mxdxt_o
        self.pred_traj_rel = torch.from_numpy(_seq_list_rel[:, :, obs_len:]).type(torch.float)#mxdxt_d
        self.loss_mask_obs = torch.from_numpy(_seq_list_mask[:,:obs_len]).type(torch.long)#mxt_o
        self.loss_mask_pred = torch.from_numpy(_seq_list_mask[:, obs_len:]).type(torch.long)#mxt_d
        self.seq_inv_obs = torch.from_numpy(_seq_gan[:, :obs_len]).type(torch.long)#sxt_o
        self.seq_inv_pred = torch.from_numpy(_seq_gan[:, obs_len:]).type(torch.long)#sxt_d
        self.non_linear_mask = torch.from_numpy(_non_linear_ped).type(torch.long)#m,非线性指的是预测轨迹为非线性

        cum_st_idx = [0] + np.cumsum(_num_peds_in_seq).tolist()
        self.st_end = [(x, y) for x, y in zip(cum_st_idx[:-1], cum_st_idx[1:])]

        self.v_obs = []
        self.a_obs = []
        self.v_pred = []
        self.a_pred = []
        print('Processing Data...')
        pbar = tqdm(range(len(self.st_end)),file=sys.stdout)
        for ss in pbar:
            s, e = self.st_end[ss]
            v_o,a_o = seq_to_graph(self.obs_traj[s:e],self.obs_traj_rel[s:e],self.loss_mask_obs[s:e])
            self.v_obs.append(v_o.clone())
            self.a_obs.append(a_o.clone())
            v_p,a_p = seq_to_graph(self.pred_traj[s:e],self.pred_traj_rel[s:e],self.loss_mask_pred[s:e])
            self.v_pred.append(v_p.clone()) #txnxd
            self.a_pred.append(a_p.clone()) #txnxn
        pbar.close()

    def __len__(self):
        return len(self.v_obs)


    def __getitem__(self, index):
        start,end = self.st_end[index]
        # out = [
        #     self.obs_traj[start:end],self.pred_traj[start:end], #nxdxs
        #     self.obs_traj_rel[start:end],self.pred_traj_rel[start:end],
        #     self.loss_mask_obs[start:end],self.loss_mask_pred[start:end],
        #     self.non_linear_mask[start:end],self.seq_inv_obs[index],self.seq_inv_pred[index],
        #     self.v_obs[index],self.a_obs[index],self.v_pred[index],self.a_pred[index]
        # ]

        return self.obs_traj[start:end],self.pred_traj[start:end],self.obs_traj_rel[start:end],self.pred_traj_rel[start:end],self.loss_mask_obs[start:end], \
            self.loss_mask_pred[start:end], self.non_linear_mask[start:end],self.seq_inv_obs[index],self.seq_inv_pred[index], \
            self.v_obs[index],self.a_obs[index],self.v_pred[index],self.a_pred[index]

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


