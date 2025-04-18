'''
@File: ttools.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 20, 2025
@HomePage: https://github.com/YanJieWen
'''

import numpy as np
import math

import torch
import torch.nn.functional as F

def rel2abs(nodes,init_node,tinv):
    #type: (np.array,np.array) -> np.array
    '''
    相对位移转为绝对位移
    :param nodes: txnxd
    :param init_node: nxd
    :param tinv:t
    :return:txnxd
    '''
    _nodes = nodes * tinv[:, None, None]
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s, ped, :] = np.sum(_nodes[:s + 1, ped, :], axis=0) + init_node[ped, :]
    return nodes_

def world2pixel(world,h):
    #type: (np.array,np.array) -> np.array
    '''
    根据单应性矩阵将世界坐标转为像素坐标
    :param world: txnxd
    :param h:3x3
    :return:tx2xn
    '''
    h_inv = np.linalg.inv(h)
    world_pos = np.concatenate((world, np.ones((world.shape))[..., -1:]), axis=-1)  # txnxd
    pixel_pos = np.matmul(h_inv[None, :, :], world_pos.transpose(0, 2, 1))  # txdxn
    pixel_pos = pixel_pos[:, :2] / pixel_pos[:, -1:]
    return np.asarray(pixel_pos,int)


def ade(pred,gt):
    '''
    计算平均位移损失,以1个场景为基础
    :param pred: txnx2
    :param gt: txnx2
    :return: float
    '''
    T,N,_ = pred.shape
    sum_all = 0
    for n in range(N):
        for t in range(T):
            sum_all+=math.sqrt((pred[t,n,0]-gt[t,n,0])**2+(pred[t,n,1]-gt[t,n,1])**2)
    return sum_all/(N*T)

def fde(pred,gt):
    '''
    计算最后一个时隙的位移
    :param pred: txnx2
    :param gt:txnx2
    :return:
    '''
    T, N, _ = pred.shape
    sum_all = 0
    for n in range(N):
        for t in range(T-1,T):
            sum_all += math.sqrt((pred[t, n, 0] - gt[t, n, 0]) ** 2 + (pred[t, n, 1] - gt[t, n, 1]) ** 2)
    return sum_all/N


def bivarte_loss(preds,gts,mask):
    '''
    计算高斯损失函数-->只计算完整的轨迹
    :param preds:bxtxnmx5
    :param gts:bxtxnmx2
    :param mask:bxnm
    :return:
    '''
    sum_loss = torch.zeros(1).to(preds.device)
    for pred,gt,m in zip(preds,gts,mask):#遍历每个batch
        pred = pred[:,m]#txnx5(ux,uy,sx,sy,corr)
        gt = gt[:,m]#txnx2
        normx = gt[:,:,0]-pred[:,:,0]
        normy = gt[:,:,1]-pred[:,:,1]
        sx = torch.exp(pred[:,:,2])
        sy = torch.exp(pred[:,:,3])
        corr = torch.tanh(pred[:,:,4])
        # print(sx.mean(), sy.mean(),corr.mean())
        sxsy = sx*sy
        z = (normx/sx)**2+(normy/sy)**2-2*((corr*normx*normy)/sxsy)
        negrho = 1-corr**2
        result = torch.exp(-z/(2*negrho))
        denom = 2*np.pi*(sxsy*torch.sqrt(negrho))

        result = result/denom

        eplsilon = 1e-20
        result = -torch.log(torch.clamp(result,min=eplsilon))
        result = torch.mean(result)
        sum_loss+=result
    return sum_loss

def compute_mse_loss(preds,gts,mask):
    '''
    计算平均误差损失-->只计算完整的轨迹
    :param preds:bxtxnmx2
    :param gts:bxtxnmx2
    :param mask:bxnm
    :return:
    '''
    sum_loss = torch.zeros(1).to(preds.device)
    for pred,gt,m in zip(preds,gts,mask):
        pred = pred[:,m]
        gt = gt[:, m]
        res = F.mse_loss(pred, gt, reduction='mean')
        sum_loss+=res
    return sum_loss


def graph_loss(pred,gt,mask):
    pred  = pred[:,mask,:]
    gt = gt[:,mask,:]
    normx = gt[:, :, 0] - pred[:, :, 0]
    normy = gt[:, :, 1] - pred[:, :, 1]
    sx = torch.exp(pred[:, :, 2])
    sy = torch.exp(pred[:, :, 3])
    corr = torch.tanh(pred[:, :, 4])
    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negrho = 1 - corr ** 2
    result = torch.exp(-z / (2 * negrho))
    denom = 2 * np.pi * (sxsy * torch.sqrt(negrho))

    result = result / denom

    eplsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=eplsilon))
    result = torch.mean(result)
    return result

def mse_loss(pred,gt,mask):
    pred = pred[:, mask, :]
    gt = gt[:, mask, :]
    res = F.mse_loss(pred, gt, reduction='mean')
    return res
