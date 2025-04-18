'''
@File: train_wodist.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 30, 2025
@HomePage: https://github.com/YanJieWen
'''
import argparse
import os
import time
import random
import os.path as osp
import sys
from datetime import timedelta
from collections import OrderedDict


import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from thop import profile
from thop import clever_format

from tools import yaml_load,Logger,Config,Evaluator_wodist,load_checkpoint
from engine import Modeltrainerfp16,train_pipeline_wodist

from datasets import CreateDataset,TrajectoryDateset
from models import WDGTrans_sb

from visual import Visualer



def main():
    start_time = time.monotonic()
    #parser config
    parser = argparse.ArgumentParser('--Pedestrian Trajectory Prediction in the World View--')
    parser.add_argument('--cpath',type=str,default='./configs/base.yaml',help='Config as yaml form')
    args = parser.parse_args()
    cfg = Config(yaml_load(args.cpath))

    #init envs
    if not hasattr(cfg,'seed'):
        cfg.seed = 42
    else:
        pass
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    #build project folder
    task_name = time.strftime('%Y%m%d')+'_'+cfg.name
    project_root = 'exp'
    log_file_name = osp.join(project_root,task_name,'log.txt')
    ckpt_save_dir = osp.join(project_root,task_name)
    sys.stdout = Logger(log_file_name)
    print(f'{"="*10}\n{cfg}\n{"="*10}')
    print('=>Task name: ',task_name)
    #init dataloder-->train & val
    datasets = CreateDataset(**cfg.datasets.__dict__)
    val_data = TrajectoryDateset(datasets.val,**cfg.datasets.__dict__)
    val_dataloader = DataLoader(val_data,batch_size=cfg.datasets.batch_size,num_workers=0,shuffle=False,
                                pin_memory=True,collate_fn=val_data.collate_fn)
    train_data = TrajectoryDateset(datasets.train,**cfg.datasets.__dict__)
    train_dataloader = DataLoader(train_data,batch_size=1,num_workers=0,shuffle=True,
                                  pin_memory=True,collate_fn=train_data.collate_fn)
    #init device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #init model
    model = WDGTrans_sb(device=device,**cfg.model.__dict__).to(device)
    verbose_sample = next(iter(val_dataloader))
    # [print(x.shape) for x in model(verbose_sample)]
    flops, parms = profile(model, inputs=(verbose_sample,))
    flops, params = clever_format([flops, parms], "%.3f")
    print(f'parameters: {params}\t GFLOPS:{flops} under batch size of {int(cfg.datasets.batch_size)}')
    #init evaluator-->贪心搜索，分布解码，多采样评估（如果不进行20-of-best）直接采用均值，保存采样的20条轨迹，每经历10个epoch保存
    evalator = Evaluator_wodist(model,device=device,k=cfg.eval.k)
    #init optimizer and schedular
    scaler = amp.GradScaler() if cfg.train.fp16 else None
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=150,gamma=0.2)
    #ckpt setting
    weights = [x for x in os.listdir(ckpt_save_dir) if x.endswith('.pt')]
    if len(weights)==0:
        start_ep = 0
        print('=>Training from scratch')
    else:
        weight_path = os.path.join(ckpt_save_dir,weights[-1])
        model,optimizer,lr_scheduler,scaler,start_ep = load_checkpoint(model,optimizer,lr_scheduler,scaler,weight_path)
        if start_ep>cfg.train.epochs:
            raise ValueError('model has reached the optimal state')
        else:
            print(f'=>continue training from epoch {int(start_ep)}')
    #init trainer-->train_one_epoch
    trainer = Modeltrainerfp16(model)
    #train engine
    train_pipeline_wodist(batch_size=cfg.datasets.batch_size,start_ep=start_ep,model=model,trainer=trainer,evalator=evalator,optimizer=optimizer,
                                lr_scheduler=lr_scheduler,scaler=scaler,train_loader=train_dataloader,
                                val_loader=val_dataloader,ckpt_save_dir=ckpt_save_dir,device=device,
                                **cfg.train.__dict__)

    end_time = time.monotonic()
    dtime = timedelta(seconds=end_time-start_time)
    print('=> Task finished: {}'.format(cfg.name))
    print('Total running time: {}'.format(dtime))


    # a = next(iter(test_dataloder))
    # from models import Preprocesser
    # pre = Preprocesser(device='cpu')
    # [print(x.shape) for x in pre(a)]
    # print(pre(a)[-1][20],pre(a)[-1][20].shape)
    # print(pre(a)[0][20][:,0,:],pre(a)[0][20][:,0,:].shape)


    # [print(x[0].shape) for x in a]
    # vis = Visualer(datasets.test)
    # vis.fulltraj()



if __name__ == '__main__':
    main()