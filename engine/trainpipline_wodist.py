'''
@File: trainpipline_wodist.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 30, 2025
@HomePage: https://github.com/YanJieWen
'''
'''
@File: trainpipeline.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 29, 2025
@HomePage: https://github.com/YanJieWen
'''

import sys
import os
from tqdm import tqdm
from collections import OrderedDict

import torch
import numpy as np

from tools import ade,fde,rel2abs,save_checkpoint


def train_pipeline_wodist(batch_size,start_ep,epochs,model,trainer,evalator,optimizer,lr_scheduler,scaler,train_loader,fp16,
                   warmup,print_freq,clip_grad,val_loader,eval_freq,save_freq,ckpt_save_dir,device,**kwargs):
    print(f'{"="*30}Train Pipeline{"="*30}')
    min_ade = 9999
    pbar = tqdm(range(start_ep,epochs),file=sys.stdout)
    raw_data_dict = None
    for epoch in pbar:
        pbar.desc = f'{"="*30}Epoch num {int(epoch+1)}{"="*30}'
        trainer.train_sb(batch_size,epoch,train_loader,optimizer,scaler,fp16,warmup,print_freq,clip_grad)
        lr_scheduler.step()
        if (epoch+1)%eval_freq==0 or epoch==epochs-1:#评估
            print(f'{"="*30}Eval Mode{"="*30}')
            ade_, fde_, raw_data_dict = evalator.eval(val_loader)
            print('*' * 30)
            print('AVG ADE: ', ade_)
            print('AVG FDE: ', fde_)
            print('*' * 30)
        #save benchmark
        # if (epoch+1)%save_freq == 0:
        if ade_<min_ade:
            save_checkpoint(model,optimizer,lr_scheduler,scaler,ckpt_save_dir,epoch,fp16)
            min_ade = ade_
        else:
            pass

        torch.cuda.empty_cache()
        pbar.desc = '=> CUDA cache is released'
    return raw_data_dict

