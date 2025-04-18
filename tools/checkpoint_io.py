'''
@File: checkpoint_io.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 28, 2025
@HomePage: https://github.com/YanJieWen
'''

import os
import torch


def save_checkpoint(model,optimizer,schedular,scaler,ckpt_save_dir,epoch,fp16):
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir,exist_ok=True)
    save_files = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': schedular.state_dict(),
        'epoch':epoch,
    }
    if fp16 and scaler is not None:
        save_files['scaler'] = scaler.state_dict()
    torch.save(save_files,os.path.join(ckpt_save_dir,f'{int(epoch)}.pt'))


def load_checkpoint(model,optimizer,scheduler,scaler,ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise TypeError(f'{ckpt_path} is not a file')
    ckpt = torch.load(ckpt_path,map_location='cpu')
    expect_keys,miss_keys = model.load_state_dict(ckpt['model'],strict=False)
    print('*'*10)
    print(f'Expect keys:{expect_keys},Missing keys: {miss_keys}' )
    print('*'*10)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['lr_scheduler'])
    start_epoch = ckpt['epoch']+1
    scaler.load_state_dict(ckpt['scaler']) if 'scaler' in ckpt else None
    return model,optimizer,scheduler,scaler,start_epoch
