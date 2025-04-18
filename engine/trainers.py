'''
@File: trainers.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 28, 2025
@HomePage: https://github.com/YanJieWen
'''

import time

import torch
import torch.optim.lr_scheduler
from torch.cuda import amp

import math
import sys

from tools import AverageMeter,bivarte_loss,compute_mse_loss,graph_loss,mse_loss


def warmup_lr_scheduler(optimizer,warmup_iters,warmup_factor):
    def f(x):
        if x>=warmup_iters:
            return 1
        alpha = float(x)/warmup_iters
        return warmup_factor*(1-alpha)+alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=f)


class Modeltrainerfp16(object):
    def __init__(self,model):
        self.model = model
    def train_sb(self,batch_size,epoch,data_loader,optimizer,scaler,fp16=True,warmup=True,print_freq=50,clip_grad=1):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        lr_scheduler = None
        if epoch == 0 and warmup is True:
            warmup_factor = 1.0 / 1000
            warm_iters = min(1000, len(data_loader) - 1)
            lr_scheduler = warmup_lr_scheduler(optimizer, warm_iters, warmup_factor)
        # loss_batch = 0
        batch_count = 0
        is_fst_loss = True
        loader_len = len(data_loader)
        turn_point = int(loader_len / batch_size) * batch_size + loader_len % batch_size - 1
        for i, dataset in enumerate(data_loader):
            batch_count += 1
            optimizer.zero_grad()
            # bxtxnx5,bxn(bool),bxnx2xt,bxt

            out, out_mask = self.model(dataset)
            v_gt = dataset[-2][0].to(out.device)  # [txnx2]
            v_pred = out.squeeze(0)  # txnx5
            mask = out_mask.squeeze(0)  # n
            # print('gt:', dataset[3][0].to(mask.device)[:, 0][mask])#nxt
            # print('pred:', v_pred[:,mask,0].transpose(0,1))
            # print('gt_v:',v_gt[:,mask,0].transpose(0,1))
            # compute loss
            if batch_count % batch_size != 0 and i != turn_point:
                l = graph_loss(v_pred, v_gt, mask)
                # l = mse_loss(v_pred, v_gt, mask)
                if is_fst_loss:
                    loss = l
                    is_fst_loss = False
                else:
                    loss += l
            else:
                loss = loss / batch_size
                is_fst_loss = True
                loss.backward()
                if not math.isfinite(loss):
                    print(f'Loss is {loss},stopping training')
                    sys.exit(1)
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad, norm_type=2)
                optimizer.step()
                losses.update(loss.item())
                batch_time.update(time.time() - end)
                end = time.time()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                if (i + 1) % print_freq == 0:
                    print('Epoch: [{}][{}/{}]'
                          'Time: {:.3f} ({:.3f}),'
                          'Loss: {:.3f} ({:.3f})'
                          .format(epoch + 1, i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  losses.val, losses.avg)
                          )

    def train(self,epoch,data_loader,optimizer,scaler,fp16=True,warmup=True,print_freq=50,clip_grad=1):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        lr_scheduler = None
        if epoch == 0 and warmup is True:
            warmup_factor = 1.0/1000
            warm_iters = min(1000,len(data_loader)-1)
            lr_scheduler = warmup_lr_scheduler(optimizer,warm_iters,warmup_factor)
        for i,dataset in enumerate(data_loader):
            with amp.autocast(enabled=fp16):
                # bxtxnx5,bxn(bool),bxnx2xt,bxt
                out,out_mask = self.model(dataset)
                v_gt = dataset[-2]#[txnx2]
                v_gt = self.model.preprocess.vmask(v_gt)#bxtxnmx2
                #compute loss
                # loss = bivarte_loss(out,v_gt,out_mask)
                loss = compute_mse_loss(out,v_gt,out_mask)
                loss /= v_gt.size(0)
                if not math.isfinite(loss):
                    print(f'Loss is {loss},stopping training')
                    sys.exit(1)
            optimizer.zero_grad()
            if scaler is None:
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),clip_grad)
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            losses.update(loss.item())
            batch_time.update(time.time()-end)
            end = time.time()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if (i+1)%print_freq==0:
                print('Epoch: [{}][{}/{}]'
                      'Time: {:.3f} ({:.3f}),'
                      'Loss: {:.3f} ({:.3f})'
                      .format(epoch+1,i+1,len(data_loader),
                              batch_time.val,batch_time.avg,
                              losses.val,losses.avg)
                      )


def train_one_epoch(model,epoch,train_dataloader,cfg,optimizer,scaler):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    lr_scheduler = None
    if epoch == 0 and cfg.train.warmup is True:
        warmup_factor = 1.0 / 1000
        warm_iters = min(1000, len(train_dataloader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warm_iters, warmup_factor)
    for i, dataset in enumerate(train_dataloader):
        with amp.autocast(enabled=cfg.train.fp16):
            # bxtxnx5,bxn(bool),bxnx2xt,bxt
            out, out_mask = model(dataset)
            v_gt = dataset[-2]  # [txnx2]
            v_gt = model.preprocess.vmask(v_gt)
            # compute loss
            loss = bivarte_loss(out, v_gt, out_mask)
            loss /= out.size(0)
            if not math.isfinite(loss):
                print(f'Loss is {loss},stopping training')
                sys.exit(1)
        optimizer.zero_grad()
        if scaler is None:
            loss.backward()
            if cfg.train.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.clip_grad,norm_type=2)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if (i + 1) % cfg.train.print_freq == 0:
            print('Epoch: [{}][{}/{}]'
                  'Time: {:.3f} ({:.3f}),'
                  'Loss: {:.3f} ({:.3f})'
                  .format(epoch + 1, i + 1, len(train_dataloader),
                          batch_time.val, batch_time.avg,
                          losses.val, losses.avg)
                  )


def train_one_epoch_singlebatch(model,batch_size,epoch,train_dataloader,cfg,optimizer,scaler):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    lr_scheduler = None
    if epoch == 0 and cfg.train.warmup is True:
        warmup_factor = 1.0 / 1000
        warm_iters = min(1000, len(train_dataloader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warm_iters, warmup_factor)
    # loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(train_dataloader)
    turn_point = int(loader_len/batch_size)*batch_size+loader_len%batch_size-1
    for i, dataset in enumerate(train_dataloader):
        batch_count+=1
        optimizer.zero_grad()
        # bxtxnx5,bxn(bool),bxnx2xt,bxt
        out, out_mask = model(dataset)
        v_gt = dataset[-2][0].to(out.device)  # [txnx2]
        v_pred = out.squeeze(0) #txnx5
        mask = out_mask.squeeze(0) #n
        # compute loss
        if batch_count%batch_size!=0 and i!=turn_point:
            l = graph_loss(v_pred,v_gt,mask)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss+=l
        else:
            loss = loss/batch_size
            is_fst_loss = True
            loss.backward()
            if not math.isfinite(loss):
                print(f'Loss is {loss},stopping training')
                sys.exit(1)
            if cfg.train.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad, norm_type=2)
            optimizer.step()
            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()
            if lr_scheduler is not None:
                lr_scheduler.step()
            if (i + 1) % cfg.train.print_freq == 0:
                print('Epoch: [{}][{}/{}]'
                      'Time: {:.3f} ({:.3f}),'
                      'Loss: {:.3f} ({:.3f})'
                      .format(epoch + 1, i + 1, len(train_dataloader),
                              batch_time.val, batch_time.avg,
                              losses.val, losses.avg)
                      )