# train.py
import time

import numpy as np
import torch
from tqdm import tqdm

from config import CFG
from utils import AverageMeter, timeSince


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    pbar = tqdm(enumerate(train_loader), total=len(
        train_loader), desc='Train ')
    for step, (images, labels) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        y_preds = model(images)
        loss = criterion(y_preds, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            tqdm.write('Epoch: [{0}][{1}/{2}] '
                       'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                       'Elapsed {remain:s} '
                       'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                       'Grad: {grad_norm:.4f}  '
                       # 'LR: {lr:.6f}  '
                       .format(
                           epoch+1, step, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses,
                           remain=timeSince(start, float(
                               step+1)/len(train_loader)),
                           grad_norm=grad_norm,
                           # lr=scheduler.get_lr()[0],
                       ))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    preds = []
    start = end = time.time()
    pbar = tqdm(enumerate(valid_loader), total=len(
        valid_loader), desc='Valid ')
    for step, (images, labels) in pbar:
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.softmax(1).to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            tqdm.write('EVAL: [{0}/{1}] '
                       'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                       'Elapsed {remain:s} '
                       'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                       .format(
                           step, len(valid_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses,
                           remain=timeSince(start, float(
                               step+1)/len(valid_loader)),
                       ))
    predictions = np.concatenate(preds)
    return losses.avg, predictions
