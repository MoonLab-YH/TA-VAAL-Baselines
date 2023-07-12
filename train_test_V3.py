from config import *
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T
import models.resnet as resnet
import torch.nn as nn


##
# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    criterion = nn.BCELoss()
    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    diff = torch.sigmoid(input)
    one = torch.sign(torch.clamp(target, min=0))  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = criterion(diff, one)
    elif reduction == 'none':
        loss = criterion(diff, one)
    else:
        NotImplementedError()

    return loss


def test3(models, epoch, method, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()

    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return 100 * correct / total


def test_tsne(models, epoch, method, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'train'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    out_vec = torch.zeros(0)
    label = torch.zeros(0).long()
    with torch.no_grad():
        for (inputs, labels) in dataloaders:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _, _ = models['backbone'](inputs)
            preds = scores.cpu()
            labels = labels.cpu()
            out_vec = torch.cat([out_vec, preds])
            label = torch.cat([label, labels])
        out_vec = out_vec.numpy()
        label = label.numpy()
    return out_vec, label


iters = 0


def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss):
    models['backbone'].train()
    if method == 'lloss' or method == 'TA-VAAL':
        models['module'].train()
    global iters
    # for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss' or method == 'TA-VAAL':
            optimizers['module'].zero_grad()

        scores, embedding, features = models['backbone'](inputs)
        target_loss = criterion(scores, labels)

        if method == 'lloss' or method == 'TA-VAAL':
            if epoch > epoch_loss:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            loss = m_backbone_loss + WEIGHT * m_module_loss
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            loss = m_backbone_loss

        loss.backward()
        optimizers['backbone'].step()
        if method == 'lloss' or method == 'TA-VAAL':
            optimizers['module'].step()
    return loss


def train3(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss):
    print('>> Train a Model.')
    best_acc = 0.

    for epoch in tqdm(range(num_epochs)):

        best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)

        schedulers['backbone'].step()
        if method == 'lloss' or method == 'TA-VAAL':
            schedulers['module'].step()

        if epoch == num_epochs - 1:
            acc = test3(models, epoch, method, dataloaders, mode='test')
            # acc = test(models, dataloaders, mc, 'test')
            if best_acc < acc:
                best_acc = acc
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')
