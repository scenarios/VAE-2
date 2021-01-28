# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.criterion = nn.L1Loss(reduction='sum')

    def forward(self, predict, target):
        loss = self.criterion(predict, target) / predict.shape[0]

        return loss


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, mu, logvar):
        if isinstance(mu, list):
            assert isinstance(logvar, list)
            loss = 0.0
            for m, v in zip(mu, logvar):
                #_m = torch.mean(m, dim = [0, 2, 3])
                #_var = torch.mean(torch.sqrt(torch.exp(v)), dim = [0, 2, 3])
                loss = loss + torch.sum(0.5 * (m**2 + torch.exp(v) - v - 1)) / m.shape[0]
        else:
            loss = torch.sum(0.5 * (mu**2 + torch.exp(logvar) - logvar - 1)) / mu.shape[0]

        return loss


class lsgan_adversarial_loss(nn.Module):
    def __init__(self):
        super(lsgan_adversarial_loss, self).__init__()

        self.criterion = nn.MSELoss(reduction='sum')

    def forward(self, sample, mode):
        assert mode in ['real', 'fake']
        if mode is 'real':
            loss = self.criterion(sample, torch.ones(sample.shape, dtype=sample.dtype, device=sample.device)) / sample.shape[0]
        else:
            loss = self.criterion(sample, torch.zeros(sample.shape, dtype=sample.dtype, device=sample.device)) / sample.shape[0]

        return loss


class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))