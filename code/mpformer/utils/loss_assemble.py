import torch
from pytorch_msssim import ssim
from torch import nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


def ssim_loss(x, y, data_range=1.0, channel=1):
    return ssim(x, y, data_range=data_range, size_average=True, channel=channel)

class BMSELoss(nn.Module):
    def __init__(self, weights, thresholds, value_scale=1.0):
        super(BMSELoss, self).__init__()
        assert len(weights) == len(thresholds)
        self.weights=weights
        self.thresholds=thresholds
        self.value_scale=value_scale

    def forward(self, preds, targets):
        raw_targets = targets * self.value_scale
        weights = torch.ones_like(raw_targets)
        for weight, threshold in zip(self.weights, self.thresholds):
            weights = torch.where(raw_targets >= threshold, raw_targets.new_tensor(weight), weights)
        return torch.mean(weights * (preds - targets) ** 2)
    

class BMAELoss(nn.Module):
    def __init__(self, weights, thresholds, value_scale=1.0):
        super(BMAELoss, self).__init__()
        assert len(weights) == len(thresholds)
        self.weights=weights
        self.thresholds=thresholds
        self.value_scale=value_scale
    
    def forward(self, preds, targets):
        raw_targets = targets * self.value_scale
        weights = torch.ones_like(raw_targets)
        for weight, threshold in zip(self.weights, self.thresholds):
            weights = torch.where(raw_targets >= threshold, raw_targets.new_tensor(weight), weights)
        return torch.mean(weights * torch.abs(preds - targets))
    

class WSloss(nn.Module):
    def __init__(self, iterate=3):
        super(WSloss, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = iterate
        
    def forward(self, x, y):
        loss = []
        loss.append(1 - ssim_loss(x, y))
        for i in range(self.iterate):
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            loss.append(1 - ssim_loss(x1[0][:, :, 0], y1[0][:, :, 0]))
            loss.append(1 - ssim_loss(x1[0][:, :, 1], y1[0][:, :, 1]))
            loss.append(1 - ssim_loss(x1[0][:, :, 2], y1[0][:, :, 2]))
            loss.append(1 - ssim_loss(x0, y0))
            x, y = x0, y0
        return loss
    
    
class WSloss_linear_add_adhoc(nn.Module):
    def __init__(self, multi_scaler=[0.25, 1.0, 4.0], iterate=3):
        super(WSloss_linear_add_adhoc, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = iterate
        self.scaler = multi_scaler

    def forward(self, x, y):       
        y = y[..., 0:1].permute(0, 4, 1, 2, 3).reshape(-1, 1, y.shape[2], y.shape[3])
        x = x.permute(0, 4, 1, 2, 3).reshape(-1, 1, x.shape[2], x.shape[3])
        loss = 1 - ssim(x, y, data_range=1.0)  # Initial SSIM loss
        l, m, h = self.scaler  # Low, mid, high frequency weights

        for _ in range(self.iterate):
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)
            # Accumulate high- and low-frequency losses
            loss += (1 - ssim(x1[0][:,:,0], y1[0][:,:,0], data_range=1.0)) * m
            loss += (1 - ssim(x1[0][:,:,1], y1[0][:,:,1], data_range=1.0)) * m
            loss += (1 - ssim(x1[0][:,:,2], y1[0][:,:,2], data_range=1.0)) * h
            loss += (1 - ssim(x0, y0, data_range=1.0)) * l
            x, y = x0, y0  # Continue with the low-frequency component
        return loss


class MTloss(nn.Module):
    def __init__(self, iterate=3, scaler=0.02):
        super(MTloss, self).__init__()
        self.k_size = [5, 7, 11]
        self.iterate = iterate
        self.loss_func = nn.L1Loss()
        self.scaler = scaler

    def forward(self, x, y):
        loss = []
        loss.append(self.loss_func(x[:, 1:] - x[:, :11], y[:, 1:] - y[:, :11]) * self.scaler)
        for m in range(self.iterate):
            pf = nn.AvgPool2d(self.k_size[m], stride=1, padding=int((self.k_size[m] - 1) / 2))
            loss.append(self.loss_func(pf(x[:, 1:]) - pf(x[:, :11]), pf(y[:, 1:]) - pf(y[:, :11])) * self.scaler)
        return loss
    
    
class WTloss(nn.Module):
    def __init__(self):
        super(WTloss, self).__init__()
        self.dwt = DWTForward(J=1, wave='haar', mode='symmetric')
        self.iterate = 3
        self.loss_func = nn.L1Loss()

    def forward(self, x, y):
        y = y[..., 0:1].permute(0, 4, 1, 2, 3).reshape(-1, 1, y.shape[2], y.shape[3])
        x = x.permute(0, 4, 1, 2, 3).reshape(-1, 1, x.shape[2], x.shape[3])
        loss = self.loss_func(x, y)
        l, m, h = 0.25, 1.0, 4.0
        # Iterative wavelet decomposition
        for i in range(self.iterate):
            x0, x1 = self.dwt(x)
            y0, y1 = self.dwt(y)

            # Weighted L1 loss on high-frequency components
            loss += self.loss_func(x1[0][:, :, 0], y1[0][:, :, 0]) * m
            loss += self.loss_func(x1[0][:, :, 1], y1[0][:, :, 1]) * m
            loss += self.loss_func(x1[0][:, :, 2], y1[0][:, :, 2]) * h

            # Weighted L1 loss on low-frequency components
            loss += self.loss_func(x0, y0) * l

            # Continue with the low-frequency component
            x, y = x0, y0

        return loss


class MTloss_add_linear(nn.Module):
    def __init__(self, iterate=3, scaler=0.02):
        super(MTloss_add_linear, self).__init__()
        self.k_size = [5, 7, 11, 12, 13, 14]
        self.iterate = iterate
        self.loss_func = nn.L1Loss()
        self.scaler = scaler

    def forward(self, x, y):
        loss = self.loss_func(x[:, 1:] - x[:, :11], y[:, 1:] - y[:, :11]) * self.scaler
        for m in range(self.iterate):
            pf = nn.AvgPool2d(self.k_size[m], stride=1, padding=0)
            loss += self.loss_func(pf(x[:, 1:]) - pf(x[:, :11]), pf(y[:, 1:]) - pf(y[:, :11])) * self.scaler
        return loss
