import torch
import torch.nn.functional as F


def weight_func(x, value_lim):
    value_min, value_max = value_lim
    return torch.clamp(1.0 + (x - value_min) / (value_max - value_min) * 128.0, max=24.0)


def wdis_l1(pred, gt, reg_loss=True, value_lim=(0.0, 128.0)):
    w = weight_func(gt, value_lim)
    diff_w = torch.abs(pred - gt) * w
    if reg_loss:
        loss_n = torch.sum(diff_w, (-1, -2)) / torch.clamp(torch.sum(w, (-1, -2)), min=1e-6)
        return torch.sum(loss_n, -1).mean()
    return diff_w.mean() * gt.shape[1]


def sobel_filter_2d(x):
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3)
    x = x.unsqueeze(1)
    gx = F.conv2d(x, sobel_x, padding=1).squeeze(1)
    gy = F.conv2d(x, sobel_y, padding=1).squeeze(1)
    return gx, gy


def motion_reg(motion, gt, reg_loss=True, value_lim=(0.0, 128.0)):
    total_reg = motion.new_tensor(0.0)
    for t in range(motion.shape[1]):
        vx = motion[:, t, 0]
        vy = motion[:, t, 1]
        w = weight_func(gt[:, t], value_lim)
        gx_vx, gy_vx = sobel_filter_2d(vx)
        gx_vy, gy_vy = sobel_filter_2d(vy)
        if reg_loss:
            denom = torch.clamp(torch.sum(w, (-1, -2)), min=1e-6)
            reg_vx = torch.sum(gx_vx ** 2 + gy_vx ** 2, (-1, -2)) / denom
            reg_vy = torch.sum(gx_vy ** 2 + gy_vy ** 2, (-1, -2)) / denom
            total_reg = total_reg + reg_vx.mean() + reg_vy.mean()
        else:
            total_reg = total_reg + ((gx_vx ** 2 + gy_vx ** 2) * w).mean()
            total_reg = total_reg + ((gx_vy ** 2 + gy_vy ** 2) * w).mean()
    return total_reg


def accumulation_loss(pred_final, pred_bili, real, reg_loss=True, value_lim=(0.0, 128.0)):
    loss = wdis_l1(pred_final, real, reg_loss=reg_loss, value_lim=value_lim)
    if pred_bili is not None:
        loss = 0.5 * (loss + wdis_l1(pred_bili, real, reg_loss=reg_loss, value_lim=value_lim))
    return loss
