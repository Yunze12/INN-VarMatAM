import torch
import numpy as np
from train_config import *
from model import INN


def l2_fit(y, y0):
    return torch.mean((y - y0) ** 2)


def MMD_matrix_multiscale(x, y, widths_exponents):
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2. * xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2. * yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2. * xy, 0, np.inf)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for C, a in widths_exponents:
        XX += C ** a * ((C + dxx) / a) ** -a
        YY += C ** a * ((C + dyy) / a) ** -a
        XY += C ** a * ((C + dxy) / a) ** -a

    return XX + YY - 2. * XY


def l2_dist_matrix(x, y):
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    return torch.clamp(rx.t() + ry - 2. * xy, 0, np.inf)


def forward_mmd(y0, y1):
    return MMD_matrix_multiscale(y0, y1, mmd_forw_kernels)


def backward_mmd(x0, x1):
    return MMD_matrix_multiscale(x0, x1, mmd_back_kernels)


def loss_forward_fit(out, y):
    l_forw_fit = lambd_fit_forw * l2_fit(out[:, :ndim_y], y[:, :ndim_y])
    return l_forw_fit


def loss_forward_mmd(out, y):
    # remove gradients wrt y, for latent loss
    output_block_grad = torch.cat((out[:, :ndim_y].data,
                                   out[:, -ndim_z:]), dim=1)
    y_short = torch.cat((y[:, :ndim_y], y[:, -ndim_z:]), dim=1)
    l_forw_mmd = lambd_mmd_forw * torch.mean(forward_mmd(output_block_grad, y_short))
    return l_forw_mmd


def loss_backward_mmd(x, y):
    inn = INN().to(device)
    x_pre = inn.inverse(y)
    MMD = backward_mmd(x, x_pre)
    if mmd_back_weighted:
        MMD *= torch.exp(- 0.5 / y_uncertainty_sigma ** 2 * l2_dist_matrix(y, y))
    return lambd_mmd_back * torch.mean(MMD)


def loss_reconstruction(out_y, x):
    inn = INN().to(device)
    cat_inputs = [out_y[:, :ndim_y] + add_y_noise * noise_batch(ndim_y),
                  out_y[:, -ndim_z:] + add_z_noise * noise_batch(ndim_z)]
    x_reconstructed = inn.inverse(torch.cat(cat_inputs, 1))
    return lambd_reconstruct * l2_fit(x_reconstructed, x)


def noise_batch(ndim):
    return torch.randn(batch_size, ndim).to(device)