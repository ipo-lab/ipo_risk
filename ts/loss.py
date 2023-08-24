import math
import torch
from ts.utils import torch_transpose_batch, torch_cov2cor, torch_crossprod


def torch_portfolio_return(z, y):
    pret = z * y
    pret = pret.sum(dim=1)
    return pret


def torch_portfolio_variance(z, cov_mat):
    z = z.unsqueeze(2)
    zt = torch_transpose_batch(z)
    var = torch.matmul(zt, cov_mat)
    var = torch.matmul(var, z)
    var = var[:, 0, 0]
    return var


def loss_minvar(z, y):
    pret = torch_portfolio_return(z=z, y=y)
    loss = pret.var()
    return loss


def loss_erc(z, y):
    var = loss_minvar(z, y)
    log_loss = torch.log(z).sum(axis=1).mean()
    loss = 0.50 * var - log_loss
    return loss


def loss_maxdiv(z, y):
    var = loss_minvar(z, y)
    vol = torch_portfolio_return(z=z, y=torch.abs(y))
    vol = vol.mean()
    loss = -vol / torch.sqrt(var)
    return loss


def loss_minvar_cov(z, cov_mat):
    loss = torch_portfolio_variance(z=z, cov_mat=cov_mat)
    return loss.mean()


def loss_erc_cov(z, cov_mat):
    denom = torch_portfolio_variance(z=z, cov_mat=cov_mat)
    z = z.unsqueeze(2)
    num = z * torch.matmul(cov_mat, z)
    loss = num.squeeze(2) / denom.unsqueeze(1)
    loss = loss ** 2
    loss = loss.sum(dim=1)
    return loss.mean()


def loss_maxdiv_cov(z, cov_mat):
    vol = torch.sqrt(torch.diagonal(cov_mat, dim1=1, dim2=2))
    num = (z * vol).sum(dim=1)
    denom = torch.sqrt(torch_portfolio_variance(z=z, cov_mat=cov_mat))
    loss = num / denom
    return -loss.mean()


def loss_garch(x, y):
    x2 = x ** 2
    y2 = y ** 2
    loss = 0.5 * (torch.log(x2) + y2 / x2 + math.log(2*math.pi))
    return loss.sum()


def loss_garch_dcc(x, y):
    cor_mat = torch_cov2cor(x)
    cor_mat_inv = torch.linalg.inv(cor_mat)
    vol = torch.sqrt(torch.diagonal(x, dim1=1, dim2=2))
    zt = y / vol
    d1 = torch.log(torch.linalg.det(cor_mat))
    d2 = torch_portfolio_variance(zt, cor_mat_inv)
    d3 = (zt * zt).sum(dim=1)
    loss = 0.5 * (d1 + d2 - d3)
    return loss.sum()
