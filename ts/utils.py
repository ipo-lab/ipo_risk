import torch


def make_matrix(x):
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    return x


def torch_uniform(*size, lower=0, upper=1, dtype=torch.float64):
    r = torch.rand(*size, dtype=dtype)
    r = r * (upper - lower) + lower
    return r


def is_batch(x):
    return len(x.shape) > 2


def torch_transpose_batch(x):
    if is_batch(x):
        dim0 = 1
        dim1 = 2
    else:
        dim0 = 0
        dim1 = 1

    return torch.transpose(x, dim0=dim0, dim1=dim1)


def torch_crossprod(x, y=None):
    if y is None:
        y = x

    return torch.matmul(torch_transpose_batch(x), y)


def torch_quad_form(x, y=None, dim_x=2, dim_y=1):
    if y is None:
        y = x
    return torch.matmul(x.unsqueeze(dim_x), y.unsqueeze(dim_y))


def torch_quad_form_mat(x, mat):
    xt = torch_transpose_batch(x)
    return torch.matmul(torch.matmul(xt, mat), x)


def torch_cov2cor(cov_mat):
    if is_batch(cov_mat):
        sigma = torch.diagonal(cov_mat, dim1=1, dim2=2)
        sigma = torch.sqrt(sigma)
        sigma = torch.diag_embed(1 / sigma)
    else:
        sigma = torch.diag(cov_mat)
        sigma = torch.sqrt(sigma)
        sigma = torch.diag(1 / sigma)

    cor_mat = torch.matmul(torch.matmul(sigma, cov_mat), sigma)
    return cor_mat


def torch_cor2cov(cor_mat, sigma):
    # --- dimension handling
    if not is_batch(cor_mat):
        cor_mat = cor_mat.unsqueeze(0)
    if is_batch(sigma):
        sigma = sigma.squeeze(2)

    # --- make sigma an embedded diagonal matrix
    sigma = torch.diag_embed(sigma)
    cov_mat = torch.matmul(torch.matmul(sigma, cor_mat), sigma)
    return cov_mat