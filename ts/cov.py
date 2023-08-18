import torch
import torch.nn as nn
from math import log, ceil
from ts.utils import is_batch, torch_uniform, torch_crossprod, make_matrix, torch_quad_form, torch_quad_form_mat, torch_cov2cor, torch_cor2cov
from ts.loss import loss_garch, loss_garch_dcc


class CovFactor(nn.Module):
    def __init__(self, x, y, factor_cov_model, dtype=torch.float64):
        super().__init__()
        in_features = x.shape[1]
        out_features = y.shape[1]
        # --- init correlation parameters:
        #weight = torch.rand(in_features, out_features, dtype=dtype)
        #bias = torch_uniform(out_features, lower=-4, upper=-2, dtype=dtype)
        weight = torch.linalg.lstsq(x, y)[0]
        residuals = y - torch.matmul(x, weight)
        bias = residuals.var(dim=0)

        # --- self:
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias)
        self.factor_cov_model = factor_cov_model

    def forward(self, x):
        # --- compute factor covariance:
        factor_cov_mat = self.factor_cov_model(x=x)
        # --- quadratic form:
        cov_mat = torch_quad_form_mat(self.weight, factor_cov_mat)
        # --- diagonal bias:
        bias = torch.sigmoid(self.bias)
        bias = torch.diag(bias)
        cov_mat = cov_mat + bias.unsqueeze(0)

        return cov_mat


class CovFactorOLS(nn.Module):
    def __init__(self, x, y, factor_cov_model):
        super().__init__()
        # --- init OLS parameters:
        weight = torch.linalg.lstsq(x, y)[0]
        residuals = y - torch.matmul(x, weight)
        bias = residuals.var(dim=0)

        # --- self:
        self.weight = weight
        self.bias = bias
        self.factor_cov_model = factor_cov_model

    def forward(self, x):
        # --- compute factor covariance:
        factor_cov_mat = self.factor_cov_model(x=x)
        # --- quadratic form:
        cov_mat = torch_quad_form_mat(self.weight, factor_cov_mat)
        # --- diagonal bias:
        bias = torch.diag(self.bias)
        cov_mat = cov_mat + bias.unsqueeze(0)

        return cov_mat


class GarchNet(nn.Module):
    def __init__(self, x, alpha_order=1, beta_order=1, buffer=1, n_ahead=1):
        super().__init__()
        # --- self:
        dtype = x.dtype
        self.in_features = x.shape[1]
        self.buffer = buffer
        self.n_ahead = n_ahead

        # --- nn_parameters
        alpha = torch_uniform(alpha_order, self.in_features, lower=-4, upper=-1, dtype=dtype)
        alpha = torch.nn.Parameter(alpha)
        beta = torch_uniform(beta_order, self.in_features, lower=0, upper=4, dtype=dtype)
        beta = torch.nn.Parameter(beta)
        omega = torch_uniform(self.in_features, lower=-4, upper=-2, dtype=dtype)
        omega = torch.nn.Parameter(omega)

        # --- self:
        self.alpha = alpha
        self.beta = beta
        self.omega = omega

    def forward(self, x):
        # --- constraint hacking
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta) * 0.9999
        omega = torch.sigmoid(self.omega)

        # --- compute garch estimates
        sigma = torch_garch(x=x, alpha=alpha, beta=beta, omega=omega, buffer=self.buffer, n_ahead=self.n_ahead)

        return sigma


class GarchNetNNL(nn.Module):
    def __init__(self, x, alpha_order=1, beta_order=1, buffer=1, n_ahead=1, lr=1e-1, n_epochs=500, verbose=True):
        super().__init__()
        # --- self:
        dtype = x.dtype
        self.in_features = x.shape[1]
        self.buffer = buffer
        self.n_ahead = n_ahead
        # --- nn_parameters
        alpha = torch_uniform(alpha_order, self.in_features, lower=-4, upper=-1, dtype=dtype)
        alpha = torch.nn.Parameter(alpha)
        beta = torch_uniform(beta_order, self.in_features, lower=0, upper=4, dtype=dtype)
        beta = torch.nn.Parameter(beta)
        omega = torch_uniform(self.in_features, lower=-4, upper=-2, dtype=dtype)
        omega = torch.nn.Parameter(omega)

        # --- train by negative log-likelihood:
        optimizer = torch.optim.Adam([alpha, beta, omega], lr=lr)

        # ---- main training loop
        for epoch in range(n_epochs):
            sigma = torch_garch(x=x, alpha=torch.sigmoid(alpha), beta=torch.sigmoid(beta),
                                omega=torch.sigmoid(omega), buffer=self.buffer, n_ahead=self.n_ahead)
            loss = loss_garch(sigma, x)
            optimizer.zero_grad()
            # --- compute gradients
            loss.backward()
            # --- update parameters
            optimizer.step()
            if verbose:
                print('epoch {}, loss {}'.format(epoch, loss.item()))

        # --- nn_parameters
        self.alpha = alpha.detach()
        self.beta = beta.detach()
        self.omega = omega.detach()

    def forward(self, x):
        # --- constraint hacking
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta) * 0.9999
        omega = torch.sigmoid(self.omega)

        # --- compute garch estimates
        sigma = torch_garch(x=x, alpha=alpha, beta=beta, omega=omega, buffer=self.buffer, n_ahead=self.n_ahead)

        return sigma


class GarchCCCNet(nn.Module):
    def __init__(self, x, alpha_order=1, beta_order=1, buffer=1, n_ahead=1):
        super().__init__()
        # --- self:
        self.garch_model = GarchNet(x=x, alpha_order=alpha_order, beta_order=beta_order, buffer=buffer, n_ahead=n_ahead)
        self.cor_mat = torch_cor(x, center=True)

    def forward(self, x):
        # --- compute sigmas:
        sigma = self.garch_model(x)
        cov_mat = torch_cor2cov(cor_mat=self.cor_mat, sigma=sigma)
        return cov_mat


class GarchCCCNetNNL(nn.Module):
    def __init__(self, x, alpha_order=1, beta_order=1, buffer=1, n_ahead=1,
                 lr=1e-1, n_epochs=500, verbose=True):
        super().__init__()
        # --- self:
        self.garch_model = GarchNetNNL(x=x, alpha_order=alpha_order, beta_order=beta_order, buffer=buffer, n_ahead=n_ahead,
                                       lr=lr, n_epochs=n_epochs, verbose=verbose)
        self.cor_mat = torch_cor(x, center=True)

    def forward(self, x):
        # --- compute sigmas:
        sigma = self.garch_model(x)
        cov_mat = torch_cor2cov(cor_mat=self.cor_mat, sigma=sigma)
        return cov_mat


class GarchDCCNet(nn.Module):
    def __init__(self, x, alpha_order=1, beta_order=1, buffer=1, n_ahead=1):
        super().__init__()
        dtype = x.dtype
        # --- init garch:
        self.garch_model = GarchNet(x=x, alpha_order=alpha_order, beta_order=beta_order, buffer=buffer, n_ahead=n_ahead)
        self.cor_mat = torch_cor(x, center=True)

        # --- self dcc params:
        self.a = torch.nn.Parameter(torch_uniform(1, lower=-4, upper=-1, dtype=dtype))
        self.b = torch.nn.Parameter(torch_uniform(1, lower=0, upper=4, dtype=dtype))

    def forward(self, x):
        # --- compute sigmas:
        sigma = self.garch_model(x)

        # --- constraint hack:
        a = torch.sigmoid(self.a)
        b = torch.sigmoid(self.b)

        # compute dynamic covariance:
        cov_mat = torch_garch_dcc(x=x, sigma=sigma, cor_mat_bar=self.cor_mat, a=a,
                                  b=b, buffer=self.garch_model.buffer, n_ahead=self.garch_model.n_ahead)

        return cov_mat


class GarchDCCNetNNL(nn.Module):
    def __init__(self, x, alpha_order=1, beta_order=1, buffer=1, n_ahead=1, lr=1e-1, n_epochs=500, verbose=True):
        super().__init__()
        dtype = x.dtype
        # --- garch_model and cor_mat:
        self.garch_model = GarchNetNNL(x=x, alpha_order=alpha_order, beta_order=beta_order, buffer=buffer,
                                       n_ahead=n_ahead,  lr=lr, n_epochs=n_epochs, verbose=verbose)
        self.cor_mat = torch_cor(x, center=True)
        # --- init correlation parameters:
        a = torch_uniform(1, lower=-4, upper=-1, dtype=dtype)
        a = torch.nn.Parameter(a)
        b = torch_uniform(1, lower=0, upper=4, dtype=dtype)
        b = torch.nn.Parameter(b)

        # --- note: garch_model should be pre-trained by NNL
        # --- train DCC by negative log-likelihood:
        optimizer = torch.optim.Adam([a, b], lr=lr)
        sigma = self.garch_model(x)
        # ---- main training loop
        for epoch in range(round(n_epochs/5)):
            # compute dynamic covariance:
            cov_mat = torch_garch_dcc(x=x, sigma=sigma, cor_mat_bar=self.cor_mat,
                                      a=torch.sigmoid(a), b=torch.sigmoid(b),
                                      buffer=self.garch_model.buffer, n_ahead=self.garch_model.n_ahead)
            loss = loss_garch_dcc(cov_mat, x)
            optimizer.zero_grad()
            # --- compute gradients
            loss.backward()
            # --- update parameters
            optimizer.step()
            if verbose:
                print('epoch {}, loss {}'.format(epoch, loss.item()))

        # --- self:
        self.a = a.detach()
        self.b = b.detach()

    def forward(self, x):
        # --- compute sigmas:
        sigma = self.garch_model(x)

        # --- constraint hack:
        a = torch.sigmoid(self.a)
        b = torch.sigmoid(self.b)

        # compute dynamic covariance:
        cov_mat = torch_garch_dcc(x=x, sigma=sigma, cor_mat_bar=self.cor_mat, a=a, b=b,
                                  buffer=self.garch_model.buffer, n_ahead=self.garch_model.n_ahead)

        return cov_mat


def torch_garch(x, alpha, beta, omega, buffer=1, n_ahead=1):
    # --- prep:
    nc_x = x.shape[1]

    # --- create implied garch weight: omega is sqrt (std-dev) not variance
    omega_2 = omega ** 2
    w = (1 / (1 - beta)) * omega_2
    with torch.no_grad():
        beta_max = beta.max().item()
        num_weights = min(ceil(log(0.01, beta_max)) + 1, 100)
    weight = beta.repeat(num_weights, 1)
    weight = torch.cumprod(weight, dim=0) / beta
    weight = torch.flip(weight, (0,))

    # --- add buffer to weight:
    if buffer > 0:
        zero = torch.zeros(buffer, nc_x, dtype=x.dtype)
        weight = torch.cat([weight, zero])

    # --- moving average of squared observations
    x_2 = x ** 2
    a = torch_wma(x=x_2, weight=weight, bias=None)

    # --- variance:
    var = w + alpha * a

    # --- n_ahead greater than 1 -> weighted average between var and long-term omega:
    if n_ahead > 1:
        a_b = alpha + beta
        with torch.no_grad():
            a_b = torch.clamp(a_b, min=0, max=0.99)
        a_b_n = a_b ** n_ahead
        co2 = (1 / n_ahead) * (1 - a_b_n) / (1 - a_b)
        co1 = (1 - co2) / (1 - a_b)
        var = co1 * omega_2 + co2 * var

    # --- sigma:
    sigma = torch.sqrt(var)

    return sigma


def torch_garch_dcc(x, sigma, cor_mat_bar, a, b, buffer=1, n_ahead=1):
    # --- Dynamic correlation:
    b = b * 0.99  # stability
    a_b = a + b

    # --- standardized values:
    zt = x / sigma
    zz = torch_quad_form(zt, zt)
    d_zz = zz.shape
    nc_x = d_zz[1]
    nc_x2 = nc_x ** 2

    # --- compute weight
    if a_b.item() > 1:
        w = 0.01
    else:
        w = (1 - a_b) / (1 - b)

    # --- weight lookback:
    with torch.no_grad():
        b_max = b.max().item()
        num_weights = min(ceil(log(0.01, b_max)) + 1, 100)

    weight = b.repeat((num_weights, nc_x2))
    weight = torch.cumprod(weight, dim=0) / b
    weight = torch.flip(weight, (0, ))

    # --- add buffer to weight:
    if buffer > 0:
        zero = torch.zeros(buffer, nc_x2, dtype=x.dtype)
        weight = torch.cat([weight, zero])

    # --- compute moving average of correlations
    zz_ma = torch_wma2(zz, weight=weight)

    # --- cor is average between static and dynamic components
    cor_mat = w * cor_mat_bar + a * zz_ma

    # --- n_ahead greater than 1 -> weighted average between var and long-term omega
    if n_ahead > 1:
        with torch.no_grad():
            a_b = torch.clamp(a_b, min=0, max=0.99)
        a_b_n = a_b ** n_ahead
        co2 = (1 / n_ahead) * (1 - a_b_n) / (1 - a_b)
        co1 = (1 - co2)
        cor_mat = co1 * cor_mat_bar + co2 * cor_mat

    # --- make a proper correlation matrix:
    cor_mat = torch_cov2cor(cor_mat)

    # --- convert to covariance:
    cov_mat = torch_cor2cov(cor_mat, sigma)

    return cov_mat


def torch_ma(x, n, bias=None):
    weight = torch.ones((n, x.shape[1]), dtype=x.dtype)/n
    return torch_wma(x=x, weight=weight, bias=bias)


def torch_wma(x, weight, bias=None):
    # --- prep
    x = make_matrix(x)
    weight = make_matrix(weight)
    nr_x = x.shape[0]
    nc_x = x.shape[1]
    nr_weight = weight.shape[0]
    nc_weight = weight.shape[1]

    # --- repeat weight if necessary
    if not (nc_x == nc_weight):
        weight = weight.repeat(1, nc_x)

    # --- reshaping
    x = x.t().unsqueeze(0)
    weight = weight.t().unsqueeze(1)

    # --- conv1d:
    padding = nr_weight - 1
    groups = nc_x
    out = torch.nn.functional.conv1d(x=x, weight=weight, bias=bias, groups=groups, padding=padding)
    out = out.squeeze(0)
    out = out[:, :nr_x]
    out = out.t()

    return out


def torch_wma2(x, weight, bias=None):
    n_obs = x.shape[0]
    n_col = x.shape[1] * x.shape[2]
    out = torch_wma(x=x.view((n_obs, n_col)), weight=weight, bias=bias)
    out = out.view((n_obs, x.shape[1], x.shape[2]))
    return out


def torch_cov(x, center=True):
    if is_batch(x):
        dim = 1
        n_obs = x.shape[1]
    else:
        dim = 0
        n_obs = x.shape[0]

    # --- crossprod
    cov_mat = torch_crossprod(x)
    cov_mat = cov_mat / n_obs

    # --- centering
    if center:
        mu = torch.mean(x, dim=dim, keepdim=True)
        mu_mat = torch_crossprod(mu)
        cov_mat = cov_mat - mu_mat

    return cov_mat


def torch_cor(x, center=True):
    cov_mat = torch_cov(x, center=center)
    cor_mat = torch_cov2cor(cov_mat)
    return cor_mat



