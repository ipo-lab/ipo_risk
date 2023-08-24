import torch
import torch.nn as nn
from ts.cov import CovFactor, CovFactorOLS, GarchCCCNet, GarchCCCNetNNL, GarchDCCNet, GarchDCCNetNNL
from lqp_py.solve_box_qp_admm_torch import SolveBoxQP
from lqp_py.control import box_qp_control
from rpth.rpth import RPTHNet
from rpth.control import rpth_control


class MinVarCCC(nn.Module):
    def __init__(self, x, y, n_ahead=1, init='random', control=box_qp_control()):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchCCCNet(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactor(x=x, y=y, factor_cov_model=factor_cov_model, init=init)
        # --- portfolio model:
        self.QP = SolveBoxQP(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        p = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        A = torch.ones((n_obs, 1, n_z), dtype=dtype)
        b = torch.ones((n_obs, 1, 1), dtype=dtype)
        lb = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        ub = 100 * torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
        return z


class MinVarCCCOLS(nn.Module):
    def __init__(self, x, y, n_ahead=1, control=box_qp_control()):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchCCCNetNNL(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactorOLS(x=x, y=y, factor_cov_model=factor_cov_model)
        # --- portfolio model:
        self.QP = SolveBoxQP(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        p = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        A = torch.ones((n_obs, 1, n_z), dtype=dtype)
        b = torch.ones((n_obs, 1, 1), dtype=dtype)
        lb = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        ub = 100 * torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
        return z


class MinVarDCC(nn.Module):
    def __init__(self, x, y, n_ahead=1, init='random', control=box_qp_control()):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchDCCNet(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactor(x=x, y=y, factor_cov_model=factor_cov_model, init=init)
        # --- portfolio model:
        self.QP = SolveBoxQP(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        p = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        A = torch.ones((n_obs, 1, n_z), dtype=dtype)
        b = torch.ones((n_obs, 1, 1), dtype=dtype)
        lb = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        ub = 100 * torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
        return z


class MinVarDCCOLS(nn.Module):
    def __init__(self, x, y, n_ahead=1, control=box_qp_control()):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchDCCNetNNL(x=x,n_ahead=n_ahead)
        self.cov_model = CovFactorOLS(x=x, y=y, factor_cov_model=factor_cov_model)
        # --- portfolio model:
        self.QP = SolveBoxQP(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        p = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        A = torch.ones((n_obs, 1, n_z), dtype=dtype)
        b = torch.ones((n_obs, 1, 1), dtype=dtype)
        lb = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        ub = 100 * torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
        return z


class MaxDivCCC(nn.Module):
    def __init__(self, x, y, n_ahead=1, init='random', control=box_qp_control()):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchCCCNet(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactor(x=x, y=y, factor_cov_model=factor_cov_model, init=init)
        # --- portfolio model:
        self.QP = SolveBoxQP(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        with torch.no_grad():
            p = torch.sqrt(torch.diagonal(Q, dim1=1, dim2=2)).unsqueeze(2)
        A = torch.ones((n_obs, 1, n_z), dtype=dtype)
        b = torch.ones((n_obs, 1, 1), dtype=dtype)
        lb = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        ub = 1e8 * torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
        z = z / z.sum(dim=1, keepdims=True)
        return z


class MaxDivCCCOLS(nn.Module):
    def __init__(self, x, y, n_ahead=1, control=box_qp_control()):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchCCCNetNNL(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactorOLS(x=x, y=y, factor_cov_model=factor_cov_model)
        # --- portfolio model:
        self.QP = SolveBoxQP(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        with torch.no_grad():
            p = torch.sqrt(torch.diagonal(Q, dim1=1, dim2=2)).unsqueeze(2)
        A = torch.ones((n_obs, 1, n_z), dtype=dtype)
        b = torch.ones((n_obs, 1, 1), dtype=dtype)
        lb = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        ub = 1e8 * torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
        z = z / z.sum(dim=1, keepdims=True)
        return z


class MaxDivDCC(nn.Module):
    def __init__(self, x, y, n_ahead=1, init='random', control=box_qp_control()):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchDCCNet(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactor(x=x, y=y, factor_cov_model=factor_cov_model, init=init)
        # --- portfolio model:
        self.QP = SolveBoxQP(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        with torch.no_grad():
            p = torch.sqrt(torch.diagonal(Q, dim1=1, dim2=2)).unsqueeze(2)
        A = torch.ones((n_obs, 1, n_z), dtype=dtype)
        b = torch.ones((n_obs, 1, 1), dtype=dtype)
        lb = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        ub = 1e8 * torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
        z = z / z.sum(dim=1, keepdims=True)
        return z


class MaxDivDCCOLS(nn.Module):
    def __init__(self, x, y, n_ahead=1, control=box_qp_control()):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchDCCNetNNL(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactorOLS(x=x, y=y, factor_cov_model=factor_cov_model)
        # --- portfolio model:
        self.QP = SolveBoxQP(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        with torch.no_grad():
            p = torch.sqrt(torch.diagonal(Q, dim1=1, dim2=2)).unsqueeze(2)
        A = torch.ones((n_obs, 1, n_z), dtype=dtype)
        b = torch.ones((n_obs, 1, 1), dtype=dtype)
        lb = torch.zeros((n_obs, n_z, 1), dtype=dtype)
        ub = 1e8 * torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, p=p, A=A, b=b, lb=lb, ub=ub)
        z = z / z.sum(dim=1, keepdims=True)
        return z


class RPCCC(nn.Module):
    def __init__(self, x, y, n_ahead=1, init='random', control=rpth_control(normalize=True)):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchCCCNet(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactor(x=x, y=y, factor_cov_model=factor_cov_model, init=init)
        # --- portfolio model:
        self.QP = RPTHNet(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        r = torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, r=r)
        return z


class RPCCCOLS(nn.Module):
    def __init__(self, x, y, n_ahead=1, control=rpth_control(normalize=True)):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchCCCNetNNL(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactorOLS(x=x, y=y, factor_cov_model=factor_cov_model)
        # --- portfolio model:
        self.QP = RPTHNet(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        r = torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, r=r)
        return z


class RPDCC(nn.Module):
    def __init__(self, x, y, n_ahead=1, init='random', control=rpth_control(normalize=True)):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchDCCNet(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactor(x=x, y=y, factor_cov_model=factor_cov_model, init=init)
        # --- portfolio model:
        self.QP = RPTHNet(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        r = torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, r=r)
        return z


class RPDCCOLS(nn.Module):
    def __init__(self, x, y, n_ahead=1, control=rpth_control(normalize=True)):
        super().__init__()
        # --- init cov model:
        factor_cov_model = GarchDCCNetNNL(x=x, n_ahead=n_ahead)
        self.cov_model = CovFactorOLS(x=x, y=y, factor_cov_model=factor_cov_model)
        # --- portfolio model:
        self.QP = RPTHNet(control=control)

    def forward(self, x):
        n_obs = x.shape[0]
        dtype = x.dtype
        Q = self.cov_model(x=x)
        n_z = Q.shape[1]
        r = torch.ones((n_obs, n_z, 1), dtype=dtype)

        z = self.QP.forward(Q=Q, r=r)
        return z
