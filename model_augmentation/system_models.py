import deepSI
import torch
from torch import nn
from model_augmentation.utils import RK4_step


# -------------------------------------------------------------------------------------------
# ------------------------------- SIMULATION FUNCTION WRAPPERS ------------------------------
# -------------------------------------------------------------------------------------------

def hidden_apply_experiment(sys, data, x0, x0_meas=False, dt=None):
    # Apply experiment wrapper
    if sys.Ts is None and data.dt is None and dt is None:
        raise ValueError('Sample time of data not specified')
    if dt is None:
        dt = data.dt
    if type(data) is deepSI.system_data.System_data_list:
        lis = []
        for i in range(len(data)):
            if x0_meas:
                x0 = data[i].y[0]
            y, x = hidden_apply_one_experiment(sys,data[i].u,x0, data[i].N_samples, dt)
            lis.append(deepSI.System_data(u=data[i].u, y=y.detach(), x=x.detach()))
        retrn = deepSI.System_data_list(lis)
    else:
        if x0_meas:
            x0 = data.y[0]
        y, x = hidden_apply_one_experiment(sys, data.u, x0, data.N_samples, dt)
        retrn = deepSI.System_data(u=data.u, y=y.detach(), x=x.detach())
    return retrn


def hidden_apply_one_experiment(sys, u, x0, T, dt):
    u = torch.tensor(u,dtype=torch.float) if not torch.is_tensor(u) else u
    if u.ndim == 1:
        u = torch.unsqueeze(u, dim=1)
    x0 = torch.tensor(x0, dtype=torch.float) if not torch.is_tensor(x0) else x0
    y = torch.zeros(T, sys.Ny)
    x = torch.zeros(T+1, sys.Nx)
    x[0, :] = x0
    for k in range(T):
        y[k, :] = sys.h(x[k, :], u[k, :])
        # Discrete simulation
        if sys.Ts is not None:
            x[k + 1, :] = sys.f(x[k, :], u[k, :])
        else:
            x[k + 1, :] = RK4_step(sys.f, x[k, :], u[k, :], dt)
    return y, x[:-1, :]


# -------------------------------------------------------------------------------------------
# -------------------------------------- SYSTEM MODELS --------------------------------------
# -------------------------------------------------------------------------------------------
class lti_system:
    """
    Simple LTI-SS system
    - tuning of physical parameters, hence orthogonalization-based regularization is currently not supported
    -  for these type of applications general_nonlinear_system should be used
    """
    def __init__(self, A, B, C, D, Ts=-1):
        super(lti_system, self).__init__()
        self.A = A if torch.is_tensor(A) else torch.tensor(A, dtype=torch.float)  # shape: (Nx, Nx)
        self.B = B if torch.is_tensor(B) else torch.tensor(B, dtype=torch.float)   # shape: (Nx, Nu)
        self.C = C if torch.is_tensor(C) else torch.tensor(C, dtype=torch.float)   # shape: (Ny, Nx)
        self.D = D if torch.is_tensor(D) else torch.tensor(D, dtype=torch.float)   # shape: (Ny, Nu)
        self.Nx = self.A.shape[0]
        self.Nu = self.B.shape[1]
        self.Ny = self.C.shape[0]
        self.Ts = Ts

    def f(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - x+ (Nd, Nx)
        #  - u (Nd, Nu) |
        einsumequation = 'ik, bk->bi' if x.ndim > 1 else 'ik, k->i'
        Ax = torch.einsum(einsumequation, self.A, x)  # (Nx, Nx)*(Nd, Nx)->(Nd, Nx)
        einsumequation = 'ik, bk->bi' if u.ndim > 1 else 'ik, k->i'
        Bu = torch.einsum(einsumequation, self.B, u)   # (Nd, Nx, Nu)*(Nd, Nu)->(Nd, Nx)
        return Ax + Bu

    def h(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - y (Nd, Ny)
        #  - u (Nd, Nu) |
        einsumequation = 'ik, bk->bi' if x.ndim > 1 else 'ik, k->i'
        Cx = torch.einsum(einsumequation, self.C, x)  # (Nd, Nx, Nx)*(Nd, Nx)->(Nd, Nx)
        einsumequation = 'ik, bk->bi' if u.ndim > 1 else 'ik, k->i'
        Du = torch.einsum(einsumequation, self.D, u)  # (Nd, Nx, Nu)*(Nd, Nu)->(Nd, Nx)
        return Cx + Du

    def apply_experiment(self, data, x0=None, x0_meas=False):
        if x0 is None:
            x0 = torch.zeros(self.Nx)
        return hidden_apply_experiment(self, data, x0, x0_meas)

    def calculate_orth_matrix(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - F(x,u) (Nd*Nx, Np)  assumed LIP model: x+ = F(x,u)*theta
        #  - u (Nd, Nu) |
        raise NotImplementedError('The matrix containing the basis functions should be implemented in child!')


class general_nonlinear_system(nn.Module):
    """
    This system must be able to process the inputs and outputs for the subspace encoder,
    hence: shape({x,u,y}) = (Nd,{nx,nu,ny})
    also parameter mapping should be implemented in child
    parm 1D numpy array / tensor
    """
    def __init__(self, nu, nx, ny, Ts=-1, parmTune=False, parm=None):
        super(general_nonlinear_system, self).__init__()
        self.Nu = nu
        self.Nx = nx
        self.Ny = ny
        self.Ts = Ts
        self.parm_corr_enab = parmTune
        if self.parm_corr_enab and parm is not None:
            self.P_orig = torch.tensor(data=parm, dtype=torch.float)
            self.P = nn.Parameter(data=self.P_orig.clone())
        elif parm is not None:
            self.P_orig = torch.tensor(data=parm, dtype=torch.float)
            self.P = self.P_orig.clone()

    def f(self, x, u):
        # in:                   | out:
        #     x (Nd,Nx)         |      x+ (Nd,Nx)
        #     u (Nd,Nu)  |
        raise NotImplementedError('Function should be implemented in child')

    def h(self, x, u):
        # in:                   | out:
        #     x (Nd,Nx)         |      y (Nd,Ny)
        #     u (Nd,Nu)  |
        raise NotImplementedError('Function should be implemented in child')

    def apply_experiment(self, data, dt=None, x0=None, x0_meas=False):
        if x0 is None:
            x0 = torch.zeros(self.Nx)
        return hidden_apply_experiment(self, data, x0, x0_meas, dt=dt)

    def calculate_orth_matrix(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - F(x,u) (Nd*Nx, Np)  assumed LIP model: x+ = F(x,u)*theta
        #  - u (Nd, Nu) |
        raise NotImplementedError('The matrix containing the basis functions should be implemented in child!')
