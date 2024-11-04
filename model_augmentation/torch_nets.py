import torch
from torch import nn
from model_augmentation.system_models import lti_system
from model_augmentation.utils import assign_param
from deepSI.utils import feed_forward_nn


class simple_res_net(nn.Module):
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        # linear + non-linear part
        super(simple_res_net, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if n_hidden_layers > 0:
            self.net_lin = nn.Linear(n_in, n_out, bias=False)
            self.net_non_lin = feed_forward_nn(n_in, n_out, n_nodes_per_layer=n_nodes_per_layer,
                                               n_hidden_layers=n_hidden_layers, activation=activation)
        else:
            self.net_lin = nn.Linear(n_in, n_out)
            self.net_non_lin = None

    def forward(self, x):
        if self.net_non_lin is not None:
            return self.net_lin(x) + self.net_non_lin(x)
        else:  # linear
            return self.net_lin(x)


class contracting_REN(nn.Module):
    """ Contracting REN -- DT: """
    def __init__(self, n_in=6, n_state=8, n_out=5, n_neurons=64, activation=nn.Tanh):
        super(contracting_REN, self).__init__()
        assert n_state > 0
        self.n_in = n_in
        self.n_state = n_state
        self.n_out = n_out
        self.n_neurons = n_neurons
        self.activation = activation()
        # Use the convex parametrization of Revay (2021) - Recurrent Equilibrium Networks, Flexible Dynamic Models with Guaranteed Stability and Robustness
        # Parameters: (see sec. V.A)
        self.X = nn.Parameter(data=torch.rand((2*self.n_state+self.n_neurons,2*self.n_state+self.n_neurons)))
        self.calB_2 = nn.Parameter(data=torch.rand((self.n_state, self.n_in)))
        self.C_2 = nn.Parameter(data=torch.rand((self.n_out, self.n_state)))
        self.calD_12 = nn.Parameter(data=torch.rand((self.n_neurons, self.n_in)))
        self.D_21 = nn.Parameter(data=torch.rand((self.n_out, self.n_neurons)))
        self.D_22 = nn.Parameter(data=torch.rand((self.n_out, self.n_in)))
        self.Y_1 = nn.Parameter(data=torch.rand((self.n_state, self.n_state)))
        self.epsilon = 1e-4
        self.biasvec = nn.Parameter(data=torch.rand((self.n_state+self.n_neurons+self.n_out)))

    def calculate_system_matrices(self):
        H_ = torch.einsum('ij,ik->jk', self.X, self.X) + self.epsilon * torch.eye(2 * self.n_state + self.n_neurons)
        F_ = H_[(self.n_state + self.n_neurons):, :self.n_state]
        calB_1 = H_[(self.n_state + self.n_neurons):, self.n_state:(self.n_state + self.n_neurons)]
        calP = H_[(self.n_state + self.n_neurons):, (self.n_state + self.n_neurons):]
        calC_1 = -H_[self.n_state:(self.n_state + self.n_neurons), :self.n_state]
        H11 = H_[:self.n_state, :self.n_state]
        E_ = 0.5 * (H11 + calP + self.Y_1)
        H22 = H_[self.n_state:(self.n_state + self.n_neurons), self.n_state:(self.n_state + self.n_neurons)]
        Lambda = 0.5 * H22.diag()  # Vector!
        calD_11 = -H22.tril(diagonal=-1)
        return E_, F_, calB_1, self.calB_2, Lambda, calC_1, calD_11, self.calD_12, self.C_2, self.D_21, self.D_22

    def calculate_w(self, x, u, calC_1, calD_11, calD_12, Lambda):
        # in:         | out:
        # - x (Nd, Nx)
        # - u (Nd, Nu)
        Linv = torch.diag(1 / Lambda)
        Dtilde = Linv @ calD_11
        # the following is of shape: (Nd, N_neurons)
        C1x_p_D12u_p_b = torch.einsum('ik, bk->bi', calC_1, x) + torch.einsum('ik, bk->bi', calD_12, u) + self.biasvec[self.n_state:(self.n_state+self.n_neurons)]
        v = torch.einsum('ij,bj->bi', Linv, C1x_p_D12u_p_b)
        for i in range(self.n_neurons):
            v += torch.einsum('j,b->bj', Dtilde[:, i], self.activation(v[:, i]))
        return self.activation(v)  # (Nd, N_neurons)

    def forward(self, hidden_state, u):
        # in:         | out:
        # - x (Nd, Nxh)
        # - u (Nd, Nu)
        # partition the bias vector:
        bx = self.biasvec[:self.n_state]
        by = self.biasvec[(self.n_state+self.n_neurons):]
        # calculate system matrices:
        E, Fx, B1, B2, Lambda, C1, D11, D12, C2, D21, D22 = self.calculate_system_matrices()
        Einv = torch.inverse(E)
        # calculate w
        w = self.calculate_w(hidden_state, u, C1, D11, D12, Lambda)  # (Nd, N_neurons)
        # calculate hidden state
        Ehidden = torch.einsum('ik, bk->bi', Fx, hidden_state) + torch.einsum('ik, bk->bi', B1, w) + \
            torch.einsum('ik, bk->bi', B2, u) + bx
        hidden = torch.einsum('ik, bk->bi', Einv, Ehidden)
        # Calculate network output
        y = torch.einsum('ik, bk->bi', C2, hidden_state) + torch.einsum('ik, bk->bi', D21, w) + \
            torch.einsum('ik, bk->bi', D22, u) + by
        return hidden, y

    def init_hidden(self):
        return torch.zeros(self.n_state)


class LFR_ANN(nn.Module):
    """
    LFR-ANN (DT/CT)
    """
    def __init__(self, n_in=6, n_state=8, n_out=5, n_neurons=64, activation=nn.Tanh, initial_gain=1e-3):
        super(LFR_ANN, self).__init__()
        assert n_state > 0
        self.n_in = n_in
        self.n_state = n_state
        self.n_out = n_out
        self.n_neurons = n_neurons
        self.activation = activation()
        # LFR-ANN matrices for the LTI part. The feedthrough between w and z is assumed to be zero
        self.A = nn.Parameter(data=initial_gain * torch.rand((self.n_state, self.n_state)))
        self.Bu = nn.Parameter(data=initial_gain * torch.rand((self.n_state, self.n_in)))
        self.Bw = nn.Parameter(data=initial_gain * torch.rand((self.n_state, self.n_neurons)))
        self.Cy = nn.Parameter(data=initial_gain * torch.rand((self.n_out, self.n_state)))
        self.Cz = nn.Parameter(data=initial_gain * torch.rand((self.n_neurons, self.n_state)))
        self.Dyu = nn.Parameter(data=initial_gain * torch.rand((self.n_out, self.n_in)))
        self.Dyw = nn.Parameter(data=initial_gain * torch.rand((self.n_out, self.n_neurons)))
        self.Dzu = nn.Parameter(data=initial_gain * torch.rand((self.n_neurons, self.n_in)))
        # biases
        self.bx = nn.Parameter(data=initial_gain * torch.rand(self.n_state))
        self.bz = nn.Parameter(data=initial_gain * torch.rand(self.n_neurons))
        self.by = nn.Parameter(data=initial_gain * torch.rand(self.n_out))

    def initialize_parameters(self, A=None, Bu=None, Bw=None,Cy=None, Cz=None, Dyu=None, Dyw=None, Dzu=None, bx=None, bz=None, by=None):
        self.A.data = assign_param(self.A, A, 'A')
        self.Bu.data = assign_param(self.Bu, Bu, 'Bu')
        self.Bw.data = assign_param(self.Bw, Bw, 'Bw')
        self.Cy.data = assign_param(self.Cy, Cy, 'Cy')
        self.Cz.data = assign_param(self.Cz, Cz, 'Cz')
        self.Dyu.data = assign_param(self.Dyu, Dyu, 'Dzw')
        self.Dzu.data = assign_param(self.Dzu, Dzu, 'Dzu')
        self.Dyw.data = assign_param(self.Dyw, Dyw, 'Dyw')
        self.bx.data = assign_param(self.bx, bx, 'bx')
        self.bz.data = assign_param(self.bz, bz, 'bz')
        self.by.data = assign_param(self.by, by, 'by')

    def forward(self, hidden_state, u):
        # in:         | out:
        # - x (Nd, Nxh)
        # - u (Nd, Nu)
        z = torch.einsum('ij, bj->bi',self.Cz, hidden_state) + torch.einsum('ij, bj->bi',self.Dzu, u) + self.bz
        w = self.activation(z)
        xp = torch.einsum('ij, bj->bi', self.A, hidden_state) + torch.einsum('ij, bj->bi', self.Bu, u) + torch.einsum('ij, bj->bi', self.Bw, w) + self.bx
        y = torch.einsum('ij, bj->bi', self.Cy, hidden_state) + torch.einsum('ij, bj->bi', self.Dyu, u) + torch.einsum('ij, bj->bi', self.Dyw, w) + self.by
        return xp, y

    def get_LTI_sys(self, Ts=-1):
        A = self.A.data
        Bu = self.Bu.data
        Bw = self.Bw.data
        Cy = self.Cy.data
        Cz = self.Cz.data
        Dyu = self.Dyu.data
        Dyw = self.Dyw.data
        Dzu = self.Dzu.data
        Dzw = torch.zeros((self.n_neurons,self.n_neurons))
        Dz = torch.cat((Dzu, Dzw), dim=1)
        Dy = torch.cat((Dyu, Dyw), dim=1)
        return lti_system(A=A, B=torch.cat((Bu, Bw),dim=1), C=torch.cat((Cy, Cz)), D=torch.cat((Dz,Dy)), Ts=Ts), self.n_neurons, torch.cat((self.bx.data, self.bz.data, self.by.data))


class time_integrators(nn.Module):
    """docstring for time_integrators"""
    def __init__(self, augm_structure, dt=None):
        super(time_integrators, self).__init__()
        self.dt_checked = False
        self.dt_valued = None

        self.dt = dt  # the current time constant (most probably the same as dt_0), should be set using set_dt before applying any dataset
        self.augm_structure = augm_structure

    @property
    def dt(self):
        return self._dt
    
    @dt.setter
    def dt(self, dt):
        if not self.dt_checked:  # checking for mixing of None valued dt and valued dt
            self.dt_checked = True
            self.dt_valued = False if dt is None else True
        else:
            assert self.dt_valued == (dt is not None), 'are you mixing valued dt and None dt valued datasets?'
        self._dt = 1. if dt is None else dt


class integrator_RK4(time_integrators):
    def forward(self, x, u):  # u constant on segment, zero-order hold
        y, k1 = self.augm_structure(x, u)  # t=0
        k1 *= self.dt

        _, k2 = self.augm_structure(x + k1/2, u)  # t=dt/2
        k2 *= self.dt

        _, k3 = self.augm_structure(x + k2/2, u)  # t=dt/2
        k3 *= self.dt

        _, k4 = self.augm_structure(x + k3, u)  # t=dt
        k4 *= self.dt

        xnext = x + (k1 + 2*k2 + 2*k3 + k4)/6
        return y, xnext


class integrator_euler(time_integrators):
    def forward(self, x, u):  # u constant on segment
        y, xderiv = self.augm_structure(x, u)  # t=0
        xnext = x + self.dt * xderiv
        return y, xnext
