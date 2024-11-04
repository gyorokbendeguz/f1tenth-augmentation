import torch
from torch import nn
from model_augmentation.utils import verifySystemType, verifyNetType, initialize_augmentation_net, calculate_normalization
from model_augmentation.augmentation_encoders import default_encoder_net, state_measure_encoder, dynamic_state_meas_encoder
from model_augmentation.fit_system import augmentation_encoder, augmentation_encoder_deriv
from model_augmentation.torch_nets import contracting_REN, integrator_RK4


# -------------------------------------------------------------------------------------------
# ------------------------------------- GENERIC FUNCTIONS ----------------------------------
# -------------------------------------------------------------------------------------------

def verifyAugmentationStructure(augmentation_struct, known_sys, neur_net, nx_hidden=0):
    # Verify if the augmentation structure is valid and calculate the encoder state depending on static/dynamic augmentation.
    if augmentation_struct is SSE_LFRAugmentation: static = True; augm = 'LFR'
    elif augmentation_struct is SSE_LFRDynAugmentation: static = False; augm = 'LFR'
    elif augmentation_struct is LFR_test:
        static = False
        augm = 'LFR'
    elif augmentation_struct is LFR_test_v2:
        static = False
        augm = 'LFR'
    elif augmentation_struct is SSE_AdditiveAugmentation: static = True; augm = 'additive'
    elif augmentation_struct is SSE_AdditiveDynAugmentation: static = False; augm = 'additive'
    elif augmentation_struct is SSE_MultiplicativeAugmentation: static = True; augm = 'multiplicative'
    elif augmentation_struct is SSE_MultiplicativeDynAugmentation: static = False; augm = 'multiplicative'
    else: raise ValueError("'augmentation_structure' must be one of the types defined in 'model_augmentation.augmentationstructures'")

    initialize_augmentation_net(network=neur_net, augm_type=augm, nx=known_sys.Nx)

    if static:
        # Only learn the system states for static augmentation
        nx_encoder = known_sys.Nx
    else:
        # Learn the state of the augmented model as well for dynamic augmentation
        nx_system = known_sys.Nx
        nx_encoder = nx_system + nx_hidden
    return nx_encoder


def get_dynamic_augment_fitsys(augmentation_structure, known_system, hidden_state, neur_net, aug_kwargs={}, e_net=default_encoder_net,
                               y_lag_encoder=None, u_lag_encoder=None, enet_kwargs={}, na_right=0, nb_right=0,
                               regLambda=0, norm_data=None, norm_x0_meas=False, l2_reg=0, init_scaling_factor=0):
    nx_encoder = verifyAugmentationStructure(augmentation_structure, known_system, neur_net, hidden_state)
    if y_lag_encoder is None: y_lag_encoder = 1
    if u_lag_encoder is None: u_lag_encoder = 1
    if e_net is None:
        na_right = 1
        e_net = dynamic_state_meas_encoder
    std_x, std_y, std_u = calculate_normalization(norm_data, norm_x0_meas, known_system)
    return augmentation_encoder(nx=nx_encoder, na=y_lag_encoder, nb=u_lag_encoder, e_net=e_net,
                                e_net_kwargs=dict(nx_h=hidden_state, std_x=std_x, std_y=std_y, std_u=std_u, **enet_kwargs),
                                augm_net=augmentation_structure, na_right=na_right, nb_right=nb_right,
                                augm_net_kwargs=dict(known_system=known_system, net=neur_net, regLambda=regLambda,
                                                     nx_h=hidden_state, std_x=std_x, std_y=std_y, std_u=std_u,
                                                     l2_reg=l2_reg, init_scaling_factor=init_scaling_factor, **aug_kwargs))


def get_augmented_fitsys(augmentation_structure, known_system, neur_net, aug_kwargs={}, e_net=default_encoder_net,
                         y_lag_encoder=None, u_lag_encoder=None, enet_kwargs={}, na_right=0, nb_right=0,
                         regLambda=0, orthLambda=0, norm_data=None, norm_x0_meas=False, l2_reg=0):
    nx_encoder = verifyAugmentationStructure(augmentation_structure, known_system, neur_net)
    if y_lag_encoder is None: y_lag_encoder = nx_encoder + 1
    if u_lag_encoder is None: u_lag_encoder = nx_encoder + 1
    if e_net is None:
        y_lag_encoder = 1
        u_lag_encoder = 1
        na_right = 1
        e_net = state_measure_encoder
    std_x, std_y, std_u = calculate_normalization(norm_data, norm_x0_meas, known_system)
    return augmentation_encoder(nx=nx_encoder, na=y_lag_encoder, nb=u_lag_encoder, e_net=e_net,
                                e_net_kwargs=dict(std_x=std_x, std_y=std_y, std_u=std_u, **enet_kwargs),
                                augm_net=augmentation_structure, na_right=na_right, nb_right=nb_right,
                                augm_net_kwargs=dict(known_system=known_system, net=neur_net, regLambda=regLambda,
                                                     orthLambda=orthLambda, std_x=std_x, std_y=std_y, std_u=std_u,
                                                     l2_reg=l2_reg, **aug_kwargs))


def get_ct_augmented_fitsys(augmentation_structure, known_system, neur_net, aug_kwargs={}, e_net=default_encoder_net,
                            y_lag_encoder=None, u_lag_encoder=None, enet_kwargs={}, na_right=0, nb_right=0,
                            regLambda=0, orthLambda=0, norm_data=None, norm_x0_meas=False,
                            integrator_net=integrator_RK4, integrator_net_kwargs={}):
    nx_encoder = verifyAugmentationStructure(augmentation_structure, known_system, neur_net)
    if y_lag_encoder is None: y_lag_encoder = nx_encoder + 1
    if u_lag_encoder is None: u_lag_encoder = nx_encoder + 1
    if e_net is None:
        y_lag_encoder = 1
        u_lag_encoder = 1
        na_right = 1
        e_net = state_measure_encoder
    std_x, std_y, std_u = calculate_normalization(norm_data, norm_x0_meas, known_system)
    return augmentation_encoder_deriv(nx=nx_encoder, na=y_lag_encoder, nb=u_lag_encoder, e_net=e_net,
                                      e_net_kwargs=dict(std_x=std_x, std_y=std_y, std_u=std_u, **enet_kwargs),
                                      integrator_net=integrator_net, integrator_net_kwargs=integrator_net_kwargs,
                                      augm_net=augmentation_structure, na_right=na_right, nb_right=nb_right,
                                      augm_net_kwargs=dict(known_system=known_system, net=neur_net, regLambda=regLambda,
                                                           orthLambda=orthLambda, std_x=std_x, std_y=std_y, std_u=std_u,
                                                           **aug_kwargs))


class LFR_test(nn.Module):
    """
    testing this new concept
    """
    def __init__(self, known_system, net, std_x, std_y, std_u, regLambda=0, l2_reg=0, **kwargs):
        super(LFR_test, self).__init__()

        #Todo: verification of sys and net

        # Save parameters
        self.sys = known_system
        self.net = net
        self.Nu = self.sys.Nu
        self.Nx = self.sys.Nx
        self.Ny = self.sys.Ny
        self.Nz = self.net.n_in
        self.Nw = self.net.n_out

        # save regularization coefficient (only if first-principle model enables it)
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.regLambda = regLambda
        else:
            self.regLambda = 0

        self.l2_reg = l2_reg

        # Todo: check for LTI model
        if not self.sys.linearized_form_provided:
            x0 = torch.zeros(self.Nx)
            u0 = torch.zeros(self.Nu)
            jacobians = torch.autograd.functional.jacobian(self.sys.f, (x0, u0))
            self.A = jacobians[0]
            self.B = jacobians[1]

        # save normalization parameters
        self.Tx = torch.diag(torch.tensor(1 / std_x, dtype=torch.float))
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))
        self.Tx_inv = torch.diag(torch.tensor(std_x, dtype=torch.float))
        self.Ty_inv = torch.diag(torch.tensor(std_y, dtype=torch.float))

        # initialize interconnection matrices
        Cz_rand = torch.rand(self.Nz, self.Nx) * 2 - 1
        Dzu_rand = torch.rand(self.Nz, self.Nu) * 2 - 1
        self.Cz = nn.Parameter(data=Cz_rand)
        self.Dzu = nn.Parameter(data=Dzu_rand)
        self.Bw = nn.Parameter(data=torch.zeros(self.Nx, self.Nw))
        self.Dyw = nn.Parameter(data=torch.zeros(self.Ny, self.Nw))

        Dzw_rand = torch.rand(self.Nz, self.Nx) * 2 - 1
        self.Dzw = nn.Parameter(data=Dzw_rand)

    def compute_z(self, x, u, w_tilde):
        # in:           | out:
        #  - x (Nd, Nx) |  - z (Nd, Nz)
        #  - u (Nd, Nu) |
        zx = torch.einsum('ij, bj -> bi', self.Cz, x @ self.Tx)   # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        zu = torch.einsum('ij, bj -> bi', self.Dzu, u @ self.Tu)  # (Nz, Nu)*(Nd, Nu)->(Nd, Nz)
        zw = torch.einsum('ij, bj -> bi', self.Dzw, w_tilde @ self.Tx)
        return zx + zu + zw

    def compute_ynet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Dyw*w (Nd, Ny)
        return torch.einsum('ij, bj -> bi', self.Dyw, w) @ self.Ty_inv  # (Ny, Nw)*(Nd, Nw)->(Nd, Ny)

    def compute_xnet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Bw*w (Nd, Nx)
        return torch.einsum('ij, bj -> bi', self.Bw, w) @ self.Tx_inv  # (Nx, Nw)*(Nd, Nw)->(Nd, Nx)

    def forward(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - y  (Nd, Ny)
        #  - u (Nd, Nu) |  - x+ (Nd, Nx)
        if u.ndim == 1:
            u = torch.unsqueeze(u, dim=0)

        x_plus_fp = self.sys.f(x, u)
        if self.sys.linearized_form_provided:
            w_tilde = x_plus_fp - self.sys.linear_part(x, u)
        else:
            w_tilde = x_plus_fp - x @ self.A.T - u @ self.B.T

        # compute network contribution
        z = self.compute_z(x, u, w_tilde)
        w = self.net(z)

        # add network contributions to state transition and output calculation
        x_plus = x_plus_fp + self.compute_xnet_contribution(w)
        y_k = self.sys.h(x, u) + self.compute_ynet_contribution(w)

        return y_k, x_plus


class LFR_test_v2(nn.Module):
    def __init__(self, known_system, net, std_x, std_y, std_u, regLambda=0, l2_reg=0, **kwargs):
        super(LFR_test_v2, self).__init__()
        # Todo: verification of sys and net

        # Save parameters
        self.sys = known_system
        self.net = net
        self.Nu = self.sys.Nu
        self.Nx = self.sys.Nx
        self.Ny = self.sys.Ny
        self.Nz = self.net.n_in
        self.Nw = self.net.n_out

        # save regularization coefficient (only if first-principle model enables it)
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.regLambda = regLambda
        else:
            self.regLambda = 0

        self.l2_reg = l2_reg

        # save normalization parameters
        self.Tx = torch.diag(torch.tensor(1 / std_x, dtype=torch.float))
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))
        self.Tx_inv = torch.diag(torch.tensor(std_x, dtype=torch.float))
        self.Tu_inv = torch.diag(torch.tensor(std_u, dtype=torch.float))
        self.Ty_inv = torch.diag(torch.tensor(std_y, dtype=torch.float))

        # initialize interconnection matrices
        Cz_rand = torch.rand(self.Nz, self.Nx) * 2 - 1
        Dzu_rand = torch.rand(self.Nz, self.Nu) * 2 - 1
        self.Cz = nn.Parameter(data=Cz_rand)
        self.Dzu = nn.Parameter(data=Dzu_rand)
        self.Bw = nn.Parameter(data=torch.zeros(self.Nx, self.Nw))
        self.Dyw = nn.Parameter(data=torch.zeros(self.Ny, self.Nw))

        self.Bw_fp = nn.Parameter(data=torch.eye(self.Nx))
        Cz_tilde_init = torch.cat((torch.eye(self.Nx), torch.zeros(self.Nu, self.Nx)), dim=0)
        Dzu_tilde_init = torch.cat((torch.zeros(self.Nx, self.Nu), torch.eye(self.Nu)), dim=0)
        self.Cz_tilde = nn.Parameter(data=Cz_tilde_init)
        self.Dzu_tilde = nn.Parameter(data=Dzu_tilde_init)
        self.Dzw_tilde = nn.Parameter(data=torch.zeros(self.Nx+self.Nu, self.Nw))

    def compute_z(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - z (Nd, Nz)
        #  - u (Nd, Nu) |
        zx = torch.einsum('ij, bj -> bi', self.Cz, x @ self.Tx)   # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        zu = torch.einsum('ij, bj -> bi', self.Dzu, u @ self.Tu)  # (Nz, Nu)*(Nd, Nu)->(Nd, Nz)
        return zx + zu

    def compute_z_tilde(self, x, u, w):
        # in:           | out:
        #  - x (Nd, Nx) |  - z_tilde (Nd, Nx+Nu)
        #  - u (Nd, Nu) |
        #  - w (Nd, Nw) |
        zx = torch.einsum('ij, bj -> bi', self.Cz_tilde, x @ self.Tx)   # (Nx+Nu, Nx)*(Nd, Nx)->(Nd, Nx+Nu)
        zu = torch.einsum('ij, bj -> bi', self.Dzu_tilde, u @ self.Tu)  # (Nx+Nu, Nu)*(Nd, Nu)->(Nd, Nx+Nu)
        zw = torch.einsum('ij, bj -> bi', self.Dzw_tilde, w)            # (Nx+Nu, Nw)*(Nd, Nw)->(Nd, Nx+Nu)
        return zx + zu + zw

    def compute_fp_contribution(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - x_plus (Nd, Nx)
        #  - u (Nd, Nu) |
        x_plus = torch.einsum('ij, bj -> bi', self.Bw_fp, self.sys.f(x, u) @ self.Tx) @ self.Tx_inv  # (Nx, Nx)*(Nd, Nx)->(Nd, Nx)
        return x_plus

    def compute_ynet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Dyw*w (Nd, Ny)
        return torch.einsum('ij, bj -> bi', self.Dyw, w) @ self.Ty_inv  # (Ny, Nw)*(Nd, Nw)->(Nd, Ny)

    def compute_xnet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Bw*w (Nd, Nx)
        return torch.einsum('ij, bj -> bi', self.Bw, w) @ self.Tx_inv  # (Nx, Nw)*(Nd, Nw)->(Nd, Nx)

    def forward(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - y  (Nd, Ny)
        #  - u (Nd, Nu) |  - x+ (Nd, Nx)
        if u.ndim == 1:
            u = torch.unsqueeze(u, dim=0)

        z = self.compute_z(x, u)
        w = self.net(z)
        z_tilde = self.compute_z_tilde(x, u, w)

        x_tilde = z_tilde[:, :self.Nx] @ self.Tx_inv
        u_tilde = z_tilde[:, self.Nx:] @ self.Tu_inv

        x_plus = self.compute_fp_contribution(x_tilde, u_tilde) + self.compute_xnet_contribution(w)

        y_k = self.sys.h(x, u) + self.compute_ynet_contribution(w)

        return y_k, x_plus


# -------------------------------------------------------------------------------------------
# --------------------------------- LFR-BASED AUGMENTATION ---------------------------------
# -------------------------------------------------------------------------------------------

class SSE_LFRAugmentation(nn.Module):
    """
    Todo: Add doscstring
    """
    def __init__(self, known_system, net, std_x, std_y, std_u, regLambda=0, l2_reg=0, **kwargs):
        super(SSE_LFRAugmentation, self).__init__()

        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)
        verifyNetType(net, 'static')

        # Save parameters
        self.sys = known_system
        self.net = net
        self.Nu = self.sys.Nu
        self.Nx = self.sys.Nx
        self.Ny = self.sys.Ny
        self.Nz = self.net.n_in
        self.Nw = self.net.n_out

        # save regularization coefficient (only if first-principle model enables it)
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.regLambda = regLambda

        self.l2_reg = l2_reg

        # save normalization parameters
        self.Tx = torch.diag(torch.tensor(1 / std_x, dtype=torch.float))
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))
        self.Tx_inv = torch.diag(torch.tensor(std_x, dtype=torch.float))
        self.Ty_inv = torch.diag(torch.tensor(std_y, dtype=torch.float))

        # initialize interconnection matrices
        Cz_rand = torch.rand(self.Nz, self.Nx) * 2 - 1
        Dzu_rand = torch.rand(self.Nz, self.Nu) * 2 - 1
        self.Cz = nn.Parameter(data=Cz_rand)
        self.Dzu = nn.Parameter(data=Dzu_rand)
        # ToDO: Bw and Dyw should have an option to not be zero, just a small number
        self.Bw = nn.Parameter(data=torch.zeros(self.Nx, self.Nw))
        self.Dyw = nn.Parameter(data=torch.zeros(self.Ny, self.Nw))
        # self.Dzw = None if Dzw_is_zero else nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nw))

    def compute_z(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - z (Nd, Nz)
        #  - u (Nd, Nu) |
        zx = torch.einsum('ij, bj -> bi', self.Cz, x @ self.Tx)   # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        zu = torch.einsum('ij, bj -> bi', self.Dzu, u @ self.Tu)  # (Nz, Nu)*(Nd, Nu)->(Nd, Nz)
        # zw = torch.zeros(zu.shape) if self.Dzw is None else torch.zeros(zu.shape)
        return zx + zu  # + zw

    def compute_ynet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Dyw*w (Nd, Ny)
        return torch.einsum('ij, bj -> bi', self.Dyw, w) @ self.Ty_inv  # (Ny, Nw)*(Nd, Nw)->(Nd, Ny)

    def compute_xnet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Bw*w (Nd, Nx)
        return torch.einsum('ij, bj -> bi', self.Bw, w) @ self.Tx_inv  # (Nx, Nw)*(Nd, Nw)->(Nd, Nx)

    def forward(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - y  (Nd, Ny)
        #  - u (Nd, Nu) |  - x+ (Nd, Nx)
        if u.ndim == 1:
            u = torch.unsqueeze(u, dim=0)

        # compute network contribution
        z = self.compute_z(x, u)
        w = self.net(z)

        # add network contributions to state transition and output calculation
        x_plus = self.sys.f(x, u) + self.compute_xnet_contribution(w)
        y_k = self.sys.h(x, u) + self.compute_ynet_contribution(w)

        return y_k, x_plus


# -------------------------------------------------------------------------------------------
# ------------------------------ DYNAMIC LFR-BASED AUGMENTATION -----------------------------
# -------------------------------------------------------------------------------------------

class SSE_LFRDynAugmentation(nn.Module):
    """
    #Todo: add documentation
    """
    def __init__(self, nx_h, known_system, net, std_x, std_y, std_u, regLambda=0, l2_reg=0, init_scaling_factor=0,
                 **kwargs):
        super(SSE_LFRDynAugmentation, self).__init__()

        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)

        # Save parameters
        self.sys = known_system
        self.Nxh = nx_h
        self.net = net
        if type(self.net) is contracting_REN:
            self.nettype = 'cREN'
            self.Nz = self.net.n_in
            self.Nw = self.net.n_out
        else:
            self.nettype = 'Feedforward'
            self.Nz = self.net.n_in - self.Nxh
            self.Nw = self.net.n_out - self.Nxh
        self.Nu = self.sys.Nu
        self.Nx = self.sys.Nx
        self.Ny = self.sys.Ny

        # Save regularization parameters (if allowed by the baseline model)
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.regLambda = regLambda

        self.l2_reg = l2_reg

        # Save normalization matrices
        self.Tx = torch.diag(torch.tensor(1 / std_x, dtype=torch.float))
        self.Tx_inv = torch.diag(torch.tensor(std_x, dtype=torch.float))
        self.Ty_inv = torch.diag(torch.tensor(std_y, dtype=torch.float))
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))

        # Initialize LFR matrices
        Cz_rand = torch.rand(self.Nz, self.Nx) * 2 - 1
        Dzu_rand = torch.rand(self.Nz, self.Nu) * 2 - 1
        self.Cz = nn.Parameter(data=Cz_rand)
        self.Dzu = nn.Parameter(data=Dzu_rand)
        if init_scaling_factor > 0:
            Bw = init_scaling_factor * torch.rand(self.Nx, self.Nw)
            Dyw = init_scaling_factor * torch.rand(self.Ny, self.Nw)
        else:
            Bw = torch.zeros(self.Nx, self.Nw)
            Dyw = torch.zeros(self.Ny, self.Nw)
        self.Bw = nn.Parameter(data=Bw)
        self.Dyw = nn.Parameter(data=Dyw)
        # self.Dzw = None if Dzw_is_zero else nn.Parameter(data=initial_scaling_factor * torch.rand(self.Nz, self.Nw))

    def compute_z(self, x, w, u):
        # in:            | out:
        #  - x (Nd, Nx)  |  - z (Nd, Nz)
        #  - u (Nd, Nu)  |
        zx = torch.einsum('ij, bj -> bi', self.Cz, x @ self.Tx)   # (Nz, Nx)*(Nd, Nx)->(Nd, Nz)
        zu = torch.einsum('ij, bj -> bi', self.Dzu, u @ self.Tu)  # (Nz, Nu)*(Nd, Nu)->(Nd, Nz)
        # zw = torch.zeros(zu.shape) if self.Dzw is None else torch.zeros(zu.shape)
        return zx + zu  # + zw

    def compute_ynet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Dyw*w (Nd, Ny)
        return torch.einsum('ij, bj -> bi', self.Dyw, w) @ self.Ty_inv  # (Ny, Nw)*(Nd, Nw)->(Nd, Ny)

    def compute_xnet_contribution(self, w):
        # in:                | out:
        #  - w (Nd, Nw)      |  - Bw*w (Nd, Nx)
        return torch.einsum('ij, bj -> bi', self.Bw, w) @ self.Tx_inv  # (Nx, Nw)*(Nd, Nw)->(Nd, Nx)

    def compute_ANN(self, x_hidden, z):
        # in:                   | out:
        #  - x_hidden (Nd, Nxh) |  - x_hidden_plus (Nd, Nz)
        #  - z (Nd, Nx)         |  - w (nd, Nw)
        if self.nettype == 'cREN':
            # for contracting REN networks
            x_hidden_plus, w = self.net(hidden_state=x_hidden, u=z)  # u_net = z_model
        else:
            # simple feedforward or residual net., etc.
            net_in = torch.cat((x_hidden.view(z.shape[0], -1), z.view(z.shape[0], -1)), dim=1)
            net_out = self.net(net_in)
            x_hidden_plus = net_out[:, :self.Nxh]
            w = net_out[:, -self.Nw:]
        return x_hidden_plus, w

    def forward(self, x, u):
        # in:                 | out:
        #  - x (Nd, Nx + Nxh) |  - x+ (Nd, Nx + Nxh)
        #  - u (Nd, Nu)       |  - y  (Nd, Ny)
        # split up the state from the encoder in the state of the known part
        # and the state of the unknown (to be learned) part
        if x.ndim == 1:  # shape of x is (Nx+Nxh)
            assert x.shape != self.Nx+self.Nxh, 'dimension of state is not nx+nx_hidden... nx = ' + str(self.Nx) + \
                                                ', nx_h = ' + str(self.Nxh) + ', while x.shape = ' + str(x.shape)
            x_known = x[:self.Nx]
            x_learn = x[-self.Nxh:]
        else:
            x_known = x[:, :self.Nx]
            x_learn = x[:, -self.Nxh:]
        if u.ndim == 1:
            u = torch.unsqueeze(u, dim=0)

        # compute the input for the network
        z = self.compute_z(x=x_known, w=None, u=u)  # z = Cz x + Dzw w + Dzu u  --> Dzw = 0

        # calculate w from NN and update hidden state
        x_learn_plus, w = self.compute_ANN(x_hidden=x_learn, z=z)

        # calculate the modeled state
        x_known_plus = self.sys.f(x_known, u) + self.compute_xnet_contribution(w)

        y_k = self.sys.h(x_known, u) + self.compute_ynet_contribution(w)
        x_plus = torch.cat((x_known_plus, x_learn_plus), dim=x.ndim-1)
        return y_k, x_plus


# -------------------------------------------------------------------------------------------
# ---------------------------------- ADDITIVE AUGMENTATION ----------------------------------
# -------------------------------------------------------------------------------------------

class SSE_AdditiveAugmentation(nn.Module):
    """
    Simple augmentation structure implementing an additive scheme as
        x[k+1] = f(x[k], u[k]) + F(x[k], u[k]),
    where f is the first-principle mechanical model, and F is a neural network, x is the model state, u is the input.

    Arguments:
        known_system - first-principle model
        net - neural network (previously noted with F)
        regLambda - regularization coefficient for physical parameters in f
        orthLambda - orthogonalization coefficient that penalizes the cost fun. for F, which ouput is in the subspace of f
        std_x, std_u - standard deviation of the state and input (approximated based on the training set) for standardization
    """
    def __init__(self, known_system, net, std_x, std_u, regLambda=0, orthLambda=0, l2_reg=0, **kwargs):
        super(SSE_AdditiveAugmentation, self).__init__()

        # First verify if we have the correct system type and augmentation parameters
        verifySystemType(known_system)
        verifyNetType(net, 'static')

        # Save parameters
        self.sys = known_system
        self.xnet = net
        self.Nu = known_system.Nu
        self.Nx = known_system.Nx
        self.Ny = known_system.Ny

        # save normalization matrices
        self.Tx = torch.diag(torch.tensor(1 / std_x, dtype=torch.float))
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))
        self.Tx_inv = torch.diag(torch.tensor(std_x, dtype=torch.float))

        # check for regularization and orthogonalization coefficients (only if known_sys allows it)
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.regLambda = regLambda
            self.orthLambda = orthLambda

        self.l2_reg = l2_reg

    def calculate_xnet(self, x, u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |

        x_norm = x @ self.Tx
        u_norm = u @ self.Tu

        xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1)), dim=1)

        return self.xnet(xnet_input) @ self.Tx_inv

    def forward(self, x, u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |  - y  (Nd, Ny)
        x_plus = self.sys.f(x, u) + self.calculate_xnet(x, u)
        y_k = self.sys.h(x, u)
        return y_k, x_plus

    def calculate_orthogonalisation(self, x, u, U1):
        # in:                   | out:
        #  - x (Nd, Nx)         |  - cost
        #  - u (Nd, Nu)         |
        #  - U1 (Nd*Nx, Ntheta) |

        x_net = self.calculate_xnet(x, u).view(U1.shape[0], -1)
        orthogonal_components = U1 @ U1.T @ x_net
        cost = self.orthLambda * torch.linalg.vector_norm(orthogonal_components)**2
        return cost


# -------------------------------------------------------------------------------------------
# ------------------------------ DYNAMIC ADDITIVE AUGMENTATION ------------------------------
# -------------------------------------------------------------------------------------------

class SSE_AdditiveDynAugmentation(SSE_AdditiveAugmentation):
    """"
    Dynamic augmentation structure implementing an additive scheme as
        x[k+1] = f(x[k], u[k]) + F(x[k], xh[k], u[k]),
    where f is the first-principle mechanical model, and F is a neural network.
    xh[k] noted the hidden states which are not incorporated in the first-principle model.
    Approximated by the encoder network.
    """
    def __init__(self, nx_h, known_system, net, std_x, std_u, regLambda=0, l2_reg=0, **kwargs):
        super(SSE_AdditiveDynAugmentation, self).__init__(known_system=known_system, net=net, regLambda=regLambda,
                                                          std_x=std_x, std_u=std_u, l2_reg=l2_reg)

        # save dynamic augmentation related parameter
        self.Nx_h = nx_h

        # redefine std_x to include hidden states
        std_x = torch.cat((torch.tensor(std_x, dtype=torch.float), torch.ones(nx_h)))
        self.Tx = torch.diag(1 / std_x)
        self.Tx_inv = torch.diag(std_x)

    def forward(self, x, u):
        # in:                       | out:
        #  - x (Nd, Nx_m+Nx_h)      |  - x+ (Nd, Nx_m+Nx_h)
        #  - u (Nd, Nu)             |  - y  (Nd, Ny)

        if x.ndim == 1:  # shape of x is (Nx+Nxh)
            assert x.shape != self.Nx+self.Nxh, 'dimension of state is not nx+nx_hidden... nx = ' + str(self.Nx) + \
                                                ', nx_h = ' + str(self.Nxh) + ', while x.shape = ' + str(x.shape)
            x_meas = x[:self.Nx]
        else:
            x_meas = x[:, :self.Nx]

        x_plus_bb = self.calculate_xnet(x, u)   # black-box part with measured and hidden states
        x_plus_fp = self.sys.f(x_meas, u)       # first-principles model part

        x_plus = torch.hstack((x_plus_fp, torch.zeros(x.size(dim=0), self.Nx_h))) + x_plus_bb
        y_k = self.sys.h(x_meas, u)
        return y_k, x_plus

    def calculate_orthogonalisation(self, x, u, U1):
        # Orthogonalization-based regularization is not supported for dynamic augmentation
        # self.orthLambda = 0, but for safety reasons this function is implemented as dummy
        # ToDo: check if this is necessary at all
        return 0


# -------------------------------------------------------------------------------------------
# ------------------------------- MULTIPLICATIVE AUGMENTATION -------------------------------
# -------------------------------------------------------------------------------------------

class SSE_MultiplicativeAugmentation(nn.Module):
    """
    ToDo: add documentation
    """
    def __init__(self, known_system, net, std_x, std_u, regLambda=0, l2_reg=0, **kwargs):
        super(SSE_MultiplicativeAugmentation, self).__init__()

        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)
        verifyNetType(net, 'static')  # this may cause instabilities, we have to check later ...

        # Save parameters
        self.sys = known_system
        self.xnet = net
        self.Nu = known_system.Nu
        self.Nx = known_system.Nx
        self.Ny = known_system.Ny

        # save normalization matrices
        self.Tx = torch.diag(torch.tensor(1 / std_x, dtype=torch.float))
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))
        self.Tx_inv = torch.diag(torch.tensor(std_x, dtype=torch.float))

        # check for parameter regularization coefficient
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.regLambda = regLambda

        self.l2_reg = l2_reg

    def calculate_xplus(self, x, u, x_fp):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |
        #  - x_fp (Nd, Nx)  |

        x_norm = x @ self.Tx
        x_fp_norm = x_fp @ self.Tx
        u_norm = u @ self.Tu
        xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1),
                                x_fp_norm.view(x_fp.shape[0], -1)), dim=1)
        return self.xnet(xnet_input) @ self.Tx_inv

    def forward(self, x, u):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |  - y  (Nd, Ny)

        x_plus_fp = self.sys.f(x, u)
        x_plus = self.calculate_xplus(x, u, x_plus_fp)
        y_k = self.sys.h(x, u)
        return y_k, x_plus


# -------------------------------------------------------------------------------------------
# ------------------------------- DYNAMIC MULTIPLICATIVE AUGMENTATION -------------------------------
# -------------------------------------------------------------------------------------------

class SSE_MultiplicativeDynAugmentation(nn.Module):
    """
    ToDo: add documentation
    """
    def __init__(self, nx_h, known_system, net, std_x, std_u, regLambda=0, l2_reg=0, **kwargs):
        super(SSE_MultiplicativeDynAugmentation, self).__init__()

        # First verify if we have the correct system type and augmentationparameters
        verifySystemType(known_system)
        verifyNetType(net, 'static')  # this may cause instabilities, we have to chech later ...

        # Save parameters
        self.sys = known_system
        self.xnet = net
        self.Nu = known_system.Nu
        self.Nx = known_system.Nx
        self.Ny = known_system.Ny
        self.Nxh = nx_h

        # save normalization matrices
        std_x = torch.tensor(std_x, dtype=torch.float)
        self.Tx = torch.diag(1 / std_x)
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))
        std_xh = torch.cat((std_x, torch.ones(nx_h)))  # include the modeled + hidden states also
        self.Txh = torch.diag(1 / std_xh)
        self.Txh_inv = torch.diag(std_xh)

        # check for paramater regularization coefficient
        if hasattr(known_system, 'parm_corr_enab'):
            self.Pcorr_enab = known_system.parm_corr_enab
        else:
            self.Pcorr_enab = False
        if self.Pcorr_enab:
            self.regLambda = regLambda

        self.l2_reg = l2_reg

    def calculate_xplus(self, x, u, x_fp):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |
        #  - x_fp (Nd, Nx)  |

        x_norm = x @ self.Txh  # including the hidden state
        u_norm = u @ self.Tu
        x_fp_norm = x_fp @ self.Tx  # only the modeled state

        xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1),
                                x_fp_norm.view(x_fp.shape[0], -1)), dim=1)
        return self.xnet(xnet_input) @ self.Txh_inv

    def forward(self, x, u):
        # in:                       | out:
        #  - x (Nd, Nx_m+Nx_h)      |  - x+ (Nd, Nx_m+Nx_h)
        #  - u (Nd, Nu)             |  - y  (Nd, Ny)

        if x.ndim == 1:  # shape of x is (Nx+Nxh)
            assert x.shape != self.Nx+self.Nxh, 'dimension of state is not nx+nx_hidden... nx = ' + str(self.Nx) + \
                                                ', nx_h = ' + str(self.Nxh) + ', while x.shape = ' + str(x.shape)
            x_meas = x[:self.Nx]
        else:
            x_meas = x[:, :self.Nx]

        # first-principle model part
        x_plus_fp = self.sys.f(x_meas, u)

        x_plus = self.calculate_xplus(x, u, x_plus_fp)
        y_k = self.sys.h(x_meas, u)
        return y_k, x_plus
