import numpy as np
import torch
from torch import nn
import warnings


class default_encoder_net(nn.Module):
    """
        a simple FC net with a residual (default approach)
        used for augmentation purposes: implemented normalization and back-scaling to provide interpretable states for the physics-based model
    """
    def __init__(self, nb, nu, na, ny, nx, std_x, std_y, std_u, nx_h=0, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(default_encoder_net, self).__init__()
        from model_augmentation.torch_nets import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu, int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny, int) else ny)
        self.net = simple_res_net(n_in=nb * np.prod(self.nu, dtype=int) + na * np.prod(self.ny, dtype=int), n_out=nx,
                                  n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers,
                                  activation=activation)
        self.Ty = torch.diag(torch.tensor(1 / std_y, dtype=torch.float))
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))
        if nx_h > 0:  # there are hidden states --> they should stay normalized
            std_x = torch.cat((torch.tensor(std_x, dtype=torch.float), torch.ones(nx_h)))
        else:
            std_x = torch.tensor(std_x, dtype=torch.float)
        self.Tx_inv = torch.diag(std_x)

    def forward(self, upast, ypast):
        upast_norm = upast @ self.Tu
        ypast_norm = ypast @ self.Ty
        net_in = torch.cat([upast_norm.view(upast.shape[0], -1), ypast_norm.view(ypast.shape[0], -1)], dim=1)
        return self.net(net_in) @ self.Tx_inv


class state_measure_encoder:
    """
    for known y[k]=x[k] cases
    """
    def __init__(self, nb, nu, na, ny, nx, **kwargs):
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu, int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny, int) else ny)

    def __call__(self, upast, ypast):
        # in:                               | out:
        #  - u_past (Nd, nb+nb_right, Nu)   |  - x0 (Nd, Nx=Ny)
        #  - y_past (Nd, na+na_right, Ny)   |
        return ypast[:, -1, :]


class dynamic_state_meas_encoder(nn.Module):
    """
    for known y[k]=x[k] cases with hidden states
    an FC residual network estimates the hidden states, while the modeled states are known
    """
    def __init__(self, nb, nu, na, ny, nx, nx_h, std_x, std_y, std_u, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(dynamic_state_meas_encoder, self).__init__()
        from model_augmentation.torch_nets import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu, int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny, int) else ny)
        self.net = simple_res_net(n_in=nb * np.prod(self.nu, dtype=int) + na * np.prod(self.ny, dtype=int), n_out=nx_h,
                                  n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers,
                                  activation=activation)
        self.Ty = torch.diag(torch.tensor(1 / std_y, dtype=torch.float))
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))

    def forward(self, upast, ypast):
        # in:                               | out:
        #  - u_past (Nd, nb+nb_right, Nu)   |  - x0 (Nd, Nx_meas+Nx_hidden)
        #  - y_past (Nd, na+na_right, Ny)   |
        x_meas = ypast[:, -1, :]
        upast_norm = upast @ self.Tu
        ypast_norm = ypast @ self.Ty
        net_in = torch.cat([upast_norm.view(upast.shape[0], -1), ypast_norm.view(ypast.shape[0], -1)], dim=1)
        x_hidden = self.net(net_in)
        return torch.hstack((x_meas, x_hidden))


class lti_initialized_encoder(nn.Module):
    """
    Encoder network with LTI-SS model-based first-principle calculation. The initial estimation matches the LTI model's reconstructibility map.
    Without noise model: (A, B, C, D) system matrices
        x[k] = A^n * Phi^(-1) * ( y[k-n:k] - Gamma * u[k-n:k] ) + gamma * u[k-n:k]
    With noise model: ( assumed system dynamics are x[k+1] = A*x[k] + B*u[k] + K*e[k];  y[k] = C*x[k] + D*u[k] + e[k];  e[k] - white noise )
        Ah = A - K * C
        Bh = B - K * D
        x[k] = Ah^n * Phi^(-1) * ( y[k-n:k] - Lambda * y[k-n:k] - Gamma * u[k-n:k] ) + gamma * u[k-n:k] + lambda * y[k-n:k]

    Input parameters:
        na, nb - output lag, input lag
        nu, nx, ny - input dimension, state dimension, output dimension
        known_sys - LTI model (or linearized nonlinear model)
        n_nodes_per_layer, n_hidden_layers, activation - neural network hyperparameters
        noise_handling - bool: use noise structure or not
        std_x, std_y, std_u: standard deviation of the state, output, and input based on the training data
        nx_h: number of hidden states
    """

    def __init__(self, nb, nu, na, ny, nx, known_sys, std_x, std_y, std_u, nx_h=0, n_nodes_per_layer=64,
                 n_hidden_layers=2, activation=nn.Tanh, noise_handling=True):
        super(lti_initialized_encoder, self).__init__()
        from model_augmentation.torch_nets import simple_res_net
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu, int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny, int) else ny)
        self.nx = known_sys.Nx
        self.Ty = torch.diag(torch.tensor(1 / std_y, dtype=torch.float))
        self.Tu = torch.diag(torch.tensor(1 / std_u, dtype=torch.float))
        if self.nx != nx:
            # for dynamic augmentation
            self.dynamic = True
            std_x = torch.cat((torch.tensor(std_x, dtype=torch.float), torch.ones(nx_h)))
        else:
            # static augmentation
            self.dynamic = False
            std_x = torch.tensor(std_x, dtype=torch.float)
        self.nx_hidden = nx_h
        self.Tx_inv = torch.diag(std_x)

        self.net = simple_res_net(n_in=nb * np.prod(self.nu, dtype=int) + na * np.prod(self.ny, dtype=int), n_out=nx,
                                  n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers,
                                  activation=activation)
        self.initialize_encoder_net(self.net)  # set last layer weights + biases to zero

        # declare matrices as None, then set the values later in calculate_matrices function
        self.n = self.nx - 1  # for LTI models n=nx-1
        self.A_mx = None
        self.Phi_mx = None
        self.Gamma_mx = None
        self.gamma_mx = None
        self.Lambda_mx = None
        self.lambda_mx = None

        self.noise = noise_handling
        if self.noise:
            if hasattr(known_sys, "K"):
                self.K = known_sys.K.detach()
            else:
                warnings.warn("LTI model has no 'K' attribute. Noise model is not used!")
                self.noise = False
        # calculate the matrix values (if not needed the noise handling matrices kept as None)
        self.calculate_matrices(known_sys)

    def initialize_encoder_net(self, network):
        network.net_lin.weight.data.fill_(0.0)
        if network.net_non_lin is not None:  # has nonlinear part
            network.net_non_lin.net[-1].weight.data.fill_(0.0)
            network.net_non_lin.net[-1].bias.data.fill_(0.0)
        else:  # if only linear part is present, then it has bias value
            network.net_lin.bias.data.fill_(0.0)

    def calculate_matrices(self, lti_sys):
        if self.noise:
            K = self.K.clone()
            A = lti_sys.A.detach() - K @ lti_sys.C.detach()
            B = lti_sys.B.detach() - K @ lti_sys.D.detach()
        else:
            A = lti_sys.A.detach()
            B = lti_sys.B.detach()
        C = lti_sys.C.detach()
        D = lti_sys.D.detach()

        n = self.n
        Phi = C @ torch.matrix_power(A, n)
        for i in range(n - 1, -1, -1):
            Phi = torch.vstack((Phi, C @ torch.matrix_power(A, i)))

        Gamma = D
        gamma = torch.zeros_like(B)
        if self.noise:
            Lambda = torch.zeros((lti_sys.Ny, lti_sys.Ny))
            lambda_val = torch.zeros_like(K)
        for i in range(n):
            Gamma = torch.hstack((Gamma, C @ torch.matrix_power(A, i) @ B))
            gamma = torch.hstack((gamma, torch.matrix_power(A, i) @ B))
            if self.noise:
                Lambda = torch.hstack((Lambda, C @ torch.matrix_power(A, i) @ K))
                lambda_val = torch.hstack((lambda_val, torch.matrix_power(A, i) @ K))
        for i in range(n):
            #Gamma_row = torch.zeros((lti_sys.Ny, lti_sys.Nu * (i + 1)))
            Gamma_row = torch.zeros_like(D)
            Gamma_row = torch.hstack((Gamma_row, Gamma[-lti_sys.Ny:, :-lti_sys.Nu]))
            Gamma = torch.vstack((Gamma, Gamma_row))
            if self.noise:
                #Lambda_row = torch.zeros((lti_sys.Ny, lti_sys.Ny * (i + 1)))
                Lambda_row = torch.zeros_like(C @ K)
                Lambda_row = torch.hstack((Lambda_row, Lambda[-lti_sys.Ny:, :-lti_sys.Ny]))
                Lambda = torch.vstack((Lambda, Lambda_row))

        self.A_mx = A
        self.Phi_mx = Phi
        self.Gamma_mx = Gamma
        self.gamma_mx = gamma
        if self.noise:
            self.Lambda_mx = Lambda
            self.lambda_mx = lambda_val

            # PATCH
            self.Wu = -torch.matrix_power(A, n) @ torch.linalg.pinv(Phi) @ Gamma + gamma
            self.Wy = (torch.matrix_power(A, n) @ torch.linalg.pinv(Phi) -
                       torch.matrix_power(A, n) @ torch.linalg.pinv(Phi) @ Lambda + lambda_val)
        return

    def forward(self, upast, ypast):
        # in:                               | out:
        #  - u_past (Nd, nb+nb_right, Nu)   |  - x0 (Nd, Nx)
        #  - y_past (Nd, na+na_right, Ny)   |

        upast_n = upast[:, -self.nx:, :].view(upast.shape[0], -1)
        ypast_n = ypast[:, -self.nx:, :].view(ypast.shape[0], -1)

        upast_n = torch.fliplr(upast_n).T  # in order to upast_n.shape = (Nx*Nu, Nd),  n = Nx-1
        ypast_n = torch.fliplr(ypast_n).T  # in order to ypast_n.shape = (Nx*Ny, Nd),  n = Nx-1

        if self.noise:  # noise handling case
            # Alpha_mx = torch.linalg.solve(self.Phi_mx, ypast_n - self.Gamma_mx @ upast_n - self.Lambda_mx @ ypast_n)
            # x0_lti = torch.matrix_power(self.A_mx,
            #                             self.n) @ Alpha_mx + self.gamma_mx @ upast_n + self.lambda_mx @ ypast_n
            # PATCH
            x0_lti = upast_n.T @ self.Wu.T + ypast_n.T @ self.Wy.T
        else:  # without noise structure
            Alpha_mx = torch.linalg.solve(self.Phi_mx, ypast_n - self.Gamma_mx @ upast_n)
            x0_lti = torch.matrix_power(self.A_mx, self.n) @ Alpha_mx + self.gamma_mx @ upast_n
            # x0_lti has a shape of (Nx, Nd)  --> needs transposing
            x0_lti = x0_lti.T


        upast_norm = upast @ self.Tu
        ypast_norm = ypast @ self.Ty
        net_in = torch.cat([upast_norm.view(upast.shape[0], -1), ypast_norm.view(ypast.shape[0], -1)], dim=1)
        x0_ann = self.net(net_in) @ self.Tx_inv  # with backscaling

        if self.dynamic:
            # for dynamic augmentation the hidden states can only be approximated by the encoder
            x0_zeros = torch.zeros((x0_lti.shape[0], self.nx_hidden))
            x0_lti = torch.cat((x0_lti, x0_zeros), dim=1)

        return x0_lti + x0_ann

############################################ NOTE ############################################
# "Parameter tuning" has been implemented into simple lti_initialized_encoder

# class lti_initialized_parmtuning_encoder(lti_initialized_encoder):
#     """
#     Same in functionality as lti_initialized_encoder, but the resulting matrices from the LTI-based reconstruction are
#     also tuned as network parameters.
#
#     Input parameters:
#         na, nb - output lag, input lag
#         nu, nx, ny - input dimension, state dimension, output dimension
#         known_sys - LTI model
#         n_nodes_per_layer, n_hidden_layers, activation - neural network hyperparameters
#         noise_handling - bool: use noise structure or not
#     """
#
#     def __init__(self, nb, nu, na, ny, nx, std_x, std_y, std_u, known_sys, nx_h=0, n_nodes_per_layer=64,
#                  n_hidden_layers=2, activation=nn.Tanh, noise_handling=True):
#         super(lti_initialized_parmtuning_encoder, self).__init__(nb=nb, nu=nu, na=na, ny=ny, nx=nx, known_sys=known_sys,
#                                                                  n_nodes_per_layer=n_nodes_per_layer, nx_h=nx_h,
#                                                                  n_hidden_layers=n_hidden_layers, std_x=std_x,
#                                                                  std_y=std_y, std_u=std_u, activation=activation,
#                                                                  noise_handling=noise_handling)
#
#         Phi_inv = torch.linalg.pinv(self.Phi_mx)  # inaccurate, but later will be tuned as a parameter
#         # self.parm_Phi_inv = nn.Parameter(data=Phi_inv)
#         # self.parm_Gamma = nn.Parameter(data=self.Gamma_mx)
#         # self.parm_gamma = nn.Parameter(data=self.gamma_mx)
#         # if self.noise:
#         #     self.parm_Lambda = nn.Parameter(data=self.Lambda_mx)
#         #     self.parm_lambda = nn.Parameter(data=self.lambda_mx)
#
#         Wu = -torch.matrix_power(self.A_mx, self.n) @ Phi_inv @ self.Gamma_mx + self.gamma_mx
#         if self.noise:
#             Wy = torch.matrix_power(self.A_mx, self.n) @ Phi_inv @ (torch.eye(self.Lambda_mx.shape[0]) - self.Lambda_mx) + self.lambda_mx
#         else:
#             Wy = torch.matrix_power(self.A_mx, self.n) @ Phi_inv
#         self.Wu = nn.Parameter(data=Wu)
#         self.Wy = nn.Parameter(data=Wy)
#
#     def forward(self, upast, ypast):
#         # in:                               | out:
#         #  - u_past (Nd, nb+nb_right, Nu)   |  - x0 (Nd, Nx)
#         #  - y_past (Nd, na+na_right, Ny)   |
#
#         upast_n = upast[:, -self.nx:, :].view(upast.shape[0], -1)
#         ypast_n = ypast[:, -self.nx:, :].view(ypast.shape[0], -1)
#
#         # upast_n = torch.fliplr(upast_n).T  # in order to upast_n.shape = (Nx*Nu, Nd),  n = Nx-1
#         # ypast_n = torch.fliplr(ypast_n).T  # in order to ypast_n.shape = (Nx*Ny, Nd),  n = Nx-1
#
#         # if self.noise:  # noise handling case
#         #     x0_lti = (torch.matrix_power(self.A_mx, self.n) @ self.parm_Phi_inv @ (
#         #                 ypast_n - self.parm_Lambda @ ypast_n -
#         #                 self.parm_Gamma @ upast_n) +
#         #               self.parm_gamma @ upast_n + self.parm_lambda @ ypast_n)
#         # else:  # without noise structure
#         #     x0_lti = torch.matrix_power(self.A_mx, self.n) @ self.parm_Phi_inv @ (
#         #                 ypast_n - self.parm_Gamma @ upast_n) + self.parm_gamma @ upast_n
#         #
#         # # x0_lti has a shape of (Nx, Nd)  --> needs transposing
#         # x0_lti = x0_lti.T
#
#         x0_lti = upast_n @ self.Wu.T + ypast_n @ self.Wy.T
#
#         upast_norm = upast @ self.Tu
#         ypast_norm = ypast @ self.Ty
#         net_in = torch.cat([upast_norm.view(upast.shape[0], -1), ypast_norm.view(ypast.shape[0], -1)], dim=1)
#         x0_ann = self.net(net_in) @ self.Tx_inv
#
#         if self.dynamic:
#             # for dynamic augmentation the hidden states can only be approximated by the encoder
#             x0_zeros = torch.zeros((x0_lti.shape[0], self.nx_hidden))
#             x0_lti = torch.cat((x0_lti, x0_zeros), dim=1)
#
#         return x0_lti + x0_ann
