import torch
from torch import nn
import numpy as np
from model_augmentation.augmentation_structures import (SSE_AdditiveAugmentation, SSE_AdditiveDynAugmentation,
                                                        SSE_MultiplicativeAugmentation,
                                                        SSE_MultiplicativeDynAugmentation, SSE_LFRAugmentation,
                                                        SSE_LFRDynAugmentation)


class SSE_AdditiveAugmentation_innovation(SSE_AdditiveAugmentation):
    """
        Simple augmentation structure implementing an additive scheme as
            x[k+1] = f(x[k], u[k]) + F(x[k], u[k], e[k]),
        where f is the first-principle mechanical model, and F is a neural network, x is the model state, u is the input,
        e is the approximated noise if innovation noise structure is use, otherwise e[k]==0 is assumed.

        Arguments:
            known_system - first-principle model
            net - neural network (previously noted with F)
            regLambda - regularization coefficient for physical parameters in f
            orthLambda - orthogonalization coefficient that penalizes the cost fun. for F, which output is in
                        the subspace of f
            std_x, std_u, std_y - standard deviation of the state, input and output (based on the training set) for
                        standardization
            innov_nonlin - to use nonlinear noise structure (default False)
        """
    def __init__(self, known_system, net, std_x, std_u, std_y, regLambda=0, orthLambda=0, innov_nonlin=False, **kwargs):
        super(SSE_AdditiveAugmentation_innovation, self).__init__(known_system=known_system, net=net, std_x=std_x,
                                                                  std_u=std_u, regLambda=regLambda,
                                                                  orthLambda=orthLambda)

        # save extra normalization parameter
        self.Ty = torch.diag(torch.tensor(1 / std_y, dtype=torch.float))

        # save innovation noise structure parameters
        self.innov = True
        if innov_nonlin:
            self.lin_innov = False
            self.nonlin_innov = True
        else:
            self.lin_innov = True
            self.nonlin_innov = False
            self.noise_gain_K = nn.Linear(np.prod(self.Ny, dtype=int), self.Nx, bias=False)

    def calculate_xnet(self, x, u, eps=None):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |

        x_norm = x @ self.Tx
        u_norm = u @ self.Tu

        if self.nonlin_innov and eps is not None:
            # nonlinear innovation noise and approximated noise is available
            eps_norm = eps @ self.Ty
            xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1),
                                    eps_norm.view(u.shape[0], -1)), dim=1)
        elif self.nonlin_innov and eps is None:
            # nonlinear innovation structure, but noise is assumed to be zero
            eps = torch.zeros((x.shape[0], np.prod(self.ny, dtype=int)), dtype=torch.float32)
            xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1),
                                    eps.view(x.shape[0], -1)), dim=1)
        else:
            # no noise structure
            xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1)), dim=1)

        return self.xnet(xnet_input) @ self.Tx_inv

    def forward(self, x, u, y_train=None):
        # in:                   | out:
        #  - x (Nd, Nx)         |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)         |  - y  (Nd, Ny)
        #  - y_train (Nd, Ny)   |

        y_k = self.sys.h(x, u)
        if y_train is not None:
            eps = y_train - y_k
        else:
            eps = None

        if self.lin_innov and eps is not None:
            eps_flatten = (eps @ self.Ty).view(u.shape[0], -1)
            x_plus = self.sys.f(x, u) + self.calculate_xnet(x, u) + self.noise_gain_K(eps_flatten)
        else:
            x_plus = self.sys.f(x, u) + self.calculate_xnet(x, u, eps)

        return y_k, x_plus


class SSE_AdditiveDynAugmentation_innovation(SSE_AdditiveDynAugmentation):
    """
    Dynamic augmentation structure implementing an additive scheme as
        x[k+1] = f(x[k], u[k]) + F(x[k], xh[k], u[k], e[k]),
    where f is the first-principle mechanical model, and F is a neural network.

    xh[k] notes the hidden states which are not incorporated in the first-principle model. Approximated by the encoder
    network.

    e[k] is the approximated noise during training
    """
    def __init__(self, nx_h, known_system, net, std_x, std_u, std_y, regLambda=0, innov_nonlin=False, **kwargs):
        super(SSE_AdditiveDynAugmentation_innovation, self).__init__(nx_h=nx_h, known_system=known_system, net=net,
                                                                     std_x=std_x, std_u=std_u, regLambda=regLambda)

        # save extra normalization parameter
        self.Ty = torch.diag(torch.tensor(1 / std_y, dtype=torch.float))

        # save innovation noise structure parameters
        self.innov = True
        if innov_nonlin:
            self.lin_innov = False
            self.nonlin_innov = True
        else:
            self.lin_innov = True
            self.nonlin_innov = False
            self.noise_gain_K = nn.Linear(np.prod(self.Ny, dtype=int), self.Nx, bias=False)

    def calculate_fp_model(self, x, u, eps):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |
        #  - eps (Nd, Ny)   |

        x_plus = self.sys.f(x, u)
        if self.lin_innov and eps is not None:
            eps_flatten = (eps @ self.Ty).view(u.shape[0], -1)
            x_plus += self.noise_gain_K(eps_flatten)
        return x_plus

    def calculate_net(self, x, u, eps):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |
        #  - eps (Nd, Ny)   |

        x_norm = x @ self.Tx
        u_norm = u @ self.Tu

        if self.nonlin_innov and eps is not None:
            # nonlinear innovation structure and noise values are approximated
            eps_norm = eps @ self.Ty
            xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1),
                                    eps_norm.view(u.shape[0], -1)), dim=1)
        elif self.nonlin_innov and eps is None:
            # nonlinear innovation structure, but noise is assumed to be zero
            eps = torch.zeros((x.shape[0], np.prod(self.ny, dtype=int)), dtype=torch.float32)
            xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1),
                                    eps.view(x.shape[0], -1)), dim=1)
        else:
            # no noise structure
            xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1)), dim=1)

        return self.xnet(xnet_input) @ self.Tx_inv

    def forward(self, x, u, y_train=None):
        # in:                       | out:
        #  - x (Nd, Nx_m+Nx_h)      |  - x+ (Nd, Nx_m+Nx_h)
        #  - u (Nd, Nu)             |  - y  (Nd, Ny)
        #  - y_train (Nd, Ny)       |

        if x.ndim == 1:  # shape of x is (Nx+Nxh)
            assert x.shape != self.Nx+self.Nxh, 'dimension of state is not nx+nx_hidden... nx = ' + str(self.Nx) + \
                                                ', nx_h = ' + str(self.Nxh) + ', while x.shape = ' + str(x.shape)
            x_meas = x[:self.Nx]
        else:
            x_meas = x[:, :self.Nx]

        y_k = self.sys.h(x_meas, u)
        if y_train is not None:
            eps = y_train - y_k
        else:
            eps = None

        x_plus_bb = self.calculate_net(x, u, eps)   # black-box part with measured and hidden states
        x_plus_fp = self.calculate_fp_model(x_meas, u, eps)  # first-principles part + lin. noise model

        x_plus = torch.hstack((x_plus_fp, torch.zeros(x.size(dim=0), self.Nx_h))) + x_plus_bb

        return y_k, x_plus


class SSE_MultiplicativeAugmentation_innov(SSE_MultiplicativeAugmentation):
    """
    ToDo: add documentation
    """
    def __init__(self, known_system, net, std_x, std_u, std_y, innov_nonlin=False, regLambda=0, **kwargs):
        super(SSE_MultiplicativeAugmentation_innov, self).__init__(known_system=known_system, net=net, std_x=std_x,
                                                                   std_u=std_u, regLambda=regLambda)

        # save extra normalization parameter
        self.Ty = torch.diag(torch.tensor(1 / std_y, dtype=torch.float))

        # save innovation noise structure parameters
        self.innov = True
        if innov_nonlin:
            self.lin_innov = False
            self.nonlin_innov = True
        else:
            self.lin_innov = True
            self.nonlin_innov = False
            self.noise_gain_K = nn.Linear(np.prod(self.Ny, dtype=int), self.Nx, bias=False)

    def calculate_xplus_nonlin_innov(self, x, u, x_fp, eps):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |
        #  - x_fp (Nd, Nx)  |
        #  - eps (Nd, Ny)   |

        x_norm = x @ self.Tx
        x_fp_norm = x_fp @ self.Tx
        u_norm = u @ self.Tu
        if eps is None:
            eps = torch.zeros((x.shape[0], np.prod(self.ny, dtype=int)), dtype=torch.float32)
        eps_norm = eps @ self.Ty

        xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1),
                                x_fp_norm.view(x_fp.shape[0], -1), eps_norm.view(x.shape[0], -1)), dim=1)

        return self.xnet(xnet_input) @ self.Tx_inv

    def forward(self, x, u, y_train=None):
        # in:                   | out:
        #  - x (Nd, Nx)         |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)         |  - y  (Nd, Ny)
        #  - y_train (Nd, Ny)   |

        y_k = self.sys.h(x, u)

        if y_train is not None:
            eps = y_train - y_k
        else:
            eps = None

        x_plus_fp = self.sys.f(x, u)
        if self.lin_innov and eps is not None:
            # linear innovation structure and noise is approximated
            eps_flatten = (eps @ self.Ty).view(x.shape[0], -1)
            x_plus = self.calculate_xplus(x, u, x_plus_fp) + self.noise_gain_K(eps_flatten)
        elif self.nonlin_innov:
            # nonlinear innovation structure
            x_plus = self.calculate_xplus_nonlin_innov(x, u, x_plus_fp, eps)
        else:
            # no noise structure is used
            x_plus = self.calculate_xplus(x, u, x_plus_fp)

        return y_k, x_plus


class SSE_MultiplicativeDynAugmentation_innov(SSE_MultiplicativeDynAugmentation):
    """
    ToDo: add documentation
    """
    def __init__(self, nx_h, known_system, net, std_x, std_u, std_y, innov_nonlin=False, regLambda=0, **kwargs):
        super(SSE_MultiplicativeDynAugmentation_innov, self).__init__(nx_h=nx_h, known_system=known_system, net=net,
                                                                      std_x=std_x, std_u=std_u, regLambda=regLambda)

        # save extra normalization parameter
        self.Ty = torch.diag(torch.tensor(1 / std_y, dtype=torch.float))

        # save innovation noise structure parameters
        self.innov = True
        if innov_nonlin:
            self.lin_innov = False
            self.nonlin_innov = True
        else:
            self.lin_innov = True
            self.nonlin_innov = False
            self.noise_gain_K = nn.Linear(np.prod(self.Ny, dtype=int), self.Nx, bias=False)

    def calculate_xplus_nonlin_innov(self, x, u, x_fp, eps):
        # in:               | out:
        #  - x (Nd, Nx)     |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)     |
        #  - x_fp (Nd, Nx)  |
        #  - eps (Nd, Ny)   |

        x_norm = x @ self.Txh  # including the hidden state
        u_norm = u @ self.Tu
        x_fp_norm = x_fp @ self.Tx  # only the modeled state
        if eps is None:
            eps = torch.zeros((x.shape[0], np.prod(self.ny, dtype=int)), dtype=torch.float32)
        eps_norm = eps @ self.Ty

        xnet_input = torch.cat((x_norm.view(x.shape[0], -1), u_norm.view(u.shape[0], -1),
                                x_fp_norm.view(x_fp.shape[0], -1), eps_norm.view(u.shape[0], -1)), dim=1)

        return self.xnet(xnet_input) @ self.Txh_inv

    def forward(self, x, u, y_train=None):
        # in:                       | out:
        #  - x (Nd, Nx_m+Nx_h)      |  - x+ (Nd, Nx_m+Nx_h)
        #  - u (Nd, Nu)             |  - y  (Nd, Ny)
        #  - y_train (Nd, Ny)       |

        if x.ndim == 1:  # shape of x is (Nx+Nxh)
            assert x.shape != self.Nx+self.Nxh, 'dimension of state is not nx+nx_hidden... nx = ' + str(self.Nx) + \
                                                ', nx_h = ' + str(self.Nxh) + ', while x.shape = ' + str(x.shape)
            x_meas = x[:self.Nx]
        else:
            x_meas = x[:, :self.Nx]

        y_k = self.sys.h(x_meas, u)

        if y_train is not None:
            eps = y_train - y_k
        else:
            eps = None

        # first-principle model part
        x_plus_fp = self.sys.f(x_meas, u)

        if self.lin_innov and eps is None:
            # no noise structure
            x_plus = self.calculate_xplus(x, u, x_plus_fp)
        elif self.lin_innov and eps is not None:
            # linear noise structure
            eps_flatten = (eps @ self.Ty).view(x.shape[0], -1)
            x_plus = self.calculate_xplus(x, u, x_plus_fp) + self.noise_gain_K(eps_flatten)
        else:
            # nonlinear innovation noise structure
            x_plus = self.calculate_xplus_nonlin_innov(x, u, x_plus_fp, eps)

        return y_k, x_plus


class SSE_LFRAugmentation_innovation(SSE_LFRAugmentation):
    """
    ToDO: documentation
    """
    def __init__(self, known_system, net, std_x, std_y, std_u, innov_nonlin=False, regLambda=0, **kwargs):
        super(SSE_LFRAugmentation_innovation, self).__init__(known_system=known_system, net=net, std_x=std_x,
                                                             std_y=std_y, std_u=std_u, regLambda=regLambda)

        raise NotImplementedError("LFR-based innovation noise structure is not implemented yet...")
        # ToDo: derive the method for LFR-based augmentation

        # save innovation noise structure parameters
        self.innov = True
        if innov_nonlin:
            self.lin_innov = False
            self.nonlin_innov = True
        else:
            self.lin_innov = True
            self.nonlin_innov = False
            self.noise_gain_K = nn.Linear(np.prod(self.Ny, dtype=int), self.Nx, bias=False)

    def calculate_net(self, z, eps=None):
        # in:               | out:
        #  - z (Nd, Nz)     |  - x+ (Nd, Nx)
        #  - eps (Nd, Ny)   |
        if self.lin_innov:
            # linear innovation noise model
            w = self.net(z)
        else:
            # nonlinear innovation noise structure
            if eps is None:
                # zero noise assumed
                eps = torch.zeros((z.shape[0], np.prod(self.ny, dtype=int)), dtype=torch.float32)
            eps_norm = eps @ self.Ty
            net_in = torch.cat((z.view(z.shape[0], -1), eps_norm.view(z.shape[0], -1)), dim=1)
            w = self.net(net_in)
        return w

    def forward(self, x, u, y_train=None):
        # in:                   | out:
        #  - x (Nd, Nx)         |  - x+ (Nd, Nx)
        #  - u (Nd, Nu)         |  - y  (Nd, Ny)
        #  - y_train (Nd, Ny)   |
        if u.ndim == 1:
            u = torch.unsqueeze(u, dim=0)

        # compute network contribution
        z = self.compute_z(x, u)
        w_hat = self.calculate_net(z)

        y_k = self.sys.h(x, u) + self.compute_ynet_contribution(w_hat)
        if y_train is not None:
            eps = y_train - y_k
        else:
            eps = None

        # add network contributions to state transition and output calculation
        if eps is None:
            x_plus = self.sys.f(x, u) + self.compute_xnet_contribution(w_hat)
        elif self.lin_innov:
            # linear innovation noise structure
            eps_flatten = (eps @ self.Ty).view(x.shape[0], -1)
            x_plus = self.sys.f(x, u) + self.compute_xnet_contribution(w_hat) + self.noise_gain_K(eps_flatten)
        else:
            w_new = self.calculate_net(z, eps)
            x_plus = self.sys.f(x, u) + self.compute_xnet_contribution(w_new)

        return y_k, x_plus


class SSE_LFRDynAugmentation_innovation(SSE_LFRDynAugmentation):
    """
    ToDO: documentation
    """
    def __init__(self, nx_h, known_system, net, std_x, std_y, std_u, innov_nonlin=False, regLambda=0, **kwargs):
        super(SSE_LFRDynAugmentation_innovation, self).__init__(nx_h=nx_h, known_system=known_system, net=net,
                                                                std_x=std_x, std_y=std_y, std_u=std_u,
                                                                regLambda=regLambda)

        raise NotImplementedError("LFR-based innovation noise structure is not implemented yet...")
        # ToDo: derive the method for LFR-based augmentation
        # save innovation noise structure parameters
        self.innov = True
        if innov_nonlin:
            self.lin_innov = False
            self.nonlin_innov = True
        else:
            self.lin_innov = True
            self.nonlin_innov = False
            self.noise_gain_K = nn.Linear(np.prod(self.Ny, dtype=int), self.Nx, bias=False)

    def calculate_net(self, x_hidden, z, eps=None):
        # in:                   | out:
        #  - x_hidden (Nd, Nxh) |  - x_hidden_plus (Nd, Nz)
        #  - z (Nd, Nx)         |  - w (nd, Nw)
        #  - eps (Nd, Ny)       |
        if self.nettype == 'cREN':
            # for contracting REN networks
            # x_hidden_plus, w = self.net(hidden_state=x_hidden, u=z)  # u_net = z_model
            raise NotImplementedError("cREN is not implemented for innovation noise structure")
        else:
            # simple feedforward or residual net., etc.
            if self.lin_innov:
                # linear innovation noise structure
                net_in = torch.cat((x_hidden.view(z.shape[0], -1), z.view(z.shape[0], -1)), dim=1)
            else:
                # nonlinear innovation form
                if eps is None:
                    eps = torch.zeros((z.shape[0], np.prod(self.ny, dtype=int)), dtype=torch.float32)
                eps_norm = eps @ self.Ty
                net_in = torch.cat((x_hidden.view(z.shape[0], -1), z.view(z.shape[0], -1),
                                    eps_norm.view(z.shape[0], -1)), dim=1)
            net_out = self.net(net_in)
            x_hidden_plus = net_out[:, :self.Nxh]
            w = net_out[:, -self.Nw:]
        return x_hidden_plus, w

    def forward(self, x, u, y_train=None):
        # in:                   | out:
        #  - x (Nd, Nx + Nxh)   |  - x+ (Nd, Nx + Nxh)
        #  - u (Nd, Nu)         |  - y  (Nd, Ny)
        #  - y_train (Nd, Ny)   |

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
        x_learn_plus, w_hat = self.calculate_net(x_hidden=x_learn, z=z)

        # calculate output
        y_k = self.sys.h(x_known, u) + self.compute_ynet_contribution(w_hat)

        if y_train is not None:
            eps = y_train - y_k
        else:
            eps = None

        # calculate the modeled state
        if eps is None:
            x_known_plus = self.sys.f(x_known, u) + self.compute_xnet_contribution(w_hat)
        elif self.lin_innov:
            eps_flatten = (eps @ self.Ty).view(x.shape[0], -1)
            x_known_plus = self.sys.f(x_known, u) + self.compute_xnet_contribution(w_hat) + self.noise_gain_K(eps_flatten)
        else:
            x_learn_plus, w_new = self.calculate_net(x_hidden=x_learn, z=z, eps=eps)
            x_known_plus = self.sys.f(x_known, u) + self.compute_xnet_contribution(w_new)

        x_plus = torch.cat((x_known_plus, x_learn_plus), dim=x.ndim-1)
        return y_k, x_plus
