import warnings
import deepSI
import numpy as np
import torch
import model_augmentation


# Allowed systems:
def verifySystemType(sys):
    if issubclass(type(sys), model_augmentation.system_models.lti_system): return
    elif issubclass(type(sys), model_augmentation.system_models.general_nonlinear_system): return
    else: raise ValueError("Systems must be of the types defined in 'model_augmentation.system_models'")

def verifyNetType(net,nettype):
    if nettype in 'static':
        if type(net) is not model_augmentation.torch_nets.contracting_REN: return
        elif type(net) is not model_augmentation.torch_nets.LFR_ANN: return
        else: raise ValueError("Static network required...")
    elif nettype in 'dynamic':
        if type(net) is model_augmentation.torch_nets.contracting_REN: return
        elif type(net) is model_augmentation.torch_nets.LFR_ANN: return
        else: raise ValueError("Dynamic network required...")
    else: raise ValueError('Unknown net type, only dynamic or static supported')

# some generic functions
def to_torch_tensor(A): # Obsolete?
    if torch.is_tensor(A):
        return A
    else:
        return torch.tensor(A, dtype=torch.float)


# Function used for parameter initialization
def assign_param(A_old, A_new, nm):
    if A_new is not None:
        assert torch.is_tensor(A_new), nm + ' must be of the Tensor type'
        assert A_new.size() == A_old.size(), nm + ' must be of size' + str(A_old.size())
        return A_new.data
    else:
        return A_old.data


def RK4_step(f, x, u, h):  # Functions of the form f(x,u). See other scripts for time-varying cases
    # one step of runge-kutta integration. u is zero-order-hold
    k1 = h * f(x, u)
    k2 = h * f(x + k1 / 2, u)
    k3 = h * f(x + k2 / 2, u)
    k4 = h * f(x + k3, u)
    return x + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Calculating the SVD of the training data-set for orthogonalization-based regularization
def calculate_orthogonalisation(sys, train_data, x_meas=False, mini_batch_size=None):
    # in:               | out:
    #  - x (Nd, Nx)     |  - cost
    #  - u (Nd, Nu)     |

    if x_meas:  # when y=x
        x = train_data.y
        u = train_data.u
    else:
        sys_data = sys.apply_experiment(train_data)
        # ToDo: check if sys_data_list has x attribute
        x = sys_data.x
        u = sys_data.u

    if mini_batch_size is not None:
        batch_strt = int(torch.rand(1) * (x.shape[0] - mini_batch_size))
        batch_end = batch_strt + mini_batch_size
        x = x[batch_strt:batch_end, :]
        u = u[batch_strt:batch_end, :]
    Matrix = sys.calculate_orth_matrix(x, u)
    U1, _, _ = torch.linalg.svd(Matrix, full_matrices=False)
    return U1, torch.tensor(x, dtype=torch.float), torch.tensor(u, dtype=torch.float)


def initialize_augmentation_net(network, augm_type, nx):
    if augm_type in 'additive':
        init_additive_augmentation_net(network)
    elif augm_type in 'multiplicative':
        init_multiplicative_augmentation_net(network, nx)


# Function for initializing neural networks in additive structure
def init_additive_augmentation_net(network):
    if type(network) is model_augmentation.torch_nets.simple_res_net:
        # If the network is residual neur. net. (has linear part)
        network.net_lin.weight.data.fill_(0.0)
        if network.net_non_lin is not None:  # has nonlinear part
            network.net_non_lin.net[-1].weight.data.fill_(0.0)
            network.net_non_lin.net[-1].bias.data.fill_(0.0)
        else:  # if only linear part is present, then it has bias value
            network.net_lin.bias.data.fill_(0.0)
    elif type(network) is deepSI.utils.torch_nets.feed_forward_nn:
        # for simple feedforward nets
        network.net[-1].weight.data.fill_(0.0)
        network.net[-1].bias.data.fill_(0.0)
    else:
        warnings.warn("Neural network type should be either 'model_augmentation.torch_nets.simple_res_net'"
                      "or 'deepSI.utils.torch_nets.feed_forward_nn' for accurate initialization.")


def init_multiplicative_augmentation_net(network, nx):
    if type(network) is model_augmentation.torch_nets.simple_res_net:
        # If the network is residual neur. net. (has linear part)
        network.net_lin.weight.data.fill_(0.0)
        for i in range(nx):
            network.net_lin.weight.data[i, -nx+i].fill_(1.0)
        if network.net_non_lin is not None:  # has nonlinear part
            network.net_non_lin.net[-1].weight.data.fill_(0.0)
            network.net_non_lin.net[-1].bias.data.fill_(0.0)
        else:  # if only linear layer is present, then it has a bias value
            network.net_lin.bias.data.fill_(0.0)
    else:
        warnings.warn("Neural network type should be 'model_augmentation.torch_nets.simple_res_net'"
                      "for accurate initialization.")


def calculate_normalization(data, x0_meas, fp_system):
    if data is not None:
        std_y = np.std(data.y, axis=0)
        std_u = np.std(data.u, axis=0)
        if x0_meas:
            # for y[k] = x[k] full-state measurement
            std_x = std_y
        else:
            sim_data = fp_system.apply_experiment(data)
            std_x = np.std(sim_data.x, axis=0)
            # ToDo: check if sys_data_list has x attribute
            # ToDO: if this implementation is wrong --> for loop and add all x variables to X array maybe in a different function
    else:
        std_x = np.ones(fp_system.Nx)
        std_y = np.ones(fp_system.Ny)
        std_u = np.ones(fp_system.Nu)
    return std_x, std_y, std_u
