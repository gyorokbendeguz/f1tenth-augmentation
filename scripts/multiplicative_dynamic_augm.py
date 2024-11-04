from datetime import datetime
import os
from torch import nn
import torch
from matplotlib import pyplot as plt
from f1tenth_augmentation.utils import create_sysdata_from_file, create_rnd_train_and_val_data_from_file, plot_losses
from f1tenth_augmentation.car_models import nonlinearCar, nonlinearCar_RK4
from model_augmentation.torch_nets import simple_res_net
from model_augmentation.augmentation_structures import get_dynamic_augment_fitsys, SSE_MultiplicativeDynAugmentation


experiment = "meas"  # "sim" or "meas" for simulation or measurement data based identification
cwd = os.path.dirname(__file__)
save_name = f'../results/{experiment}/multiplicative_dynamic_augment_' + (datetime.now()).strftime("%Y%m%d-T%H%M%S")
saveFolder = os.path.join(cwd, save_name)
os.mkdir(saveFolder)

Ts = 0.025  # sampling time
T = 20  # truncation time
epoch = 1250
batch_size = 256
regLambda = 0.1
lr = 1e-3
bParmTune = False

# state + input dimensions
nx = ny = 3  # model state (first-principle model)
nu = 2  # input
nx_hidden = 1  # hidden state

# hyperparameters of the augmentation net
n_in = 2*nx + nu + nx_hidden  # state + fp x+ + inputs
n_out = nx + nx_hidden  # states
n_hidden_layers = 2  # no. of hidden layers
n_nodes_per_layer = 128  # no. nodes per layer
n_act = nn.Tanh  # activation function

# encoder hyperparameters
lag_all = 12
n_hidden_layers_enc = 2
n_nodes_per_layer_enc = 64

# load data
train_data_folder = os.path.join(cwd, f"../data/{experiment}_data/train/")
test_data_folder = os.path.join(cwd, f"../data/{experiment}_data/test/")

# split data
train, valid = create_rnd_train_and_val_data_from_file(data_folders=train_data_folder, nf=lag_all, T=T, split_fraction=0.2)
test = create_sysdata_from_file(test_data_folder)

# neural net for augmentation
ffw_net = simple_res_net(n_in=n_in, n_out=n_out, n_hidden_layers=n_hidden_layers, n_nodes_per_layer=n_nodes_per_layer,
                         activation=n_act)

# Dynamical models of the car
if experiment == "sim":
    bb_model = nonlinearCar(nx=nx, ny=ny, nu=nu, ts=Ts, parmTune=bParmTune)  # Euler's discretization is good enough
else:
    bb_model = nonlinearCar_RK4(nx=nx, ny=ny, nu=nu, ts=Ts, parmTune=bParmTune)  # use RK4 discretization

# Initialization
fit_sys = get_dynamic_augment_fitsys(augmentation_structure=SSE_MultiplicativeDynAugmentation, known_system=bb_model,
                                     neur_net=ffw_net, e_net=None, y_lag_encoder=lag_all, u_lag_encoder=lag_all,
                                     enet_kwargs=dict(n_nodes_per_layer=n_nodes_per_layer_enc, n_hidden_layers=n_hidden_layers_enc,
                                                      activation=n_act), regLambda=regLambda, hidden_state=nx_hidden,
                                     norm_data=train, norm_x0_meas=True)
fit_sys.init_model(test, auto_fit_norm=False)

# Training
print('Training the augmented model...')
fit_sys.fit(train_sys_data=train, epochs=epoch, val_sys_data=valid, batch_size=batch_size,
            loss_kwargs=dict(nf=T, online_construct=False), optimizer_kwargs=dict(lr=lr))

# Training losses
plot_losses(fit_sys, saveFolder)

# Testing
print('Applying an experiment...')
fit_sys.checkpoint_load_system(name='_best')
test_augmented_model = fit_sys.apply_experiment(test)

if bb_model.parm_corr_enab:
    P = fit_sys.hfn.sys.P.detach().numpy()
    parm_corr_data = {
        "alpha (regularization)": regLambda,
        "m": P[0],
        "Jz": P[1],
        "lr": P[2],
        "lf": P[3],
        "Cm1": P[4],
        "Cm2": P[5],
        "Cm3": P[6],
        "Cr": P[7],
        "Cf": P[8]
    }
    with torch.no_grad():
        bb_model.P.data = bb_model.P_orig

test_fp_model = bb_model.apply_experiment(test, x0_meas=True)

fig1, ax1 = plt.subplots(1, 3, layout="tight")
ax1[0].plot(test.y[:, 0], "k")
ax1[0].plot(test.y[:, 0] - test_fp_model.y[:, 0], '-.b')
ax1[0].plot(test.y[:, 0] - test_augmented_model.y[:, 0], "--r")
ax1[0].legend(["MuJoCo sim.", "FP model error", "Augmented model error"])
ax1[0].set_xlabel("Sim index")
ax1[0].set_ylabel("Longitud. vel. [m/s]")
ax1[0].set_xlim([0, test.y.shape[0]])

ax1[1].plot(test.y[:, 1], "k")
ax1[1].plot(test.y[:, 1] - test_fp_model.y[:, 1], '-.b')
ax1[1].plot(test.y[:, 1] - test_augmented_model.y[:, 1], "--r")
ax1[1].set_xlabel("Sim index")
ax1[1].set_ylabel("Lateral vel. [m/s]")
ax1[1].set_xlim([0, test.y.shape[0]])

ax1[2].plot(test.y[:, 2], "k")
ax1[2].plot(test.y[:, 2] - test_fp_model.y[:, 2], '-.b')
ax1[2].plot(test.y[:, 2] - test_augmented_model.y[:, 2], "--r")
ax1[2].set_xlabel("Sim index")
ax1[2].set_ylabel("Ang. vel. [rad/s]")
ax1[2].set_xlim([0, test.y.shape[0]])
plt.savefig(saveFolder + '/errors.png')
plt.show(block=False)

print(f"RMS simulation FP model: {test_fp_model.RMS(test):.2}")
print(f"RMS simulation augmented model: {test_augmented_model.RMS(test):.2}")
print(f"NRMS simulation FP model: {test_fp_model.NRMS(test):.2%}")
print(f"NRMS simulation augmented model: {test_augmented_model.NRMS(test):.2%}")

fig2, ax2 = plt.subplots(2, 3, layout="tight")
ax2[0, 0].plot(test.y[:, 0], "k")
ax2[0, 0].plot(test_fp_model.y[:, 0], "-.b")
ax2[0, 0].set_xlabel("Sim index")
ax2[0, 0].set_ylabel("Longitud. vel. [m/s]")
ax2[0, 0].legend(["MuJoCo", "FP model"])
ax2[0, 0].set_xlim(0, test.y.shape[0])

ax2[0, 1].plot(test.y[:, 1], "k")
ax2[0, 1].plot(test_fp_model.y[:, 1], "-.b")
ax2[0, 1].set_xlabel("Sim index")
ax2[0, 1].set_ylabel("Lateral vel. [m/s]")
ax2[0, 1].set_xlim(0, test.y.shape[0])

ax2[0, 2].plot(test.y[:, 2], "k")
ax2[0, 2].plot(test_fp_model.y[:, 2], "-.b")
ax2[0, 2].set_xlabel("Sim index")
ax2[0, 2].set_ylabel("Ang. vel. [rad/s]")
ax2[0, 2].set_xlim(0, test.y.shape[0])

ax2[1, 0].plot(test.y[:, 0], "k")
ax2[1, 0].plot(test_augmented_model.y[:, 0], "--r")
ax2[1, 0].set_xlabel("Sim index")
ax2[1, 0].set_ylabel("Longitud. vel. [m/s]")
ax2[1, 0].legend(["MuJoCo", "Augmented model"])
ax2[1, 0].set_xlim(0, test.y.shape[0])

ax2[1, 1].plot(test.y[:, 1], "k")
ax2[1, 1].plot(test_augmented_model.y[:, 1], "--r")
ax2[1, 1].set_xlabel("Sim index")
ax2[1, 1].set_ylabel("Lateral vel. [m/s]")
ax2[1, 1].set_xlim(0, test.y.shape[0])

ax2[1, 2].plot(test.y[:, 2], "k")
ax2[1, 2].plot(test_augmented_model.y[:, 2], "--r")
ax2[1, 2].set_xlabel("Sim index")
ax2[1, 2].set_ylabel("Ang. vel. [rad/s]")
ax2[1, 2].set_xlim(0, test.y.shape[0])
plt.savefig(saveFolder + '/outputs.png')
plt.show()

augmentation_data = {
    "Augmentation": "DYNAMIC MULTIPLICATIVE",
    "truncation time": T,
    "epochs": epoch,
    "batch size": batch_size,
    "nodes per layer": n_nodes_per_layer,
    "no. layers": n_hidden_layers,
    "activation function": str(n_act),
    "no. hidden states": nx_hidden,
    "ANN inputs": n_in,
    "ANN outputs": n_out,
    "Test RMS": test_augmented_model.RMS(test),
    "Test NRMS": test_augmented_model.NRMS(test),
    "Correction enabled": bb_model.parm_corr_enab
}

text = 'Training properties: \n'
with open(saveFolder + '/info.txt', 'w') as f:
    f.write(text)
    for key, value in augmentation_data.items():
        f.write('%s: %s\n' % (key, value))
    if bb_model.parm_corr_enab:
        for key, value in parm_corr_data.items():
            f.write('%s: %s\n' % (key, value))

# save system
fit_sys.checkpoint_load_system(name='_best')
fit_sys.save_system(saveFolder + '/best.pt')
