import os
import deepSI
import numpy as np
from datetime import datetime
from torch import nn
from model_augmentation.augmentation_structures import get_dynamic_augment_fitsys, SSE_LFRDynAugmentation
from matplotlib import pyplot as plt
from f1tenth_augmentation.utils import plot_losses
from model_augmentation.augmentation_encoders import lti_initialized_encoder
from MSD_system.MSD_systems import generate_MSD_fp_model
from model_augmentation.torch_nets import simple_res_net


SIGMA = 1
strEncoderType = "physics-based-noise"
USE_NOISE_MODEL = True

cwd = os.path.dirname(__file__)
data_file_path = os.path.join(cwd, "..", "data", "MSD_data", f"MSD_sigma{SIGMA}")
train_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_train.npz"))
val_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_val.npz"))
test_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_test.npz"))

# expanding dims (should have implemented in data gen.)
train_data.y = np.expand_dims(train_data.y, axis=1)
val_data.y = np.expand_dims(val_data.y, axis=1)
test_data.y = np.expand_dims(test_data.y, axis=1)

saveFolder = os.path.join(cwd, '..', f'results/MSD/MSD_benchmark_' + (datetime.now()).strftime("%Y%m%d-T%H%M%S"))
os.mkdir(saveFolder)
T = 200  # truncation time
epoch = 5000  # 5000 in CDC paper
batch_size = 1024
enc_lag = 7
nx_hidden = 2

bSaveTrainingData = True

# hyperparameters of the augmentation net
n_z = 5  # LFR-LTI input
n_w = 4  # LFR-LTI output
n_hidden_layers = 3  # no. of hidden layers
n_nodes_per_layer = 64  # no. nodes per layer
n_act = nn.Tanh  # activation function

# neural net for augmentation
ffw_net = simple_res_net(n_in=n_z+nx_hidden, n_out=n_w+nx_hidden, n_hidden_layers=n_hidden_layers,
                         n_nodes_per_layer=n_nodes_per_layer, activation=n_act)

# first-principles model
fp_system = generate_MSD_fp_model(sigma=SIGMA)

# Initialization
fit_sys = get_dynamic_augment_fitsys(augmentation_structure=SSE_LFRDynAugmentation, known_system=fp_system,
                                     hidden_state=nx_hidden, neur_net=ffw_net, e_net=lti_initialized_encoder,
                                     enet_kwargs=dict(known_sys=fp_system, noise_handling=USE_NOISE_MODEL),
                                     y_lag_encoder=enc_lag, u_lag_encoder=enc_lag, na_right=1,
                                     norm_data=train_data, l2_reg=1e-6, init_scaling_factor=1e-3)
# fit_sys = get_dynamic_augment_fitsys(augmentation_structure=SSE_LFRDynAugmentation, known_system=fp_system,
#                                      hidden_state=nx_hidden, neur_net=ffw_net, y_lag_encoder=enc_lag,
#                                      u_lag_encoder=enc_lag, na_right=1, init_scaling_factor=1e-3,
#                                      norm_data=train_data, l2_reg=1e-6)
fit_sys.init_model(test_data, auto_fit_norm=False)

# Training
print('Training the augmented model...')
fit_sys.fit(train_sys_data=train_data, epochs=epoch, val_sys_data=val_data, batch_size=batch_size,
            loss_kwargs=dict(nf=T, online_construct=False), optimizer_kwargs=dict(lr=1e-3))

# Training losses
plot_losses(fit_sys, saveFolder)

# Testing
fit_sys.checkpoint_load_system(name='_best')
print('Applying an experiment...')
test_augmented_model = fit_sys.apply_experiment(test_data)
test_fp_model = fp_system.apply_experiment(test_data)

print(f"FP model test error (NRMS): {test_fp_model.NRMS(test_data):.2%}")
print(f"Test error (NRMS): {test_augmented_model.NRMS(test_data):.2%}")


plt.figure()
plt.plot(test_data.y)
plt.plot(test_data.y - test_fp_model.y)
plt.plot(test_data.y - test_augmented_model.y)
plt.legend(["Sim.", "LTI model error", "Augmented model error"])
plt.xlabel("Sim index")
plt.ylabel("Position [m]")
plt.xlim([0, test_data.y.shape[0]])
plt.savefig(saveFolder + "/errors.png")
plt.show()

augmentation_data = {
    "Augmentation": "DYNAMIC LFR-BASED",
    "truncation time": T,
    "epochs": epoch,
    "batch size": batch_size,
    "nodes per layer": n_nodes_per_layer,
    "no. layers": n_hidden_layers,
    "activation function": str(n_act),
    "n_z": n_z,
    "n_w": n_w,
    "Test RMS": test_augmented_model.RMS(test_data),
    "Test NRMS": test_augmented_model.NRMS(test_data),
    "Sigma value": SIGMA,
    "Encoder type": strEncoderType,
    "Noise handling": USE_NOISE_MODEL
}

text = 'Training properties: \n'
with open(saveFolder + '/info.txt', 'w') as f:
    f.write(text)
    for key, value in augmentation_data.items():
        f.write('%s: %s\n' % (key, value))

# save system
fit_sys.checkpoint_load_system(name='_best')
fit_sys.save_system(saveFolder + '/best.pt')
fit_sys.checkpoint_load_system(name='_last')
fit_sys.save_system(saveFolder + '/last.pt')
