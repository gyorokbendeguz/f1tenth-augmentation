import deepSI
import os
import numpy as np
import random
from matplotlib import pyplot as plt


def csv_read(folder, path_parts=[]):
    """
    Reads the data into a .csv file.

    Arguments:
        - folder (string): path of the reading
        - path_parts (string): extra paths in the folder

    Returns:
        -data (np.array): data which been read from .csv file
    """

    for part in path_parts:
        folder = os.path.join(folder, part)
    with open(os.path.join(folder), 'r') as in_file:
        data = np.genfromtxt(in_file, delimiter=",")
    return data


def create_sysdata_from_file(data_folders):
    """
    Creates system (deepSI) data from time-series data.

    Arguments:
        - data_folders (string): path direction of the time-series data file
    Returns:
        - system_data (deepSI.system_data): training data
    """

    input_data = []
    output_data = []
    data_names_list = os.listdir(data_folders)
    for name in data_names_list:
        folder = os.path.join(data_folders, name)
        input_data.append(csv_read(folder, ['input.csv']))
        output_data.append(csv_read(folder, ['output.csv']))

    sys_data_list = []
    for input, output in zip(input_data, output_data):
        # start from idx where tire dynamics are defined
        idxCut = np.where(np.abs(output[:, 1]) > 0.25)[0][0]
        sys_data_part = deepSI.System_data(u=input[idxCut:, 1:], y=output[idxCut:, 1:])
        sys_data_list.append(sys_data_part)

    system_data = deepSI.System_data_list(sys_data_list=sys_data_list)
    return system_data

def create_orth_data_from_file(data_folder):
    input_data = []
    output_data = []
    data_names_list = os.listdir(data_folder)
    for name in data_names_list:
        if name == 'circle_0.5' or name == 'circle_1.0' or name == 'eight_0.45' or name =='eight_0.95':
            folder = os.path.join(data_folder, name)
            input_data.append(csv_read(folder, ['input.csv']))
            output_data.append(csv_read(folder, ['output.csv']))

    sys_data_list = []
    for input, output in zip(input_data, output_data):
        idxCut = np.where(np.abs(output[:, 1]) > 0.25)[0][0]
        sys_data_part = deepSI.System_data(u=input[idxCut:, 1:], y=output[idxCut:, 1:])
        sys_data_list.append(sys_data_part)

    system_data = deepSI.System_data_list(sys_data_list=sys_data_list)
    return system_data


def split_list(inputs, outputs, nf, T, split_fraction):
    """
    Splits a list(two) randomly in between and extract the remaining parts.

    Arguments:
        - inputs (list): input time-series data
        - outputs (list): output time-series data
        - nf (float): the number of steps the encoder must calculate from the past
        - T-truncation time (float): the number of steps the encoder must predict to the future
        - split_fraction (float): the ratio to split the data (for 0.2 the validation will be 20% and the training is the remaining 80%)
    Returns:
        - train_data_in1 (list): first half of the input training data
        - train_data_in2 (list): second half of the input training data
        - valid_data_in (list): input validation data between the first and second training data
        - train_data_out1 (list): first half of the output training data
        - train_data_out2 (list): second half of the output training data
        - valid_data_out (list): output validation data between the first and second training data
    """

    split_index = random.randint(nf+T, int(len(inputs)*(1-split_fraction)-(nf+T)))

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, axis=1)

    if outputs.ndim == 1:
        outputs = np.expand_dims(outputs, axis=1)

    valid_data_in = inputs[split_index:split_index + int(len(inputs) * split_fraction), :]
    train_data_in1 = inputs[:split_index, :]
    train_data_in2 = inputs[split_index + int(len(inputs) * split_fraction):, :]

    valid_data_out = outputs[split_index:split_index + int(len(outputs) * split_fraction), :]
    train_data_out1 = outputs[:split_index, :]
    train_data_out2 = outputs[split_index + int(len(outputs) * split_fraction):, :]

    return train_data_in1, train_data_in2, valid_data_in, train_data_out1, train_data_out2, valid_data_out


def create_rnd_train_and_val_data_from_file(data_folders, nf, T, split_fraction=0.2):
    """
    Creates random training and validation data from time-series data.

    Arguments:
        - data_folders (string): path direction of the time-series data file
        - nf (float): the number of steps the encoder must calculate from the past
        - T-truncation time (float): the number of steps the encoder must predict to the future
        - split_fraction (float): the ratio to split the data (for 0.2 the validation will be 20% and the training is the remaining 80%)
    Returns:
        - train_data (deepSI.system_data): training data
        - valid_data (deepSI.system_data): validation data
    """

    input_train_data = []
    input_valid_data = []
    output_train_data = []
    output_valid_data = []

    data_names_list = os.listdir(data_folders)
    for name in data_names_list:

        folder = os.path.join(data_folders, name)
        inputs = csv_read(folder, ['input.csv'])
        outputs = csv_read(folder, ['output.csv'])

        train_data_in1, train_data_in2, valid_data_in, train_data_out1, train_data_out2, valid_data_out = split_list(inputs, outputs, nf=nf, T=T, split_fraction=split_fraction)

        input_train_data.append(train_data_in1)
        input_train_data.append(train_data_in2)
        output_train_data.append(train_data_out1)
        output_train_data.append(train_data_out2)
        input_valid_data.append(valid_data_in)
        output_valid_data.append(valid_data_out)

    train_data_list = []
    valid_data_list = []
    for input1, output1 in zip(input_train_data, output_train_data):
        # start from idx where tire dynamics are defined
        idxCut = np.where(np.abs(output1[:, 1]) > 0.25)[0][0]
        sys_data_part_train = deepSI.System_data(u=np.array(input1[idxCut:, 1:]), y=np.array(output1[idxCut:, 1:]))
        train_data_list.append(sys_data_part_train)

    for input2, output2 in zip(input_valid_data, output_valid_data):
        # start from idx where tire dynamics are defined
        idxCut = np.where(np.abs(output2[:, 1]) > 0.25)[0][0]
        sys_data_part_valid = deepSI.System_data(u=np.array(input2[idxCut:, 1:]), y=np.array(output2[idxCut:, 1:]))
        valid_data_list.append(sys_data_part_valid)

    train_data = deepSI.System_data_list(sys_data_list=train_data_list)
    valid_data = deepSI.System_data_list(sys_data_list=valid_data_list)

    return train_data, valid_data


def plot_losses(fitsys, saveFolder=None):
    fitsys.checkpoint_load_system(name='_last')
    train_losses = fitsys.Loss_train[:]
    val_losses = fitsys.Loss_val[:]

    fig_losses, axs = plt.subplots(2, 1, sharex=True)

    axs[0].semilogy(train_losses, label='Training loss')
    axs[0].legend()
    axs[0].set_ylabel('MS')

    axs[1].semilogy(val_losses, label='Validation loss')
    axs[1].legend()
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('NRMS')
    axs[1].legend()

    fig_losses.tight_layout()
    if saveFolder is not None:
        plt.savefig(saveFolder + '/losses.png')
    plt.show(block=False)


def RungeKutta4_step(t, h, y_n, model, u_n):
    """
    Executes 1 step of the Runge-Kutta method with input params kept as a constant.

    Arguments:
         - t (float): current time
         - h (float): time step
         - y_n (float): current output
         - model (function): dynamics of the model
         - u_n: current input
    Returns:
        - t_n1: next time
        - y_n1: next output
    """

    k1 = model(t, y_n, u_n)
    k2 = model(t + h / 2, y_n + h * k1 / 2, u_n)
    k3 = model(t + h / 2, y_n + h * k2 / 2, u_n)
    k4 = model(t + h, y_n + h * k3, u_n)

    y_n1 = y_n + 1 / 6 * h * (k1 + 2 * k2 + 2 * k3 + k4)
    t_n1 = t + h

    return t_n1, y_n1


def dynamics(t, states, inputs):
    """
    Function that describes the model dynamics.

    Arguments:
         - states (np.array): 1D vector (6) of the state variables
    Returns:
        - delta (np.array): 1D vector (3) of the changes in the position variables
    """

    # retrieve inputs & states
    x_pos = states[0]
    y_pos = states[1]
    phi = states[2]
    v_long = inputs[0]
    v_lat = inputs[1]
    yaw_rate = inputs[2]

    # dynamics:
    d_phi = yaw_rate
    d_x = v_long * np.cos(phi) - v_lat * np.sin(phi)
    d_y = v_long * np.sin(phi) + v_lat * np.cos(phi)

    delta = np.array([d_x, d_y, d_phi])

    return delta


def _normalize(angle):
    """
    Normalizes the given angle into the [-pi/2, pi/2] range.

    Arguments:
        - angle (float): angle to normalize, in radian
    Returns:
        - angle (float): normalized angle, in radian
    """

    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi

    return angle
