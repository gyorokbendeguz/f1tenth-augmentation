import numpy as np

def multisine_test(data_length, amplitude_max=30, ts=0.02):
    freqencies = [1, 2, 5, 7.5, 10, 12.5, 15, 25]
    inputs = np.empty((data_length, 1))
    amplitudes = np.linspace(0, amplitude_max, data_length)
    for i in range(data_length):
        u_k = 0
        time = i * ts
        for freq in freqencies:
            u_k += np.cos(time * freq) / len(freqencies)
        ampl = amplitudes[i]
        inputs[i, 0] = ampl * u_k
    return inputs

def multisine_train(data_length, ts=0.02):
    freqencies = [1, 2, 5, 7.5, 10, 12.5, 15, 25]
    inputs = np.empty((data_length, 1))
    amplitudes = [1, 2, 3, 5, 7.5, 10, 15, 20, 25]
    j = 0
    for i in range(data_length):
        u_k = 0
        time = i * ts
        for freq in freqencies:
            u_k += np.cos(time * freq) / len(freqencies)
        if i > (j+1) * data_length / len(amplitudes):
            j += 1
        ampl = amplitudes[j]
        inputs[i, 0] = ampl * u_k + np.random.normal(loc=0, scale=ampl/20)
    return inputs

if __name__ == "__main__":
    from matplotlib import pyplot as plt

    u = multisine_train(40000)

    plt.figure()
    plt.plot(u)
    plt.show()
