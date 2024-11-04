import torch
from deepSI import System_deriv
import numpy as np
from model_augmentation.system_models import lti_system


class true_system(System_deriv):
    def __init__(self, dt=0.02, noise=True, sigma=0.0):
        super(true_system, self).__init__(nx=6, nu=1, ny=1, dt=dt)
        m1 = 0.5
        m2 = 0.4
        m3 = 0.02
        ki = 100
        bi = 0.5
        self.m1 = m1
        self.a = 50

        self.A = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
                           [(-ki-ki)/m1, ki/m1, 0, (-bi-bi)/m1, bi/m1, 0],
                           [ki/m2, (-ki-ki)/m2, ki/m2, bi/m2, (-bi-bi)/m2, bi/m2],
                           [0, ki/m3, -ki/m3, 0, bi/m3, -bi/m3]])

        self.B = np.array([[0], [0], [0], [1/m1], [0], [0]])

        self.K = np.array([0, 0, 0, 0.71, -0.64, 0.28])
        self.sigma = sigma

        self.noise = noise
        self.noise_val = 0

    def deriv(self, x, u):
        x1, x2, x3, x1dot, x2dot, x3dot = x
        x_vector = np.array([[x1], [x2], [x3], [x1dot], [x2dot], [x3dot]])
        xdot_vector = self.A @ x_vector + self.B * u + np.array([[0], [0], [0], [-self.a/self.m1 * x1**3], [0], [0]])
        return [xdot_vector[0, 0], xdot_vector[1, 0], xdot_vector[2, 0], xdot_vector[3, 0], xdot_vector[4, 0], xdot_vector[5, 0]]

    def f(self, x, u):
        assert self.dt is not None, 'please set dt or in the __init__ or in sys_data.dt'
        if self.method == 'RK4':
            x = np.array(x)
            k1 = self.dt*np.array(self.deriv(x, u))
            k2 = self.dt*np.array(self.deriv(x+k1/2, u))
            k3 = self.dt*np.array(self.deriv(x+k2/2, u))
            k4 = self.dt*np.array(self.deriv(x+k3, u))
            x_plus = x + (k1+2*k2+2*k3+k4)/6
            if self.sigma > 0:
                x_plus += self.sigma * self.K * self.noise_val
            return x_plus
        else:
            raise NotImplementedError

    def h(self, x, u):
        if self.noise:
            self.noise_val = np.random.normal(scale=1.5e-4)
        else:
            self.noise_val = 0
        x1, x2, x3, x1dot, x2dot, x3dot = x
        return x2 + self.noise_val


def generate_MSD_fp_model(sigma=0):
    dt = 0.02
    A = np.array([[0.9227, 0.0383, 0.01909, 0.0004505], [0.04788, 0.9515, 0.0005632, 0.01943],
                  [-7.524, 3.706, 0.8851, 0.05683], [4.632, -4.745, 0.07104, 0.9277]])
    B = np.array([[0.0003895], [6.5e-06], [0.03819], [0.001126]])
    C = np.array([[0, 1, 0, 0]])
    D = np.array([[0]])
    fp_model = lti_system(A=A, B=B, C=C, D=D, Ts=dt)

    K = sigma * np.array([[0], [0], [0.71], [-0.64]])
    fp_model.K = torch.tensor(data=K, dtype=torch.float)
    return fp_model
