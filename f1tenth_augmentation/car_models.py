from model_augmentation.system_models import general_nonlinear_system
from model_augmentation.utils import RK4_step
import torch
from torch import nn
import os


class nonlinearCar(general_nonlinear_system):
    def __init__(self, nu, nx, ny, ts, parmTune=False):
        super(nonlinearCar, self).__init__(nu=nu, nx=nx, ny=ny, Ts=ts)
        #  parameter: m, Jz, lr, lf, Cm1, Cm2, Cm3, Cr, Cf
        self.P_orig = torch.tensor([2.93, 0.0796, 0.168, 0.163, 41.796, 2.0152, 0.4328, 41.7372, 29.4662], dtype=torch.float)
        self.parm_corr_enab = parmTune
        if self.parm_corr_enab:
            self.P = nn.Parameter(data=self.P_orig.clone())
        else:
            self.P = self.P_orig
        self.ts = ts
        self.linearized_form_provided = True

    def f(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - x+ (Nd, Nx)
        #  - u (Nd, Nu) |
        #  - P (Np)     |

        if x.dim() == 1:
            v_xi = x[0]
            v_eta = x[1]
            omega = x[2]
        else:
            v_xi = x[:, 0]
            v_eta = x[:, 1]
            omega = x[:, 2]
        if u.dim() == 1:
            delta = u[0]
            d = u[1]
        else:
            delta = u[:, 0]
            d = u[:, 1]

        Parameters = self.P.clone()
        m = torch.take(Parameters, torch.tensor([0]))
        Jz = torch.take(Parameters, torch.tensor([1]))
        lr = torch.take(Parameters, torch.tensor([2]))
        lf = torch.take(Parameters, torch.tensor([3]))
        Cm1 = torch.take(Parameters, torch.tensor([4]))
        Cm2 = torch.take(Parameters, torch.tensor([5]))
        Cm3 = torch.take(Parameters, torch.tensor([6]))
        Cr = torch.take(Parameters, torch.tensor([7]))
        Cf = torch.take(Parameters, torch.tensor([8]))

        Fxi = Cm1 * d - Cm2 * v_xi - torch.sign(v_xi) * Cm3

        alpha_r = (-v_eta + lr * omega) / v_xi
        alpha_f = delta - (v_eta + lf * omega) / v_xi
        Fr_eta = Cr * alpha_r
        Ff_eta = Cf * alpha_f

        idx_tire_model = torch.where(torch.abs(v_xi) > 0.1, 1.0, 0.0)
        Fr_eta = Fr_eta*idx_tire_model
        Ff_eta = Ff_eta*idx_tire_model

        v_xid = 1 / m * (Fxi + Fxi * torch.cos(delta) - Ff_eta * torch.sin(delta) + m * v_eta * omega)
        v_etad = 1 / m * (Fr_eta + Fxi * torch.sin(delta) + Ff_eta * torch.cos(delta) - m * v_xi * omega)
        omega_d = 1 / Jz * (Ff_eta * lf * torch.cos(delta) + Fxi * lf * torch.sin(delta) - Fr_eta * lr)

        if x.dim() == 1:
            xplus = x + self.ts*torch.hstack((v_xid, v_etad, omega_d))
        else:
            xplus = (x.T + self.ts*torch.vstack((v_xid, v_etad, omega_d))).T
        return xplus

    def h(self, x, u):
        return x

    def calculate_orth_matrix(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - F(x,u) (Nd*Nx, Np)  assumed LIP model: x+ = F(x,u)*theta
        #  - u (Nd, Nu) |
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float)
        if not torch.is_tensor(u):
            u = torch.tensor(u, dtype=torch.float)
        F = torch.zeros((3*x.shape[0], 10))  # Nx=3, Np = 10
        for i in range(x.shape[0]):
            v_xi = x[i, 0]
            v_eta = x[i, 1]
            ome = x[i, 2]
            delta = u[i, 0]
            d = u[i, 1]
            F[3*i, 0] = 0.0059*v_xi - 0.1217*d + 0.0013*torch.sign(v_xi) + 0.0858*torch.sin(delta)*(delta - (0.1630*ome + v_eta)/v_xi) + 0.0029*torch.cos(delta)*(2.0152*v_xi - 41.7960*d + 0.4328*torch.sign(v_xi))
            F[3*i, 3] = (0.2514*ome*torch.sin(delta))/v_xi
            F[3*i, 4] = 0.0085*d + 0.0085*d*torch.cos(delta)
            F[3*i, 5] = - 0.0085*v_xi - 0.0085*v_xi*torch.cos(delta)
            F[3*i, 6] = - 0.0085*torch.sign(v_xi) - 0.0085*torch.cos(delta)*torch.sign(v_xi)
            F[3*i, 8] = -0.0085*torch.sin(delta)*(delta - (0.1630*ome + v_eta)/v_xi)
            F[3*i, 9] = 0.3566*d + 0.9828*v_xi - 0.0037*torch.sign(v_xi) + 0.0037*torch.cos(delta)*torch.sign(v_xi) - 0.2514*torch.sin(delta)*(delta - (0.1630*ome + v_eta)/v_xi) + 0.0250*ome*v_eta - 0.0171*torch.cos(delta)*(2.0152*v_xi - 41.7960*d + 0.4328*torch.sign(v_xi)) - 0.3566*d*torch.cos(delta) + 0.0172*v_xi*torch.cos(delta) - (0.0410*ome*torch.sin(delta))/v_xi

            F[3*i+1, 0] = 0.0858*torch.cos(delta)*(delta - (0.1630*ome + v_eta)/v_xi) + 0.0029*torch.sin(delta)*(2.0152*v_xi - 41.7960*d + 0.4328*torch.sign(v_xi)) - (0.1215*(0.1680*ome - v_eta))/v_xi
            F[3*i+1, 2] = (0.3561*ome)/v_xi
            F[3*i+1, 3] = (0.2514*ome*torch.cos(delta))/v_xi
            F[3*i+1, 4] = 0.0085*d*torch.sin(delta)
            F[3*i+1, 5] = -0.0085*v_xi*torch.sin(delta)
            F[3*i+1, 6] = -0.0085*torch.sin(delta)*torch.sign(v_xi)
            F[3*i+1, 7] = (0.0085*(0.1680*ome - v_eta))/v_xi
            F[3*i+1, 8] = -0.0085*torch.cos(delta)*(delta - (0.1630*ome + v_eta)/v_xi)
            F[3*i+1, 9] = v_eta - 0.2514*torch.cos(delta)*(delta - (0.1630*ome + v_eta)/v_xi) + 0.0037*torch.sin(delta)*torch.sign(v_xi) - 0.0250*ome*v_xi - 0.0171*torch.sin(delta)*(2.0152*v_xi - 41.7960*d + 0.4328*torch.sign(v_xi)) + (0.3561*(0.1680*ome - v_eta))/v_xi - 0.3566*d*torch.sin(delta) - (0.0598*ome)/v_xi + 0.0172*v_xi*torch.sin(delta) - (0.0410*ome*torch.cos(delta))/v_xi

            F[3*i+2, 1] = 27.7050*d - 1.3358*v_xi - 0.2869*torch.sign(v_xi) - 18.9507*torch.cos(delta)*(delta - (0.1630*ome + v_eta)/v_xi) + 0.6431*torch.sin(delta)*(2.0152*v_xi - 41.7960*d + 0.4328*torch.sign(v_xi))
            F[3*i+2, 2] = 0.6329*v_xi - 13.1269*d + 0.1359*torch.sign(v_xi)
            F[3*i+2, 3] = 9.2545*torch.cos(delta)*(delta - (0.1630*ome + v_eta)/v_xi) - 0.3141*torch.sin(delta)*(2.0152*v_xi - 41.7960*d + 0.4328*torch.sign(v_xi)) - (1.5085*ome*torch.cos(delta))/v_xi
            F[3*i+2, 4] = 0.0512*d*torch.sin(delta) - 0.0528*d
            F[3*i+2, 5] = 0.0528*v_xi - 0.0512*v_xi*torch.sin(delta)
            F[3*i+2, 6] = 0.0528*torch.sign(v_xi) - 0.0512*torch.sin(delta)*torch.sign(v_xi)
            F[3*i+2, 8] = 0.0512*torch.cos(delta)*(delta - (0.1630*ome + v_eta)/v_xi)
            F[3*i+2, 9] = ome + 0.0222*torch.sin(delta)*torch.sign(v_xi) - 0.0512*torch.sin(delta)*(2.0152*v_xi - 41.7960*d + 0.4328*torch.sign(v_xi)) - 2.1397*d*torch.sin(delta) + 0.1032*v_xi*torch.sin(delta) + (0.2459*ome*torch.cos(delta))/v_xi
        return F

    def linear_part(self, x, u):
        if x.dim() == 1:
            v_xi = x[0]
            v_eta = x[1]
            omega = x[2]
        else:
            v_xi = x[:, 0]
            v_eta = x[:, 1]
            omega = x[:, 2]
        if u.dim() == 1:
            delta = u[0]
            d = u[1]
        else:
            delta = u[:, 0]
            d = u[:, 1]

        Parameters = self.P.clone()
        m = torch.take(Parameters, torch.tensor([0]))
        Jz = torch.take(Parameters, torch.tensor([1]))
        lr = torch.take(Parameters, torch.tensor([2]))
        lf = torch.take(Parameters, torch.tensor([3]))
        Cm1 = torch.take(Parameters, torch.tensor([4]))
        Cm2 = torch.take(Parameters, torch.tensor([5]))
        Cr = torch.take(Parameters, torch.tensor([7]))
        Cf = torch.take(Parameters, torch.tensor([8]))

        v_xi_0 = 1.14
        d_0 = 0.055

        v_xi_d = (-2 * Cm2 / m) * (v_xi - v_xi_0) + (2 * Cm1 / m) * (d - d_0)
        v_eta_d = -(Cr + Cf)/(m*v_xi_0) * v_eta + (lr*Cr - lf*Cf)/(Jz*v_xi_0) * omega + Cf/m * delta
        omega_d = ((Cr*lr - Cf*lf)/(m*v_xi_0) - v_xi_0) * v_eta - (lf*lf*Cf + lr*lr*Cr)/(Jz*v_xi_0) * omega + Cf*lf/Jz * delta

        if x.dim() == 1:
            xplus = x + self.ts*torch.hstack((v_xi_d, v_eta_d, omega_d))
        else:
            xplus = (x.T + self.ts*torch.vstack((v_xi_d, v_eta_d, omega_d))).T
        return xplus


class nonlinearCar_RK4(nonlinearCar):
    def __init__(self, nu, nx, ny, ts, parmTune=False):
        super(nonlinearCar_RK4, self).__init__(nu=nu, nx=nx, ny=ny, ts=ts, parmTune=parmTune)

    def f_deriv(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - x+ (Nd, Nx)
        #  - u (Nd, Nu) |
        #  - P (Np)     |

        if x.dim() == 1:
            v_xi = x[0]
            v_eta = x[1]
            omega = x[2]
        else:
            v_xi = x[:, 0]
            v_eta = x[:, 1]
            omega = x[:, 2]
        if u.dim() == 1:
            delta = u[0]
            d = u[1]
        else:
            delta = u[:, 0]
            d = u[:, 1]

        Parameters = self.P.clone()
        m = torch.take(Parameters, torch.tensor([0]))
        Jz = torch.take(Parameters, torch.tensor([1]))
        lr = torch.take(Parameters, torch.tensor([2]))
        lf = torch.take(Parameters, torch.tensor([3]))
        Cm1 = torch.take(Parameters, torch.tensor([4]))
        Cm2 = torch.take(Parameters, torch.tensor([5]))
        Cm3 = torch.take(Parameters, torch.tensor([6]))
        Cr = torch.take(Parameters, torch.tensor([7]))
        Cf = torch.take(Parameters, torch.tensor([8]))

        Fxi = Cm1 * d - Cm2 * v_xi - torch.sign(v_xi) * Cm3

        alpha_r = (-v_eta + lr * omega) / v_xi
        alpha_f = delta - (v_eta + lf * omega) / v_xi
        Fr_eta = Cr * alpha_r
        Ff_eta = Cf * alpha_f

        idx_tire_model = torch.where(torch.abs(v_xi) > 0.1, 1.0, 0.0)
        Fr_eta = Fr_eta*idx_tire_model
        Ff_eta = Ff_eta*idx_tire_model

        v_xid = 1 / m * (Fxi + Fxi * torch.cos(delta) - Ff_eta * torch.sin(delta) + m * v_eta * omega)
        v_etad = 1 / m * (Fr_eta + Fxi * torch.sin(delta) + Ff_eta * torch.cos(delta) - m * v_xi * omega)
        omega_d = 1 / Jz * (Ff_eta * lf * torch.cos(delta) + Fxi * lf * torch.sin(delta) - Fr_eta * lr)

        if x.dim() == 1:
            x_deriv = torch.hstack((v_xid, v_etad, omega_d))
        else:
            x_deriv = torch.vstack((v_xid, v_etad, omega_d)).T
        return x_deriv

    def f(self, x, u):
        return RK4_step(self.f_deriv, x, u, self.ts)


class nonlinearCar_deriv(nonlinearCar_RK4):
    def __init__(self, nu, nx, ny, ts, parmTune=False):
        super(nonlinearCar_deriv, self).__init__(nu=nu, nx=nx, ny=ny, ts=None, parmTune=parmTune)

    def f(self, x, u):
        return self.f_deriv(x, u)


class LPV_car(general_nonlinear_system):
    def __init__(self, nu, nx, ny):
        P = torch.tensor([2.93, 0.0796, 0.168, 0.163, 41.796, 2.0152, 0.4328, 41.7372, 29.4662], dtype=torch.float)
        super(LPV_car, self).__init__(nu=nu, nx=nx, ny=ny, Ts=None, parm=P, parmTune=False)

    def f(self, x, u):
        # in:           | out:
        #  - x (Nd, Nx) |  - x_dot (Nd, Nx)
        #  - u (Nd, Nu) |

        if x.dim() == 1:
            v_xi = x[0]
            v_eta = x[1]
            omega = x[2]
        else:
            v_xi = x[:, 0]
            v_eta = x[:, 1]
            omega = x[:, 2]
        if u.dim() == 1:
            delta = u[0]
            d = u[1]
        else:
            delta = u[:, 0]
            d = u[:, 1]

        Parameters = self.P.clone()
        m = torch.take(Parameters, torch.tensor([0]))
        Jz = torch.take(Parameters, torch.tensor([1]))
        lr = torch.take(Parameters, torch.tensor([2]))
        lf = torch.take(Parameters, torch.tensor([3]))
        Cm1 = torch.take(Parameters, torch.tensor([4]))
        Cm2 = torch.take(Parameters, torch.tensor([5]))
        Cm3 = torch.take(Parameters, torch.tensor([6]))
        Cr = torch.take(Parameters, torch.tensor([7]))
        Cf = torch.take(Parameters, torch.tensor([8]))

        alpha_f = delta - (v_eta + lf * omega) / v_xi
        Ff_eta = Cf * alpha_f
        idx_tire_model = torch.where(torch.abs(v_xi) > 0.1, 1.0, 0.0)
        F_feta = Ff_eta*idx_tire_model

        v_xid = (-Cm2 * (1 + torch.cos(delta))/m * v_xi + Cm1 * (1 + torch.cos(delta)) / m * d -
                 Cm3 * (1 + torch.cos(delta)) / m - F_feta * torch.sin(delta) / m + v_eta * omega)

        v_xi_mod = v_xi + 0.001
        v_etad = (-(Cf + Cr) / (m * v_xi_mod) * v_eta + (-v_xi_mod - (lf * Cf - lr*Cr) / (m * v_xi_mod)) * omega +
                  Cf / m * delta)
        omega_d = ((lr*Cr - lf*Cf) / (Jz * v_xi_mod) * v_eta - (lr**2 * Cr + lf**2 * Cf) / (Jz * v_xi_mod) * omega +
                   lf * Cf / Jz * delta)

        if x.dim() == 1:
            x_deriv = torch.hstack((v_xid, v_etad, omega_d))
        else:
            x_deriv = torch.vstack((v_xid, v_etad, omega_d)).T
        return x_deriv

    def h(self, x, u):
        return x


if __name__ == '__main__':

    fp_car = nonlinearCar(nu=2, nx=3, ny=3, ts=0.025, parmTune=False)

    x0 = torch.ones(fp_car.Nx)
    u0 = torch.zeros(fp_car.Nu)
    fp_out_zero = fp_car.f(x0, u0)
    A = torch.autograd.functional.jacobian(fp_car.f, (x0, u0))
    print(A[0])

