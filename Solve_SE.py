from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

from QuantumOptics import QuantumOptics
from PulseShapes import PulseShapeBox

PulseShapeBox = PulseShapeBox()


class solve_SE(QuantumOptics):
    def __init__(self, gate_type='iSWAP', modulate_type='omega', omega=None, alpha=None, g=None, dim=None):
        if dim is None:
            if gate_type == 'iSWAP':
                self.dim = 2
            else:
                self.dim = 3
        else:
            self.dim = dim
        super().__init__(mode_num=3, dim=self.dim)

        if g is None:
            g = [10., 100., 100.]
        if alpha is None:
            alpha = [-130., -240., -240.]
        if omega is None:
            omega = [5790., 4800., 4799.9]
        self.omega = omega
        self.alpha = alpha
        self.g = g
        self.gate_type = gate_type
        self.modulation_type = modulate_type

        self.add_modulations = []
        # 示例：self.add_modulations.append([PulseShapeBox.square, [100, 0, 1], self.adag_a[0]])

        self.dressed_button = True
        self.saved_psi_t_solved = None
        self.t_span = np.arange(0, 1, 0.001)

        self.generate_hamiltonian()

    def generate_hamiltonian(self):
        self.hamiltonian = self.omega[0] * self.adag_a[0] \
                           + self.alpha[0] / 2 * self.adag_a[0] @ (self.adag_a[0] - self.I) \
                           + self.omega[1] * self.adag_a[1] \
                           + self.alpha[1] / 2 * self.adag_a[1] @ (self.adag_a[1] - self.I) \
                           + self.omega[2] * self.adag_a[2] \
                           + self.alpha[2] / 2 * self.adag_a[2] @ (self.adag_a[2] - self.I) \
                           + self.g[0] * (self.adag[1] @ self.a[2] + self.a[1] @ self.adag[2]) \
                           + self.g[1] * (self.adag[0] @ self.a[2] + self.a[0] @ self.adag[2]) \
                           + self.g[2] * (self.adag[0] @ self.a[1] + self.a[0] @ self.adag[1])
        # 闲置参数下的哈密顿量

    def modulation_added(self, t, pulse_func=PulseShapeBox.square, pulse_parameters=None, pulse_mat=None):
        if pulse_parameters is None:
            pulse_parameters = [100, 0., 1.]
        if pulse_mat is None:
            pulse_mat = self.adag_a[0]
        return pulse_mat * pulse_func(t=t, pulse_parameters=pulse_parameters)

    def solve_ode(self, psi0, show_plot_button=False):
        # 示例：psi0=self.state[0,1,0], t_span=np.arange(0,1,0.001)
        t_span = self.t_span

        if self.dressed_button:
            psi0 = self.dressed_state(self.hamiltonian, psi0, inverse_calculate=True)

        def schrodinger_eq(psi_list, tau):
            H = self.hamiltonian
            for modu_info in self.add_modulations:
                added_H = self.modulation_added(tau, pulse_func=modu_info[0], pulse_parameters=modu_info[1],
                                                pulse_mat=modu_info[2])
                H = H + added_H

            dpsi_dt = H @ self.list_to_psi(psi_list) / 1j
            return self.psi_to_list(dpsi_dt)

        psi_t_list_solved = np.array(odeint(schrodinger_eq, self.psi_to_list(psi0), t_span))

        # 画图与存储
        self.saved_psi_t_solved = np.zeros((psi_t_list_solved.shape[0], psi0.shape[0]), dtype=complex)

        if not self.dressed_button:
            for t in range(0, psi_t_list_solved.shape[0]):
                self.saved_psi_t_solved[t, :] = (self.list_to_psi(psi_t_list_solved[t, :])).reshape(-1)
        else:
            H_dress = self.hamiltonian
            for modu_info_t in self.add_modulations:
                H_dress = H_dress + self.modulation_added(0, pulse_func=modu_info_t[0],
                                                          pulse_parameters=modu_info_t[1], pulse_mat=modu_info_t[2])
                # 基于t=0的哈密顿量做缀饰态
            for t in range(0, psi_t_list_solved.shape[0]):
                self.saved_psi_t_solved[t, :] = self.dressed_state(H_dress, self.list_to_psi(psi_t_list_solved[t, :])) \
                    .reshape(-1)

        plot_list = np.real(self.saved_psi_t_solved * self.saved_psi_t_solved.conj())
        if show_plot_button:
            # print(np.around(plot_list[-1, :], 5))
            plt.plot(self.t_span, np.real(self.saved_psi_t_solved * self.saved_psi_t_solved.conj()))
            plt.show()

        return self.saved_psi_t_solved[-1, :].reshape(-1, 1)

    def phase_calibration(self):
        theta1 = theta2 = 0

        if self.gate_type == 'iSWAP':
            psi_f = self.solve_ode(self.state([0, 1, 0]))
            theta2 = np.angle(psi_f.reshape(-1)[np.argmax(abs(psi_f))]) - np.angle([1j]).item()         # 解1得2
            psi_f = self.solve_ode(self.state([0, 0, 1]))
            theta1 = np.angle(psi_f.reshape(-1)[np.argmax(abs(psi_f))]) - np.angle([1j]).item()

        if self.gate_type == 'CZ':
            psi_f = self.solve_ode(self.state([0, 1, 0]))
            theta1 = np.angle(psi_f.reshape(-1)[np.argmax(abs(psi_f))])         # 解1得2
            psi_f = self.solve_ode(self.state([0, 0, 1]))
            theta2 = np.angle(psi_f.reshape(-1)[np.argmax(abs(psi_f))])

        return theta1, theta2
