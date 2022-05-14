import numpy as np
import sys


class QuantumOptics:
    def __init__(self, dim=5, mode_num=3):
        self.dim = dim
        self.mode_num = mode_num

        self.I_single_mode = np.eye(dim)
        self.a_single_mode = np.sqrt(np.diag(np.arange(dim + 1))[1:, :-1])

        self.I = np.eye(dim ** mode_num)

        self.a = np.zeros([mode_num, dim ** mode_num, dim ** mode_num])
        self.adag = np.zeros([mode_num, dim ** mode_num, dim ** mode_num])
        self.adag_a = np.zeros([mode_num, dim ** mode_num, dim ** mode_num])
        self.a_adag = np.zeros([mode_num, dim ** mode_num, dim ** mode_num])
        for i in range(mode_num):
            krons_list = [self.I_single_mode] * mode_num
            krons_list[i] = self.a_single_mode
            self.a[i] = self.krons(krons_list=krons_list)
            self.adag[i] = self.a[i].transpose().conj()
            self.adag_a[i] = np.matmul(self.adag[i], self.a[i])
            self.a_adag[i] = np.matmul(self.a[i], self.adag[i])

        self.psi = None
        self.rho = None

        self.hamiltonian = self.I

    def state(self, working_list, by='excitation', saving=False):
        if by == 'matrix element':  # 此时 working_list 只输入一个i
            self.psi = np.zeros((self.dim ** self.mode_num, 1))
            self.psi[working_list, 0] = 1
            self.rho = np.matmul(self.psi, self.psi.transpose().conj())
            return self.psi

        if by == 'excitation':
            amplitudes_list = []
            for i in range(self.mode_num):
                amplitudes_1_mode = [0] * self.dim
                amplitudes_1_mode[working_list[i]] = 1
                amplitudes_list.append(amplitudes_1_mode)

            for i in range(len(amplitudes_list)):
                amplitudes_list[i] = np.array(amplitudes_list[i])
            psi = self.krons(amplitudes_list).reshape(-1, 1)
            if saving:
                self.psi = psi
                self.rho = np.matmul(self.psi, self.psi.transpose().conj())
            return psi

        elif by == 'amplitudes':  # by = 'amplitudes'
            amplitudes_list = working_list

            for i in range(len(amplitudes_list)):
                amplitudes_list[i] = np.array(amplitudes_list[i])
            psi = self.krons(amplitudes_list).reshape(-1, 1)
            if saving:
                self.psi = psi
                self.rho = np.matmul(self.psi, self.psi.transpose().conj())
            return psi

    def lindblad(self, gamma, Nth, rho, mode_id=None):  # only term II
        if mode_id is None:
            mode_id = range(self.mode_num)

        termII = 0
        for id in mode_id:
            termII += gamma / 2 * (Nth + 1) * (2 * self.a[id] @ rho @ self.adag[id]
                                               - self.adag_a[id] @ rho - rho @ self.adag_a[id]) \
                      + gamma / 2 * Nth * (2 * self.adag[id] @ rho @ self.a[id]
                                           - self.a_adag[id] @ rho - rho @ self.a_adag[id])
        return termII

    def eff_space_rho(self, hamiltonian, rho_bare, eig_values_list):  # eig_values_list: 关注的能级编号，按照从低到高从0开始排列即可
        V_list, U = np.linalg.eigh(hamiltonian)
        # V = np.diag(V_list)

        P__list = np.zeros(self.dim ** self.mode_num)
        for i in eig_values_list:
            P__list[i] = 1
        P_ = np.diag(P__list)

        P = U @ P_ @ U.conj().transpose()

        return P.conj().transpose() @ rho_bare @ P

    def phase_gate(self, theta, mode_id=0):
        single_phase_gate = np.zeros_like(self.I_single_mode, dtype=complex)
        single_phase_gate[:2, :2] = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
        krons_list = [self.I_single_mode] * self.mode_num
        krons_list[mode_id] = single_phase_gate
        return self.krons(krons_list=krons_list)

    @staticmethod
    def commutator(mat0, mat1):
        return np.matmul(mat0, mat1) - np.matmul(mat1, mat0)

    @staticmethod
    def krons(krons_list):
        output = 1
        for i in range(0, len(krons_list)):
            output = np.kron(output, krons_list[i])
        return output

    @staticmethod
    def list_to_psi(psi_list):
        dim = int(len(psi_list) / 2)  # int是直接舍去小数部分
        if dim * 2 != len(psi_list): sys.exit("维度错误！")
        return (np.array(psi_list, dtype=complex)[:dim] + 1j * np.array(psi_list, dtype=complex)[dim:]).reshape(dim, 1)

    @staticmethod
    def psi_to_list(psi):
        real = np.real(psi).reshape(psi.size)
        imag = np.imag(psi).reshape(psi.size)
        return np.concatenate((real, imag))

    @staticmethod
    def complex_mat_to_list(c_mat):  # 全部为np格式
        real = np.real(c_mat).reshape(c_mat.size)
        imag = np.imag(c_mat).reshape(c_mat.size)
        return np.concatenate((real, imag))

    @staticmethod
    def list_to_complex_mat(working_list, mat_type='square', mat_shape=None):  # 输入输出全部为np格式
        if mat_type == 'square':
            mat_dim = int(np.around(np.sqrt(len(working_list) / 2)))
            mat_shape = [mat_dim, mat_dim]

        real = working_list[:(mat_shape[0] * mat_shape[1])].reshape(mat_shape[0], mat_shape[1])
        imag = working_list[(mat_shape[0] * mat_shape[1]):(mat_shape[0] * mat_shape[1] * 2)] \
            .reshape(mat_shape[0], mat_shape[1])
        return real + 1j * imag

    @staticmethod
    def projective_prob_state(state_a, state_b):
        p = state_a.conj().T @ state_b
        return (p * p.conj()).item()

    @staticmethod
    def dressed_state(H, psi, inverse_calculate=False):
        values, U = np.linalg.eigh(H)
        # H @ U[:,i] = values[i] * U[:,i] 或 H @ U = diag(values) @ U
        # psi_eigen_base = U.conj().T @ psi

        # psi_dressed = np.zeros_like(psi, dtype=complex)
        # for i in range(psi.size):
        #     max_id = np.argmax(abs(U[:, i]))        # 表示U[:,i]这个新基对应的旧基为 ...1_maxid...
        #     psi_dressed[max_id, 0] = (U[:, i].conj().T @ psi).item()

        U_new = np.zeros_like(U, dtype=complex)
        for i in range(psi.size):
            max_id = np.argmax(abs(U[:, i]))
            U_new[:, max_id] = U[:, i]

        if inverse_calculate:
            U_new = U_new.T.conj()

        psi_new = U_new.conj().T @ psi
        return psi_new

    @staticmethod
    def partial_trace(rho, dim1, dim2, tracing_over='later'):
        rho = rho.reshape(dim1, dim2, dim1, dim2)

        if tracing_over == 'later':
            rho = rho.transpose((0, 2, 1, 3))
        else:
            rho = rho.transpose((1, 3, 0, 2))
            [dim2, dim1] = [dim1, dim2]

        rho = rho.reshape(dim1 ** 2, dim2 ** 2)
        rho = rho @ np.eye(dim2).reshape(dim2 ** 2, 1)
        rho = rho.reshape(dim1, dim1)

        return rho

