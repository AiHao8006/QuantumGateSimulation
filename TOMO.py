import numpy as np
import itertools
import multiprocessing as mp
from QuantumOptics import QuantumOptics


class Tomo(QuantumOptics):      # physical_process：输入初态，输出密度矩阵
    def __init__(self, physical_process, dim=2, mode_num=2, saving=False,
                 calculating_in_subspace=False, dim_tot=None, mode_num_added=None):
        super(Tomo, self).__init__(dim=dim, mode_num=mode_num)

        self.rho_dim = dim ** mode_num
        self.saving = saving
        self.physical_process = physical_process

        self.calculating_in_subspace = calculating_in_subspace
        self.dim_tot = dim_tot
        self.mode_num_added = mode_num_added

        self.lambda_mat = None
        self.chi = None
        self.chi_values = None
        self.chi_vectors = None
        self.U0 = None

        # generating beta matrix, which has 3 delta-function terms for any rho_dim or mode_num
        self.beta = np.zeros((self.rho_dim ** 4, self.rho_dim ** 4))
        for b1, b2, b3, a1 in itertools.product(range(self.rho_dim), range(self.rho_dim),
                                                range(self.rho_dim), range(self.rho_dim)):
            a2 = b1
            a3 = b2
            a4 = a1
            b4 = b3

            m = a1 * self.rho_dim + b1
            j = a2 * self.rho_dim + b2
            n = a3 * self.rho_dim + b3
            k = a4 * self.rho_dim + b4

            p = m * self.rho_dim ** 2 + n
            q = j * self.rho_dim ** 2 + k

            self.beta[q, p] = 1

        if self.saving:
            np.save('tomo_saved/beta.npy', self.beta)

    def physical_process_ab_rhof(self, a, b):
        psi_a = self.state(a, by='matrix element')
        psi_b = self.state(b, by='matrix element')
        psi_p = (psi_a + psi_b) / np.sqrt(2)
        psi_m = (psi_a + 1j * psi_b) / np.sqrt(2)

        if self.calculating_in_subspace:
            rho_a_final = self.process_2QubitSpaceRedefine(self.physical_process, psi_a,
                                                           self.dim_tot, self.mode_num_added)
            rho_b_final = self.process_2QubitSpaceRedefine(self.physical_process, psi_b,
                                                           self.dim_tot, self.mode_num_added)
            rho_p_final = self.process_2QubitSpaceRedefine(self.physical_process, psi_p,
                                                           self.dim_tot, self.mode_num_added)
            rho_m_final = self.process_2QubitSpaceRedefine(self.physical_process, psi_m,
                                                           self.dim_tot, self.mode_num_added)
        else:
            rho_a_final = self.physical_process(psi_a)
            rho_b_final = self.physical_process(psi_b)
            rho_p_final = self.physical_process(psi_p)
            rho_m_final = self.physical_process(psi_m)

        rho_final = rho_p_final + 1j * rho_m_final - (1 + 1j) / 2 * rho_a_final - (1 + 1j) / 2 * rho_b_final
        return rho_final

    def physical_process_ab_lambda_mat(self, a, b):
        lambda_mat = np.zeros((self.rho_dim ** 2, self.rho_dim ** 2), dtype=complex)

        rho_final = self.physical_process_ab_rhof(a, b)

        j = a * self.rho_dim + b
        lambda_mat[j, :] = rho_final.reshape(1, -1)

        # print("a,b=", bin(a)[2:], bin(b)[2:])
        # print("rho_final=\n", np.around(rho_final, 2))
        # print("angle= ", np.around(np.angle(rho_final.reshape(-1)[np.argmax(abs(rho_final))])/np.pi, 4), " pi")
        # print("-"*30, "\n\n")

        return lambda_mat

    def measure(self):
        lambda_mat = np.zeros((self.rho_dim ** 2, self.rho_dim ** 2), dtype=complex)
        for a, b in itertools.product(range(self.rho_dim), range(self.rho_dim)):
            lambda_mat += self.physical_process_ab_lambda_mat(a, b)

        self.lambda_mat = lambda_mat
        if self.saving:
            np.save('tomo_saved/lambda.npy', lambda_mat)

    def measure_mp(self):
        my_pool = mp.Pool(6)
        memory_list = []
        for a, b in itertools.product(range(self.rho_dim), range(self.rho_dim)):
            result = my_pool.apply_async(self.physical_process_ab_lambda_mat, args=(a, b))
            memory_list.append(result)
        my_pool.close()
        my_pool.join()

        lambda_mat = np.zeros((self.rho_dim ** 2, self.rho_dim ** 2), dtype=complex)
        for r in memory_list:
            lambda_mat += r.get()

        self.lambda_mat = lambda_mat
        if self.saving:
            np.save('tomo_saved/lambda.npy', lambda_mat)

    def chi_mat(self):
        lambda_vec = self.lambda_mat.reshape(-1, 1)
        self.chi = (np.linalg.pinv(self.beta) @ lambda_vec).reshape(self.rho_dim ** 2, self.rho_dim ** 2)

        if self.saving:
            np.save('tomo_saved/chi.npy', self.chi)

    def fidelity(self, U0):     # dim_space = None 意味着在全空间计算
        self.chi_values, self.chi_vectors = np.linalg.eig(self.chi)

        sum_M_Mdag = np.zeros((self.rho_dim, self.rho_dim), dtype=complex)
        sum_Tr_M2 = 0.

        for i in range(self.rho_dim ** 2):
            Ei = np.zeros((self.rho_dim, self.rho_dim), dtype=complex)
            for a, b in itertools.product(range(self.rho_dim), range(self.rho_dim)):
                Ej_tilde = np.matmul(self.state(a, by='matrix element'),
                                     self.state(b, by='matrix element').transpose().conj())

                j = a * self.rho_dim + b
                Ei += Ej_tilde * self.chi_vectors[j, i]
            Ei *= np.sqrt(self.chi_values[i])

            Mi = U0.transpose().conj() @ Ei
            sum_M_Mdag += Mi @ (Mi.transpose().conj())
            sum_Tr_M2 += abs(np.trace(Mi)) ** 2
        fidelity = (np.trace(sum_M_Mdag) + sum_Tr_M2) / self.rho_dim / (self.rho_dim + 1)
        # print('\nFidelity:', round(abs(fidelity), 5))       # final print

        return fidelity

    def generat_U0(self, gate_name='iSWAP-subspace'):
        if gate_name == 'iSWAP':
            # generating U0 for iSWAP gate
            U0 = np.zeros((self.rho_dim, self.rho_dim), dtype=complex)
            for i, j, k in itertools.product(range(self.dim), range(self.dim), range(self.dim)):
                if [i, j, k] == [0, 1, 0]:
                    U0 += np.matmul(self.state([0, 1, 0]), self.state([0, 0, 1]).transpose().conj())
                elif [i, j, k] == [0, 0, 1]:
                    U0 += np.matmul(self.state([0, 0, 1]), self.state([0, 1, 0]).transpose().conj())
                else:
                    U0 += np.matmul(self.state([i, j, k]), self.state([i, j, k]).transpose().conj())
            self.U0 = U0
            return U0

        if gate_name == 'iSWAP-subspace':
            U0 = np.array([[1, 0, 0, 0],
                           [0, 0, 1j, 0],
                           [0, 1j, 0, 0],
                           [0, 0, 0, 1]], dtype=complex)
            self.U0 = U0
            return U0

    def process_2QubitSpaceRedefine(self, process_tot, psi0_sub, dim_tot, mode_num_added=None):
        psi0_sub = psi0_sub.reshape(2, 2)
        psi0_tot = np.zeros([dim_tot, dim_tot], dtype=complex)
        psi0_tot[:2, :2] = psi0_sub
        psi0_tot = psi0_tot.reshape(dim_tot ** 2, 1)
        if mode_num_added is not None:
            ground_state_added = np.zeros([dim_tot, 1], dtype=complex)
            ground_state_added[0] = 1
            psi0_tot = self.krons([ground_state_added] * mode_num_added + [psi0_tot])

        rho_f_tot = process_tot(psi0_tot)

        if mode_num_added is not None:
            rho_f_tot = self.partial_trace(rho_f_tot, dim_tot ** mode_num_added, dim_tot ** 2, tracing_over='former')
        rho_f_tot = rho_f_tot.reshape([dim_tot, dim_tot, dim_tot, dim_tot])
        rho_f_sub = rho_f_tot[:2, :2, :2, :2]
        rho_f_sub = rho_f_sub.reshape(2 ** 2, 2 ** 2)
        return rho_f_sub
