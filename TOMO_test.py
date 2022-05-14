import numpy as np
from TOMO import Tomo

U1 = np.array([[0, 1], [1, 0]])
# U2d = np.kron(U0, U0)
U2d = np.array([[1, 0, 0, 0],
                [0, 0, 1j, 0],
                [0, 1j, 0, 0],
                [0, 0, 0, 1]], dtype=complex)
U3d = np.kron(U1, U2d)


def U3d_process(psi):
    rho = psi @ psi.conj().T
    return U3d @ rho @ U3d.transpose().conj()


tomo = Tomo(U3d_process, calculating_in_subspace=True, dim_tot=2, mode_num_added=1)
tomo.generat_U0()

if __name__ == '__main__':
    tomo.measure_mp()
    tomo.chi_mat()
    tomo.fidelity(tomo.U0)
