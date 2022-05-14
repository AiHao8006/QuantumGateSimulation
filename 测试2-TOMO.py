import numpy as np
from TOMO import Tomo
from Solve_SE import solve_SE
from PulseShapes import PulseShapeBox

solver = solve_SE()
solver.add_modulations.append([PulseShapeBox.square, [6400-5790, 0.2, 0.63], solver.adag_a[0]])

theta1 = theta2 = 0


def precess(psi):
    psi_tf = solver.solve_ode(psi)
    psi_tf = solver.phase_gate(theta=theta1, mode_id=1) @ solver.phase_gate(theta=theta2, mode_id=2) @ psi_tf
    print(np.around(psi, 2), '\n', np.around(psi_tf, 2), '\n')
    rho_tf = psi_tf @ psi_tf.T.conj()
    return rho_tf


tomo = Tomo(precess, saving=True, calculating_in_subspace=True, dim_tot=2, mode_num_added=1)
tomo.generat_U0()

if __name__ == '__main__':
    tomo.physical_process_ab_lambda_mat(0, 1)
