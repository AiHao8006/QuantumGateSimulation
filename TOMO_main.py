import numpy as np
import pandas as pd
from TOMO import Tomo
from Solve_SE import solve_SE
from PulseShapes import PulseShapeBox

dim = 3

solver = solve_SE(dim=dim)
solver.add_modulations.append([PulseShapeBox.square, [6400-5790, 0.285, 0.715], solver.adag_a[0]])

theta1, theta2 = solver.phase_calibration()
print("完成相位标定：theta1={}, theta2={}".format(np.around(theta1, 2), np.around(theta2, 2)))
# theta1 = theta2 = 0


def precess(psi):       # 这里相当于 physical_process（可以用主方程直接替换），当然输入输出的维度也是在solver空间中的
    psi_tf = solver.solve_ode(psi)
    psi_tf = solver.phase_gate(theta=(-theta1), mode_id=1) @ solver.phase_gate(theta=(-theta2), mode_id=2) @ psi_tf
    rho_tf = psi_tf @ psi_tf.T.conj()
    return rho_tf


tomo = Tomo(precess, dim=2, saving=True, calculating_in_subspace=True, dim_tot=dim, mode_num_added=1)
tomo.generat_U0()

if __name__ == '__main__':
    tomo.measure_mp()
    tomo.chi_mat()
    Fid = tomo.fidelity(tomo.U0)

    # Fid = 0

    df = pd.DataFrame(columns=['Gate type',     'omega',        'alpha',        'g',        'Pulse on', 'Pulse shape',
                               'Pulse max',     't_i',      't_f',      'Fidelity',             'Other info'],
                      data=[['iSWAP',            solver.omega,   solver.alpha,   solver.g,   'omega0',   'square',
                            6400-5790,          0.285,      0.715,      np.real(Fid),           None]])

    df_all = pd.read_excel(r"Data_saved.xlsx", index_col=False)
    df_all = pd.concat([df_all, df])
    df_all.to_excel("Data_saved.xlsx", index=False)
