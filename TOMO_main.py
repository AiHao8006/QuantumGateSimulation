import numpy as np
import pandas as pd
from TOMO import Tomo
from Solve_SE import solve_SE
from PulseShapes import PulseShapeBox

dim = 3
other_info = "testing CZ 1st time"

solver = solve_SE(omega=[5560, 4800, 4799.9-240], gate_type='CZ', dim=dim)
solver.add_modulations.append([PulseShapeBox.square, [6400-5560, 0.2, 0.683], solver.adag_a[0]])

theta1, theta2 = solver.phase_calibration()
print("完成相位标定：theta1={}, theta2={}".format(np.around(theta1, 2), np.around(theta2, 2)))
# theta1 = theta2 = 0


def precess(psi):       # 这里相当于 physical_process（可以用主方程直接替换），当然输入输出的维度也是在solver空间中的
    psi_tf = solver.solve_ode(psi)
    psi_tf = solver.phase_gate(theta=(-theta1), mode_id=1) @ solver.phase_gate(theta=(-theta2), mode_id=2) @ psi_tf
    rho_tf = psi_tf @ psi_tf.T.conj()
    return rho_tf


tomo = Tomo(precess, dim=2, calculating_in_subspace=True, dim_tot=dim, mode_num_added=1, print_or_not=True)
tomo.generat_U0(gate_name='CZ-subspace')

if __name__ == '__main__':
    tomo.measure_mp()
    tomo.chi_mat()
    Fid = tomo.fidelity(tomo.U0)

    # Fid = 0

    df = pd.DataFrame(columns=['Gate type',     'omega',        'alpha',        'g',        'Pulse on', 'Pulse shape',
                               'Pulse max',     't_i',      't_f',      'Fidelity',             'Other info'],
                      data=[['CZ',            solver.omega,   solver.alpha,   solver.g,   'omega0',   'square',
                            6400-5560,          0.2,      0.683,      np.real(Fid),           other_info]])

    df_all = pd.read_excel(r"Data_saved_CZ.xlsx", index_col=False)
    df_all = pd.concat([df_all, df])
    df_all.to_excel("Data_saved_CZ.xlsx", index=False)
