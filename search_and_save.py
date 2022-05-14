import numpy as np
import pandas as pd
import multiprocessing as mp
from TOMO import Tomo
from Solve_SE import solve_SE
from PulseShapes import PulseShapeBox

dim = 3


def tomo_one_time_t(t):
    solver = solve_SE(dim=dim)
    tomo = Tomo(None, dim=2, saving=True, calculating_in_subspace=True, dim_tot=dim, mode_num_added=1)

    solver.add_modulations = [[PulseShapeBox.square, [6400 - 5790, 0.5 - t, 0.5 + t], solver.adag_a[0]]]

    theta1, theta2 = solver.phase_calibration()

    # print("完成相位标定：theta1={}, theta2={}".format(np.around(theta1, 2), np.around(theta2, 2)))
    # theta1 = theta2 = 0

    def precess(psi):  # 这里相当于 physical_process（可以用主方程直接替换），当然输入输出的维度也是在solver空间中的
        psi_tf = solver.solve_ode(psi)
        psi_tf = solver.phase_gate(theta=(-theta1), mode_id=1) @ solver.phase_gate(theta=(-theta2), mode_id=2) @ psi_tf
        rho_tf = psi_tf @ psi_tf.T.conj()
        return rho_tf

    tomo.physical_process = precess
    tomo.generat_U0()

    tomo.measure()
    tomo.chi_mat()
    Fid = tomo.fidelity(tomo.U0)

    print("time: {}~{}; fidelity:{}.".format(np.around(0.5-tau, 4), np.around(0.5+tau, 4), np.around(Fid, 5)))

    # Fid = 0

    df_result = pd.DataFrame(columns=['Gate type', 'omega', 'alpha', 'g', 'Pulse on', 'Pulse shape',
                                        'Pulse max', 't_i', 't_f', 'Fidelity', 'Other info'],
                            data=[['iSWAP', solver.omega, solver.alpha, solver.g, 'omega0', 'square',
                                        6400 - 5790, 0.5-t, 0.5+t, np.real(Fid), None]])
    return df_result


if __name__ == '__main__':
    p = mp.Pool(6)
    results = []
    for i in range(30):
        tau = 0.2 + i / 1000
        result = p.apply_async(tomo_one_time_t, args=(tau,))
        results.append(result)
    p.close()
    p.join()

    df_all = pd.read_excel("Data_saved.xlsx", index_col=False)
    for r in results:
        df = r.get()
        df_all = pd.concat([df_all, df])
    df_all.to_excel("Data_saved2.xlsx", index=False)
