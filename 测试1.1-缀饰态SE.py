import numpy as np
from Solve_SE import solve_SE
from PulseShapes import PulseShapeBox

solver = solve_SE()
solver.add_modulations.append([PulseShapeBox.square, [6400-5790, 0.2, 0.63], solver.adag_a[0]])


"""
这是在测试 inverse_dressed
values, U = np.linalg.eigh(solver.hamiltonian)
psi_0 = U[:, 1].reshape(-1, 1)
print(np.around(np.real(psi_0), 2))
#
psi_1 = solver.dressed_state(solver.hamiltonian, solver.state([0,0,1]),inverse_calculate=True)
print(np.around(np.real(psi_1), 2))

plot_tf = solver.solve_ode(psi_1, show_plot_button=True)
print(plot_tf)
"""

psi_0 = solver.state([0,1,0])
solver.solve_ode(psi_0, show_plot_button=True)
