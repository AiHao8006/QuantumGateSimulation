import numpy as np
from TOMO import Tomo
from Solve_SE import solve_SE
from PulseShapes import PulseShapeBox

dim = 3
solver = solve_SE(omega=[5560, 4800, 4799.9-240], gate_type='CZ', dim=dim)
solver.add_modulations.append([PulseShapeBox.square, [6400-5560, 0.2, 0.683], solver.adag_a[0]])

psi_f = solver.solve_ode(solver.state([0, 1, 1]), show_plot_button=True)
print(np.around(abs(psi_f), 5))

theta1, theta2 = solver.phase_calibration()
print(theta1, theta2)
print(np.exp(1j*(theta1 + theta2)))
