import numpy as np
from TOMO import Tomo
from Solve_SE import solve_SE
from PulseShapes import PulseShapeBox

solver = solve_SE()
solver.dressed_button = True
solver.add_modulations.append([PulseShapeBox.square, [6400-5790, 0.285, 0.715], solver.adag_a[0]])

# psi_f = solver.solve_ode(solver.state([0, 1, 1]), show_plot_button=True)
# print(psi_f)

theta1, theta2 = solver.phase_calibration()
print(theta1, theta2)
print(np.exp(1j*(theta1 + theta2)))
