import numpy as np
from Solve_SE import solve_SE
from PulseShapes import PulseShapeBox

solver = solve_SE()
solver.add_modulations.append([PulseShapeBox.square, [6400-5790, 0.5, 0.6], solver.adag_a[0]])

H = solver.hamiltonian
for modu_info in solver.add_modulations:
    H += solver.modulation_added(0.2, pulse_func=modu_info[0], pulse_parameters=modu_info[1],
                                 pulse_mat=modu_info[2])
print(np.real(H).astype(int))

H = solver.hamiltonian
for modu_info in solver.add_modulations:
    H += solver.modulation_added(0.5, pulse_func=modu_info[0], pulse_parameters=modu_info[1],
                                 pulse_mat=modu_info[2])
print(np.real(H).astype(int))
