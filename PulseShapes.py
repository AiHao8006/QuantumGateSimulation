import numpy as np


class PulseShapeBox:
    def __init__(self):
        pass

    @staticmethod
    def square(t, pulse_parameters):          # pulse_parameters = [value, ti, tf]
        max_value = pulse_parameters[0]
        ti = pulse_parameters[1]
        tf = pulse_parameters[2]

        if ti <= t <= tf:
            return max_value
        else:
            return 0

    @staticmethod
    def sin(t, pulse_parameters):          # pulse_parameters = [value, ti, tf]
        max_value = pulse_parameters[0]
        ti = pulse_parameters[1]
        tf = pulse_parameters[2]

        if ti <= t <= tf:
            return max_value * np.sin(np.pi * (t - ti) / (tf - ti))
        else:
            return 0
