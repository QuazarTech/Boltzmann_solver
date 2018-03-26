"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk = 0
    v2_bulk = params.v0 * 0.1 * af.sin(2 * np.pi * q1/ params.L_x)
    v3_bulk = params.v0 * 0.1 * af.cos(2 * np.pi * q1/ params.L_x)

    n = n_b + 0 * q1**0

    f = n * (m / (2 * np.pi * k * T_b))**(3 / 2) \
          * af.exp(-m * (v1 - v1_bulk)**2 / (2 * k * T_b)) \
          * af.exp(-m * (v2 - v2_bulk)**2 / (2 * k * T_b)) \
          * af.exp(-m * (v3 - v3_bulk)**2 / (2 * k * T_b))

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    E2 = 0 * q1**0
    E3 = 0 * q1**0

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = params.B1 * q1**0
    B2 = params.B1 * 0.1 * af.sin(2 * np.pi * q1/ params.L_x)
    B3 = params.B1 * 0.1 * af.cos(2 * np.pi * q1/ params.L_x)

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
