"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np

# Problem Parameters:
# n0_num  = 1
# B10_num = 1
# c_num   = 5
# mu_num  = 1
# e_num   = 1
# mi_num  = 1
# me_num  = 1 / 10
# L1_num  = 2 / 0.9 * pi
# k1_num  = 2 * pi / L1_num

# ('Eigenvalue   = ', -3.3840660338578e-16 - 0.57232851937906*I)
# (delta_u2_e, ' = ', -1.2414728973845785e-16 + 0.21327009745028128*I)
# (delta_u3_e, ' = ', 0.21327009745028208 - 4.517274353310467e-17*I)
# (delta_u2_i, ' = ', -2.7044279698172115e-17 + 0.5272181185265511*I)
# (delta_u3_i, ' = ', 0.5272181185265513)
# (delta_B2, ' = ', 1.6097225887857537e-16 - 0.35456653155176815*I)
# (delta_B3, ' = ', -0.35456653155176954 + 1.5767846538405862e-16*I)
# (delta_E2, ' = ', -0.2254761533604373 + 2.902543682224072e-16*I)
# (delta_E3, ' = ', -4.8385010106448195e-17 + 0.2254761533604362*I)

def initialize_f(q1, q2, v1, v2, v3, params):
    
    m = params.mass
    k = params.boltzmann_constant

    n_b = params.density_background
    T_b = params.temperature_background

    k = params.boltzmann_constant

    v1_bulk   = 0

    # Assigning separate bulk velocities
    v2_bulk_i =   params.amplitude * -2.7044279698172115e-17 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0.5272181185265511 * af.sin(params.k_q1 * q1)

    v2_bulk_e =   params.amplitude * -1.2414728973845785e-16 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0.21327009745028128 * af.sin(params.k_q1 * q1)
    
    v3_bulk_i =   params.amplitude * 0.5272181185265513 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0  * af.sin(params.k_q1 * q1)

    v3_bulk_e =   params.amplitude * 0.21327009745028208 * af.cos(params.k_q1 * q1) \
                - params.amplitude * 0 * af.sin(params.k_q1 * q1)

    n = n_b + 0 * q1**0

    f_e = n * (m[0, 0] / (2 * np.pi * k * T_b)) \
            * af.exp(-m[0, 0] * (v2[:, 0] - v2_bulk_e)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 0] * (v3[:, 0] - v3_bulk_e)**2 / (2 * k * T_b))

    f_i = n * (m[0, 1] / (2 * np.pi * k * T_b)) \
            * af.exp(-m[0, 1] * (v2[:, 1] - v2_bulk_i)**2 / (2 * k * T_b)) \
            * af.exp(-m[0, 1] * (v3[:, 1] - v3_bulk_i)**2 / (2 * k * T_b))

    f = af.join(1, f_e, f_i)

    af.eval(f)
    return (f)

def initialize_E(q1, q2, params):

    E1 = 0 * q1**0
    
    E2 =   params.amplitude * -0.2254761533604373 * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0  * af.sin(params.k_q1 * q1)
    
    E3 =   params.amplitude * 0   * af.cos(params.k_q1 * q1) \
         - params.amplitude * 0.2254761533604362 * af.sin(params.k_q1 * q1)

    af.eval(E1, E2, E3)
    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    dt = params.dt
    B1 = params.B0 * q1**0

    omega = -3.3840660338578e-16 - 0.57232851937906 * 1j

    B2 = (params.amplitude * (1.6097225887857537e-16 - 0.35456653155176815*1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B3 = (params.amplitude * (-0.35456653155176954 + 1.5767846538405862e-16 * 1j) * \
          np.exp(  1j * params.k_q1 * np.array(q1)
                 + omega * dt / 2
                )).real

    B2 = af.moddims(af.to_array(B2), 1, 1, q1.shape[2], q1.shape[3])
    B3 = af.moddims(af.to_array(B3), 1, 1, q1.shape[2], q1.shape[3])

    af.eval(B1, B2, B3)
    return(B1, B2, B3)
