"""
Functions which are used in assigning the I.C's to
the system.
"""

import arrayfire as af
import numpy as np
from petsc4py import PETSc

import domain

def initialize_f(q1, q2, p1, p2, p3, params):
   
    PETSc.Sys.Print("Initializing f")
    k = params.boltzmann_constant
    
    params.mu          = 0.*q1 + params.initial_mu
    params.T           = 0.*q1 + params.initial_temperature
    params.vel_drift_x = 0.*q1
    params.vel_drift_y = 0.*q1

    params.mu_ee       = params.mu.copy()
    params.T_ee        = params.T.copy()
    params.vel_drift_x = 0.*q1 + 0e-3
    params.vel_drift_y = 0.*q1 + 0e-3

    params.p_x, params.p_y = params.get_p_x_and_p_y(p1, p2)
    params.E_band   = params.band_energy(p1, p2)
    params.vel_band = params.band_velocity(p1, p2)

    # Evaluating velocity space resolution for each species:
    dp1 = []; dp2 = []; dp3 = []
    N_p1 = domain.N_p1; N_p2 = domain.N_p2; N_p3 = domain.N_p3
    p1_start = domain.p1_start; p1_end = domain.p1_end
    p2_start = domain.p2_start; p2_end = domain.p2_end
    p3_start = domain.p3_start; p3_end = domain.p3_end

    N_species = len(params.mass)
    for i in range(N_species):
        dp1.append((p1_end[i] - p1_start[i]) / N_p1)
        dp2.append((p2_end[i] - p2_start[i]) / N_p2)
        dp3.append((p3_end[i] - p3_start[i]) / N_p3)


    theta = af.atan(params.p_y / params.p_x)
    p_f   = params.fermi_momentum_magnitude(theta)
    
    if (params.p_space_grid == 'cartesian'):
        dp_x = dp1[0]; dp_y = dp2[0]; dp_z = dp3[0]
        params.integral_measure = \
          (4./(2.*np.pi*params.h_bar)**2) * dp_z * dp_y * dp_x

    elif (params.p_space_grid == 'polar2D'):
     	# In polar2D coordinates, p1 = radius and p2 = theta
        # Integral : \int delta(r - r_F) F(r, theta) r dr dtheta
        r = p1; theta = p2
        dp_r = dp1[0]; dp_theta = dp2[0]

        if (params.zero_temperature):
            # Assumption : F(r, theta) = delta(r-r_F)*F(theta)
            params.integral_measure = \
              (4./(2.*np.pi*params.h_bar)**2) * p_f * dp_theta

        else:
            params.integral_measure = \
              (4./(2.*np.pi*params.h_bar)**2) * r * dp_r * dp_theta
            

    else : 
        raise NotImplementedError('Unsupported coordinate system in p_space')



    f = (1./(af.exp( (params.E_band - params.vel_drift_x*params.p_x
                                    - params.vel_drift_y*params.p_y
                                    - params.mu
                    )/(k*params.T) 
                  ) + 1.
           ))

    af.eval(f)
    return(f)


def initialize_E(q1, q2, params):
    
    E1 = 0.*q1
    E2 = 0.*q1
    E3 = 0.*q1

    return(E1, E2, E3)

def initialize_B(q1, q2, params):

    B1 = 0.*q1
    B2 = 0.*q1
    B3 = 0.*q1

    return(B1, B2, B3)
