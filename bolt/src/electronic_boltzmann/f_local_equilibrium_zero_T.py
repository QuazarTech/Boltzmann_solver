"""Contains the function which returns the Source/Sink term."""

from petsc4py import PETSc
import numpy as np
import arrayfire as af

from .matrix_inverse import inverse_4x4_matrix
from .matrix_inverse import inverse_3x3_matrix

from bolt.src.utils.integral_over_p import integral_over_p

import domain

@af.broadcast
def f0_defect_constant_T(f, p_x, p_y, p_z, params):
    """
    Return the local equilibrium distribution corresponding to the tau_D
    relaxation time when lattice temperature, T, is set to constant.
    Parameters:
    -----------
    f : Distribution function array
        shape:(N_v, N_s, N_q1, N_q2)
    
    p_x : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    p_y : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    p_z : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)
    
    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function

    """

    mu = params.mu
    T  = params.T

    for n in range(params.collision_nonlinear_iters):

        E_upper = params.E_band
        k       = params.boltzmann_constant

        tmp         = ((E_upper - mu)/(k*T))
        denominator = (k*T**2.*(af.exp(tmp) + 2. + af.exp(-tmp)) )

        # TODO: Multiply with the integral measure dp_x * dp_y
        a00 = integral_over_p(T/denominator, params.integral_measure)

        fermi_dirac = 1./(af.exp( (E_upper - mu)/(k*T) ) + 1.)
        af.eval(fermi_dirac)

        zeroth_moment = f - fermi_dirac

        eqn_mass_conservation = integral_over_p(zeroth_moment,
                                                params.integral_measure
                                               )

        N_g = domain.N_ghost
        error_mass_conservation = af.max(af.abs(eqn_mass_conservation)[0, 0, N_g:-N_g, N_g:-N_g])

        print("    rank = ", params.rank,
	      "||residual_defect|| = ", error_mass_conservation
	     )

        res      = eqn_mass_conservation
        dres_dmu = -a00

        delta_mu = -res/dres_dmu

        mu = mu + delta_mu

        af.eval(mu)

    # Solved for mu. Now store in params
    params.mu = mu

    # Print final residual
    fermi_dirac = 1./(af.exp( (E_upper - mu)/(k*T) ) + 1.)
    af.eval(fermi_dirac)

    zeroth_moment = f - fermi_dirac

    eqn_mass_conservation   = integral_over_p(zeroth_moment,
                                              params.integral_measure
                                             )

    N_g = domain.N_ghost
    error_mass_conservation = af.max(af.abs(eqn_mass_conservation)[0, 0, N_g:-N_g, N_g:-N_g])

    print("    rank = ", params.rank,
	  "||residual_defect|| = ", error_mass_conservation
	 )
    print("    rank = ", params.rank,
          "mu = ", af.mean(params.mu[0, 0, N_g:-N_g, N_g:-N_g]),
          "T = ", af.mean(params.T[0, 0, N_g:-N_g, N_g:-N_g])
         )
    PETSc.Sys.Print("    ------------------")

    return(fermi_dirac)


@af.broadcast
def f0_ee_constant_T(f, p_x, p_y, p_z, params):
    """
    Return the local equilibrium distribution corresponding to the tau_ee
    relaxation time when lattice temperature, T, is set to constant.
    Parameters:
    -----------
    f : Distribution function array
        shape:(N_v, N_s, N_q1, N_q2)
    
    p_x : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    p_y : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    p_z : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)
    
    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function

    """

    # Initial guess
    mu_ee       = params.mu_ee
    T_ee        = params.T_ee
    vel_drift_x = params.vel_drift_x 
    vel_drift_y = params.vel_drift_y
    
    for n in range(params.collision_nonlinear_iters):

        E_upper = params.E_band
        k       = params.boltzmann_constant
        
        tmp1        = (E_upper - mu_ee - p_x*vel_drift_x - p_y*vel_drift_y)
        tmp         = (tmp1/(k*T_ee))
        denominator = (k*T_ee**2.*(af.exp(tmp) + 2. + af.exp(-tmp)) )
        
        a_0 = T_ee      / denominator
        a_1 = tmp1      / denominator
        a_2 = T_ee * p_x / denominator
        a_3 = T_ee * p_y / denominator

        af.eval(a_0, a_1, a_2, a_3)


        # TODO: Multiply with the integral measure dp_x * dp_y
        a_00 = integral_over_p(a_0, params.integral_measure)
        ##a_01 = af.sum(a_1, 0)
        a_02 = integral_over_p(a_2, params.integral_measure)
        a_03 = integral_over_p(a_3, params.integral_measure)

        #a_10 = af.sum(E_upper * a_0, 0)
        #a_11 = af.sum(E_upper * a_1, 0)
        #a_12 = af.sum(E_upper * a_2, 0)
        #a_13 = af.sum(E_upper * a_3, 0)

        a_20 = integral_over_p(p_x * a_0, params.integral_measure)
        ##a_21 = af.sum(p_x * a_1, 0)
        a_22 = integral_over_p(p_x * a_2, params.integral_measure)
        a_23 = integral_over_p(p_x * a_3, params.integral_measure)

        a_30 = integral_over_p(p_y * a_0, params.integral_measure)
        ##a_31 = af.sum(p_y * a_1, 0)
        a_32 = integral_over_p(p_y * a_2, params.integral_measure)
        a_33 = integral_over_p(p_y * a_3, params.integral_measure)

        A = [ [a_00, a_02, a_03], \
              [a_20, a_22, a_23], \
              [a_30, a_32, a_33]  \
            ]
        
        
        fermi_dirac = 1./(af.exp( (  E_upper - mu_ee
                                   - vel_drift_x*p_x - vel_drift_y*p_y 
                                  )/(k*T_ee) 
                                ) + 1.
                         )
        af.eval(fermi_dirac)

        zeroth_moment  =         (f - fermi_dirac)
        #second_moment  = E_upper*(f - fermi_dirac)
        first_moment_x =      p_x*(f - fermi_dirac)
        first_moment_y =      p_y*(f - fermi_dirac)

        eqn_mass_conservation   = integral_over_p(zeroth_moment,  params.integral_measure)
        #eqn_energy_conservation = af.sum(second_moment,  0)
        eqn_mom_x_conservation  = integral_over_p(first_moment_x, params.integral_measure)
        eqn_mom_y_conservation  = integral_over_p(first_moment_y, params.integral_measure)

        residual = [eqn_mass_conservation, \
                    eqn_mom_x_conservation, \
                    eqn_mom_y_conservation]

        error_norm = np.max([af.max(af.abs(residual[0])),
                             af.max(af.abs(residual[1])),
                             af.max(af.abs(residual[2]))
                            ]
                           )
        print("    rank = ", params.rank,
	      "||residual_ee|| = ", error_norm
	     )

#        if (error_norm < 1e-13):
#            params.mu_ee       = mu_ee      
#            params.T_ee        = T_ee       
#            params.vel_drift_x = vel_drift_x
#            params.vel_drift_y = vel_drift_y
#            return(fermi_dirac)

        b_0 = eqn_mass_conservation  
        #b_1 = eqn_energy_conservation
        b_2 = eqn_mom_x_conservation 
        b_3 = eqn_mom_y_conservation 
        b   = [b_0, b_2, b_3]

        # Solve Ax = b
        # where A == Jacobian,
        #       x == delta guess (correction to guess), 
        #       b = -residual

        A_inv = inverse_3x3_matrix(A)
        
        x_0 = A_inv[0][0]*b[0] + A_inv[0][1]*b[1] + A_inv[0][2]*b[2]
        #x_1 = A_inv[1][0]*b[0] + A_inv[1][1]*b[1] + A_inv[1][2]*b[2] + A_inv[1][3]*b[3]
        x_2 = A_inv[1][0]*b[0] + A_inv[1][1]*b[1] + A_inv[1][2]*b[2]
        x_3 = A_inv[2][0]*b[0] + A_inv[2][1]*b[1] + A_inv[2][2]*b[2]

        delta_mu = x_0
        #delta_T  = x_1
        delta_vx = x_2
        delta_vy = x_3
        
        mu_ee       = mu_ee       + delta_mu
        #T_ee        = T_ee        + delta_T
        vel_drift_x = vel_drift_x + delta_vx
        vel_drift_y = vel_drift_y + delta_vy

        af.eval(mu_ee, vel_drift_x, vel_drift_y)

    # Solved for (mu_ee, T_ee, vel_drift_x, vel_drift_y). Now store in params
    params.mu_ee       = mu_ee      
    #params.T_ee        = T_ee       
    params.vel_drift_x = vel_drift_x
    params.vel_drift_y = vel_drift_y

    fermi_dirac = 1./(af.exp( (  E_upper - mu_ee
                               - vel_drift_x*p_x - vel_drift_y*p_y 
                              )/(k*T_ee) 
                            ) + 1.
                     )
    af.eval(fermi_dirac)

    zeroth_moment  =          f - fermi_dirac
    #second_moment  = E_upper*(f - fermi_dirac)
    first_moment_x =      p_x*(f - fermi_dirac)
    first_moment_y =      p_y*(f - fermi_dirac)
    
    eqn_mass_conservation   = integral_over_p(zeroth_moment,  params.integral_measure)
    #eqn_energy_conservation = af.sum(second_moment,  0)
    eqn_mom_x_conservation  = integral_over_p(first_moment_x, params.integral_measure)
    eqn_mom_y_conservation  = integral_over_p(first_moment_y, params.integral_measure)

    residual = [eqn_mass_conservation, \
                eqn_mom_x_conservation, \
                eqn_mom_y_conservation
               ]

    error_norm = np.max([af.max(af.abs(residual[0])),
                         af.max(af.abs(residual[1])),
                         af.max(af.abs(residual[2]))
                        ]
                       )
    print("    rank = ", params.rank,
	  "||residual_ee|| = ", error_norm
	 )
    N_g = domain.N_ghost
    print("    rank = ", params.rank,
          "mu_ee = ", af.mean(params.mu_ee[0, 0, N_g:-N_g, N_g:-N_g]),
          "T_ee = ", af.mean(params.T_ee[0, 0, N_g:-N_g, N_g:-N_g]),
          "<v_x> = ", af.mean(params.vel_drift_x[0, 0, N_g:-N_g, N_g:-N_g]),
          "<v_y> = ", af.mean(params.vel_drift_y[0, 0, N_g:-N_g, N_g:-N_g])
         )
    PETSc.Sys.Print("    ------------------")

    return(fermi_dirac)

