"""Contains the function which returns the Source/Sink term."""

from petsc4py import PETSc
import numpy as np
import arrayfire as af

#from bolt.lib.nonlinear.compute_moments import compute_moments

from .f_local_equilibrium import f0_ee, f0_defect
from .f_local_equilibrium_zero_T import f0_ee_constant_T, f0_defect_constant_T

from .matrix_inverse import inverse_4x4_matrix
from .matrix_inverse import inverse_3x3_matrix
from bolt.src.utils.integral_over_p import integral_over_p

import domain


def RTA(f, t, q1, q2, p1, p2, p3, moments, params, flag = False):
    """
    Return RTA (Relaxation Time Approximation) operator 
     - (f-f0_D)/tau_D - (f-f0_ee)/tau_ee

    Parameters:
    -----------
    f : Distribution function array
        shape:(N_v, N_s, N_q1, N_q2)
    
    t : Time elapsed
    
    q1 : The array that holds data for the q1 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    q2 : The array that holds data for the q2 dimension in q-space
         shape:(1, 1, N_q1, N_q2)

    p1 : The array that holds data for the v1 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    p2 : The array that holds data for the v2 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    p3 : The array that holds data for the v3 dimension in v-space
         shape:(N_v, N_s, 1, 1)

    params: The parameters file/object that is originally declared by the user.
            This can be used to inject other functions/attributes into the function
    
    flag: Toggle used for evaluating tau = 0 cases need to be evaluated. When set to True, this
          function is made to return f0, thus setting f = f0 wherever tau = 0
    """

    p_x, p_y = params.p_x, params.p_y
    p_z = p3

    #if(af.any_true(params.tau_defect(q1, q2, p_x, p_y, p_z) == 0)):
    #    if (flag == False):
    #        return(0.*f)
    #
    #    f = f0_defect_constant_T(f, p_x, p_y, p_z, params)
    #
    #    return(f)

    if (params.disable_collision_op):
        # Activate the following line to disable the collision operator
        C_f = 0.*f
    else:
        # Activate the following lines to enable normal operation of collision
        # operator
        if (params.p_dim==1):
            C_f = -(  f - f0_defect_constant_T(f, p_x, p_y, p_z, params) \
                   ) / params.tau_defect(q1, q2, p_x, p_y, p_z) \
                  -(  f - f0_ee_constant_T(f, p_x, p_y, p_z, params)
                  ) / params.tau_ee(q1, q2, p_x, p_y, p_z)

        elif (params.p_dim==2):
            C_f = -(  f - f0_defect(f, p_x, p_y, p_z, params) \
                   ) / params.tau_defect(q1, q2, p_x, p_y, p_z) \
                  -(  f - f0_ee_constant_T(f, p_x, p_y, p_z, params)
                  ) / params.tau_ee(q1, q2, p_x, p_y, p_z)

    # When (f - f0) is NaN. Dividing by np.inf doesn't give 0
    # TODO: WORKAROUND
 

    af.eval(C_f)
    return(C_f)

