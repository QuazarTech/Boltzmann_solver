import numpy as np
import arrayfire as af
import domain

in_q1_left   = 'mirror+dirichlet'
in_q1_right  = 'mirror+dirichlet'
in_q2_bottom = 'mirror'
in_q2_top    = 'mirror'

@af.broadcast
def f_left(f, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu
    
    vel_drift_x = params.vel_drift_x_in

    fermi_dirac = (1./(af.exp( (E_upper - vel_drift_x*p1 - mu)/(k*T) ) + 1.)
                  )
    
    q2_contact_start = params.contact_start
    q2_contact_end   = params.contact_end
    cond = ((q2 >= q2_contact_start) & \
            (q2 <= q2_contact_end) \
           )
    
    f_left = cond*fermi_dirac + (1 - cond)*f

    af.eval(f_left)
    return(f_left)

@af.broadcast
def f_right(f, q1, q2, p1, p2, p3, params):

    k       = params.boltzmann_constant
    E_upper = params.E_band
    T       = params.initial_temperature
    mu      = params.initial_mu

    vel_drift_x = params.vel_drift_x_out

    fermi_dirac = (1./(af.exp( (E_upper - vel_drift_x*p1 - mu)/(k*T) ) + 1.)
                  )
    
    q2_contact_start = params.contact_start
    q2_contact_end   = params.contact_end
    cond = ((q2 >= q2_contact_start) & \
            (q2 <= q2_contact_end) \
           )
    
    f_right = cond*fermi_dirac + (1 - cond)*f

    af.eval(f_right)
    return(f_right)
