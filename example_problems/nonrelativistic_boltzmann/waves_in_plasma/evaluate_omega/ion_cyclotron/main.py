import arrayfire as af
import numpy as np
import math
from petsc4py import PETSc

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.linear.linear_solver import linear_solver

import domain
import boundary_conditions
import params.k_zero_point_one as params
import initialize.k_zero_point_one as initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

dq1 = params.L_x / domain.N_q1
dq2 = params.L_y / domain.N_q2

# Timestep as set by the CFL condition:
dt_fvm = params.N_cfl * min(dq1, dq2) \
                      / max(domain.p1_end + domain.p2_end + domain.p3_end) # joining elements of the list

dt_fdtd = params.N_cfl * min(dq1, dq2) \
                       / params.c # lightspeed

dt        = min(dt_fvm, dt_fdtd)
params.dt = dt

# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )

nls = nonlinear_solver(system)

time_elapsed = 0

while(abs(time_elapsed - params.t_final) > 1e-12):
    
    nls.strang_timestep(dt)
    time_elapsed += dt

    # We step by delta_dt to get the values at dt_dump
    delta_dt =   (1 - math.modf(time_elapsed/params.dt_dump_moments)[0]) \
               * params.dt_dump_moments

    if(delta_dt<dt):
        nls.strang_timestep(delta_dt)
        time_elapsed += delta_dt

    PETSc.Sys.Print('Computing For Time =', time_elapsed / params.t0, "|t0| units(t0)")

nls.dump_moments('moments_data')
nls.dump_EM_fields('EM_fields_data')
