import os
import arrayfire as af
import numpy as np
import math
import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc
from mpi4py import MPI
MPI.WTIME_IS_GLOBAL=True

from bolt.lib.physical_system import physical_system

from bolt.lib.nonlinear.nonlinear_solver \
    import nonlinear_solver
#from bolt.lib.nonlinear.fields.fields.fields \
#    import fields_solver.compute_electrostatic_fields
from bolt.lib.utils.restart_latest import latest_output, format_time

import domain
import boundary_conditions
import initialize
import params

import bolt.src.electronic_boltzmann.advection_terms \
    as advection_terms
import bolt.src.electronic_boltzmann.collision_operator \
    as collision_operator
import bolt.src.electronic_boltzmann.moments \
    as moments


# Create required folders if they do not exist already
#if not os.path.isdir("dump_f"):
#    os.system("mkdir dump_f")
#if not os.path.isdir("dump_moments"):
#    os.system("mkdir dump_moments")
#if not os.path.isdir("dump_lagrange_multipliers"):
#    os.system("mkdir dump_lagrange_multipliers")
#if not os.path.isdir("images"):
#    os.system("mkdir images")


# Defining the physical system to be solved:
system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.RTA,
                         moments
                        )

# Declaring a nonlinear system object which will evolve the defined physical system:
nls         = nonlinear_solver(system)
N_g         = domain.N_ghost
params.rank = nls._comm.rank

# Time parameters:
dt      = params.dt
t_final = params.t_final
params.current_time = time_elapsed   = 0.0
params.time_step    = time_step = 0
dump_counter = 0
dump_time_array = []


using_latest_restart = False
if(params.latest_restart == True):
    latest_f, time_elapsed = latest_output('')
    print(time_elapsed)
    if(latest_f is not None and  time_elapsed is not None):
      nls.load_distribution_function(latest_f)
      dump_time_array = np.loadtxt("dump_time_array.txt").tolist()
      using_latest_restart = True


if using_latest_restart == False:
    if(params.t_restart == 0 or params.latest_restart == True):
        time_elapsed = 0
        formatted_time = format_time(time_elapsed)
        nls.dump_distribution_function('dump_f/t=' + formatted_time)
        nls.dump_moments('dump_moments/t=' + formatted_time)
        nls.dump_aux_arrays([params.mu,
                             params.mu_ee,
                             params.T_ee,
                             params.vel_drift_x, params.vel_drift_y
                            ],
                             'lagrange_multipliers',
                             'dump_lagrange_multipliers/t=' + formatted_time
                            )
        dump_time_array.append(time_elapsed)
        if (params.rank==0):
            np.savetxt("dump_time_array.txt", dump_time_array)
    else:
        time_elapsed = params.t_restart
        formatted_time = format_time(time_elapsed)
        nls.load_distribution_function('dump_f/t=' + formatted_time)

# Checking that the file writing intervals are greater than dt:
assert(params.dt_dump_f >= dt)
assert(params.dt_dump_moments >= dt)
assert(params.dt_dump_fields >= dt)


#if (params.restart):
#    nls.load_distribution_function(params.restart_file)

density = nls.compute_moments('density')
print("rank = ", params.rank, "\n",
      "     <mu>    = ", af.mean(params.mu[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
      "     max(mu) = ", af.max(params.mu[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
      "     <n>     = ", af.mean(density[0, 0, N_g:-N_g, N_g:-N_g]), "i\n",
      "     max(n)  = ", af.max(density[0, 0, N_g:-N_g, N_g:-N_g]), "\n"
     )

nls.f = af.select(nls.f < 1e-20, 1e-20, nls.f)
while(time_elapsed < t_final):

    # Refine to machine error
    if (time_step==0):
        params.collision_nonlinear_iters = 10
    else:
        params.collision_nonlinear_iters = params.collision_operator_nonlinear_iters

    dump_steps = params.dump_steps

    if(params.dt_dump_moments != 0):
        # We step by delta_dt to get the values at dt_dump
        delta_dt =   (1 - math.modf(time_elapsed/params.dt_dump_moments)[0]) \
                   * params.dt_dump_moments

        if((delta_dt-dt)<1e-5):
            nls.strang_timestep(delta_dt)
            time_elapsed += delta_dt
            formatted_time = format_time(time_elapsed)            
            nls.dump_moments('dump_moments/t=' + formatted_time)
            nls.dump_aux_arrays([params.mu,
                             params.mu_ee,
                             params.T_ee,
                             params.vel_drift_x, params.vel_drift_y
                                ],
                             'lagrange_multipliers',
                             'dump_lagrange_multipliers/t=' + formatted_time
                            )
            dump_time_array.append(time_elapsed)
            if (params.rank==0):
                np.savetxt("dump_time_array.txt", dump_time_array)

    if(math.modf(time_elapsed/params.dt_dump_f)[0] < 1e-12):
        formatted_time = format_time(time_elapsed)
        nls.dump_distribution_function('dump_f/t=' + formatted_time)        

    PETSc.Sys.Print("Time step =", time_step, ", Time =", time_elapsed)

    nls.strang_timestep(dt)
    time_elapsed        = time_elapsed + dt
    time_step           = time_step + 1
    params.time_step    = time_step
    params.current_time = time_elapsed

    # Floors
    nls.f     = af.select(nls.f < 1e-20, 1e-20, nls.f)

    density = nls.compute_moments('density')
    print("rank = ", params.rank, "\n",
          "     <mu>    = ", af.mean(params.mu[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(mu) = ", af.max(params.mu[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
          "     <n>     = ", af.mean(density[0, 0, N_g:-N_g, N_g:-N_g]), "\n",
          "     max(n)  = ", af.max(density[0, 0, N_g:-N_g, N_g:-N_g]), "\n"
         )
    PETSc.Sys.Print("--------------------\n")

#nls.dump_distribution_function('dump_f/t_laststep')
