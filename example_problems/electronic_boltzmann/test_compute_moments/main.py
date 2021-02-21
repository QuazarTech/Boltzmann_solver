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

from bolt.lib.nonlinear.compute_moments import compute_moments
from bolt.lib.utils.calculate_q import calculate_q


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


nls.dump_moments('dump_moments/moments')
nls.dump_aux_arrays([params.mu,
                     params.mu_ee,
                     params.T_ee,
                     params.vel_drift_x, params.vel_drift_y
                    ],
                   'lagrange_multipliers',
                   'dump_lagrange_multipliers/lagrange_multipliers'
                   )
nls.dump_distribution_function('dump_f/f')        

