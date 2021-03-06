import arrayfire as af
import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver

import domain
import boundary_conditions
import params
import initialize

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'gist_heat'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 20
pl.rcParams['font.sans-serif'] = 'serif'
pl.rcParams['text.usetex']     = True
pl.rcParams['axes.linewidth']  = 1.5
pl.rcParams['axes.titlesize']  = 'medium'
pl.rcParams['axes.labelsize']  = 'medium'

pl.rcParams['xtick.major.size'] = 8
pl.rcParams['xtick.minor.size'] = 4
pl.rcParams['xtick.major.pad']  = 8
pl.rcParams['xtick.minor.pad']  = 8
pl.rcParams['xtick.color']      = 'k'
pl.rcParams['xtick.labelsize']  = 'medium'
pl.rcParams['xtick.direction']  = 'in'

pl.rcParams['ytick.major.size'] = 8
pl.rcParams['ytick.minor.size'] = 4
pl.rcParams['ytick.major.pad']  = 8
pl.rcParams['ytick.minor.pad']  = 8
pl.rcParams['ytick.color']      = 'k'
pl.rcParams['ytick.labelsize']  = 'medium'
pl.rcParams['ytick.direction']  = 'in'

N     = 2**np.arange(6, 11) 
error = np.zeros(N.size)

for i in range(N.size):
    
    domain.N_q1 = int(N[i])
    domain.N_q2 = int(N[i])

    # Defining the physical system to be solved:
    system = physical_system(domain,
                             boundary_conditions,
                             params,
                             initialize,
                             advection_terms,
                             collision_operator.BGK,
                             moments
                            )

    # Declaring a linear system object which will evolve the defined physical system:
    nls = nonlinear_solver(system)
    N_g = nls.N_ghost

    # Time parameters:
    dt      = 0.01 * 32/nls.N_q1
    t_final = 2.0 

    time_array = np.arange(dt, t_final + dt, dt)
    n_initial = nls.compute_moments('density')    
    # Checking that time array doesn't cross final time:
    if(time_array[-1]>t_final):
        time_array = np.delete(time_array, -1)

    f_reference = nls.f.copy()
    
    for time_index, t0 in enumerate(time_array):
        nls.strang_timestep(dt)

    error[i] = af.mean(af.abs(  nls.f[:, :, N_g:-N_g, N_g:-N_g]
                              - f_reference[:, :, N_g:-N_g, N_g:-N_g]
                             )
                      )


#    error[i] = af.mean(af.abs(  nls.compute_moments('density')[:, :, N_g:-N_g, N_g:-N_g]
#                              - n_initial[:, :, N_g:-N_g, N_g:-N_g]
#                             )
#                      )

    print(error)

print('Error Obtained:')
print('L1 norm of error:', error)

print('\nConvergence Rate:')
print('Order of convergence:', np.polyfit(np.log10(N), np.log10(error), 1)[0])

pl.loglog(N, error, '-o', label = 'Numerical')
pl.loglog(N, error[0]*64**2/N**2, '--', color = 'black', label = r'$O(N^{-2})$')
pl.xlabel(r'$N$')
pl.ylabel('Error')
pl.legend()
pl.savefig('convergenceplot.png')
pl.savefig('convergenceplot.svg')
