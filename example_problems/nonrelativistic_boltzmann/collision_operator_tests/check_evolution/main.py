import arrayfire as af
import numpy as np
from scipy import integrate
import matplotlib as mpl 
mpl.use('agg')
import pylab as pl

from bolt.lib.physical_system import physical_system
from bolt.lib.nonlinear.nonlinear_solver import nonlinear_solver
from bolt.lib.linear.linear_solver import linear_solver

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
pl.rcParams['image.cmap']      = 'jet'
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

@af.broadcast
def f0(v1, n, v_bulk, T):
    return(n * (1 / (2 * np.pi * T))**(1 / 2) \
             * np.exp(-(v1 - v_bulk)**2 / (2 * T))
          )

def BGK(f, t, v1, params):
    dv1 = v1.ravel()[1] - v1.ravel()[0]
    n   = np.sum(f) * dv1

    v_bulk = np.sum(f * v1) * dv1 / n

    T = (  np.sum(f * v1**2) * dv1
         - n * v_bulk**2
        ) / n

    f_MB = f0(v1, n, v_bulk, T)
    tau  = params.tau(0, 0, 0, 0, 0)

    C_f = -(f - f_MB) / tau
    return(C_f)


def set_advection_to_zero(f, t, q1, q2, v1, v2, v3, params):
    return(0 * v1**0, 0 * v2**0)

advection_terms.A_q = set_advection_to_zero
advection_terms.C_q = set_advection_to_zero

# Time parameters:
dt      = 0.0001
t_final = 0.1

system = physical_system(domain,
                         boundary_conditions,
                         params,
                         initialize,
                         advection_terms,
                         collision_operator.BGK,
                         moments
                        )
N_g = system.N_ghost

# Declaring a linear system object which will evolve the defined physical system:
nls = nonlinear_solver(system)
time_array = np.arange(0, t_final + dt, dt)
    
sol = integrate.odeint(BGK, np.array(nls.f), time_array ,
                       args = (np.array(nls.p1_center), params),
                       printmessg=1, rtol = 1e-13, atol = 1e-13,
                      )

error = np.zeros(time_array.size)

for time_index, t0 in enumerate(time_array[1:]):
    f_odeint = sol[time_index, :].ravel()
    # error[time_index] = np.mean(abs(np.array(nls.f).ravel() - f_odeint))
    
    pl.plot(np.array(nls.p1_center).ravel(), np.array(nls.f).ravel())
    pl.plot(np.array(nls.p1_center).ravel(), f_odeint, '--', color = 'black')

    pl.xlabel(r'$v_1$')
    pl.ylabel(r'$f$')
    pl.title('Distribution at Time = ' + str(t0 - dt))
    pl.legend([r'With $\texttt{bolt.nonlinear\_solver}$', r'With $\texttt{odeint}$'])
    pl.savefig('images/%04d'%time_index + '.png')
    pl.clf()

    nls.strang_timestep(dt)
