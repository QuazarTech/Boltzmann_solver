import numpy as np
import matplotlib as mpl
mpl.use('agg')
import pylab as pl
import h5py
import domain

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 6
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 30
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

# dt      = 0.001
# t_final = 0.02
# time    = np.arange(dt, t_final + dt, dt)

h5f = h5py.File('dump/0000.h5', 'r')
q1  = h5f['q1'][:].reshape(domain.N_q1 + 6, domain.N_q2 + 6)
q2  = h5f['q2'][:].reshape(domain.N_q1 + 6, domain.N_q2 + 6)
n   = h5f['n'][:].reshape(domain.N_q1 + 6, domain.N_q2 + 6)
h5f.close()

# pl.contourf(q1[3:-3, 3:-3],
#             q2[3:-3, 3:-3],
#             n[3:-3, 3:-3],
#             100,
#             cmap = 'gist_heat'
#            )
# pl.title('Time = 0')
# pl.xlabel(r'$x$')
# pl.ylabel(r'$y$')
# pl.axes().set_aspect('equal')
# pl.savefig('images/0000.png')
# pl.clf()

# for time_index, t0 in enumerate(time):
#     h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'r')
#     n   = h5f['n'][:].reshape(domain.N_q1 + 6, domain.N_q2 + 6)
#     h5f.close()

h5f = h5py.File('dump/2500.h5', 'r')
n   = h5f['n'][:].reshape(domain.N_q1 + 6, domain.N_q2 + 6)
h5f.close()

pl.contourf(q1[3:-3, 3:-3],
            q2[3:-3, 3:-3],
            abs(n)[3:-3, 3:-3],
            100,
            cmap = 'gist_heat'
           )
# pl.title('Time =' + str(t0))
pl.xlim([3, 6])
pl.xlabel(r'$x / L$')
pl.ylabel(r'$y / L$')
pl.axes().set_aspect('equal')
pl.savefig('plot.png', bbox_inches = 'tight')
# pl.savefig('images/%04d'%(time_index+1) + '.png')
pl.clf()

# pl.plot(q2[66, 10:-10], n[66, 10:-10])
# pl.title('Time = 0')
# pl.xlabel(r'$x$')
# pl.ylabel(r'$n$')
# pl.savefig('images/0000.png')
# pl.clf()

# for time_index, t0 in enumerate(time):
#     h5f = h5py.File('dump/%04d'%(time_index+1) + '.h5', 'r')
#     n   = h5f['n'][:]
#     h5f.close()

#     pl.plot(q2[66, 10:-10], n[66, 10:-10])
#     pl.title('Time = ' + str(t0))
#     pl.xlabel(r'$x$')
#     pl.ylabel(r'$n$')
#     pl.savefig('images/%04d'%(time_index+1) + '.png')
#     pl.clf()
