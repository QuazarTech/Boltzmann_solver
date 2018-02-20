import numpy as np
import h5py
import matplotlib as mpl
mpl.use('agg')
import pylab as pl
import domain

# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 300
pl.rcParams['image.cmap']      = 'bwr'
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

dq1 = (domain.q1_end - domain.q1_start) / domain.N_q1
q1  = domain.q1_start + (0.5 + np.arange(domain.N_q1)) * dq1
dp1 = (domain.p1_end - domain.p1_start) / domain.N_p1
p1  = domain.p1_start + (0.5 + np.arange(domain.N_p1)) * dp1

p1, q1 = np.meshgrid(p1, q1)

h5f       = h5py.File('dump/0000.h5', 'r')
f_initial = h5f['distribution_function'][:][0, :, :].reshape(domain.N_q1, domain.N_p1)
h5f.close()

time_array  = np.arange(0, 10.01, 0.01)

for time_index, t0 in enumerate(time_array):

    h5f = h5py.File('dump/%04d'%(10 * time_index) + '.h5', 'r')
    f   = h5f['distribution_function'][:][0, :, :].reshape(domain.N_q1, domain.N_p1)
    h5f.close()

    k_v = np.fft.fftfreq(domain.N_p1, 20/domain.N_p1)

    # pl.contourf(p1, q1, abs(f-f_initial), np.linspace(0, 5e-6, 100))
    pl.semilogy(k_v[:int(domain.N_p1/2)], abs(np.fft.fft(f[16, :].ravel()) / domain.N_p1)[:int(domain.N_p1/2)])
    pl.axvline(x = np.max(k_v), linestyle = '--', color = 'black')
    pl.ylim([1e-14, 1])
    pl.xlabel(r'$k_v$')
    pl.ylabel(r'$|\hat{f(v)}|$')
    pl.title('Time = %.2f'%(t0))
    pl.savefig('images/' + '%04d'%time_index + '.png')
    pl.clf()