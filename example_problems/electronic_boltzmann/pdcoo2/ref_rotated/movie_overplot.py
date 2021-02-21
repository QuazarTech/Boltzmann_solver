#import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import os
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
matplotlib.use('agg')
import pylab as pl
#import yt
#yt.enable_parallelism()

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc
import PetscBinaryIO

import domain
#import boundary_conditions
#import params
#import initialize


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 12, 7.5
pl.rcParams['figure.dpi']      = 100
pl.rcParams['image.cmap']      = 'jet'
pl.rcParams['lines.linewidth'] = 1.5
pl.rcParams['font.family']     = 'serif'
pl.rcParams['font.weight']     = 'bold'
pl.rcParams['font.size']       = 25
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

N_q1 = domain.N_q1
N_q2 = domain.N_q2

q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (domain.q1_end - domain.q1_start)/N_q1
q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2

q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)

N_p1 = domain.N_p1
N_p2 = domain.N_p2

p1 = domain.p1_start[0] + (0.5 + np.arange(N_p1)) * (domain.p1_end[0] - \
        domain.p1_start[0])/N_p1
p2 = domain.p2_start[0] + (0.5 + np.arange(N_p2)) * (domain.p2_end[0] - \
        domain.p2_start[0])/N_p2


filepath = os.getcwd()
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

print ("moment files : ", moment_files.size)
print ("lagrange multiplier files : ", lagrange_multiplier_files.size)


filepath = os.getcwd() + '/../L_1.0_5.25_tau_ee_inf_tau_eph_inf_DC_rotated_fermi'
moment_files_2 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files_2 = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

time_array = np.loadtxt("dump_time_array.txt")

io = PetscBinaryIO.PetscBinaryIO()

for file_number, dump_file in enumerate(moment_files[:]):

    #file_number = -1
    print("file number = ", file_number, "of ", moment_files.size)

    moments = io.readBinaryFile(moment_files[file_number])
    moments = moments[0].reshape(N_q2, N_q1, 3)
    
    density = moments[:, :, 0]
    j_x     = moments[:, :, 1]
    j_y     = moments[:, :, 2]

    lagrange_multipliers = \
        io.readBinaryFile(lagrange_multiplier_files[file_number])
    lagrange_multipliers = lagrange_multipliers[0].reshape(N_q2, N_q1, 5)
    
    vel_drift_x  = lagrange_multipliers[:, :, 3]
    vel_drift_y  = lagrange_multipliers[:, :, 4]

    moments = io.readBinaryFile(moment_files_2[file_number])
    moments = moments[0].reshape(N_q1, N_q2, 3)
    
    density_2 = moments[:, :, 0].T
    j_x_2     = moments[:, :, 1].T
    j_y_2     = moments[:, :, 2].T

    lagrange_multipliers = \
        io.readBinaryFile(lagrange_multiplier_files_2[file_number])
    lagrange_multipliers = lagrange_multipliers[0].reshape(N_q1, N_q2, 5)
    
    vel_drift_x_2  = lagrange_multipliers[:, :, 3].T
    vel_drift_y_2  = lagrange_multipliers[:, :, 4].T


    pl.subplot(211)
    pl.contourf(q1_meshgrid, q2_meshgrid, density.T, 100, cmap='bwr')
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
    pl.streamplot(q1, q2, 
                  vel_drift_x, vel_drift_y,
                  density=2, color='k',
                  linewidth=0.7, arrowsize=1
                 )
    
    pl.xlim([q1[0], q1[-1]])
    pl.ylim([q2[0], q2[-1]])
    
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')
    
    pl.subplot(212)
    pl.contourf(q1_meshgrid, q2_meshgrid, density_2.T, 100, cmap='bwr')
    pl.title(r'Time = ' + "%.2f"%(time_array[file_number]) + " ps")
    pl.streamplot(q1, q2, 
                  vel_drift_x_2, vel_drift_y_2,
                  density=2, color='k',
                  linewidth=0.7, arrowsize=1
                 )
    
    pl.xlim([q1[0], q1[-1]])
    pl.ylim([q2[0], q2[-1]])
    
    pl.gca().set_aspect('equal')
    pl.xlabel(r'$x\;(\mu \mathrm{m})$')
    pl.ylabel(r'$y\;(\mu \mathrm{m})$')

    #pl.tight_layout()
    pl.suptitle('$\\tau_\mathrm{mc} = \infty$, $\\tau_\mathrm{mr} = \infty$')
    pl.savefig('images/dump_' + '%06d'%file_number + '.png')
    pl.clf()

