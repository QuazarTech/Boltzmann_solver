import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import os
import sys
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib import transforms, colors
matplotlib.use('agg')
import pylab as pl
#import yt
#yt.enable_parallelism()

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc
import PetscBinaryIO

import domain
#import boundary_conditions
import params
#import initialize
#import coords


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

io = PetscBinaryIO.PetscBinaryIO()

N_p1 = domain.N_p1
N_p2 = domain.N_p2

p1 = domain.p1_start[0] + (0.5 + np.arange(N_p1)) * (domain.p1_end[0] - \
        domain.p1_start[0])/N_p1
p2 = domain.p2_start[0] + (0.5 + np.arange(N_p2)) * (domain.p2_end[0] - \
        domain.p2_start[0])/N_p2

print ('Momentum space : ', p1[-1], p2[int(N_p2/2)])

heights   = np.arange(0.25, 0.351, 0.025)
heights_1 = np.arange(0.375, 0.4751, 0.025)
heights_2 = np.arange(0.500, 0.601, 0.025)
heights_3 = np.arange(0.650, 1.801, 0.025)
heights_4 = np.arange(1.800, 2.850, 0.05)
heights_5 = np.arange(3.000, 3.5001, 0.25)
heights = np.append(heights, heights_1)
heights = np.append(heights, heights_2)
heights = np.append(heights, heights_3)
heights = np.append(heights, heights_4)
heights = np.append(heights, heights_5)

for index in heights:
    filepath = \
    '/home/mchandra/gitansh/bolt_master/example_problems/electronic_boltzmann/pdcoo2/L_1.0_%.3f_tau_ee_inf_tau_eph_inf_DC_rotated'%index

    moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))

    #print ("moment files : ", moment_files.size)

    q1_end = index

    N_q2 = domain.N_q2
    N_q1 = int(N_q2*q1_end/domain.q2_end)
    
    q1 = domain.q1_start + (0.5 + np.arange(N_q1)) * (q1_end - domain.q1_start)/N_q1
    q2 = domain.q2_start + (0.5 + np.arange(N_q2)) * (domain.q2_end - domain.q2_start)/N_q2
    
    q2_meshgrid, q1_meshgrid = np.meshgrid(q2, q1)
    
    dump_step = params.dt_dump_moments
    
    #time_array = np.loadtxt(filepath+"/dump_time_array.txt")
    time_array = dump_step * np.arange(0, moment_files.size, 1)

    print("Index : ", index, ", moment files : ", moment_files.size)
    
    q1_index = 0; q2_index = (q2 < .25)
    
    #sensor_1_array = []
    #for file_number, dump_file in enumerate(moment_files[:]):
    
    #    #file_number = -1
    #    print("file number = ", file_number, "of ", moment_files.size)
    
    #    moments = io.readBinaryFile(moment_files[file_number])
    #    moments = moments[0].reshape(N_q2, N_q1, 3)
        
    #    density = moments[:, :, 0]
    #    j_x     = moments[:, :, 1]
    #    j_y     = moments[:, :, 2]
    
    #    signal  = np.mean(density[q2_index, q1_index])
        
    #    sensor_1_array.append(signal)
    
    #sensor_1_array = np.array(sensor_1_array)
    
    #time_indices = time_array > 300
    
    #pl.plot(time_array[time_indices], sensor_1_array[time_indices] - np.mean(sensor_1_array[time_indices])) 
    #pl.ylim([1e-8, -1e-8])
    
    #pl.gca().set_aspect('equal')
    #pl.ylabel(r'$n$')
    #pl.xlabel(r'Time (ps)')
    #pl.suptitle('$\\tau_\mathrm{mc} = \infty$, $\\tau_\mathrm{mr} = \infty$ ps')
    #pl.savefig('images/iv.png')
    #pl.clf()

