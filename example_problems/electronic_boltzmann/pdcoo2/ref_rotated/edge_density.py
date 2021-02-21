import arrayfire as af
import numpy as np
from scipy.signal import correlate
import glob
import h5py
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
matplotlib.use('agg')
import pylab as pl
#import yt
#yt.enable_parallelism()
import os

import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

import PetscBinaryIO

import domain
import boundary_conditions
import params
import initialize


# Optimized plot parameters to make beautiful plots:
pl.rcParams['figure.figsize']  = 8, 8
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

source_start = params.contact_start
source_end   = params.contact_end

drain_start  = params.contact_start
drain_end    = params.contact_end

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q2 > drain_start)  & (q2 < drain_end )

# Left needs to be near source, right sensor near drain
sensor_1_left_start = 8.5 # um
sensor_1_left_end   = 9.5 # um

sensor_1_right_start = 8.5 # um
sensor_1_right_end   = 9.5 # um

sensor_1_left_indices  = (q2 > sensor_1_left_start ) & (q2 < sensor_1_left_end)
sensor_1_right_indices = (q2 > sensor_1_right_start) & (q2 < sensor_1_right_end)

io = PetscBinaryIO.PetscBinaryIO()

filepath = os.getcwd()
moment_files 		  = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))

dt = params.dt
dump_interval = params.dump_steps

sensor_1_signal_array = []
print("Reading sensor signal...")
for file_number, dump_file in enumerate(moment_files):

    moments = io.readBinaryFile(dump_file)
    moments = moments[0].reshape(N_q2, N_q1, 3)

    density = moments[:, :, 0]
    
    source = np.mean(density[source_indices, 0])
    drain  = np.mean(density[drain_indices, -1])

    sensor_1_left   = np.mean(density[0,  0])
    sensor_1_right  = np.mean(density[0, -1])

    sensor_1_signal = sensor_1_left - sensor_1_right

    sensor_1_signal_array.append(sensor_1_signal)

time_array = np.loadtxt("dump_time_array.txt")
AC_freq = 1./100
input_signal_array = np.sin(2.*np.pi*AC_freq*time_array)
sensor_1_signal_array = np.array(sensor_1_signal_array)
half_time = (int)(time_array.size/2)

input_normalized = \
        input_signal_array/np.max(np.abs(input_signal_array[half_time:]))
sensor_normalized = \
    sensor_1_signal_array#/np.max(np.abs(sensor_1_signal_array[half_time:]))

