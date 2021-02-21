print ("Execution started, loading libs...")

import arrayfire as af
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
import boundary_conditions
import params
import initialize


import petsc4py, sys; petsc4py.init(sys.argv)
from petsc4py import PETSc

#from bolt.lib.physical_system import physical_system

#from bolt.lib.nonlinear_solver.nonlinear_solver \
#    import nonlinear_solver
#from bolt.lib.nonlinear_solver.EM_fields_solver.electrostatic \
#    import compute_electrostatic_fields

import domain
import boundary_conditions
import params
import initialize

#import bolt.src.electronic_boltzmann.advection_terms as advection_terms

#import bolt.src.electronic_boltzmann.collision_operator \
#    as collision_operator

#import bolt.src.electronic_boltzmann.moment_defs as moment_defs

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

print ("Loaded libs")

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

source_start = params.contact_start
source_end   = params.contact_end

drain_start  = params.contact_start
drain_end    = params.contact_end

source_indices =  (q2 > source_start) & (q2 < source_end)
drain_indices  =  (q2 > drain_start)  & (q2 < drain_end )

probe_1_indices = (q2 < source_end)


#Dump files
filepath = '.'
moment_files              = np.sort(glob.glob(filepath+'/dump_moments/*.bin'))
lagrange_multiplier_files = \
        np.sort(glob.glob(filepath+'/dump_lagrange_multipliers/*.bin'))


print ("Moment files size : ", moment_files.size)
skip_files = 60000

moment_files = moment_files[skip_files:]
lagrange_multiplier_files = lagrange_multiplier_files[skip_files:]

dt = params.dt
dump_interval = params.dump_steps

#time_array = np.loadtxt("dump_time_array.txt")

time_array = dump_interval * dt * np.arange(0, moment_files.size, 1)

io = PetscBinaryIO.PetscBinaryIO()

left_edge_density  = []
right_edge_density = []
left_edge_current  = []

for file_number, dump_file in enumerate(moment_files):

    dump_file = moment_files[file_number]
    print (dump_file)

    moments = io.readBinaryFile(dump_file)
    moments = moments[0].reshape(N_q2, N_q1, 3)

    print("file number = ", file_number, "of ", moment_files.size, "shape = ", moments.shape)

    density = moments[:, :, 0]


    left_edge_density.append(density[:, 0])
    right_edge_density.append(density[:, -1])


#    lagrange_multipliers = \
#        io.readBinaryFile(lagrange_multiplier_files[file_number])
#    lagrange_multipliers = lagrange_multipliers[0].reshape(N_q2, N_q1, 7)

#    print("file number = ", file_number, "of ", lagrange_multiplier_files.size,
#            "shape = ", lagrange_multipliers.shape)

#    j_x     = lagrange_multipliers[:, :, 5]
#    j_y     = lagrange_multipliers[:, :, 6]

#    j_mag = (j_x**2. + j_y**2.)**0.5

#    left_edge_current.append(j_mag[:, 0])


voltage_left  = np.array(left_edge_density)
voltage_right = np.array(right_edge_density)
#current_left  = np.array(left_edge_current)

print("Left : ", voltage_left.shape)

np.savez_compressed("voltages_L_1.0_5.25_tau_ee_inf_tau_eph_inf.txt",
        left=voltage_left, right=voltage_right)
#np.savez_compressed('current_L_1.0_5.25_tau_ee_inf_tau_eph_inf.txt',
#        current=current_left)

