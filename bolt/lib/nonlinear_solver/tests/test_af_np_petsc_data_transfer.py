#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af
import petsc4py, sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
af.info()

N_q1    = 64
N_q2    = 128
N_q3    = 96
dof     = 16*16*16
N_ghost = 3
print("---------------------")
print("N_q1 + 2*N_ghost =", N_q1 + 2*N_ghost) 
print("N_q2 + 2*N_ghost =", N_q2 + 2*N_ghost) 
print("N_q3 + 2*N_ghost =", N_q3 + 2*N_ghost) 
print("dof              =", dof)
print("---------------------\n")

da = PETSc.DMDA().create([N_q1, N_q2],
                          stencil_width=N_ghost,
                          boundary_type=('periodic',
                                         'periodic'),
                          stencil_type=1,
                          dof = dof
                        )

glob_vec  = da.createGlobalVec() # No ghost zones in [N_q1, N_q2]
local_vec = da.createLocalVec()  # Has ghost zones in [N_q1, N_q2]

# Initialize glob_vec to random numbers
glob_vec.setRandom()

local_vec_array_old = da.getVecArray(local_vec)
glob_vec_array_old  = da.getVecArray(glob_vec)

#1d np arrays pointing to respective vecs
local_vec_array  = local_vec.getArray()
global_vec_array = glob_vec.getArray()

((i_q1_start, i_q2_start), (N_q1_local, N_q2_local)) = da.getCorners()

af_array_old = af.randu(N_q1_local+2*N_ghost,
			N_q2_local+2*N_ghost,
			dof,
                        dtype=af.Dtype.f64
		       )
af.eval(af_array_old)

af_array_new = af.randu(dof, 
                        N_q1_local+2*N_ghost,
			N_q2_local+2*N_ghost,
                        dtype=af.Dtype.f64
		       )
af.eval(af_array_new)
af.sync()

print("af_array_old.shape = ", af_array_old.shape)
print("af_array_new.shape = ", af_array_new.shape)

af.print_mem_info()

def communicate_old(af_array, da, glob_vec, local_vec):
    # af_array is (N_q2+2*N_ghost, N_q1+2*N_ghost, dof)

    # Global value is non-inclusive of the ghost-zones:
    glob_vec_array_old[:] = np.array(af_array[N_ghost:-N_ghost, 
                                              N_ghost:-N_ghost
                                             ]
                                    )
    
    # The following function takes care of periodic boundary conditions,
    # and interzonal communications:
    da.globalToLocal(glob_vec, local_vec)

    # Converting back from PETSc.Vec to af.Array:
    af_array_after_comm = af.to_array(local_vec_array_old[:])

    af.eval(af_array_after_comm)

    return(af_array_after_comm)

def communicate_new(af_array, da, glob_vec, local_vec):
    # af_array is (dof, N_q1, N_q2)
#    af_array_1d = af.moddims(af_array,
#                               (N_q1_local + 2*N_ghost)
#			     * (N_q2_local + 2*N_ghost)
#			     * dof
#                            )
#
#    local_vec_array[:] = af_array_1d.to_ndarray()
    
    # First flatten af_array
    af_array_1d = af.moddims(af_array[:, 
                                      N_ghost:-N_ghost,
                                      N_ghost:-N_ghost
				     ],
                               N_q1_local 
			     * N_q2_local
			     * dof
                            )
    
    # Convert to a np array and copy to petsc
    #global_vec_array[:] = af_array_1d.to_ndarray()
    af_array_1d.to_ndarray(global_vec_array)

    # Communication: Global -> Local
    da.globalToLocal(glob_vec, local_vec)

    # local_vec_array_new now has the data. Need to convert to af
    af_array_after_comm_1d = af.interop.np_to_af_array(local_vec_array)
#    af_array_after_comm    = af.moddims(af_array_after_comm_1d,
#                                        dof,
#                                        N_q1_local + 2*N_ghost,
#                                        N_q2_local + 2*N_ghost
#                                       )
#
#    af.eval(af_array_after_comm)

#    return(af_array_after_comm)

#communicate_old(af_array_old, da, glob_vec, local_vec)
communicate_new(af_array_new, da, glob_vec, local_vec)
af.sync()

af.print_mem_info()

iters = 1

tic_old = af.time()
for n in range(iters):
    communicate_old(af_array_old, da, glob_vec, local_vec)
af.sync()
toc_old = af.time()
time_old = toc_old - tic_old

tic_new = af.time()
for n in range(iters):
    communicate_new(af_array_new, da, glob_vec, local_vec)
af.sync()
toc_new = af.time()
time_new = toc_new - tic_new

print(" ")
print("comm_old = ", format(time_old/iters, '.4f'), "secs/iter")
print("comm_new = ", format(time_new/iters, '.4f'), "secs/iter")
print(" ")

## Now need to transfer data from glob_vec to local_vec
#da.globalToLocal(glob_vec, local_vec)
## Communication complete. All interprocessor ghost zones in local_vec are
## now filled.
#
## Problem statement: Accessing local_vec using numpy arrays
## ---------- Method 1----------- 
#local_array_method_1 = da.getVecArray(local_vec)
#
## Note: da.getVecArray() does NOT return a numpy array
##       local_array_method_1[:] is a numpy array
#local_array_method_1_np = local_array_method_1[:]
#
#print("---------------------")
#print("local_array_method_1_np.shape   = ", local_array_method_1_np.shape)
#print("local_array_method_1_np.strides = ", local_array_method_1_np.strides)
#print("---------------------\n")
#
## ---------- Method 2----------- 
#local_array_method_2_np = local_vec.getArray()
#
## Note: local_vec.getArray() *directly* returns a numpy array of shape 1D and
## size (N_q1 + 2*N_ghost) x (N_q2 + 2*N_ghost) * dof
#
#print("---------------------")
#print("local_array_method_2_np.shape   = ", local_array_method_2_np.shape)
#print("local_array_method_2_np.strides = ", local_array_method_2_np.strides)
#print("---------------------\n")
#
## Now need to reshape
#local_array_method_2_np_reshaped = \
#        local_array_method_2_np.reshape([N_q2+2*N_ghost,
#                                         N_q1+2*N_ghost,
#                                         dof
#                                        ]
#                                       )
#print("---------------------")
#print("local_array_method_2_np_reshaped.shape   = ", \
#        local_array_method_2_np_reshaped.shape)
#print("local_array_method_2_np_reshaped.strides = ", \
#        local_array_method_2_np_reshaped.strides)
#print("---------------------\n")
#
#print("Conclusion:")
#print(" * Natural order is set by PETSc where dof is the fastest index and q2 is the slowest index")
#print(" * Optimal layout for np arrays is (q2, q1, dof)")
#print(" * Optimal layout for af arrays is (dof, q1, q2)\n")
#
## Strategy for efficient af<->PETSc transfer:
## 1) We need to transfer data from array_af with dims (dof, N_q1, N_q2) to a PETSc vec: glob_vec
## 2) Reshape it into 1D : array_af_1d
## 2) Get the associated 1d np array of glob_vec: glob_array = glob_vec.getArray()
## 3) Fast transfer to np (and hence to PETSc): array_af_1d.to_ndarray(glob_array_np)
## 4) Done! : glob_vec now has data from af_array
## 5) Now perform globalToLocal(glob_vec, local_vec)
## 6) Get 1d np array: local_array = local_vec.getArray()
## 7) Transfer data from 1d np array to 1d af array: af_array_ghosted_1d = af.to_array(local_array)
## 8) Reshape af_array_ghosted_1d to dims (dof, N_q1+2*N_ghost, N_q2+2*N_ghost)
#
#local_array_method_2_af = af.to_array(local_array_method_2_np_reshaped)
#print("---------------------")
#print("local_array_method_2_af.shape   = ", local_array_method_2_af.shape)
#print("local_array_method_2_af.strides = ", local_array_method_2_af.strides())
#print("---------------------\n")
#
#local_array_method_2_af_reshaped = af.to_array(local_array_method_2_np)
#local_array_method_2_af_reshaped = af.moddims(local_array_method_2_af_reshaped,
#                                              dof,
#                                              N_q1+2*N_ghost,
#                                              N_q2+2*N_ghost
#                                             )
#
#print("---------------------")
#print("local_array_method_2_af_reshaped.shape   = ",
#        local_array_method_2_af_reshaped.shape)
#print("local_array_method_2_af_reshaped.strides = ",
#        local_array_method_2_af_reshaped.strides())
#print("---------------------\n")
#
#local_array_af_to_np = local_array_method_2_af.to_ndarray()
#print("---------------------")
#print("local_array_af_to_np.shape   = ", local_array_af_to_np.shape)
#print("local_array_af_to_np.strides = ", local_array_af_to_np.strides)
#print("---------------------\n")

#da = PETSc.DMDA().create([N_q1, N_q2, N_q3],
#                          stencil_width=N_ghost,
#                          boundary_type=('periodic',
#                                         'periodic',
#					 'periodic'),
#                          stencil_type=1,
#                          dof = dof
#                        )
#
#glob_vec  = da.createGlobalVec() # No ghost zones in [N_q1, N_q2]
#local_vec = da.createLocalVec()  # Has ghost zones in [N_q1, N_q2]
#
## Initialize glob_vec to random numbers
#glob_vec.setRandom()
#
## Now need to transfer data from glob_vec to local_vec
#da.globalToLocal(glob_vec, local_vec)
## Communication complete. All interprocessor ghost zones in local_vec are
## now filled.
#
## Problem statement: Accessing local_vec using numpy arrays
## ---------- Method 1----------- 
#local_array_method_1 = da.getVecArray(local_vec)
#
## Note: da.getVecArray() does NOT return a numpy array
##       local_array_method_1[:] is a numpy array
#local_array_method_1_np = local_array_method_1[:]
#
#print("---------------------")
#print("local_array_method_1_np.shape   = ", local_array_method_1_np.shape)
#print("local_array_method_1_np.strides = ", local_array_method_1_np.strides)
#print("---------------------\n")
#
## ---------- Method 2----------- 
#local_array_method_2_np = local_vec.getArray()
#
## Note: local_vec.getArray() *directly* returns a numpy array of shape 1D and
## size (N_q1 + 2*N_ghost) x (N_q2 + 2*N_ghost) * dof
#
#print("---------------------")
#print("local_array_method_2_np.shape   = ", local_array_method_2_np.shape)
#print("local_array_method_2_np.strides = ", local_array_method_2_np.strides)
#print("---------------------\n")
#
## Now need to reshape
#local_array_method_2_np_reshaped = \
#        local_array_method_2_np.reshape([N_q3+2*N_ghost,
#	                                 N_q2+2*N_ghost,
#                                         N_q1+2*N_ghost,
#                                         dof
#                                        ]
#                                       )
#print("---------------------")
#print("local_array_method_2_np_reshaped.shape   = ", \
#        local_array_method_2_np_reshaped.shape)
#print("local_array_method_2_np_reshaped.strides = ", \
#        local_array_method_2_np_reshaped.strides)
#print("---------------------\n")
#
#print("Conclusion:")
#print(" * Natural order is set by PETSc where dof is the fastest index and q2 is the slowest index")
#print(" * Optimal layout for np arrays is (q3, q2, q1, dof)")
#print(" * Optimal layout for af arrays is (dof, q1, q2, q3)\n")
