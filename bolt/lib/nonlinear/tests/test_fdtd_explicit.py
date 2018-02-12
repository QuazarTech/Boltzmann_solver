#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This checks that the explicit time-stepping of the
FDTD algorithm works as intended. Since Maxwell's
equation have wave like solutions, in this test we evolve
the initial state for a single timeperiod and compare the
final solution state with the initial state.

We check the fall off in error with the increase in resolution
(convergence rate) to validate the explicit FDTD algorithm.
"""

import numpy as np
import arrayfire as af
from petsc4py import PETSc

from bolt.lib.nonlinear.fields.fields import fields_solver
from bolt.lib.physical_system import physical_system

from input_files import domain
from input_files import params
from input_files import initialize_fdtd_mode1
from input_files import initialize_fdtd_mode2
from input_files import boundary_conditions

import bolt.src.nonrelativistic_boltzmann.advection_terms as advection_terms
import bolt.src.nonrelativistic_boltzmann.collision_operator as collision_operator
import bolt.src.nonrelativistic_boltzmann.moments as moments

class test(object):

    def __init__(self, N, initialize, params):

        domain.N_q1 = int(N)
        domain.N_q2 = int(N)

        system = physical_system(domain,
                                 boundary_conditions,
                                 params,
                                 initialize,
                                 advection_terms,
                                 collision_operator.BGK,
                                 moments
                                )

        self.fields_solver = fields_solver(system, 
                                           None, 
                                           False
                                          )

        return

def test_fdtd_mode1():

    error_B1 = np.zeros(3)
    error_B2 = np.zeros(3)
    error_E3 = np.zeros(3)

    N = 2**np.arange(5, 8)

    for i in range(N.size):

        dt   = (1 / int(N[i])) * np.sqrt(9/5) / 2
        time = np.arange(dt, np.sqrt(9/5) + dt, dt)

        params.dt = dt

        obj = test(N[i], initialize_fdtd_mode1, params)
        N_g = obj.fields_solver.N_g

        E3_initial = obj.fields_solver.yee_grid_EM_fields[2].copy()
        B1_initial = obj.fields_solver.yee_grid_EM_fields[3].copy()
        B2_initial = obj.fields_solver.yee_grid_EM_fields[4].copy()

        for time_index, t0 in enumerate(time):
            J1 = J2 = J3 = 0 * obj.fields_solver.q1_center**0
            obj.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, dt)
        
        error_B1[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[3, :, N_g:-N_g, N_g:-N_g] -
                                    B1_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B1_initial.elements())

        error_B2[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[4, :, N_g:-N_g, N_g:-N_g] -
                                    B2_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B2_initial.elements())

        error_E3[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[2, :, N_g:-N_g, N_g:-N_g] -
                                    E3_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E3_initial.elements())

    poly_B1 = np.polyfit(np.log10(N), np.log10(error_B1), 1)
    poly_B2 = np.polyfit(np.log10(N), np.log10(error_B2), 1)
    poly_E3 = np.polyfit(np.log10(N), np.log10(error_E3), 1)

    assert (abs(poly_B1[0] + 2) < 0.2)
    assert (abs(poly_B2[0] + 2) < 0.2) 
    assert (abs(poly_E3[0] + 2) < 0.2)

def test_fdtd_mode2():

    error_E1 = np.zeros(3)
    error_E2 = np.zeros(3)
    error_B3 = np.zeros(3)

    N = 2**np.arange(5, 8)

    for i in range(N.size):

        dt   = (1 / int(N[i])) * np.sqrt(9/5) / 2
        time = np.arange(dt, np.sqrt(9/5) + dt, dt)

        params.dt = dt

        obj = test(N[i], initialize_fdtd_mode2, params)
        N_g = obj.fields_solver.N_g

        B3_initial = obj.fields_solver.yee_grid_EM_fields[5].copy()
        E1_initial = obj.fields_solver.yee_grid_EM_fields[0].copy()
        E2_initial = obj.fields_solver.yee_grid_EM_fields[1].copy()

        for time_index, t0 in enumerate(time):
            J1 = J2 = J3 = 0 * obj.fields_solver.q1_center**0
            obj.fields_solver.evolve_electrodynamic_fields(J1, J2, J3, dt)

        error_E1[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[0, :, N_g:-N_g, N_g:-N_g] -
                                    E1_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E1_initial.elements())

        error_E2[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[1, :, N_g:-N_g, N_g:-N_g] -
                                    E2_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (E2_initial.elements())

        error_B3[i] = af.sum(af.abs(obj.fields_solver.yee_grid_EM_fields[5, :, N_g:-N_g, N_g:-N_g] -
                                    B3_initial[:, :, N_g:-N_g, N_g:-N_g]
                                   )
                            ) / (B3_initial.elements())

    poly_E1 = np.polyfit(np.log10(N), np.log10(error_E1), 1)
    poly_E2 = np.polyfit(np.log10(N), np.log10(error_E2), 1)
    poly_B3 = np.polyfit(np.log10(N), np.log10(error_B3), 1)

    assert (abs(poly_E1[0] + 2) < 0.2)
    assert (abs(poly_E2[0] + 2) < 0.2)
    assert (abs(poly_B3[0] + 2) < 0.2)