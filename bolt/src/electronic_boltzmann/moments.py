#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import arrayfire as af

from bolt.src.utils.integral_over_p import integral_over_p

import params

def density(f, p1, p2, p3, integral_measure):
    theta = af.atan(params.p_y / params.p_x)
    p_f   = params.fermi_momentum_magnitude(theta)
    return(integral_over_p(f+p_f/2, integral_measure))

def j_x(f, p1, p2, p3, integral_measure):
    theta = af.atan(params.p_y / params.p_x)
    p_f   = params.fermi_momentum_magnitude(theta)
    return(integral_over_p(f * params.fermi_velocity * params.p_x/p_f, integral_measure))

def j_y(f, p1, p2, p3, integral_measure):
    theta = af.atan(params.p_y / params.p_x)
    p_f   = params.fermi_momentum_magnitude(theta)
    return(integral_over_p(f * params.fermi_velocity * params.p_y/p_f, integral_measure))
