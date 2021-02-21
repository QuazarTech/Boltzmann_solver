#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

import arrayfire as af

def integral_over_p(array, integral_measure):
    return(af.sum(array*integral_measure, 0))
