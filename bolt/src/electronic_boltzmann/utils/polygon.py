import arrayfire as af
import numpy as np

def polygon(n, theta, rotation = 0, shift = 0):
    '''
    Returns a polygon of unit edge length on a polar coordinate system.
    Inputs : 

    n        : number of sides of the polygon
    thera    : the angle grid
    rotation : initial rotation
    shift    : initial shift in center ##TODO
    '''

    numerator   = np.cos(np.pi/n)
    denominator = af.cos((theta - rotation) - (2*np.pi/n)*af.floor((n*(theta - rotation) + np.pi)/(2*np.pi)))

    result = numerator/denominator

    return (result)


