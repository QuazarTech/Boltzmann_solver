import numpy as np
import params
import arrayfire as af
import pylab as pl

import domain

from bolt.src.electronic_boltzmann.utils.polygon import polygon

p2_start = -np.pi
p2_end   = np.pi
N_p2     = domain.N_p2
theta = \
    p2_start + (0.5 + np.arange(N_p2))*(p2_end - p2_start)/N_p2

theta = af.from_ndarray(theta)

hexa = polygon(6, theta, rotation = np.pi/6)
hexa = hexa.to_ndarray()

#pl.polar(theta, hexa)

#pl.plot(theta, hexa * np.cos(theta))
#pl.plot(theta, hexa * np.sin(theta))

pl.gca().set_aspect(1)
pl.plot(hexa*np.cos(theta), hexa*np.sin(theta))

#p_hat_0_old = np.loadtxt("P_hat_0_old.txt")
#p_hat_1_old = np.loadtxt("P_hat_1_old.txt")

#p_hat_0 = np.loadtxt("P_hat_0.txt")
#p_hat_1 = np.loadtxt("P_hat_1.txt")

#pl.plot(theta, p_hat_0_old)
#pl.plot(theta, p_hat_1_old)

#pl.plot(theta, p_hat_0, '--')
#pl.plot(theta, p_hat_1, '--')


pl.savefig("images/test.png")
