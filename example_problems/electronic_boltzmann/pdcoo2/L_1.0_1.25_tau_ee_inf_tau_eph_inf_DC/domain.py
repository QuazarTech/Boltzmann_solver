import numpy as np
import params

q1_start = 0.
q1_end   = 1.0
N_q1     = 20

q2_start = 0.
q2_end   = 1.25
N_q2     = 25

# If N_p1 > 1, mirror boundary conditions require p1 to be
# symmetric about zero
# TODO : Check and fix discrepancy between this and the claim
# that p1_center = mu in polar representation
N_p1     =  1 # Set equal to 1 for 1D polar

# In the cartesian representation of momentum space,
# p1 = p_x (magnitude of momentum)
# p1_start and p1_end are set such that p1_center is 0

# Uncomment the following for the cartesian representation of momentum space
#p1_start = [-0.04]
#p1_end   =  [0.04]


# In the 2D polar representation of momentum space,
# p1 = p_r (magnitude of momentum)
# p1_start and p1_end are set such that p1_center is mu

# Uncomment the following for the 2D polar representation of momentum space
#p1_start = [params.initial_mu - \
#        16.*params.boltzmann_constant*params.initial_temperature]
#p1_end   = [params.initial_mu + \
#        16.*params.boltzmann_constant*params.initial_temperature]

# Uncomment the following for the 1D polar representation of momentum space
p1_start = [0.5*params.initial_mu/params.fermi_velocity]
p1_end   = [1.5*params.initial_mu/params.fermi_velocity]


# If N_p2 > 1, mirror boundary conditions require p2 to be
# symmetric about zero
N_p2     =  2048

# In the cartesian representation of momentum space,
# p2 = p_y (magnitude of momentum)
# p2_start and p2_end are set such that p2_center is 0
#p2_start = [-0.04]
#p2_end   =  [0.04]

# In the 2D polar representation of momentum space,
# p2 = p_theta (angle of momentum)
# N_p_theta MUST be even.
#p2_start =  [-np.pi]
#p2_end   =  [np.pi]
p2_start =  [-3.14159265359]
p2_end   =  [3.14159265359]

# If N_p3 > 1, mirror boundary conditions require p3 to be
# symmetric about zero

p3_start = [-0.5]
p3_end   =  [0.5]
N_p3     =  1

N_ghost = 2
