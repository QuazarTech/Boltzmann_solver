import numpy as np
import arrayfire as af

from bolt.src.electronic_boltzmann.utils.polygon import polygon
from bolt.src.electronic_boltzmann.utils.unit_vectors import normal_to_hexagon_unit_vec

instantaneous_collisions = False #TODO : Remove from lib
hybrid_model_enabled     = False #TODO : Remove from lib
source_enabled           = True
disable_collision_op     = False

fields_enabled = False
# Can be defined as 'electrostatic', 'user-defined'.
# The initial conditions need to be specified under initialize
# Ensure that the initial conditions specified satisfy
# Maxwell's constraint equations
fields_initialize = 'user-defined'

# Can be defined as 'electrostatic' and 'fdtd'
# To turn feedback from Electric fields on, set fields_solver = 'LCA'
# and set charge_electron
fields_type   = 'electrostatic'
fields_solver = 'SNES'

# Can be defined as 'strang' and 'lie'
time_splitting = 'strang'

# Method in q-space
solver_method_in_q = 'FVM'
solver_method_in_p = 'FVM'

reconstruction_method_in_q = 'minmod'
reconstruction_method_in_p = 'minmod'

riemann_solver_in_q = 'upwind-flux'
riemann_solver_in_p = 'upwind-flux'

electrostatic_solver_every_nth_step = 1000000


# Time parameters:
dt      = 0.025/4 # ps
t_final = 25.     # ps


# File-writing Parameters:
dump_steps = 5
dump_dist_after = 1600
# Set to zero for no file-writing
dt_dump_f       = 1000*dt #ps
# ALWAYS set dump moments and dump fields at same frequency:
dt_dump_moments = dt_dump_fields = 5*dt #ps


# Dimensionality considered in velocity space:
p_dim = 1
p_space_grid = 'polar2D' # Supports 'cartesian' or 'polar2D' grids
# Set p-space start and end points accordingly in domain.py
#TODO : Use only polar2D for PdCoO2


zero_temperature    = (p_dim==1)
dispersion          = 'linear' # 'linear' or 'quadratic'
fermi_surface_shape = 'circle' # Supports 'circle' or 'hexagon'


# Number of devices(GPUs/Accelerators) on each node:
num_devices = 6

# Constants:
mass_particle      = 0.910938356 # x 1e-30 kg
h_bar              = 1.0545718e-4 # x aJ ps
boltzmann_constant = 1
charge             = [0.*-0.160217662] # x aC
mass               = [0.] #TODO : Not used in electronic_boltzmann
                          # Remove from lib
speed_of_light     = 300. # x [um/ps]
fermi_velocity     = speed_of_light/300
epsilon0           = 8.854187817 # x [aC^2 / (aJ um) ]

epsilon_relative      = 3.9 # SiO2
backgate_potential    = -10 # V
global_chem_potential = 0.03
contact_start         = 0.0 # um
contact_end           = 0.25 # um
contact_geometry      = "straight" # Contacts on either side of the device
                                   # For contacts on the same side, use 
                                   # contact_geometry = "turn_around"

initial_temperature = 12e-5
initial_mu          = 0.015
vel_drift_x_in      = 1e-4*fermi_velocity
vel_drift_x_out     = 1e-4*fermi_velocity
AC_freq             = 1./100 # ps^-1

B3_mean = 1. # T

# Spatial quantities (will be initialized to shape = [q1, q2] in initalize.py)
mu          = None # chemical potential used in the e-ph operator
T           = None # Electron temperature used in the e-ph operator
mu_ee       = None # chemical potential used in the e-e operator
T_ee        = None # Electron temperature used in the e-e operator
vel_drift_x = None
vel_drift_y = None
phi         = None # Electric potential in the plane of graphene sheet
p_x         = None
p_y         = None
#integral_measure = None

# Momentum quantities (will be initialized to shape = [p1*p2*p3] in initialize.py)
E_band   = None
vel_band = None

collision_operator_nonlinear_iters  = 2

# Variation of collisional-timescale parameter through phase space:
@af.broadcast
def tau_defect(q1, q2, p1, p2, p3):
    return(np.inf * q1**0 * p1**0)

@af.broadcast
def tau_ee(q1, q2, p1, p2, p3):
    return(np.inf * q1**0 * p1**0)

def tau(q1, q2, p1, p2, p3):
    return(tau_defect(q1, q2, p1, p2, p3))


def fermi_momentum_magnitude(theta):
    if (fermi_surface_shape == 'circle'):
        p_f = initial_mu/fermi_velocity # Fermi momentum
    
    elif (fermi_surface_shape == 'hexagon'):
        n = 6 # No. of sides of polygon
        p_f = (initial_mu/fermi_velocity) * polygon(n, theta, rotation = np.pi/6)
        # Note : Rotation by pi/6 results in a hexagon with horizontal top & bottom edges
        #TODO : If cartesian coordinates are being used, convert to polar to calculate p_f
    else : 
        raise NotImplementedError('Unsupported shape of fermi surface')
    return(p_f)


def band_energy(p1, p2):
    # Note :This function is only meant to be called once to initialize E_band

    if (p_space_grid == 'cartesian'):
        p_x = p1
        p_y = p2
    elif (p_space_grid == 'polar2D'):
    	# In polar2D coordinates, p1 = radius and p2 = theta
        r = p1
        theta = p2
        p_x = r * af.cos(theta)
        p_y = r * af.sin(theta)
    else : 
        raise NotImplementedError('Unsupported coordinate system in p_space')
    
    p = af.sqrt(p_x**2. + p_y**2.)
    if (dispersion == 'linear'):

        E_upper = p*fermi_velocity

    elif (dispersion == 'quadratic'):
    
        m = effective_mass(p1, p2)
        E_upper = p**2/(2.*m)

    if (zero_temperature):

        E_upper = initial_mu * p**0.

    af.eval(E_upper)
    return(E_upper)


def effective_mass(p1, p2):

    if (p_space_grid == 'cartesian'):
        p_x = p1
        p_y = p2
        
        theta = af.atan(p_y/p_x)

    elif (p_space_grid == 'polar2D'):
    	# In polar2D coordinates, p1 = radius and p2 = theta
        r = p1; theta = p2
    else : 
        raise NotImplementedError('Unsupported coordinate system in p_space')
    
    if (fermi_surface_shape == 'hexagon'):
        
        n = 6 # No. of side of polygon
        mass = mass_particle * polygon(n, theta, rotation = np.pi/6)
        # Note : Rotation by pi/6 results in a hexagon with horizontal top & bottom edges

    elif (fermi_surface_shape == 'circle'):
        
    # For now, just return the free electron mass
        mass = mass_particle

    return(mass)

def band_velocity(p1, p2):
    # Note :This function is only meant to be called once to initialize the vel vectors

    if (p_space_grid == 'cartesian'):
        p_x_local = p1
        p_y_local = p2
        
        theta = af.atan(p_y_local/p_x_local)

    elif (p_space_grid == 'polar2D'):
    	# In polar2D coordinates, p1 = radius and p2 = theta
        r = p1; theta = p2
    else : 
        raise NotImplementedError('Unsupported coordinate system in p_space')
    
    p     = af.sqrt(p_x**2. + p_y**2.)
    p_hat = [p_x / (p + 1e-20), p_y / (p + 1e-20)]

    if (fermi_surface_shape == 'circle'):

        v_f_hat = p_hat

    elif (fermi_surface_shape == 'hexagon'):

        v_f_hat = normal_to_hexagon_unit_vec(theta)

    # Quadratic dispersion        
    m = effective_mass(p1, p2)
    v_f = p/m

    if (dispersion == 'linear' or zero_temperature):

        v_f = fermi_velocity

    upper_band_velocity = [v_f * v_f_hat[0], v_f * v_f_hat[1]]

    return(upper_band_velocity)

def get_p_x_and_p_y(p1, p2):

    if (p_space_grid == 'cartesian'):
        p_x = p1
        p_y = p2
    elif (p_space_grid == 'polar2D'):
    	# In polar2D coordinates, p1 = radius and p2 = theta
        r = p1; theta = p2
        
        if (zero_temperature):
            # Get p_x and p_y at the Fermi surface
            r = fermi_momentum_magnitude(theta)
            
        p_x = r * af.cos(theta)
        p_y = r * af.sin(theta)

    else : 
        raise NotImplementedError('Unsupported coordinate system in p_space')

    return([p_x, p_y])

# Restart(Set to zero for no-restart):
latest_restart = True
t_restart = 0

@af.broadcast
def fermi_dirac(mu, E_band):

    k = boltzmann_constant
    T = initial_temperature

    f = (1./(af.exp( (E_band - mu
                     )/(k*T) 
                   ) + 1.
            )
        )

    af.eval(f)
    return(f)
