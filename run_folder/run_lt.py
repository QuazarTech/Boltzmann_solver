import setup_simulation
import lts.initialize as initialize
import lts.evolve as evolve
import lts.export as export
import h5py
import params

config     = setup_simulation.configuration_object(params)
time_array = setup_simulation.time_array(config)

delta_f_hat_initial              = initialize.init_delta_f_hat(config)
delta_rho_hat, delta_f_hat_final = evolve.time_integration(config, delta_f_hat_initial, time_array)

f_dist = export.export_5D_distribution_function(config, delta_f_hat_final)
h5f    = h5py.File('lt_distribution_function.h5', 'w')
h5f.create_dataset('distribution_function', data = f_dist)
h5f.close()

h5f = h5py.File('lt_density_data.h5', 'w')
h5f.create_dataset('density_amplitude', data = delta_rho_hat)
h5f.create_dataset('time', data = time_array)
h5f.close()