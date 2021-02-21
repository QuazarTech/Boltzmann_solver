import os
import numpy as np

# TODO : Set t_final and number of gpus in jobscript depending on device size
start_index = 1.50 + 0.025
end_index   = 1.5751
step        = 0.05

current_filepath = \
        '/home/mchandra/gitansh/bolt_master/example_problems/electronic_boltzmann/pdcoo2/ref_rotated'
source_filepath = \
        '/home/mchandra/gitansh/bolt_master/example_problems/electronic_boltzmann/pdcoo2/ref_rotated'

for index in np.arange(start_index, end_index, step):
    filepath = \
    '/home/mchandra/gitansh/bolt_master/example_problems/electronic_boltzmann/pdcoo2/L_1.0_%.3f_tau_ee_inf_tau_eph_inf_DC_rotated'%index

    # If folder does not exist, make one and add all files from source folder

    if not os.path.isdir(filepath):
        os.mkdir(filepath)
        os.mkdir(filepath+"/dump_f")
        os.mkdir(filepath+"/dump_moments")
        os.mkdir(filepath+"/dump_lagrange_multipliers")
        os.mkdir(filepath+"/images")

        os.system("cp " + source_filepath + "/* " + filepath+"/.")
   
        # Change required files
        # Code copied from here : 
        # https://stackoverflow.com/questions/4454298/prepend-a-line-to-an-existing-file-in-python
        
        f = open(filepath + "/domain.py", "r")
        old = f.read() # read everything in the file
        f.close() 
        
        f = open(filepath + "/domain.py", "w")
        f.write("q1_end = " + str(index) + " \n")
        f.write(old)
        f.close()

    # Schedule job after changing to run directory so that generated slurm file
    # is stored in that directory
    os.chdir(filepath)
    os.system("sbatch job_script")
    os.chdir(current_filepath) # Return to job firing script's directory

