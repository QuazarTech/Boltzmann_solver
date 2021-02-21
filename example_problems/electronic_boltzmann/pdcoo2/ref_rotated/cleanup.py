import os
import numpy as np
import glob

# TODO : Set t_final and number of gpus in jobscript depending on device size
start_index = 3.00
end_index   = 3.5001
step        = 0.1


for index in np.arange(start_index, end_index, step):
    print (index)
    filepath = \
    '/home/mchandra/gitansh/bolt_master/example_problems/electronic_boltzmann/pdcoo2/L_1.0_%.2f_tau_ee_inf_tau_eph_inf_DC_rotated'%index

    # List all dump_f files
    dist_func_files = np.sort(glob.glob(filepath+'/dump_f/*.bin'))
    dist_func_info_files = np.sort(glob.glob(filepath+'/dump_f/*.bin.info'))

    # Save first and last files
    os.system("mkdir " + filepath + "/dump_f/save")
    os.system("mv " + str(dist_func_files[0]) + " " + filepath+"/dump_f/save/.")
    os.system("mv " + str(dist_func_files[-1]) + " " + filepath+"/dump_f/save/.")
    os.system("mv " + str(dist_func_info_files[0]) + " " + filepath+"/dump_f/save/.")
    os.system("mv " + str(dist_func_info_files[-1]) + " " + filepath+"/dump_f/save/.")
    os.system("rm " + filepath + "/dump_f/t*")

    # Remove all files in dump_f except first and last dist funcs
    #os.system("rm " + filepath+"/output.txt")
