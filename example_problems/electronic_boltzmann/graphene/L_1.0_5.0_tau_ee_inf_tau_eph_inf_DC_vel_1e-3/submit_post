#!/bin/bash
#SBATCH -t 16:00:00       # walltime                                             
#SBATCH -N 8              # Number of nodes                                      
#SBATCH -n 32             # Number of MPI ranks                                   
#SBATCH -o out_%j         # Pathname of stdout                                   
#SBATCH -e err_%j         # Pathname of stderr                                   
#SBATCH -A w19_mcpgpu     # Allocation                                           
#SBATCH --qos=standard    # QoS (standard, interactive, standby, large*, long*, high*)
                          # See https://hpc.lanl.gov/scheduling_policies         
#SBATCH --mail-user=mc0710@gmail.com
#SBATCH --mail-type=BEGIN                                                        
#SBATCH --mail-type=END                                                          
#SBATCH --mail-type=FAIL                                                         
                                                                                 
module list                                                                      
pwd                                                                              
date                                                                             
                                                                                 
# Arrayfire library                                                              
export AF_PATH=/usr/projects/p18_ebhlight3d/arrayfire/arrayfire_install          
export PETSC_DIR=/usr/projects/p18_ebhlight3d/petsc_3.10.0_install               
export LD_LIBRARY_PATH=$AF_PATH/lib64:$LD_LIBRARY_PATH                           
                                                                                 
# Bolt as library                                                                
export PYTHONPATH=$PYTHONPATH:/users/manic/bolt_master
                                                                                 
source activate /usr/projects/p18_ebhlight3d/bolt_env_2019.1/                    
                                                                                 
srun python main.py
