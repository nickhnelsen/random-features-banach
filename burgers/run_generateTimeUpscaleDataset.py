import numpy as np
import time
from solve_burgP import solnmap_burgP, solnmap_heatP
from utilities.fileio import fileio_init, fileio_end


def write_data(dir_name, n, m, ntest, K, nu, tau_a, tau_g, al_a, al_g, burg_flag, c, f1, f2, T, origin_path):
    
    # Initialization
    test_temp = np.load(origin_path)
    input_test = test_temp[:,:,0] # all input testing functions
    print('End init., begin writing...')

    # Form upscaled output testing/validation data and write to file
    output_test = np.zeros((K, ntest))
    test = np.zeros((K, ntest, 2))
    if burg_flag:
        for i in range(ntest):
            output_test[:, i] = solnmap_burgP(input_test[:, i], nu=nu, tmax=T, fudge1=f1, fudge2=f2)
    else: # heat
        for i in range(ntest):
            output_test[:, i] = solnmap_heatP(input_test[:, i], nu=nu, tmax=T, c=c)
    test[:, :, 0] = input_test
    del input_test
    test[:, :, 1] = output_test
    del output_test
    np.save(dir_name + 'test.npy', test)
    del test
    print('Finished with: Testing data.')

    # End
    print('Finished with: All data. End program.')
    return


if __name__ == "__main__":
    # Log start time
    start_total_time = time.time() 
    
    # USER INPUT: PARAMETERS
    origin_path = 'datasets/44/test.npy' 
    dataset_num = 3     # dataset number in file directory
    burg_flag = True   # boolean flag: True for Burgers', False for heat/AD equation
    
    # Random feature model hyperparameters
    n = 1024            # number of train data (default: 1024)
    m = 1024            # number of random features (default: 1024)
    ntest = 5000        # number of test data (default: 5000)
    
    # PDE problem setup
    T = 2.0               # final time t=T (1, 1.5, 2)
    K = 1 + 1024         # spatial grid size (default: 1024 for high resolution)
    nu = 10e-3           # viscosity (default: 5e-4, 2.5e-3 to 3.5e-3 for easier problem; 1e-2-3 for heat)
    c = 1.66             # wavespeed for advection-diffusion equation (default: 0.33, 1.66)
    f1 = 0.6*5*5     # advective CFL (factor 5 for nu=5e-4, factor 25 for nu=3.5e-3, 75 for 1e-2)
    f2 = 0.6*5*5      # diffusive CFL (factor 5 for nu=5e-4, factor 25 for nu=3.5e-3, 75 for 1e-2)
    
    # Input measure
    tau_a = 7          # inverse length scale (default: 10, 7)
    al_a = 2.5          # regularity (default: 2.5, 2.5)
    
    # Measure for random feature map
    tau_g = 5         # inverse length scale (default: 7.5, 10, 5, 4)
    al_g = 2          # regularity (default: 1.8, 2, 2, 2)
        
    ################# BEGIN DATA GENERATION #################
    if burg_flag:
        dir_name = 'datasets/timeupscale/' + str(dataset_num) + '/'
        log_str = 'This is a RUN SCRIPT for Burgers solution map time upscaling data generation.\n'
    else: # heat
        dir_name = 'datasets/timeupscale/' + str(dataset_num) + '_heat/'
        log_str = 'This is a RUN SCRIPT for Heat solution map time upscaling data generation.\n'
    hyp_dict = {'n':n, 'm':m, 'ntest':ntest, 'K':K, 'nu':nu, 'tau_a':tau_a, 'tau_g':tau_g, 'al_a':al_a, 'al_g':al_g, 'burg_flag':burg_flag, 'c':c, 'fudge1':f1, 'fudge2':f2, 'T':T, 'origin_path':origin_path}
    
    # Begin log
    log_file, stdoutOrigin = fileio_init(dir_name, hyp_dict)
    print(log_str)
    
    # Save parameter dictionary to numpy array file
    # Reference: https://stackoverflow.com/questions/40219946/python-save-dictionaries-through-numpy-save
    write_data(dir_name, n, m, ntest, K, nu, tau_a, tau_g, al_a, al_g, burg_flag, c, f1, f2, T, origin_path)

    # End log
    print('Total data generation time:', time.time() - start_total_time, 'seconds.') 
    fileio_end(log_file, stdoutOrigin)
    ################# END DATA GENERATION #################
