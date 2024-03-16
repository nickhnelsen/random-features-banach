import numpy as np
import time
from GRF1D import GaussianRandomField1D
from solve_burgP import solnmap_burgP, solnmap_heatP
from utilities.fileio import fileio_init, fileio_end


def write_data(dir_name, n, m, ntest, K, nu, tau_a, tau_g, al_a, al_g, seed, burg_flag, c, f1, f2, T):
    
    # Initialization
    np.random.seed(seed)
    print('End init., begin writing...')
    
    # Form Gaussian random fields for the random feature mappings and write to file
    grf_rf = GaussianRandomField1D(tau_g, al_g, bc=2)
    theta_g = np.random.standard_normal((K, m, 2))      # features (two GRFs for each j=1,..,m)
    grf_g = np.zeros(theta_g.shape)                     # GRF data structure
    for i in range(m):
        grf_g[:, i, 0] = grf_rf.draw(theta_g[:, i, 0])
        grf_g[:, i, 1] = grf_rf.draw(theta_g[:, i, 1])
    np.save(dir_name + 'grftheta.npy', grf_g)
    del grf_g, theta_g, grf_rf
    print('Finished with: GRF data.')

    # Form training data and write to file
    grf_a = GaussianRandomField1D(tau_a, al_a, bc=2)
    theta_a = np.random.standard_normal((K, n))
    input_train = np.zeros((K, n))
    output_train = np.zeros((K, n))
    train = np.zeros((K, n, 2))
    if burg_flag:
        for i in range(n):
            input_train[:, i] = grf_a.draw(theta_a[:, i])
            output_train[:, i] = solnmap_burgP(input_train[:, i], nu=nu, tmax=T, fudge1=f1, fudge2=f2)
    else: # heat
        for i in range(n):
            input_train[:, i] = grf_a.draw(theta_a[:, i])
            output_train[:, i] = solnmap_heatP(input_train[:, i], nu=nu, tmax=T, c=c)
    del theta_a
    train[:, :, 0] = input_train
    del input_train
    train[:, :, 1] = output_train
    del output_train
    np.save(dir_name + 'train.npy', train)
    del train
    print('Finished with: Training data.')

    # Form testing/validation data and write to file
    theta_a_test = np.random.standard_normal((K, ntest))
    input_test = np.zeros((K, ntest))
    output_test = np.zeros((K, ntest))
    test = np.zeros((K, ntest, 2))
    if burg_flag:
        for i in range(ntest):
            input_test[:, i] = grf_a.draw(theta_a_test[:, i])
            output_test[:, i] = solnmap_burgP(input_test[:, i], nu=nu, tmax=T, fudge1=f1, fudge2=f2)
    else: # heat
        for i in range(ntest):
            input_test[:, i] = grf_a.draw(theta_a_test[:, i])
            output_test[:, i] = solnmap_heatP(input_test[:, i], nu=nu, tmax=T, c=c)
    del theta_a_test
    test[:, :, 0] = input_test
    del input_test
    test[:, :, 1] = output_test
    del output_test
    np.save(dir_name + 'test.npy', test)
    del test, grf_a
    print('Finished with: Testing data.')

    # End
    print('Finished with: All data. End program.')
    return


if __name__ == "__main__":
    # Log start time
    start_total_time = time.time() 
    
    # USER INPUT: PARAMETERS
    dataset_num = 7     # dataset number in file directory
    burg_flag = True   # boolean flag: True for Burgers', False for heat/AD equation
    
    # Random feature model hyperparameters
    n = 1024            # number of train data (default: 1024)
    m = 1024            # number of random features (default: 1024)
    ntest = 5000        # number of test data (default: 5000)
    
    # PDE problem setup
    T = 0.5               # final time t=T (default: 1, 0.33, 0.5, 0.75)
    K = 1 + 1024         # spatial grid size (default: 1024 for high resolution)
    nu = 10e-3           # viscosity (default: 5e-4, 2.5e-3 to 3.5e-3 for easier problem; 1e-2-3 for heat)
    c = 1.66             # wavespeed for advection-diffusion equation (default: 0.33, 1.66)
    f1 = 0.6*5     # advective CFL (factor 5 for nu=5e-4, factor 25 for nu=3.5e-3, 75 for 1e-2)
    f2 = 0.6*5      # diffusive CFL (factor 5 for nu=5e-4, factor 25 for nu=3.5e-3, 75 for 1e-2)
    
    # Input measure
    tau_a = 7          # inverse length scale (default: 10, 7)
    al_a = 2.5          # regularity (default: 2.5, 2.5)
    
    # Measure for random feature map
    tau_g = 5         # inverse length scale (default: 7.5, 10, 5, 4)
    al_g = 2          # regularity (default: 1.8, 2, 2, 2)
    
    # Random seeding
    seed = 1234321 + dataset_num         # fixed seed for random number generator
    
    ################# BEGIN DATA GENERATION #################
    if burg_flag:
        dir_name = 'datasets/' + str(dataset_num) + '/'
        log_str = 'This is a RUN SCRIPT for Burgers solution map data generation.\n'
    else: # heat
        dir_name = 'datasets/' + str(dataset_num) + '_heat/'
        log_str = 'This is a RUN SCRIPT for Heat solution map data generation.\n'
    hyp_dict = {'n':n, 'm':m, 'ntest':ntest, 'K':K, 'nu':nu, 'tau_a':tau_a, 'tau_g':tau_g, 'al_a':al_a, 'al_g':al_g, 'seed':seed, 'burg_flag':burg_flag, 'c':c, 'fudge1':f1, 'fudge2':f2, 'T':T}
    
    # Begin log
    log_file, stdoutOrigin = fileio_init(dir_name, hyp_dict)
    print(log_str)
    
    # Save parameter dictionary to numpy array file
    # Reference: https://stackoverflow.com/questions/40219946/python-save-dictionaries-through-numpy-save
    np.save(dir_name + 'params.npy', hyp_dict)
    write_data(dir_name, n, m, ntest, K, nu, tau_a, tau_g, al_a, al_g, seed, burg_flag, c, f1, f2, T)

    # End log
    print('Total data generation time:', time.time() - start_total_time, 'seconds.') 
    fileio_end(log_file, stdoutOrigin)
    ################# END DATA GENERATION #################
