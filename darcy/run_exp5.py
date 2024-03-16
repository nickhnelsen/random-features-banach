"""
A run script for experiment 5: expected relative error vs number of training samples
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all') # close all open figures
import time

# Import common function files from subdirectories
from utilities.plot_suite import Plotter
from utilities.fileio import fileio_init, fileio_end
plotter = Plotter() # set plotter class
from RFM_largedata import RandomFeatureModel
    
    
# Loop for the experiment
def exp5(n_list, K=65, m=75, ntest=1000, lamreg=0, sig_plus=1/12, sig_minus=-1/3, denom_scale=0.1, params_path='data/params.mat', data_path='data/data.mat', rf_choice=0):
    '''
    Expected relative error vs number of training samples
    '''
    N = len(n_list)
    er_store = np.zeros([N, 5]) # 5 columns: n, e_train, b_train, e_test, b_test
    for loop in range(N):
        n = n_list[loop]
        er_store[loop,0] = n
        print('Running n =', n)
        start_total_time = time.time() 

        # Train
        rfm = RandomFeatureModel(K, n, m, ntest, lamreg, sig_plus, sig_minus, denom_scale, params_path, data_path, rf_choice)
        rfm.fit()
        
        # Train error
        er_store[loop, 1:3] = rfm.relative_error_train()
        
        # Test error
        er_store[loop, 3:] = rfm.relative_error_test()
        
        # Print
        print('Time elapsed: ', time.time() - start_total_time, 'seconds.') 
        print('Best expected relative error (Train, Test):' , (er_store[loop, 1], er_store[loop, 3]), '\n')
    
    return er_store    


def exp5rates(m_list=[10*i for i in range(1,11)], n=100, K=65, ntest=500, lamreg=0, sig_plus=1/12, sig_minus=-1/3, denom_scale=0.1, params_path='data/params.mat', data_path='data/data.mat', rf_choice=0):
    '''
    Expected relative error vs number of random features for fixed sample size
    '''
    M = len(m_list)
    er_store = np.zeros([M, 5]) # 5 columns: n, e_train, b_train, e_test, b_test
    for loop in range(M):
        m = m_list[loop]
        er_store[loop,0] = m
        print('Running m =', m)
        start_total_time = time.time() 
    
        # Train
        rfm = RandomFeatureModel(K, n, m, ntest, lamreg, sig_plus, sig_minus, denom_scale, params_path, data_path, rf_choice)
        rfm.fit()
    # =============================================================================
    #         er_store_reg = rfm.regsweep([1e-8, 1e-9, 1e-10])
    #         ind_arr = np.argmin(er_store_reg, axis=0)[3] # smallest test index
    # =============================================================================
        
    # =============================================================================
    #         # Train error
    #         er_store[loop, 1:3] = er_store_reg[ind_arr, 1:3]
    #         
    #         # Test error
    #         er_store[loop, 3:] = er_store_reg[ind_arr, 3:]
    # =============================================================================
        # Train error
        er_store[loop, 1:3] = rfm.relative_error_train()
        
        # Test error
        er_store[loop, 3:] = rfm.relative_error_test()
        
        # Print
        print('Time elapsed: ', time.time() - start_total_time, 'seconds.') 
        print('Best expected relative error (Train, Test):' , (er_store[loop, 1], er_store[loop, 3]), '\n')
    
    return er_store    


# Run main script
if __name__ == "__main__":
    # Log start time for program
    start_total_time = time.time() 
    
    # Fixed hyperparameters
    K = 1 + 64
    ntest = 500
    lamreg_guess = 1e-9
    sig_plus = 1/12
    sig_minus = -1/3
    denom_scale = 0.10
    params_path = 'data/params.mat'
    data_path = 'data/data.mat'
    rf_choice = 0
    if rf_choice == 0:
        rf_type = 'pc'
    else:
        rf_type = 'flow'

    # USER INPUT
    exp_num = 555
    test_type = 1
    if test_type==0:    # n vs error
        m = 10
        n_list = [1, 3, 5] + [10*i for i in range(1, 11)]
        hyp_dict = {'m':m, 'n_list':n_list, 'K':K, 'ntest':ntest, 'lamreg_guess':lamreg_guess, 'sig_plus':sig_plus, 'sig_minus':sig_minus, 'denom_scale':denom_scale, 'params_path':params_path, 'data_path':data_path, 'rf_type':rf_type}
        descrip_str = 'mrand_sweep_' + str(m)
        str_begin = 'RUN SCRIPT for experiment 5: expected relative error vs number of training samples.\n'

    else:               # m vs error
        n = 100
        m_list = [10*i for i in range(1,11)]
        hyp_dict = {'m_list':m_list, 'n':n, 'K':K, 'ntest':ntest, 'lamreg_guess':lamreg_guess, 'sig_plus':sig_plus, 'sig_minus':sig_minus, 'denom_scale':denom_scale, 'params_path':params_path, 'data_path':data_path, 'rf_type':rf_type}
        descrip_str = 'mrate_sweep_n' + str(n)
        str_begin = 'RUN SCRIPT for experiment 5: expected relative error vs number of random features.\n'

    dir_name = 'experiments/exp' + str(exp_num) + '/' + descrip_str + '/'
  
    # Begin log
    log_file, stdoutOrigin = fileio_init(dir_name, hyp_dict)
    print(str_begin)
    if test_type==0:
        results = exp5(n_list, K, m, ntest, lamreg_guess, sig_plus, sig_minus, denom_scale, params_path, data_path, rf_choice)
        # Write to file
        leg_str = r'$m = %d$' % m
        np.save(dir_name + 'results.npy', results)
        f1 = plotter.plot_oneD(1, results[:,0], results[:,3], xlab_str1D=r'$n$, number of training samples', ylab_str1D=r'Expected relative test error', semilogy=True, legendlab_str=leg_str)
        plotter.save_plot(f1, dir_name + 'err.pdf')
    else:
        results = exp5rates(m_list, n, K, ntest, lamreg_guess, sig_plus, sig_minus, denom_scale, params_path, data_path, rf_choice)
        # Write to file
        leg_str = r'$n = %d$' % n
        np.save(dir_name + 'results_mrate.npy', results)
        f1 = plotter.plot_oneD(1, results[:,0], results[:,3], xlab_str1D=r'$m$, number of random features', ylab_str1D=r'Expected relative test error', semilogy=True, legendlab_str=leg_str)
        plotter.save_plot(f1, dir_name + 'err_mrate.pdf')
    
    # End log
    print('Total Script Runtime: ', time.time() - start_total_time, 'seconds.') 
    fileio_end(log_file, stdoutOrigin)
