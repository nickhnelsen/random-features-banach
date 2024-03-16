import numpy as np
import time

# Custom imports to this file
# import inspect
from RFM_burg import RandomFeatureModel
from utilities.fileio import fileio_init, fileio_end


# Run main script
if __name__ == "__main__":
    
    # USER INIT
    TODAY = 20200507
    exp_num = 5
    dataset_num = 4
    QUAD_ORDER = 1
    K = 129
    # n_list = np.array([10, 100, 300, 500, 1000])
    # n_list = np.array([20, 30, 50])
    m_list = np.array([16, 32, 64, 128, 256, 512, 1024])
    ntest = 4000
    lamreg = 0
    al_rf = 4.0                   # default: 4.0, 1.2, 1.5, 2.0
    nu_rf = 2.5e-3              # default: 2.5e-3, 1e-3, 5e-4, 1e-4
    
    # Log start time
    start_total_time = time.time()
    
    # Derived
    dir_path='datasets/' + str(dataset_num) + '/'
    hyp_dict = {'K':K,'m_list':m_list, 'n_list':n_list, 'ntest':ntest, 'lamreg':lamreg, 'al_rf':al_rf, 'nu_rf':nu_rf, 'data_path':dir_path, 'quad_order':QUAD_ORDER}
    str_begin = 'RUN SCRIPT for RFM m vs n sweep for Burgers equation.\n'
    save_path = 'experiments/' + str(TODAY) + '/mvsn' + str(exp_num) + '/'
    
    # Begin log
    log_file, stdoutOrigin = fileio_init(save_path, hyp_dict)
    print(str_begin)
    
    # Compute
    e_store = np.zeros([len(n_list), len(m_list)])
    for loop in range(len(n_list)):
        n = n_list[loop]
        print('Outer loop, n =', n)
        for ind in range(len(m_list)):
            m = m_list[ind]
            print('Inner loop, m =', m)
            start = time.time() 
            rfm = RandomFeatureModel(K, n, m, ntest, lamreg, nu_rf, al_rf, dir_path=dir_path)
            rfm.fit()
            etest_temp, _ = rfm.relative_error_test()
            print('Expected relative test error:', etest_temp)
            e_store[loop, ind] = etest_temp
            print('Inner loop time elapsed: ', time.time() - start, 'seconds.') # print run time
    
    np.save(save_path + 'error_mvsn.npy', e_store)
    
    # End log
    print('Total Script Runtime: ', time.time() - start_total_time, 'seconds.') 
    # print('The choice of RF map is shown in the function defn. below:\n') # Print RF function used
    # print(inspect.getsource(rf_fourier))
    fileio_end(log_file, stdoutOrigin)
