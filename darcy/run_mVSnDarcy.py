import numpy as np
import time

# Custom imports to this file
from RFM_largedata import RandomFeatureModel
from utilities.fileio import fileio_init, fileio_end


# Run main script
if __name__ == "__main__":
    
    # USER INIT
    TODAY = 20200511
    exp_num = 4
    K = 1 + 32
    n_list = np.array([5, 10, 20, 30, 50, 100, 300, 500])
    # n_list = np.array([300, 500])
    # m_list = np.array([8, 16, 32, 64, 128, 256, 512])
    m_list = np.array([512]) # 512, 1024
    ntest = 1000
    lamreg = 1e-8
    sig_plus = 1/12
    sig_minus = -1/3
    denom_scale = 0.15
    QUAD_ORDER = 1
    
    # Log start time
    start_total_time = time.time()
    
    # Derived
    hyp_dict = {'m_list':m_list, 'n':n_list, 'K':K, 'ntest':ntest, 'lamreg_init':lamreg, 'sig_plus':sig_plus,'sig_minus':sig_minus, 'denom_scale':denom_scale, 'rf_choice':0, 'quad_order':QUAD_ORDER}
    str_begin = 'RUN SCRIPT for RFM m vs n sweep for Darcy equation.\n'
    save_path = 'experiments/' + str(TODAY) + '/mvsn' + str(exp_num) + '/'
    
    # Begin log
    log_file, stdoutOrigin = fileio_init(save_path, hyp_dict)
    print(str_begin)
    
    # Compute
    e_store = np.zeros([len(n_list), len(m_list)])
    for loop in range(len(n_list)):
        n = n_list[loop]
        print('\n Outer loop, n =', n)
        for ind in range(len(m_list)):
            m = m_list[ind]
            print('Inner loop, m =', m)
            start = time.time() 
            rfm = RandomFeatureModel(K, n, m, ntest, lamreg, sig_plus, sig_minus, denom_scale, rf_choice = 0)
            rfm.fit()
            etest_temp, _ = rfm.relative_error_test()
            print('Expected relative test error:', etest_temp)
            e_store[loop, ind] = etest_temp
            print('Inner loop time elapsed: ', time.time() - start, 'seconds.') # print run time
    
    np.save(save_path + 'error_mvsn.npy', e_store)
    
    # End log
    print('Total Script Runtime: ', time.time() - start_total_time, 'seconds.') 
    fileio_end(log_file, stdoutOrigin)
