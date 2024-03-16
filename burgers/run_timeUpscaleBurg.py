import numpy as np
import time

# Custom imports to this file
from RFM_burg import RandomFeatureModel
from utilities.fileio import fileio_init, fileio_end



# Run main script
if __name__ == "__main__":
    
    # USER INIT
    TODAY = 20200508
    exp_num = 3
    tuexp_num = 44
    K = 1 + 128
    n = 512
    m = 1024
    ntest = 4000
    lamreg = 0
    al_rf = 4.0                   # default: 4.0, 1.2, 1.5, 2.0
    nu_rf = 2.5e-3              # default: 2.5e-3, 1e-3, 5e-4, 1e-4
    
    # Log start time
    start_total_time = time.time()
    
    # Derived
    str_begin = 'RUN SCRIPT for RFM time upscale testing for Burgers equation.\n'
    al_path = 'experiments/' + str(20200507) + '/exp' + str(tuexp_num) + '/'
    save_path = 'experiments/' + str(TODAY) + '/timeupscaleresults_' + str(exp_num) + '/'
    hyp_dict = {'m':m, 'n':n, 'ntest':ntest, 'lamreg':lamreg, 'al_rf':al_rf, 'nu_rf':nu_rf, 'al_path':al_path}
    
    # Begin log
    log_file, stdoutOrigin = fileio_init(save_path, hyp_dict)
    print(str_begin)
    
    # Compute
    al_model = np.load(al_path + 'al_model.npy')
    e_store = np.zeros(3) # t=1, 1.5, 2
    t_list = [1.0, 1.5, 2.0]
    for loop in range(3):
        n_comp = loop + 1
        dir_path_temp ='datasets/timeupscale/' + str(n_comp) + '/'
        print('Testing on t =', t_list[loop])
        rfm = RandomFeatureModel(K, n, m, ntest, lamreg, nu_rf, al_rf, dir_path=dir_path_temp)
        rfm.al_model = al_model
        start = time.time() 
        etest_temp = rfm.relative_error_timeupscale_test(n_comp)
        print('Expected relative test error:', etest_temp)
        e_store[loop] = etest_temp
        print('Time Elapsed: ', time.time() - start, 'seconds.') # print run time
    
    np.save(save_path + 'error_timeupscale.npy', e_store)
    
    # End log
    print('Total Script Runtime: ', time.time() - start_total_time, 'seconds.') 
    fileio_end(log_file, stdoutOrigin)
