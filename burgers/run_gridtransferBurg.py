import numpy as np
import time

# Custom imports to this file
import inspect
from RFM_burg import RandomFeatureModel, rf_fourier
from utilities.fileio import fileio_init, fileio_end


# Run main script
if __name__ == "__main__":
    
    # USER INIT
    TODAY = 20200506
    exp_num = 1
    dataset_num = 4
    QUAD_ORDER = 1
    n = 512
    m = 1024
    ntest = 4000
    lamreg = 0
    al_rf = 4.0                   # default: 4.0, 1.2, 1.5, 2.0
    nu_rf = 2.5e-3              # default: 2.5e-3, 1e-3, 5e-4, 1e-4
    
    # Log start time
    start_total_time = time.time()
    
    # Derived
    dir_path='datasets/' + str(dataset_num) + '/'
    dat_path='experiments/gridtransfer_coeff/'
    hyp_dict = {'m':m, 'n':n, 'ntest':ntest, 'lamreg':lamreg, 'al_rf':al_rf, 'nu_rf':nu_rf, 'data_path':dir_path, 'quad_order':QUAD_ORDER}
    str_begin = 'RUN SCRIPT for RFM grid transfer sweep for Burgers equation.\n'
    save_path = 'experiments/' + str(TODAY) + '/gridtransfer' + str(exp_num) + '/'
    
    # Begin log
    log_file, stdoutOrigin = fileio_init(save_path, hyp_dict)
    print(str_begin)
    
    # Compute
    al_model = np.load(dat_path + 'al_model_gridsweep.npy')[2:]     # K = 65, ..., 1025
    al_res = np.delete(al_model,[1,3],axis=0)
    K_list = [17, 33, 65, 129, 257, 513, 1025]
    e_store = np.zeros([len(K_list), 3])
    trainK = [65, 257, 1025]
    for loop in range(3):
        al_temp = al_res[loop, :]
        print('Training grid size K =', trainK[loop])
        for ind in range(len(K_list)):
            K = K_list[ind]
            print('Testing on grid size K =', K)
            rfm = RandomFeatureModel(K, n, m, ntest, lamreg, nu_rf, al_rf, dir_path=dir_path)
            rfm.al_model = al_temp
            start = time.time() 
            etest_temp, _ = rfm.relative_error_test()
            print('Expected relative test error:', etest_temp)
            e_store[ind, loop] = etest_temp
            print('Time Elapsed: ', time.time() - start, 'seconds.') # print run time
    
    np.save(save_path + 'error_gridtransfer.npy', e_store)
    
    # End log
    print('Total Script Runtime: ', time.time() - start_total_time, 'seconds.') 
    print('The choice of RF map is shown in the function defn. below:\n') # Print RF function used
    print(inspect.getsource(rf_fourier))
    fileio_end(log_file, stdoutOrigin)
