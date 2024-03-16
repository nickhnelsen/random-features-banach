import numpy as np
import time

# Custom imports to this file
from RFM_largedata import RandomFeatureModel
from utilities.fileio import fileio_init, fileio_end


# Run main script
if __name__ == "__main__":
    
    # USER INIT
    TODAY = 20200512
    exp_num = 2
    n = 128             # fixed
    m = 256             # fixed
    ntest = 2000        # fixed
    lamreg = 1e-8
    sig_plus = 1/12
    sig_minus = -1/3
    denom_scale = 0.15
    K_list = [9, 17, 33, 65, 129, 257]
    
    # Log start time
    start_total_time = time.time()
    
    # Derived
    dat_path='experiments/gridtransfer_coeffdata/'
    hyp_dict = {'m':m, 'n':n, 'K_list':K_list, 'ntest':ntest, 'lamreg_init':lamreg, 'sig_plus':sig_plus,'sig_minus':sig_minus, 'denom_scale':denom_scale, 'rf_choice':0}
    str_begin = 'RUN SCRIPT for RFM grid transfer sweep for Darcy equation.\n'
    save_path = 'experiments/' + str(TODAY) + '/gridtransfer' + str(exp_num) + '/'
    
    # Begin log
    log_file, stdoutOrigin = fileio_init(save_path, hyp_dict)
    print(str_begin)
    
    # Compute
    al_model = np.load(dat_path + 'al_model_gridsweep.npy')
    al_res = np.delete(al_model,[0,2,3],axis=0) # only K = 17, 129
    num_m = 2
    e_store = np.zeros([len(K_list), num_m])
    trainK = [17, 129]
    for loop in range(num_m):
        al_temp = al_res[loop, :]
        print('Training grid size K =', trainK[loop])
        for ind in range(len(K_list)):
            K = K_list[ind]
            print('Testing on grid size K =', K)
            rfm = RandomFeatureModel(K, n, m, ntest, lamreg, sig_plus, sig_minus, denom_scale, rf_choice = 0)
            rfm.al_model = al_temp
            start = time.time() 
            etest_temp, _ = rfm.relative_error_test()
            print('Expected relative test error:', etest_temp)
            e_store[ind, loop] = etest_temp
            print('Time Elapsed: ', time.time() - start, 'seconds.') # print run time
    
    np.save(save_path + 'error_gridtransfer.npy', e_store)
    
    # End log
    print('Total Script Runtime: ', time.time() - start_total_time, 'seconds.') 
    fileio_end(log_file, stdoutOrigin)
