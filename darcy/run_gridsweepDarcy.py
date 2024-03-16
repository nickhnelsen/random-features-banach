"""
A run script for RFM grid resolution sweep for Darcy equation.
"""

import numpy as np
import time

# Custom imports to this file
from RFM_largedata import RandomFeatureModel
from utilities.fileio import fileio_init, fileio_end


# Run main script
if __name__ == "__main__":
    
    # USER INIT
    TODAY = 20200512
    exp_num = 5
    n = 128         # FIXED
    m = 128          # 64, 128, 256
    ntest = 1000    # FIXED
    lamreg = 1e-8   # FIXED
    sig_plus = 1/12
    sig_minus = -1/3
    denom_scale = 0.15
    # K_list = [9, 17, 33, 65, 129]
    K_list = [257]
    
    # Log start time
    start_total_time = time.time()
    
    # Derived
    hyp_dict = {'m':m, 'n':n, 'K_list':K_list, 'ntest':ntest, 'lamreg_init':lamreg, 'sig_plus':sig_plus,'sig_minus':sig_minus, 'denom_scale':denom_scale, 'rf_choice':0}
    str_begin = 'RUN SCRIPT for RFM grid sweep for Darcy equation.\n'
    save_path = 'experiments/' + str(TODAY) + '/gridsweep' + str(exp_num) + '/'
    
    # Begin log
    log_file, stdoutOrigin = fileio_init(save_path, hyp_dict)
    print(str_begin)
    
    # Compute
    rfm = RandomFeatureModel(9, n, m, ntest, lamreg, sig_plus, sig_minus, denom_scale, rf_choice = 0)
    e_grid, almodel_grid = rfm.gridsweep(K_list)
    np.save(save_path + 'error_gridsweep.npy', e_grid)
    np.save(save_path + 'al_model_gridsweep.npy', almodel_grid)
    
    # End log
    print('Total Script Runtime: ', time.time() - start_total_time, 'seconds.') 
    fileio_end(log_file, stdoutOrigin)
