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
    exp_num = 000
    dataset_num = 4
    QUAD_ORDER = 1
    n = 1024
    m = 1024
    ntest = 4000
    lamreg = 0
    al_rf = 4.0                   # default: 4.0, 1.2, 1.5, 2.0
    nu_rf = 2.5e-3              # default: 2.5e-3, 1e-3, 5e-4, 1e-4
    
    # Log start time
    start_total_time = time.time()
    
    # Derived
    dir_path='datasets/' + str(dataset_num) + '/'
    hyp_dict = {'m':m, 'n':n, 'ntest':ntest, 'lamreg':lamreg, 'al_rf':al_rf, 'nu_rf':nu_rf, 'data_path':dir_path, 'quad_order':QUAD_ORDER}
    str_begin = 'RUN SCRIPT for RFM grid sweep for Burgers equation.\n'
    save_path = 'experiments/' + str(TODAY) + '/gridsweep' + str(exp_num) + '/'
    
    # Begin log
    log_file, stdoutOrigin = fileio_init(save_path, hyp_dict)
    print(str_begin)
    
    # Compute
    rfm = RandomFeatureModel(17, n, m, ntest, lamreg, nu_rf, al_rf, dir_path=dir_path)
    e_grid, almodel_grid = rfm.gridsweep(quad_order=QUAD_ORDER)
    np.save(save_path + 'error_gridsweep.npy', e_grid)
    np.save(save_path + 'al_model_gridsweep.npy', almodel_grid)
    
    # End log
    print('Total Script Runtime: ', time.time() - start_total_time, 'seconds.') 
    print('The choice of RF map is shown in the function defn. below:\n') # Print RF function used
    print(inspect.getsource(rf_fourier))
    fileio_end(log_file, stdoutOrigin)
