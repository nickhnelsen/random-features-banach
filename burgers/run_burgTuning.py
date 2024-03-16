import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
plt.close('all') # close all open figures
import time

# Custom imports to this file
import inspect
from RFM_burg import RandomFeatureModel, InnerProduct1D, rf_fourier
from solve_burgP import solnmap_burgP
from utilities.fileio import fileio_init, fileio_end
from utilities.plot_suiteSIAM import Plotter
plotter = Plotter(xlab_str=r'$x$', ylab_str=r'$t$') # set plotter class

# %% Init

TODAY = 20200507
exp_num = 44111
dataset_num = 44111
QUAD_ORDER = 1
K = 1 + 128
n = 512
m = 1024
ntest = 4000
lamreg = 0
al_rf = 4                   # default: 4, 1.2, 1.5, 2
nu_rf = 2.5e-3              # default: 2.5e-3, 1e-3, 5e-4, 1e-4

# %% Train and Test

# Log start time
start_total_time = time.time()

# Derived
dir_path='datasets/' + str(dataset_num) + '/'
hyp_dict = {'m':m, 'n':n, 'K':K, 'ntest':ntest, 'lamreg':lamreg, 'al_rf':al_rf, 'nu_rf':nu_rf, 'data_path':dir_path, 'quad_order':QUAD_ORDER}
str_begin = 'RUN SCRIPT for tuning the RFM for Burgers equation.\n'
save_path = 'experiments/' + str(TODAY) + '/exp' + str(exp_num) + '/'

# Begin log
log_file, stdoutOrigin = fileio_init(save_path, hyp_dict)
print(str_begin)

# Compute
start = time.time() 
rfm = RandomFeatureModel(K, n, m, ntest, lamreg, nu_rf, al_rf, dir_path=dir_path)
rfm.fit(order=QUAD_ORDER)
e_reg = rfm.regsweep([])
# e_reg = rfm.regsweep([1e-8])
print('Time Elapsed: ', time.time() - start, 'seconds.') # print run time
print('2-norm of coeff:', linalg.norm(rfm.al_model),'; Max coeff:', np.max(np.abs(rfm.al_model)))
input_test, output_test = rfm.get_testpairs()

# %% One point test error

plt.close('all') # close all open figures
order = QUAD_ORDER
np.random.seed(None)
ind_test = np.random.randint(0, rfm.ntest)
if rfm.tmax == None:
    rfm.tmax = 1

# Random input
a_test = input_test[:,ind_test]
y_test = output_test[:,ind_test]
y_pred = rfm.predict(a_test)

# =============================================================================
# # Evolve one step by composition
# a_test = input_test[:,ind_test]
# y_test = solnmap_burgP(a_test, rfm.nu_burg, tmax=2*rfm.tmax)
# y_pred = rfm.predict(rfm.predict(a_test))
# =============================================================================

# =============================================================================
# # Evolve two steps by composition
# a_test = input_test[:,ind_test]
# y_test = solnmap_burgP(a_test, rfm.nu_burg, tmax=3*rfm.tmax)
# y_pred = rfm.predict(rfm.predict(rfm.predict(a_test)))
# =============================================================================

# =============================================================================
# # Evolve three steps by composition
# a_test = input_test[:,ind_test]
# y_test = solnmap_burgP(a_test, rfm.nu_burg, tmax=4*rfm.tmax)
# y_pred = rfm.predict(rfm.predict(rfm.predict(rfm.predict(a_test))))
# =============================================================================

# =============================================================================
# # Evolve four steps by composition
# a_test = input_test[:,ind_test]
# y_test = solnmap_burgP(a_test, rfm.nu_burg, tmax=5*rfm.tmax)
# y_pred = rfm.predict(rfm.predict(rfm.predict(rfm.predict(rfm.predict(a_test)))))
# =============================================================================

# =============================================================================
# # Evolve five steps by composition
# a_test = input_test[:,ind_test]
# y_test = solnmap_burgP(a_test, rfm.nu_burg, tmax=6*rfm.tmax)
# y_pred = rfm.predict(rfm.predict(rfm.predict(rfm.predict(rfm.predict(rfm.predict(a_test))))))
# =============================================================================

# =============================================================================
# # User prescribed input
# a_test = (np.exp(-25*np.sin(rfm.X - 0.5)**2)*np.sin(20*np.pi*rfm.X)) * 0 + 1*np.exp(-70*(rfm.X-.33)**2)
# a_test = a_test - a_test.mean()
# y_test = solnmap_burgP(a_test, nu=rfm.nu_burg)
# y_pred = rfm.predict(a_test)
# =============================================================================

res = y_test-y_pred
ip = InnerProduct1D(1/(rfm.K - 1), order)
print('One test point relative error:' , np.sqrt(ip.L2(res,res)/ip.L2(y_test,y_test)))

fsd_list = [False, True]
out_list = ['output_test.pdf', 'output_test_wide.pdf']
pw_list = ['pwerror_test.pdf', 'pwerror_test_wide.pdf']
for loop in range(2):
    fsd = fsd_list[loop]
    out_name = out_list[loop]
    pw_name = pw_list[loop]
    plotter.plot_oneD(1, rfm.X, a_test, legendlab_str=r'Input', linestyle_str='k:', fig_sz_default=fsd)
    if K > 65:
        plotter.plot_oneD(1, rfm.X, y_pred, legendlab_str=r'Test', linestyle_str='b', fig_sz_default=fsd)
        f2 = plotter.plot_oneD(2, rfm.X, np.abs(res)**2, legendlab_str=r'Squared PW Error', linestyle_str='k', fig_sz_default=fsd)
        f3 = plotter.plot_oneD(3, rfm.X, res, legendlab_str=r'PW Error', linestyle_str='k', fig_sz_default=fsd)
    else:
        plotter.plot_oneD(1, rfm.X, y_pred, legendlab_str=r'RFM', linestyle_str='bo-.',fig_sz_default=fsd)
        f2 = plotter.plot_oneD(2, rfm.X, np.abs(res)**2, legendlab_str=r'Squared PW Error', fig_sz_default=fsd)
        f3 = plotter.plot_oneD(3, rfm.X, res, legendlab_str=r'PW Error', fig_sz_default=fsd)
    f1 = plotter.plot_oneD(1, rfm.X, y_test, legendlab_str=r'Truth', linestyle_str='r--', fig_sz_default=fsd, LW_set=4)
    
    # Save to file
    plotter.save_plot(f1, save_path + out_name)
    plotter.save_plot(f3, save_path + pw_name)
    plt.close(f1)
    plt.close(f2)
    plt.close(f3)
    
# Save data arrays to file
np.save(save_path + 'AstarA.npy',rfm.AstarA)
np.save(save_path + 'AstarY.npy',rfm.AstarY)
np.save(save_path + 'al_model.npy',rfm.al_model)
np.save(save_path + 'error_regsweep.npy',e_reg)

# End log
print('Total Script Runtime: ', time.time() - start_total_time, 'seconds.') 
print('The choice of RF map is shown in the function defn. below:\n') # Print RF function used
print(inspect.getsource(rf_fourier))
fileio_end(log_file, stdoutOrigin)

# %% Misc. Tests

#################### See Below ####################




# %% Global expected relative train and test errors

# =============================================================================
# start = time.time() 
# 
# e_train, b_train = rfm.relative_error_train(order)
# print(e_train, b_train)
# e_test, b_test = rfm.relative_error_test(order)
# print(e_test, b_test)
# 
# print('Time Elapsed: ', time.time() - start, 'seconds.') # print run time
# =============================================================================

# %% Grid sweep

# =============================================================================
# K_list = [33, 65, 129, 257, 1025]
# 
# start = time.time() 
# e_grid = rfm.gridsweep(K_list)
# 
# print('Time Elapsed: ', time.time() - start, 'seconds.') # print run time
# =============================================================================

# %% Test on high resolution data from low resolution trained model

# =============================================================================
# K_fine = 1 + 1024
# rfm_test = RandomFeatureModel(K_fine, rfm.n, rfm.m, min(rfm.ntest_max, rfm.ntest + 100), rfm.lamreg, rfm.nu_rf, rfm.al_rf, dir_path=rfm.dir_path)
# rfm_test.al_model = rfm.al_model
# start = time.time() 
# print('Expected relative error (Test, Boch. Test):', rfm_test.relative_error_test(order))
# print('Time Elapsed: ', time.time() - start, 'seconds.') # print run time
# input_testf, output_testf = rfm_test.get_testpairs()
# 
# a_testf = input_testf[:,ind_test]
# y_testf = output_testf[:,ind_test]
# y_predf = rfm_test.predict(a_testf)
# resf = y_testf - y_predf
# ip = InnerProduct1D(1/(rfm_test.K - 1), order)
# print('One test point relative error:' , np.sqrt(ip.L2(resf,resf)/ip.L2(y_testf,y_testf)))
# plotter.plot_oneD(10, rfm_test.X, a_testf, legendlab_str=r'IC', linestyle_str='b')
# plotter.plot_oneD(10, rfm_test.X, y_testf, legendlab_str=r'Truth', linestyle_str='r')
# if K_fine > 65:
#     f10 = plotter.plot_oneD(10, rfm_test.X, y_predf, legendlab_str=r'RFM', linestyle_str='k')
#     f20 = plotter.plot_oneD(20, rfm_test.X, np.abs(resf)**2, legendlab_str=r'Squared PW Error', linestyle_str='k')
#     f30 = plotter.plot_oneD(30, rfm_test.X, resf, legendlab_str=r'PW Error', linestyle_str='k')
# else:
#     f10 = plotter.plot_oneD(10, rfm_test.X, y_predf, legendlab_str=r'RFM')
#     f20 = plotter.plot_oneD(20, rfm_test.X, np.abs(resf)**2, legendlab_str=r'Squared PW Error')
#     f30 = plotter.plot_oneD(30, rfm_test.X, resf, legendlab_str=r'PW Error')
# 
# # Save to file
# # plotter.save_plot(f10, 'output_gt_test7.pdf')
# =============================================================================
