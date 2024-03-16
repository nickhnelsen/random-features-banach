import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
plt.close('all') # close all open figures
from matplotlib import cm

# Import common function files from subdirectories
from utilities.plot_suiteSIAM import Plotter
plotter = Plotter() # set plotter class


# %% Grid Sweep Plots: Burgers

# =============================================================================
# def get_path(num):
#     return 'experiments/' + str(exp_date) + '/gridsweep' + str(num) + '/'
# 
# exp_date = 20200503
# testind = 3
# dir_path_output = 'figures/'
# name_al = 'al_model_gridsweep.npy'
# name_er = 'error_gridsweep.npy'
# er1 = np.load(get_path(1) + name_er)
# er2 = np.load(get_path(2) + name_er)
# er3 = np.load(get_path(3) + name_er)
# al1 = np.load(get_path(1) + name_al)
# al2 = np.load(get_path(2) + name_al)
# al3 = np.load(get_path(3) + name_al)
# 
# K_list = er1[:,0]
# sly = False
# MSZ = None
# mfc = False
# =============================================================================

# =============================================================================
# plotter.plot_oneD(1, K_list, er1[:,testind], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=256$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc)
# plotter.plot_oneD(1, K_list, er2[:,testind], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=512$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc)
# f1 = plotter.plot_oneD(1, K_list, er3[:,testind], xlab_str1D=r'Resolution', ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=1024$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc)
# plt.xlim(0, 1060)
# # plt.ylim(0, 0.065)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f1, dir_path_output + 'gridsweep_burg_dataset4color.pdf')
# 
# 
# plotter.plot_oneD(2, K_list, linalg.norm(al1,axis=1) ,legendlab_str=r'$m=256$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly)
# plotter.plot_oneD(2, K_list, linalg.norm(al2,axis=1), legendlab_str=r'$m=512$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly)
# f2 = plotter.plot_oneD(2, K_list, linalg.norm(al3,axis=1), xlab_str1D=r'Resolution', ylab_str1D=r'2-Norm of Learned Parameter',legendlab_str=r'$m=1024$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly)
# plt.xlim(0, 1060)
# # plt.ylim(0, 0.065)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f2, dir_path_output + 'gridsweep_burg_dataset4_alphacolor.pdf')
# 
# 
# sly = True
# plotter.plot_oneD(3, K_list[:-1], linalg.norm(al1-al1[-1,:],axis=1)[:-1]/linalg.norm(al1[-1,:]) ,legendlab_str=r'$m=256$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly)
# plotter.plot_oneD(3, K_list[:-1], linalg.norm(al2-al2[-1,:],axis=1)[:-1]/linalg.norm(al2[-1,:]), legendlab_str=r'$m=512$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly)
# f3 = plotter.plot_oneD(3, K_list[:-1], linalg.norm(al3-al3[-1,:],axis=1)[:-1]/linalg.norm(al3[-1,:]), xlab_str1D=r'Resolution', ylab_str1D=r'$\| \alpha^{(K)}-\alpha^{(1025)} \|_2/\|\alpha^{(1025)}\|_2$',legendlab_str=r'$m=1024$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly)
# plt.xlim(0, 550) 
# # plt.ylim(0, 0.065)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f3, dir_path_output + 'gridsweep_burg_dataset4_alphadiffcolor.pdf')
# =============================================================================

# %% Grid Sweep Plots: for Andrew

# =============================================================================
# def get_path(num):
#     return 'experiments/' + str(exp_date) + '/gridsweep' + str(num) + '/'
# 
# exp_date = 20200505
# testind = 3
# dir_path_output = 'figures/for_AMS/'
# name_al = 'al_model_gridsweep.npy'
# name_er = 'error_gridsweep.npy'
# er0 = np.load(get_path(0) + name_er)
# er1 = np.load(get_path(1) + name_er)
# er2 = np.load(get_path(2) + name_er)
# er3 = np.load(get_path(3) + name_er)
# er4 = np.load(get_path(4) + name_er)
# 
# 
# K_list = er1[:,0]
# sly = False
# MSZ = None
# mfc = False
# fsd = True
# 
# # plotter.plot_oneD(1, np.delete(K_list,1), np.delete(er0[:,testind],1), ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=32$',linestyle_str='mH:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc,fig_sz_default=fsd)
# plotter.plot_oneD(1, np.delete(K_list,0), np.delete(er1[:,testind],0), ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$N=64$',linestyle_str='g^-', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, K_list, er2[:,testind], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$N=128$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, K_list, er3[:,testind], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$N=256$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# f1 = plotter.plot_oneD(1, K_list, er4[:,testind], xlab_str1D=r'Resolution', ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$N=512$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plt.xlim(0, 1060)
# # plt.ylim(0, 0.065)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f1, dir_path_output + 'testerror_burgers.png',0)
# =============================================================================

# %% Input-Output for Andrew's OneWorld talk

# # from RFM_burg import RandomFeatureModel, InnerProduct1D
# from solve_burgP import solnmap_burgP
# from GRF1D import GaussianRandomField1D

# dir_path_output = 'figures/'

# K = 1 + 1024
# T = 1           # final time t=T (default: 1, 0.33, 0.5, 0.75)
# nu = 4e-3       # default: 5e-4, 2.5e-3 to 3.5e-3 for easier problem
# tau = 7
# al = 2.5
# f1 = 0.6*5*5
# f2 = 0.6*5*5

# x = np.arange(0, 1 + 1/(K-1), 1/(K-1))
# grf = GaussianRandomField1D(tau, al, bc=2)

# np.random.seed(None)
# theta = np.random.standard_normal(K)

# IC = grf.draw(theta)
# u = solnmap_burgP(IC, nu=nu, tmax=T, fudge1=f1, fudge2=f2)

# fsd = True
# f1 = plotter.plot_oneD(1, x, IC, xlab_str1D=r'', linestyle_str='k', fig_sz_default=fsd)
# plt.ylim(-0.45, 0.4)
# f2 = plotter.plot_oneD(2, x, u, xlab_str1D=r'', linestyle_str='k', fig_sz_default=fsd)
# plt.ylim(-0.45, 0.4)

# # plotter.save_plot(f1, dir_path_output + 'input_burgers_wide.png', 0)
# # plotter.save_plot(f2, dir_path_output + 'output_burgers_wide.png', 0)

# %% Grid Transfer plots

# =============================================================================
# def get_path(num):
#     return 'experiments/' + str(exp_date) + '/gridtransfer' + str(num) + '/'
# 
# exp_date = 20200506
# testind = 3
# dir_path_output = 'figures/'
# name_er = 'error_gridtransfer.npy'
# er = np.load(get_path(1) + name_er)
# 
# 
# K_list = np.array([17, 33, 65, 129, 257, 513, 1025])
# sly = False
# MSZ = None
# mfc = False
# fsd = False
# 
# plotter.plot_oneD(1, K_list, er[:,0], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'Train on $K=65$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, K_list, er[:,1], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'Train on $K=257$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# f1 = plotter.plot_oneD(1, K_list, er[:,2], xlab_str1D=r'Resolution', ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'Train on $K=1025$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plt.xlim(0, 1060)
# # plt.ylim(0, 0.065)
# plt.legend().set_draggable(True)
# plotter.save_plot(f1, dir_path_output + 'gridtransfer_burg_dataset4.pdf')
# =============================================================================

# %% Burgers example input output RFM prediction plots

# =============================================================================
# # Custom imports to this file
# import time
# from RFM_burg import RandomFeatureModel, InnerProduct1D
# from solve_burgP import solnmap_burgP
# from utilities.fileio import fileio_init, fileio_end
# from utilities.plot_suiteSIAM import Plotter
# plotter = Plotter(xlab_str=r'$x$', ylab_str=r'$t$') # set plotter class
# 
# sample_num = 7
# QUAD_ORDER = 1
# n = 512
# m = 1024
# ntest = 4000
# lamreg = 0
# al_rf = 4                   # default: 1.2, 1.5, 2
# nu_rf = 2.5e-3              # default: 1e-3, 5e-4, 1e-4
# K_ind = -1 + 7
# K_list = [17, 33, 65, 129, 257, 513, 1025]
# dataset_num = 4
# 
# # Derived
# dir_path='datasets/' + str(dataset_num) + '/'
# K = K_list[K_ind]
# hyp_dict = {'m':m, 'n':n, 'K':K, 'ntest':ntest, 'lamreg':lamreg, 'al_rf':al_rf, 'nu_rf':nu_rf, 'data_path':dir_path, 'quad_order':QUAD_ORDER}
# save_path = 'figures/samples/' + str(sample_num) + '/'
# load_path = 'experiments/' + str(20200505) + '/gridsweep4/'
# 
# # Begin log
# start_total_time = time.time()
# log_file, stdoutOrigin = fileio_init(save_path, hyp_dict)
# 
# # Compute
# al_matrix = np.load(load_path + 'al_model_gridsweep.npy')
# al_K = al_matrix[K_ind,:]
# rfm = RandomFeatureModel(K, n, m, ntest, lamreg, nu_rf, al_rf, dir_path=dir_path)
# rfm.al_model = al_K
# input_test, output_test = rfm.get_testpairs()
# 
# # One sample point test error
# plt.close('all') # close all open figures
# order = QUAD_ORDER
# np.random.seed(None)
# ind_test = np.random.randint(0, rfm.ntest_max)
# if rfm.tmax == None:
#     rfm.tmax = 1
# 
# # Random input
# a_test = input_test[:,ind_test]
# y_test = output_test[:,ind_test]
# y_pred = rfm.predict(a_test)
# 
# # =============================================================================
# # # Evolve one step by composition
# # a_test = input_test[:,ind_test]
# # y_test = solnmap_burgP(a_test, rfm.nu_burg, tmax=2*rfm.tmax)
# # y_pred = rfm.predict(rfm.predict(a_test))
# # =============================================================================
# 
# # =============================================================================
# # # Evolve two steps by composition
# # a_test = input_test[:,ind_test]
# # y_test = solnmap_burgP(a_test, rfm.nu_burg, tmax=3*rfm.tmax)
# # y_pred = rfm.predict(rfm.predict(rfm.predict(a_test)))
# # =============================================================================
# 
# # =============================================================================
# # # Evolve three steps by composition
# # a_test = input_test[:,ind_test]
# # y_test = solnmap_burgP(a_test, rfm.nu_burg, tmax=4*rfm.tmax)
# # y_pred = rfm.predict(rfm.predict(rfm.predict(rfm.predict(a_test))))
# # =============================================================================
# 
# # =============================================================================
# # # Evolve four steps by composition
# # a_test = input_test[:,ind_test]
# # y_test = solnmap_burgP(a_test, rfm.nu_burg, tmax=5*rfm.tmax)
# # y_pred = rfm.predict(rfm.predict(rfm.predict(rfm.predict(rfm.predict(a_test)))))
# # =============================================================================
# 
# # =============================================================================
# # # Evolve five steps by composition
# # a_test = input_test[:,ind_test]
# # y_test = solnmap_burgP(a_test, rfm.nu_burg, tmax=6*rfm.tmax)
# # y_pred = rfm.predict(rfm.predict(rfm.predict(rfm.predict(rfm.predict(rfm.predict(a_test))))))
# # =============================================================================
# 
# # =============================================================================
# # # User prescribed input
# # a_test = (np.exp(-25*np.sin(rfm.X - 0.5)**2)*np.sin(20*np.pi*rfm.X)) * 0 + 1*np.exp(-70*(rfm.X-.33)**2)
# # a_test = a_test - a_test.mean()
# # y_test = solnmap_burgP(a_test, nu=rfm.nu_burg)
# # y_pred = rfm.predict(a_test)
# # =============================================================================
# 
# res = y_test-y_pred
# ip = InnerProduct1D(1/(rfm.K - 1), order)
# print('One test point relative error:' , np.sqrt(ip.L2(res,res)/ip.L2(y_test,y_test)))
# 
# fsd_list = [False, True]
# out_list = ['prediction_onesample.pdf', 'prediction_onesample_wide.pdf']
# pw_list = ['pwerror_onesample.pdf', 'pwerror_onesample_wide.pdf']
# for loop in range(2):
#     fsd = fsd_list[loop]
#     out_name = out_list[loop]
#     pw_name = pw_list[loop]
#     plotter.plot_oneD(1, rfm.X, a_test, legendlab_str=r'Input', linestyle_str='k:', fig_sz_default=fsd)
#     if K > 65:
#         plotter.plot_oneD(1, rfm.X, y_pred, legendlab_str=r'Test', linestyle_str='b', fig_sz_default=fsd)
#         f2 = plotter.plot_oneD(2, rfm.X, np.abs(res)**2, linestyle_str='k', fig_sz_default=fsd)
#         f3 = plotter.plot_oneD(3, rfm.X, res, linestyle_str='k', fig_sz_default=fsd)
#     else:
#         plotter.plot_oneD(1, rfm.X, y_pred, legendlab_str=r'RFM', linestyle_str='bo-.',fig_sz_default=fsd)
#         f2 = plotter.plot_oneD(2, rfm.X, np.abs(res)**2, fig_sz_default=fsd)
#         f3 = plotter.plot_oneD(3, rfm.X, res, fig_sz_default=fsd)
#     f1 = plotter.plot_oneD(1, rfm.X, y_test, legendlab_str=r'Truth', linestyle_str='r--', fig_sz_default=fsd, LW_set=4)
#     
#     # Save to file
#     plotter.save_plot(f1, save_path + out_name)
#     plotter.save_plot(f3, save_path + pw_name)
#     plt.close(f1)
#     plt.close(f2)
#     plt.close(f3)
#     
# # End log
# print('Total Script Runtime: ', time.time() - start_total_time, 'seconds.') 
# fileio_end(log_file, stdoutOrigin)
# =============================================================================

# %% m vs n

# =============================================================================
# def get_path(num):
#     return 'experiments/' + str(exp_date) + '/mvsn' + str(num) + '/'
# 
# exp_date = 20200507
# testind = 3
# dir_path_output = 'figures/'
# name_er = 'error_mvsn.npy'
# er1 = np.load(get_path(1) + name_er) # n=10, 100
# # er2 = np.load(get_path(2) + name_er) # n=300, 500
# er3 = np.load(get_path(3) + name_er) # n=1000
# er2 = np.load(get_path(4) + name_er) # n= 20, 30, 50
# er = np.vstack((er1[0,:], er2[:-1,:], er1[-1,:], er3)) # 10, 20, 30, 100, 1000
# 
# m_list = np.array([16, 32, 64, 128, 256, 512, 1024])
# n_list = np.array([10, 100, 300, 500, 1000])
# sly = False
# MSZ = None
# mfc = False
# fsd = True
# 
# plotter.plot_oneD(1, m_list, er[0,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=10$',linestyle_str='mH-', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, m_list, er[1,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=20$',linestyle_str='g^:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, m_list, er[2,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=30$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, m_list, er[3,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=100$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# f1 = plotter.plot_oneD(1, m_list, er[4,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=1000$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, xlab_str1D=r'$m$')
# 
# plt.xlim(0, 1060)
# # plt.ylim(0, 0.065)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f1, dir_path_output + 'mvsn_burg_dataset4_wide.pdf')
# =============================================================================
