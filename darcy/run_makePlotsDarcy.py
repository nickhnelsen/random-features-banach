import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
plt.close('all') # close all open figures
from matplotlib import cm

# Import common function files from subdirectories
from utilities.plot_suiteSIAM import Plotter
plotter = Plotter() # set plotter class

# %% Grid Sweep Plots: Darcy

# =============================================================================
# def get_path(num):
#     return 'experiments/' + str(exp_date) + '/gridsweep' + str(num) + '/'
# 
# exp_date = 20200512
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
# fsd = True
# 
# plotter.plot_oneD(1, K_list, er1[:,testind], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=64$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, K_list, er2[:,testind], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=128$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# f1 = plotter.plot_oneD(1, K_list, er3[:,testind], xlab_str1D=r'Resolution', ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=256$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plt.xlim(0, 135)
# # plt.ylim(0, 0.065)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f1, dir_path_output + 'gridsweep_darcy.pdf')
# 
# 
# plotter.plot_oneD(2, K_list, linalg.norm(al1,axis=1) ,legendlab_str=r'$m=64$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly)
# plotter.plot_oneD(2, K_list, linalg.norm(al2,axis=1), legendlab_str=r'$m=128$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly)
# f2 = plotter.plot_oneD(2, K_list, linalg.norm(al3,axis=1), xlab_str1D=r'Resolution', ylab_str1D=r'2-Norm of Learned Parameter',legendlab_str=r'$m=256$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly)
# plt.xlim(0, 70)
# # plt.ylim(0, 0.065)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f2, dir_path_output + 'gridsweep_burg_dataset4_alphacolor.pdf')
# 
# 
# sly = True
# plotter.plot_oneD(3, K_list[:-1], linalg.norm(al1-al1[-1,:],axis=1)[:-1]/linalg.norm(al1[-1,:]) ,legendlab_str=r'$m=64$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, fig_sz_default=fsd)
# plotter.plot_oneD(3, K_list[:-1], linalg.norm(al2-al2[-1,:],axis=1)[:-1]/linalg.norm(al2[-1,:]), legendlab_str=r'$m=128$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, fig_sz_default=fsd)
# f3 = plotter.plot_oneD(3, K_list[:-1], linalg.norm(al3-al3[-1,:],axis=1)[:-1]/linalg.norm(al3[-1,:]), xlab_str1D=r'Resolution', ylab_str1D=r'$\| \alpha^{(r)}-\alpha^{(129)} \|_2/\|\alpha^{(129)}\|_2$',legendlab_str=r'$m=256$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, fig_sz_default=fsd)
# plt.xlim(0, 70) 
# plt.ylim(1e-2, 1)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f3, dir_path_output + 'gridsweep_darcy_alphadiffwide.pdf')
# =============================================================================

# %% Grid Transfer plots

def get_path(num):
    return 'experiments/' + str(exp_date) + '/gridtransfer' + str(num) + '/'

exp_date = 20200512
dir_path_output = 'figures/'
name_er = 'error_gridtransfer.npy'
er = np.load(get_path(1) + name_er)

K_list = np.array([9, 17, 33, 65, 129, 257])
sly = False
MSZ = None
mfc = False
fsd = True

plotter.plot_oneD(1, K_list, er[:,0], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'Train on $r=17$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
f1 = plotter.plot_oneD(1, K_list, er[:,1], xlab_str1D=r'Resolution', ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'Train on $r=129$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
plt.xlim(0, 265)
# plt.ylim(0, 0.065)
plt.legend().set_draggable(True)
# plotter.save_plot(f1, dir_path_output + 'gridtransfer_darcy_wide.pdf')

# %% Darcy example input output RFM prediction plots

# =============================================================================
# # Custom imports to this file
# import time
# from RFM_largedata import RandomFeatureModel, InnerProduct
# from utilities.fileio import fileio_init, fileio_end
# from utilities.plot_suiteSIAM import Plotter
# plotter = Plotter() # set plotter class
# 
# 
# # USER CHOICE
# sample_num = 15
# K = 1 + 256
# ntest = 1000*0 + 4000
# QUAD_ORDER = 1
# 
# # From file "exp2"
# n = 256
# m = 350
# lamreg = 1e-8
# sig_plus = 1/12
# sig_minus = -1/3
# denom_scale = 0.15
# 
# # Derived
# hyp_dict = {'m':m, 'n':n, 'K':K, 'ntest':ntest, 'lamreg_init':lamreg, 'sig_plus':sig_plus,'sig_minus':sig_minus, 'denom_scale':denom_scale, 'rf_choice':0, 'quad_order':QUAD_ORDER}
# str_begin = 'RUN SCRIPT for making for Darcy prediction four plots.\n'
# save_path = 'figures/samples/' + str(sample_num) + '/'
# load_path = 'experiments/' + str(20200511) + '/exp2/'
# 
# # Begin log
# start_total_time = time.time()
# log_file, stdoutOrigin = fileio_init(save_path, hyp_dict)
# 
# # Compute
# al_choice = np.load(load_path + 'al_model.npy')
# rfm = RandomFeatureModel(K, n, m, ntest, lamreg, sig_plus, sig_minus, denom_scale, rf_choice = 0)
# rfm.al_model = al_choice
# input_test, output_test = rfm.get_testpairs()
# 
# plt.close('all') # close all open figures
# order = QUAD_ORDER
# np.random.seed(None)
# ind_test = np.random.randint(0, rfm.ntest)
# print('Random input index is: ', ind_test) 
# 
# # Random input
# a_test = input_test[:,:,ind_test]
# y_test = output_test[:,:,ind_test]
# start_eval_time = time.time()
# y_pred = rfm.predict(a_test)
# print('One sample evaluation time: ', time.time() - start_eval_time, 'seconds.') 
# 
# res = y_test - y_pred
# ip = InnerProduct(1/(rfm.K - 1), order)
# print('One test point relative error:' , np.sqrt(ip.L2(res,res)/ip.L2(y_test,y_test)))
# 
# f1 = plotter.plot_Heat(1, rfm.X, rfm.Y, y_test)
# f2 = plotter.plot_Heat(2, rfm.X, rfm.Y, y_pred)
# f3 = plotter.plot_Heat(3, rfm.X, rfm.Y, a_test, cmap_set2D = cm.viridis, cb_ticks_sn=False)
# f4 = plotter.plot_Heat(4, rfm.X, rfm.Y, res, cmap_set2D = cm.coolwarm, norm_set=0)
# 
# # Save to file
# plotter.save_plot(f1, save_path + 'truth.png', 0)
# plotter.save_plot(f2, save_path + 'prediction.png', 0)
# plotter.save_plot(f3, save_path + 'input.png', 0)
# plotter.save_plot(f4, save_path + 'pwerror.png', 0)
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
# exp_date = 20200511
# dir_path_output = 'figures/'
# name_er = 'error_mvsn.npy'
# er1 = np.load(get_path(1) + name_er) # n=5, 10, 20
# er2 = np.load(get_path(2) + name_er) # n= 30, 50 ,100
# er3 = np.load(get_path(3) + name_er) # n= 300, 500
# er4 = np.load(get_path(4) + name_er) # m = 512
# er = np.vstack((er1, er2, er3))
# er = np.hstack((er, er4))
# 
# m_list = np.array([8, 16, 32, 64, 128, 256, 512])
# n_list = np.array([5, 10, 20, 30, 50, 100, 300, 500]) 
# er_plot = np.delete(er, [3, 5, 6], 0) # n = 5, 10, 20, 50, 500
# 
# sly = True
# MSZ = None
# mfc = False
# fsd = True
# 
# plotter.plot_oneD(1, m_list, er_plot[0,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=5$',linestyle_str='mH-', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, m_list, er_plot[1,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=10$',linestyle_str='g^:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, m_list, er_plot[2,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=20$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, m_list, er_plot[3,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=50$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# f1 = plotter.plot_oneD(1, m_list, er_plot[4,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=500$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, xlab_str1D=r'$m$')
# 
# plt.xlim(0, 530)
# # plt.ylim(0.01, 0.4)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f1, dir_path_output + 'mvsn_darcy_wide.pdf')
# =============================================================================
