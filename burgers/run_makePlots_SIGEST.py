import numpy as np
import matplotlib.pyplot as plt
plt.close('all') # close all open figures

# Import common function files from subdirectories
from utilities.plot_suiteSIAM_sigest import Plotter
plotter = Plotter() # set plotter class

# %% m vs n: burgers

# def get_path(num):
#     return 'experiments/' + str(exp_date) + '/mvsn' + str(num) + '/'

# FSZ=16
# LW=3

# exp_date = 20200507
# dir_path_output = '/home/nnelsen/NelsenNH_PhD_StuartAM/writing/paper-rf-sigest/new_figures/'
# name_er = 'error_mvsn.npy'


# # ---sweep m---
# er1 = np.load(get_path(1) + name_er) # n=10, 100
# er3 = np.load(get_path(3) + name_er) # n=1000
# er4 = np.load(get_path(4) + name_er) # n= 20, 30, 50
# er = np.vstack((er1[0,:], er4[:-1,:], er1[-1,:], er3)) # 10, 20, 30, 100, 1000

# m_list = np.array([16, 32, 64, 128, 256, 512, 1024])
# sly = False
# MSZ = None
# mfc = False
# fsd = not True
# ll = True


# plotter.plot_oneD(1, m_list, er[0,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=10$',linestyle_str='mH-', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
# plotter.plot_oneD(1, m_list, er[1,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=20$',linestyle_str='g^:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
# plotter.plot_oneD(1, m_list, er[2,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=30$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
# plotter.plot_oneD(1, m_list, er[3,:],legendlab_str=r'$n=100$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
# f1 = plotter.plot_oneD(1, m_list, er[4,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=1000$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, xlab_str1D=r'$m$', loglog=ll)

# plt.loglog(m_list, 8e-1*m_list**(-0.5), ls=(0, (3, 1, 1, 1, 1, 1)), linewidth=LW, color='darkgray', label=r'$m^{-1/2}$')

# plt.xlim(10, 1250)
# # plt.ylim(0, 0.065)
# plt.legend().set_draggable(True)
# # plt.grid(axis='both')
# # plotter.save_plot(f1, dir_path_output + 'sigest_burg_msweep.pdf')


# # ---sweep n---
# er1 = np.load(get_path(1) + name_er) # n=10, 100
# er2 = np.load(get_path(2) + name_er) # n=300, 500
# er3 = np.load(get_path(3) + name_er) # n=1000
# er4 = np.load(get_path(4) + name_er) # n= 20, 30, 50
# er = np.vstack((er1[0,:], er4, er1[-1,:], er2, er3)) # 10, 20, 30, 100, 1000
# er = er.T

# m_list = np.array([16, 32, 64, 128, 256, 512, 1024])
# n_list = np.array([10, 20, 30, 50, 100, 300, 500, 1000])
# sly = False
# MSZ = None
# mfc = False
# fsd = not True
# ll = True


# plotter.plot_oneD(10, n_list, er[-5,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=64$',linestyle_str='mH-', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
# plotter.plot_oneD(10, n_list, er[-4,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=128$',linestyle_str='g^:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
# plotter.plot_oneD(10, n_list, er[-3,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=256$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
# plotter.plot_oneD(10, n_list, er[-2,:],legendlab_str=r'$m=512$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
# f10 = plotter.plot_oneD(10, n_list, er[-1,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=1024$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, xlab_str1D=r'$n$', loglog=ll)

# plt.loglog(n_list, 3e-1*n_list**(-0.5), ls=(0, (3, 1, 1, 1, 1, 1)), linewidth=LW, color='darkgray', label=r'$n^{-1/2}$')

# plt.xlim(10, 1250)
# # plt.ylim(0, 0.065)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f10, dir_path_output + 'sigest_burg_nsweep.pdf')



# %% m vs n: darcy

# er_plot = np.delete(er, [3, 5, 6], 0) # n = 5, 10, 20, 50, 500

# plotter.plot_oneD(1, m_list, er_plot[0,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=5$',linestyle_str='mH-', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, m_list, er_plot[1,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=10$',linestyle_str='g^:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, m_list, er_plot[2,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=20$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# plotter.plot_oneD(1, m_list, er_plot[3,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=50$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd)
# f1 = plotter.plot_oneD(1, m_list, er_plot[4,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=500$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, xlab_str1D=r'$m$')

# plt.xlim(0, 530)

def get_path(num):
    return '../darcy/python_version/experiments/' + str(exp_date) + '/mvsn' + str(num) + '/'

FSZ=16
LW=3

sly = False
MSZ = None
mfc = False
fsd = not True
ll = True

exp_date = 20200511
dir_path_output = '/home/nnelsen/NelsenNH_PhD_StuartAM/writing/paper-rf-sigest/new_figures/'
name_er = 'error_mvsn.npy'

m_list = np.array([8, 16, 32, 64, 128, 256, 512])
n_list = np.array([5, 10, 20, 30, 50, 100, 300, 500]) 

er1 = np.load(get_path(1) + name_er) # n=5, 10, 20
er2 = np.load(get_path(2) + name_er) # n= 30, 50 ,100
er3 = np.load(get_path(3) + name_er) # n= 300, 500
er4 = np.load(get_path(4) + name_er) # m = 512
er = np.vstack((er1, er2, er3))
er = np.hstack((er, er4))

# ---sweep m---
plotter.plot_oneD(1, m_list, er[-8,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=5$',linestyle_str='mH-', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
plotter.plot_oneD(1, m_list, er[-7,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=10$',linestyle_str='g^:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
plotter.plot_oneD(1, m_list, er[-6,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=20$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
plotter.plot_oneD(1, m_list, er[-4,:],legendlab_str=r'$n=50$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
f1 = plotter.plot_oneD(1, m_list, er[-1,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$n=500$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, xlab_str1D=r'$m$', loglog=ll)

plt.loglog(m_list, 3.5e-1*m_list**(-0.5), ls=(0, (3, 1, 1, 1, 1, 1)), linewidth=LW, color='darkgray', label=r'$m^{-1/2}$')

plt.xlim(4, 600)
# plt.ylim(0, 0.065)
plt.legend().set_draggable(True)
# plt.grid(axis='both')
# plotter.save_plot(f1, dir_path_output + 'sigest_darcy_msweep.pdf')


# # ---sweep n---
er = er.T
plotter.plot_oneD(10, n_list, er[-5,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=32$',linestyle_str='mH-', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
plotter.plot_oneD(10, n_list, er[-4,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=64$',linestyle_str='g^:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
plotter.plot_oneD(10, n_list, er[-3,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=128$',linestyle_str='bD-.', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
plotter.plot_oneD(10, n_list, er[-2,:],legendlab_str=r'$m=256$',linestyle_str='rs--', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, loglog=ll)
f10 = plotter.plot_oneD(10, n_list, er[-1,:], ylab_str1D=r'Expected Relative Test Error',legendlab_str=r'$m=512$',linestyle_str='ko:', MSZ_set=MSZ, semilogy=sly, markerfc=mfc, fig_sz_default=fsd, xlab_str1D=r'$n$', loglog=ll)

plt.loglog(n_list, 5.5e-2*n_list**(-0.1), ls=(0, (3, 1, 1, 1, 1, 1)), linewidth=LW, color='darkgray', label=r'$n^{-0.1}$')

plt.xlim(4, 633)
# plt.ylim(0, 0.065)
plt.legend().set_draggable(True)
# plt.grid(axis='both')
# plotter.save_plot(f10, dir_path_output + 'sigest_darcy_nsweep.pdf')
