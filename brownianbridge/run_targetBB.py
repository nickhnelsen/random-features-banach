# Import modules/packages
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
plt.close('all') # close all open figures
from matplotlib import cm

# Import common function files from subdirectories
from utilities.plot_suiteSIAM import Plotter
plotter = Plotter() # set plotter class

# %% Functions

def BB(x, theta = np.zeros((1000,1))):
    '''
    Brownian bridge function BB: \R \to \R
    Input
    x: dim (n,1) numpy array
    theta: dim (trunc,m) numpy array, \theta_{ij} \sim N(0,1) idd
    
    Output
    output: dim (n,m) numpy array of BB(x)
    '''
    trunc, m = theta.shape
    if linalg.norm(theta, 2) == 0:
        np.random.seed(1234)
        theta = np.random.standard_normal(theta.shape) # \theta_j \sim N(0,1) idd
    j = np.arange(1,trunc+1)[:,None] # dim (trunc,1) numpy array, j=1 to j=trunc
    coeff = theta/(np.pi*j) # dim (trunc,m) numpy array
    output = np.sqrt(2)*np.sin(np.pi*(x@j.T))@coeff # shape (n, m)
    return output


def kernel(x,y):
    '''
    Covariance kernel for the Brownian bridge
    Input
    x: dim(n,1) numpy arrays
    y: dim(m,1) numpy arrays
    
    Output
    output: dim(n,m) numpy array
    '''
    output = np.minimum(x,y.T)-x@y.T
    return output


def rfm(x, coeff, theta):
    '''
    Evaluation of the random feature model for Brownian bridge features
    -- Requires ``BB'' function
    Input
    x: dim (n,1) numpy array
    coeff: dim (m,1) numpy array
    theta: dim (trunc,m) numpy array, \theta_{ij} \sim N(0,1) idd
    
    Output
    output: dim(n,1) numpy array
    '''
    m = coeff.shape[0]
    features = BB(x, theta)
    output = 1/m*features@coeff
    return output


def rep(x, coeff, data_x):
    '''
    Evaluation of Representer Thm model
    Input
    x: dim (n,1) numpy array
    data_x: dim (N,1) numpy array
    coeff: dim (N,1) numpy array
    
    Output
    output: dim(n,1) numpy array
    '''
    output = kernel(x, data_x)@coeff
    return output


def mrep(x, coeff, data_x, theta):
    '''
    Evaluation of Empirical Representer Thm model
    Input
    x: dim (n,1) numpy array
    data_x: dim (N,1) numpy array
    coeff: dim (N,1) numpy array
    theta: dim (trunc,m) numpy array, \theta_{ij} \sim N(0,1) idd

    Output
    output: dim(n,1) numpy array
    '''
    m = theta_draw.shape[1]
    dat_features = BB(data_x, theta) # dim (N,m)
    eval_features = BB(x, theta) # dim (n,m)
    empir_kernel_mat = 1/m*eval_features@dat_features.T
    output = empir_kernel_mat@coeff
    return output


def heaviside(r):
    """ True Heaviside step function"""
    return np.heaviside(r,0.5)


def smoothheaviside(r):
    """ Smoothed Heaviside step function"""
    ep=0.001
    return 1/2 + (1/np.pi)*np.arctan(r/ep)


def sawtooth(r):
    """ Sawtooth function from Yarotsky"""
#    return np.where(r < 1/2, np.maximum(0,2*r), np.maximum(0,2*(1-r)))
    return np.maximum(0,np.minimum(2*r,2-2*r))

# %% Setup

n = 1024 # number of data points in training set
m = 5000 # number of random features
# n_trunc = m # number of terms in all series truncations in this script
x_spacing = 0 # enter ``0'' for uniform draw, ``1'' for equally spaced, ``2'' for Chebyshev nodes
dir_name = 'figures_target/'
TEST_NUM = '5'

# Target functions
n_trunc_target = 1000
def target_rkhs(x):
    '''
    Truncated series of functions in RKHS of Moore-Aronszajn form
    Input
    x: dim (n,1) numpy array
    '''
    kernel_matrix = kernel(x,np.linspace(0,1,n_trunc_target)[:,None])
    np.random.seed(1177777) # default seed value: 1177777 (only for the case n_trunc=1000)
    beta = np.random.standard_normal((n_trunc_target,1))
    return kernel_matrix@beta

# Choose target from above list
# F = target_rkhs
np.random.seed(1171642)
ttt = np.random.standard_normal((n_trunc_target,1))
F = lambda x: BB(x, ttt)

# %% Derived quantities
    
# Draw data
delta = 1e-2
xmin = 0 + 0*delta
xmax = 1 - 0*delta
x_fine = np.linspace(xmin,xmax,1000)[:,None]
if x_spacing==0: # uniform draw
#        x = np.sort(np.random.uniform(0, 1, [n,1]), axis=0) # matrix of ``nx_1d'' input vectors (columns) of dimension m=1 drawn uniformly
    x = np.sort(np.random.uniform(xmin, xmax, [n,1]), axis=0) # matrix of ``nx_1d'' input vectors (columns) of dimension m=1 drawn uniformly

elif x_spacing==1: # equally spaced
#        x = np.arange(1/(n+1), 1, 1/(n+1))[:, None] # equally spaced x-data points
    x = np.linspace(xmin, xmax, n)[:, None] # equally spaced x-data points
else: # Chebyshev nodes
#        x = 0.5+0.5*np.cos(np.pi*(1+2*(n-np.arange(1,n+1)[:,None]))/(2*n)) # Chebyshev nodes
    x = 0.5*(xmin+xmax)+0.5*(xmax-xmin)*np.cos(np.pi*(1+2*(n-np.arange(1,n+1)[:,None]))/(2*n)) # Chebyshev nodes
y = F(x)

# %% Solve linear equations 

# Draw features
theta_draw = np.random.standard_normal((n_trunc_target,m))
features = BB(x, theta_draw)
A_matrix = features/m

# Representer theorem model
K_matrix = kernel(x,x)

# Empirical Representer theorem model
Km_matrix = features@(features.T)/m

# Print condition numbers
print(np.linalg.cond(K_matrix), np.linalg.cond(Km_matrix), np.linalg.cond(A_matrix))

# %% 
# Solve LS
coeff = linalg.lstsq(A_matrix, y)[0] # rfm
coeff_rep = linalg.lstsq(K_matrix, y)[0] # representer thm
coeff_mrep = linalg.lstsq(Km_matrix, y)[0] # empirical

# %% Plotting individual solutions

# Form RFM solution
train = rfm(x, coeff, theta_draw)
test = rfm(x_fine, coeff, theta_draw)

# Form Representer solution
train_rep = K_matrix@coeff_rep
test_rep = rep(x_fine, coeff_rep, x)


# =============================================================================
# # Plotting (small data)
# fdef = True
# MSZ = 10
# mfc = False
# plotter.plot_oneD(1, x_fine, test, legendlab_str=r'Test', linestyle_str='b', fig_sz_default=fdef)
# plotter.plot_oneD(1, x_fine, F(x_fine), legendlab_str=r'Truth', linestyle_str='r--', fig_sz_default=fdef)
# f1 = plotter.plot_oneD(1, x, train, legendlab_str=r'Train', linestyle_str='ko', fig_sz_default=fdef,markerfc=mfc, MSZ_set=MSZ)
# plt.xlim(-0.05, 1.05)
# plt.ylim(-1.2, 0.5)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f1,'rfm64_5000_wide.pdf')
# 
# plotter.plot_oneD(2, x_fine, test_rep, legendlab_str=r'Test', linestyle_str='b', fig_sz_default=fdef)
# plotter.plot_oneD(2, x_fine, F(x_fine), legendlab_str=r'Truth', linestyle_str='r--', fig_sz_default=fdef)
# f2 = plotter.plot_oneD(2, x, train_rep, legendlab_str=r'Train', linestyle_str='ko', fig_sz_default=fdef, markerfc=mfc, MSZ_set=MSZ)
# plt.xlim(-0.05, 1.05)
# plt.ylim(-1.2, 0.5)
# plt.legend().set_draggable(True)
# # plotter.save_plot(f2,'rep64_color.pdf')
# =============================================================================


# Plotting (large data)
fdef = True
MSZ = 10
mfc = False
plotter.plot_oneD(1, x_fine, test, legendlab_str=r'Test', linestyle_str='b', fig_sz_default=fdef)
f1 = plotter.plot_oneD(1, x_fine, F(x_fine), legendlab_str=r'Truth', linestyle_str='r--', fig_sz_default=fdef)
plt.xlim(-0.05, 1.05)
plt.ylim(-1.2, 0.5)
plt.legend().set_draggable(True)
# plotter.save_plot(f1,'rfm1024_5000_wide.pdf')

plotter.plot_oneD(2, x_fine, test_rep, legendlab_str=r'Test', linestyle_str='b', fig_sz_default=fdef)
f2 = plotter.plot_oneD(2, x_fine, F(x_fine), legendlab_str=r'Truth', linestyle_str='r--', fig_sz_default=fdef)
plt.xlim(-0.05, 1.05)
plt.ylim(-1.2, 0.5)
plt.legend().set_draggable(True)
# plotter.save_plot(f2,'rep1024_color.pdf')


# Plot RFM
# fig1 = plt.figure(1)
# plt.plot(x_fine,test,'b',linewidth=LW, markersize=MSZ, label=r'Test')
# plt.plot(x_fine,F(x_fine),'r', linewidth=LW, markersize=MSZ, label=r'Exact')
# #plt.plot(x,y,'ks', linewidth=LW, markersize=MSZ-1, label=r'Exact')
# plt.plot(x,train,'ko',linewidth=LW, markersize=MSZ, label=r'Train')
# plt.xlabel(r'$x$')
# plt.legend(loc='best')
# plt.title(r'Random Feature Model: $n = %d, \ m = %d$' %(n,m))


# # Plot Representer
# fig2 = plt.figure(2)
# plt.plot(x_fine,test_rep,'b',linewidth=LW, markersize=MSZ, label=r'Test')
# plt.plot(x_fine,F(x_fine),'r', linewidth=LW, markersize=MSZ, label=r'Exact')
# #plt.plot(x,y,'ks', linewidth=LW, markersize=MSZ-1, label=r'Exact')
# plt.plot(x,train_rep,'ko',linewidth=LW, markersize=MSZ, label=r'Train')
# plt.xlabel(r'$x$')
# plt.legend(loc='best')
# plt.title(r'Representer Theorem Model: $n = %d$' % n)

# =============================================================================
# # Write to file
# fig1.savefig(dir_name+'rfm_test'+TEST_NUM+FIG_SUFFIX_VEC, format=FORMAT_VEC_SET, dpi=DPI_SET, bbox_inches=BBOX_SET)
# fig2.savefig(dir_name+'rep_test'+TEST_NUM+FIG_SUFFIX_VEC, format=FORMAT_VEC_SET, dpi=DPI_SET, bbox_inches=BBOX_SET)
# =============================================================================

# =============================================================================
# # Misc
# plt.figure(1111)
# plt.plot(x,BB(x),'k')
# plt.plot(x,F(x),'b')
# plt.plot(x,kernel(x,np.array([[0.6]])),'g')
# plt.plot(x_fine, target_rkhs(x_fine),'r')
# plt.title(r'Misc. tests')
# =============================================================================
