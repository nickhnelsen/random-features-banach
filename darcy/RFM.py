"""
A Class for the Random Feature Model on Function Space (Darcy Flow Elliptic PDE Problem)
    -- Requires high resolution data from MATLAB (which is then subsampled)
    -- Accelerated with Numba (@numba.njit decorators)
"""

import numpy as np
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
plt.close('all') # close all open figures
#import os
#import sys
import time
from matplotlib import cm
from matplotlib.colors import Normalize
#from mpl_toolkits.mplot3d import Axes3D  

# Define and set custom LaTeX style
styleNHN = {
        "pgf.rcfonts":False,
        "pgf.texsystem": "pdflatex",   
        "text.usetex": True,                
        "font.family": "serif"
        }
mpl.rcParams.update(styleNHN)

# Plotting defaults
ALW = 0.75  # AxesLineWidth
FSZ = 12    # Fontsize
LW = 2      # LineWidth
MSZ = 5     # MarkerSize
SMALL_SIZE = 8    # Tiny font size
MEDIUM_SIZE = 10  # Small font size
BIGGER_SIZE = 14  # Large font size
SHRINK_SIZE = 0.75 # Colorbar scaling
ASPECT_SIZE = 15 # Colorbar width
DPI_SET = 300 # Default DPI for non-vector graphics figures
FORMAT_VEC_SET = 'pdf' # Default format for saving vector graphics figures
FORMAT_IM_SET = 'png' # Default format for saving raster graphics figures
BBOX_SET = 'tight' # Bounding box setting for plots
FIG_SUFFIX_VEC = '.pdf' # default file extension for vector graphics figures
FIG_SUFFIX_IM = '.png' # default file extension for pixel/raster graphics figures
NFRAME_SET = 50 # (50) total number of frames for movie writing
NFPS_SET = 20 # (20) frames per second for movie writing
CMAPCHOICE2D_SET = cm.inferno # colorbar choices: inferno, viridis, hot, gray, magma, coolwarm
CMAPCHOICE3D_SET = cm.coolwarm # colorbar choices: viridis, coolwarm, plasma
plt.rc('font', size=FSZ)         # controls default text sizes
plt.rc('axes', titlesize=FSZ)    # fontsize of the axes title
plt.rc('axes', labelsize=FSZ)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FSZ)   # fontsize of the x-tick labels
plt.rc('ytick', labelsize=FSZ)   # fontsize of the y-tick labels
plt.rc('legend', fontsize=FSZ)   # legend fontsize
plt.rc('figure', titlesize=FSZ)  # fontsize of the figure title
plt.rcParams['axes.linewidth'] = ALW    # sets the default axes lindewidth to ``ALW''
plt.rcParams["mathtext.fontset"] = 'cm' # Computer Modern mathtext font (applies when ``usetex=False'')

# Import common function files from subdirectories
from utilities.plot_suite import Plotter
plotter = Plotter() # set plotter class

# Custom imports to this file
from scipy.fft import idct, dst
from scipy.interpolate import RectBivariateSpline
import scipy.io
import numba as nb
import timeit

# %% Functions and classes


@nb.njit
def temp_trapz2D_static(U, dx):
    '''
    * only to be used inside numba jitted functions inside RandomFeatureModel class
    '''
    return np.trapz(np.trapz(U, dx=dx), dx=dx)

    
@nb.njit(fastmath=True)
def temp_simps_static(y, dx):
    '''
    * only to be used inside numba jitted functions inside RandomFeatureModel class
    '''
    s = y[0] + y[-1]
    n = y.shape[0]//2
    for i in range(n-1):
        s += 4*y[i*2 + 1]
        s += 2*y[i*2 + 2]
    s += 4*y[(n-1)*2 + 1]
    return (dx/3)*s


class RandomFeatureModel:
    '''
    Class implementation of the random feature model. Requires ``params.mat, data.mat'' files.
    '''

    def __init__(self, K, lamreg=1e-6, sig_plus=1, sig_minus=-1, denom_scale=0.15, params_path='params.mat', data_path='data.mat'):
        '''
        Arguments:
            K:              (int), number of mesh points in one spatial direction (one plus a power of two)
            
            lamreg:         (float), regularization/penalty hyperparameter strength
            
            sig_plus:       (float), upper sigmoidal threshold
            
            sig_minus:      (float), lower sigmoidal threshold
            
            denom_scale:    (float), denominator scaling inside the sigmoid function (controls bandwidth)
            
            params_path:    (str), file path name to parameter file
            
            data_path:      (str), file path name to data file


        Attributes:
            [arguments]:    (various), see ``Arguments'' above for description

            K_fine:         (int), number of high resolution mesh points in one spatial direction (one plus a power of two)
            
            grf_g:          (K, K, m, 2), precomputed Gaussian random fields 
            
            F_darcy:        (K, K), PDE forcing (scalar function)
            
            n:              (int), number of data (default: 1024)
            
            m:              (int), number of random features (default: 512)
            
            ntest:          (int), number of test points (default: 1024)
            
            rngseed:        (int), fixed seed for random number generator
            
            tau_a:          (float), inverse length scale (20, 5, 3)
            
            al_a:           (float), regularity (3, 3.5, 2)
            
            tau_g:          (float), inverse length scale (default: 15, 7.5)
            
            al_g:           (float), regularity (default: 2, 2)
            
            a_plus:         (float), coeff upper threshold (default: -1, 1e0, 1e0)
            
            a_minus:        (float), coeff lower threshold (default: -2.5, 1e-2, 1e-1)
            
            al_model:       (m,) numpy array, random feature model expansion coefficents/parameters to be learned
            
            RF_train:       (K, K, n, m) numpy array, random feature map evaluated on all training points for eachi=1,...,m
        
            AstarA:         (m, m) numpy array, normal equation matrix
            
            AstarY:         (m,), RHS in normal equations
            
            X, Y:           (K, K), coordinate grids in physical space
        '''
        
        # From input arguments 
        self.K = K
        self.lamreg = lamreg # hyperparameter
        self.sig_plus = sig_plus # hyperparameter
        self.sig_minus = sig_minus # hyperparameter
        self.denom_scale = denom_scale # hyperparameter
        self.params_path = params_path
        self.data_path = data_path
        
        # Read from the parameter and data files
        grf_full = scipy.io.loadmat(data_path)['grf_g']
        self.K_fine = grf_full.shape[0]
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        self.grf_g = grf_full[np.ix_(slice_subsample, slice_subsample)]
        self.F_darcy = scipy.io.loadmat(data_path)['F_d'][np.ix_(slice_subsample, slice_subsample)]
        P = scipy.io.loadmat(params_path)
        _, _, _, n, m, ntest, rndseed, tau_a, al_a, tau_g, al_g, a_plus, a_minus = tuple(P.values())
        self.n = n.item()
        self.m = m.item()
        self.ntest = ntest.item()
        self.rndseed = rndseed.item()
        self.tau_a = tau_a.item()
        self.al_a = al_a.item()
        self.tau_g = tau_g.item()
        self.al_g = al_g.item()
        self.a_plus = a_plus.item()
        self.a_minus = a_minus.item()
        
        # Set random seed
#        np.random.seed(self.rndseed)   
        
        # Non-input argument attributes of RFM
        self.al_model = np.zeros(self.m)
        self.RF_train = 0 # init to zero now, update to 4-tensor in ``fit'' method
        self.AstarA = 0 # init to zero now
        self.AstarY = 0 # init to zero now

        # Make physical grid
        x = y = np.arange(0, 1 + 1/(K-1), 1/(K-1))
        self.X, self.Y = np.meshgrid(x, y)
    
    
    @staticmethod
    @nb.njit(fastmath=True)
    def formAstarA_trapz_static(ind, n, AstarA, RF_train, h):
        '''
        Output:
            Returns temporary AstarA matrix with partial entries filled in (trapezoid rule).
        '''
        for i in range(ind + 1):
            for k in range(n):
                AstarA[i,ind] += temp_trapz2D_static(RF_train[:,:,k,i]*RF_train[:,:,k,ind], h)
            AstarA[ind,i] = AstarA[i,ind]
        return AstarA
    
    @staticmethod
    @nb.njit(fastmath=True)
    def formAstarA_simps_static(ind, n, AstarA, RF_train, h):
        '''
        Output:
            Returns temporary AstarA matrix with partial entries filled in (simpson rule).
        '''
        for i in range(ind + 1):
            for k in range(n):
                AstarA[i,ind] += temp_simps_static(temp_simps_static(RF_train[:,:,k,i]*RF_train[:,:,k,ind], dx=h), dx=h)
            AstarA[ind,i] = AstarA[i,ind]
        return AstarA
    
    def fit(self, order=1):
        '''
        Solve the (regularized) normal equations given the training data.
        Input:
            order: (int), integration order for inner products (default 1, trapezoid rule)
        No Output: 
            --this method only updates the class attributes ``al_model, AstarA, AstarY, RF_train''
        Data: loaded inside function
            input_train: (K, K, n), high contrast coefficents
            output_train: (K, K, n), darcy solution fields
        '''
        # Get data
        input_train, output_train, _, _ = self.get_inputoutputpairs()
        
        # Derived
        self.n = input_train.shape[-1] # update
        
        # Allocate data structures
        RF_train = np.zeros((self.K, self.K, self.n, self.m)) # need high RAM to store this when K>=256
        AstarY = np.zeros(self.m)
        AstarA = np.zeros((self.m, self.m))
        
        # Set inner product order
        ip = InnerProduct(1/(self.K - 1), order)
                
        # Form b=(A*)Y and store random features
        for i in range(self.m):
            for k in range(self.n):
                rf_temp = self.rf_local(input_train[:,:,k], self.grf_g[:,:,i,:])
                RF_train[:,:,k,i] = np.copy(rf_temp)
                AstarY[i] += ip.L2(output_train[:,:,k], rf_temp)
        
        # Form A*A symmetric positive semi-definite matrix
        temp = np.copy(AstarA)
        if order==1: # trapz
            for j in range(self.m):
                temp = RandomFeatureModel.formAstarA_trapz_static(j, self.n, temp, RF_train, 1/(self.K - 1))
            AstarA = temp/self.m
        else: # simps
            for j in range(self.m):
                temp = RandomFeatureModel.formAstarA_simps_static(j, self.n, temp, RF_train, 1/(self.K - 1))
            AstarA = temp/self.m

        # Update class attributes
        self.RF_train = RF_train 
        self.AstarY = AstarY
        self.AstarA = AstarA
        
        # Solve linear system
        if self.lamreg == 0:
            self.al_model = linalg.pinv(AstarA)@AstarY
        else:
            self.al_model = linalg.solve(AstarA + self.lamreg*np.eye(self.m), AstarY, assume_a='sym')
            
            
    def predict(self, a):
        '''
        Evaluate random feature model on a given coefficent function ``a''.
        '''
        output = np.zeros((self.K, self.K))
        for j in range(self.m):
            output = output + self.al_model[j]*self.rf_local(a, self.grf_g[:,:,j,:])
        return output/self.m
    
    
    def get_inputoutputpairs(self):
        '''
        Extract train and test input/output pairs as a 4-tuple.
        '''
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        S = scipy.io.loadmat(self.data_path)
        input_train = S['input_train'][np.ix_(slice_subsample, slice_subsample)]
        input_test = S['input_test'][np.ix_(slice_subsample, slice_subsample)]
        output_train = S['output_train'][np.ix_(slice_subsample, slice_subsample)]
        output_test = S['output_test'][np.ix_(slice_subsample, slice_subsample)]
        S = []
        return (input_train.astype(np.float64), output_train.astype(np.float64), input_test.astype(np.float64), output_test.astype(np.float64))

    
    def relative_error(self, input_tensor, output_tensor, order=1):
        '''
        Compute the expected relative error on an arbitrary set of input/output pairs.
        '''
        ip = InnerProduct(1/(self.K - 1), order)
        er = 0
        num_samples = input_tensor.shape[-1]
        for i in range(num_samples):
            y_i = output_tensor[:,:,i]
            resid = np.abs(y_i - self.predict(input_tensor[:,:,i]))
            er += np.sqrt(ip.L2(resid, resid)/ip.L2(y_i, y_i))
        er /= num_samples
        return er
    
    
    def relative_error_test(self, order=1):
        '''
        Compute the expected relative error and Bochner error on the test set.
        '''
        _, _, input_tensor, output_tensor = self.get_inputoutputpairs() # test pairs
        ip = InnerProduct(1/(self.K - 1), order)
        er = 0
        boch_num = 0
        boch_den = 0
        num_samples = input_tensor.shape[-1]
        if self.ntest != num_samples:
            self.ntest = num_samples
        for i in range(num_samples):
            y_i = output_tensor[:,:,i]
            resid = np.abs(y_i - self.predict(input_tensor[:,:,i]))
            num = ip.L2(resid, resid)
            den = ip.L2(y_i, y_i)
            er += np.sqrt(num/den)
            boch_num += num
            boch_den += den
        er /= num_samples
        er_bochner = np.sqrt(boch_num/boch_den)
        return er, er_bochner
    
    
    @staticmethod
    @nb.njit(fastmath=True)
    def predict_train(ind_a, K, m, al_model, RF_train):
        '''
        Evaluate random feature model on a given coefficent training sample ``a_{ind_a}'' (precomputed).
        '''
        output = np.zeros((K, K))
        for j in range(m):
            output = output + al_model[j]*RF_train[:,:,ind_a,j]
        return output/m
    
    def relative_error_train(self, order=1):
        '''
        Compute the expected relative error and Bochner error on the training set (using pre-computed random features).
        '''
        _, output_tensor, _, _ = self.get_inputoutputpairs()
        ip = InnerProduct(1/(self.K - 1), order)
        er = 0
        boch_num = 0
        boch_den = 0
        num_samples = output_tensor.shape[-1]
        if self.n != num_samples:
            self.n = num_samples
        for i in range(num_samples):
            y_i = output_tensor[:,:,i]
            resid = np.abs(y_i - RandomFeatureModel.predict_train(i, self.K, self.m, self.al_model, self.RF_train))
            num = ip.L2(resid, resid)
            den = ip.L2(y_i, y_i)
            er += np.sqrt(num/den)
            boch_num += num
            boch_den += den
        er /= num_samples
        er_bochner = np.sqrt(boch_num/boch_den)
        return er, er_bochner
    
    
    def rf_local(self, a_input, gr_field):
        '''
        Shortcut function for random feature mapping. Cannot be numba-ized.
        '''
        return rf(a_input, gr_field, f_num=self.F_darcy, sig_plus=self.sig_plus, sig_minus=self.sig_minus, denom_scale=self.denom_scale)
    
    
    def regsweep(self, lambda_list=[1e-5, 1e-6, 1e-7, 1e-8, 1e-9]):
        '''
        Regularization hyperparameter sweep. Requires model to be fit first at least once. Updates model parameters to best performing ones.
        Input:
            lambda_list: (list), list of lambda values
        Output:
            er_store : (len(lambda_list), 5) numpy array, error storage
        '''
        er_store = np.zeros([len(lambda_list), 5]) # lamreg, e_train, b_train, e_test, b_test
        for loop in range(len(lambda_list)):
            reg = lambda_list[loop]
            er_store[loop,0] = reg 
            print('Running \lambda = ', reg)
            
            # Solve linear system
            self.al_model = linalg.solve(self.AstarA + reg*np.eye(self.m), self.AstarY, assume_a='sym')
            
            # Training error
            er_store[loop, 1:3] = self.relative_error_train()
            
            # Test error
            er_store[loop, 3:] = self.relative_error_test()
            
            # Print
            print('Expected relative error (Train, Test): ' , (er_store[loop, 1], er_store[loop, 3]))
            
        # Find lambda with smallest test error
        ind_arr = np.argmin(er_store, axis=0)[3] # smallest test index
        best_reg = er_store[ind_arr, 0]
        self.lamreg = best_reg
        
        # Update class attributes with best lambda
        self.al_model = linalg.solve(self.AstarA + best_reg*np.eye(self.m), self.AstarY, assume_a='sym')
        
        return er_store
            

def gridsweep(K_list=[9, 17, 33, 65, 129], lamreg=1e-6, sig_plus=1, sig_minus=-1, denom_scale=0.15):
    '''
    Grid sweep to show mesh invariant relative error.
    '''
    er_store = np.zeros([len(K_list), 5]) # K, e_train, b_train, e_test, b_test
    for loop in range(len(K_list)):
        K = K_list[loop]
        er_store[loop,0] = K
        print('Running grid size s = ', K)
        
        rfm = RandomFeatureModel(K, lamreg, sig_plus, sig_minus, denom_scale)
        rfm.fit()
        
        # Train error
        er_store[loop, 1:3] = rfm.relative_error_train()
        
        # Test error
        er_store[loop, 3:] = rfm.relative_error_test()
        
        # Print
        print('Expected relative error (Train, Test): ' , (er_store[loop, 1], er_store[loop, 3]))
    
    return er_store    
    
            
class InnerProduct:
    '''
    Class implementation of L^2, H^1, and H_0^1 inner products from samples of data on [0,1]^2.
    '''

    def __init__(self, h, ORDER=1):
        '''
        Initializes the class.
        Arguments:
            h:          Mesh size in both the x and y directions.
            
            ORDER:      The order of accuracy of the numerical quadrature: [1] for trapezoid rule, [2] for Simpson's rule.

        Parameters:
            h:          (float), Mesh size in both the x and y directions.

            order:      (int), The order of accuracy of the numerical quadrature: [1] for trapezoid rule, [2] for Simpson's rule.
                
            quad_type:  (str), Type of numerical quadrature chosen.
        '''
        
        self.h = h
        self.order = ORDER
        if ORDER == 1: # trapezoid
            self.quad_type = 'trapezoid'
        else: # simpson
            self.quad_type = 'simpson'
            
            
    @staticmethod
    @nb.njit
    def trapz2D_static(U, h):
        '''
        2D trapezoid rule 
        '''
        return np.trapz(np.trapz(U, dx=h), dx=h)
        
    @staticmethod
    @nb.njit(fastmath=True)
    def simps_static(y, dx):
        s = y[0] + y[-1]
        n = y.shape[0]//2
        for i in range(n-1):
            s += 4*y[i*2 + 1]
            s += 2*y[i*2 + 2]
        s += 4*y[(n-1)*2 + 1]
        return (dx/3)*s
            
    def L2(self, F, G):
        '''
        L^2 inner product
        https://stackoverflow.com/questions/50440592/is-there-any-good-way-to-optimize-the-speed-of-this-python-code
        '''
        if self.order == 1:
            out = InnerProduct.trapz2D_static(F*G, self.h)
        else:
            out = InnerProduct.simps_static(InnerProduct.simps_static(F*G, dx=self.h), dx=self.h)
        return out
        
    
    def H01(self, F, G):
        '''
        H_0^1 inner product
        '''
        Fx, Fy = np.gradient(F, self.h, edge_order=2)
        Gx, Gy = np.gradient(G, self.h, edge_order=2)
        integrand = Fx*Gx + Fy*Gy
        if self.order == 1:
            out = InnerProduct.trapz2D_static(integrand, self.h)
        else:
            out = InnerProduct.simps_static(InnerProduct.simps_static(integrand, dx=self.h), dx=self.h)
        return out
    
    
    def H1(self, F, G):
        '''
        H^1 inner product
        '''
        return self.L2(F, G) + self.H01(F, G)


class GaussianRandomField:
    '''
    Return a sample of a Gaussian random field on [0,1]^2 with: 
        -- mean function m = 0
        -- covariance operator C = (-Delta + tau^2)^(-alpha),
    where Delta is the Laplacian with zero Dirichlet or Neumann boundary conditions.
    '''

    def __init__(self, tau, alpha, bc=0):
        '''
        Initializes the class.
        Arguments:
            tau:        (float), inverse lengthscale for Gaussian measure covariance operator
            
            alpha:      (alpha), regularity of covariance operator
            
            bc:         (int), ``0'' for Neumann BCs or ``1'' for Dirichlet BCs

        Parameters:
            tau:        (float), inverse lengthscale for Gaussian measure covariance operator
            
            alpha:      (alpha), regularity of covariance operator
            
            bc:         (int), ``0'' for Neumann BCs or ``1'' for Dirichlet BCs

            bc_name:    (str), ``neumann'' for Neumann BCs or ``dirichlet'' for Dirichlet BCs
        '''
        
        self.tau = tau
        self.alpha = alpha
        if bc == 0: # neumann
            self.bc = 0
            self.bc_name = 'neumann'
        else: # dirichlet
            self.bc = 1
            self.bc_name = 'dirichlet'
            

    def draw(self, theta):
        '''
        Draw a sample Gaussian Random Field on [0,1]^2 with desired BCs
        Input:
            theta: (N, N) numpy array of N(0,1) iid Gaussian random variables
        Output:
            grf: (N,N) numpy array, a GRF on the grid meshgrid(np.arange(0,1+1/(N-1),1/(N-1)),np.arange(0,1+1/(N-1),1/(N-1)))
        '''
        
        # Length of random variables matrix in 2D KL expansion
        N = theta.shape[0]
        
        # Choose BCs
        if self.bc == 0: # neumann
            # Define the (square root of) the eigenvalues of the covariance operator
            K1, K2 = np.meshgrid(np.arange(N), np.arange(N)) # k= (K1, K2)
            coef = (self.tau**(self.alpha - 1))*(np.pi**2*(K1**2 + K2**2) + self.tau**2)**(-self.alpha/2) # (alpha-d/2) scaling
            # Construct the KL (discrete cosine transform) coefficients
            B = N*coef*theta # multiply by N to satisfy the DCT Type II definition
            B[0,:] = np.sqrt(2)*B[0,:] # adjust for wavenumber 0
            B[:,0] = np.sqrt(2)*B[:,0] # adjust for wavenumber 0
            B[0,0] = 0 # set k=(0,0) constant mode to zero (to satisfy zero mean field)
            
            # Inverse (fast FFT-based) 2D discrete cosine transform
            grf_temp = idct2(B) # sums B*2/N*cos(k1*pi*x)*cos(k2*pi*y) over all k1, k2 = 0, ..., N-1
            
            # Interpolate to physical grid containing the boundary of the domain [0,1]^2
            X1 = Y1 = np.arange(1/(2*N),(2*N-1)/(2*N)+1/N, 1/N) # IDCT grid
            X2 = Y2 = np.arange(0,1+1/(N-1),1/(N-1)) # physical domain grid
            func_interp = RectBivariateSpline(X1, Y1, grf_temp)
            grf = func_interp(X2, Y2)
            
        else: # dirichlet
            # Define the (square root of) the eigenvalues of the covariance operator
            K1, K2 = np.meshgrid(np.arange(1,N-1), np.arange(1,N-1)) # k= (K1, K2) (does not include first of last wavenumbers)
            coef = (self.tau**(self.alpha - 1))*(np.pi**2*(K1**2 + K2**2) + self.tau**2)**(-self.alpha/2) # (alpha-d/2) scaling
            
            # Construct the KL (discrete sine transform) coefficients
            B = (N-1)*coef*theta[1:-1, 1:-1] # multiply by (N-1) to satisfy the DST Type I definition
            
            # Inverse (fast FFT--based) 2D discrete sine transform
            U_noBC = dst2(B) # sums B*2/(N-1)*sin(k1*pi*x)*sin(k2*pi*y) over all k1,k2=1,...,N-2
            
            # Impose zero boundary data on the output
            grf = np.zeros((N,N))     
            grf[1:-1,1:-1] = np.copy(U_noBC)
        
        return grf


def make_coeff(grf, a_plus=1, a_minus=1e-2):
    '''
    Given a (Gaussian random) field in 2D, return the high contrast coefficent obtained by thresholding the field about the zero level set.
    Input:
        grf: (K, K) numpy array
        a_plus, a_minus: (float), upper and lower thresholds, respectively
    Output:
        _ : (K, K) high contrast interface field
    '''
    return a_plus*(grf > 0).astype(np.float64) + a_minus*(grf <= 0).astype(np.float64)


def dstnhn(x):
    '''
    Orthonormalized disrete sine transform type I (does NOT include zeros at boundaries in x), where dst(dst(x))=x
    Input:
        x: (n, m) numpy array
    Output:
        output: (n, m) numpy array
    '''
# =============================================================================
#     n, m = x.shape
#     y1 = np.vstack( (np.zeros((1,m)),x) )
#     y2 = np.imag(np.fft.fft(y1, n=2*n+2, axis=0));
#     output = np.sqrt(2/(n+1))*y2[1:n+1,:]
#     return output
# =============================================================================
    return dst(x, 1, axis=0, norm='ortho')


def dst2(x):
    '''
    2D Orthonormalized discrete sine transform type I
    '''
    return dstnhn(dstnhn(x).T).T


def idct2(block):
    '''
    2D inverse discrete cosine transform Type 2 (Orthonormalized), equivalent to MATLAB's ``idct2''
    Input:
        block: (N, M) numpy array
    Output:
        _ : (N, M) numpy array
    '''
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def solve_poisson_fast(F):
    '''
    Solve the Poisson equation -\lap U = F on [0,1]^2 with zero Dirichlet boundary conditions
        --- Uses orthonormalized fast discrete sine transform (FFT--based)
    Input:
        F: (N,N) numpy array, forcing function on the grid
    Output:
        U: (N,N) numpy array, Poisson solution on the grid
    '''
    N = F.shape[0] # grid point length
    h = 1/(N-1) # mesh size
    K1, K2 = np.meshgrid(np.arange(1,N-1), np.arange(1,N-1)) # Fourier wavenumbers
    Fst = dst2(F[1:-1,1:-1])  
#    eig = 4/(h**2)*(np.sin(K1/2*np.pi/(N-1))**2 + np.sin(K2/2*np.pi/(N-1))**2) # FD eigs
    eig = 2/(h**2)*(2 - np.cos(K1*np.pi/(N-1)) - np.cos(K2*np.pi/(N-1))) # FD eigs (faster)
#    eig = (np.pi*K1)**2+(np.pi*K2)**2 # Exact Laplacian eigs (spectral method)
    Ust = Fst/eig
    U = np.zeros((N,N))     
    U[1:-1,1:-1] = dst2(Ust)
    return U


@nb.njit
def mollify(a):
    '''
    Mollify/smooth the input coefficient function ``a'' with the heat operator equipped with Neumann BCs on unit square [0,1]^2
    Input:
        a: (K, K) numpy array
    Output:
        a_smooth: (K, K) numpy array
    '''
    
    # Set viscosity
    nu = 1e-4
    
    # Extract mesh size
    K = a.shape[0]
    h = 1/(K-1)
    
    # Form mesh in time
    dt = 3e-2 # 34 time steps, CFL valid for grids up to K = 257**2
    Nt = np.arange(0, 1, dt).shape[0]
    
    # Set initial condition
    U_old = np.copy(a)
    U_new = np.zeros((K, K), dtype = np.float64)
    
    # Finite difference scheme
    for t in range(Nt - 1):
#        # Reset
#        U_new = np.zeros((K, K), dtype = np.float64)
        
        # Update interior of spatial domain
        U_new[1:-1,1:-1] = U_old[1:-1,1:-1] + dt*( nu/(h**2)*(-4*U_old[1:-1,1:-1] + U_old[2:,1:-1] + U_old[:-2,1:-1] + U_old[1:-1,2:] + U_old[1:-1,:-2]) )
        
        # Update homogeneous Neumann boundaries (NOT including corners)
        U_new[1:-1,-1] = U_old[1:-1,-1] + dt*( nu/(h**2)*(-4*U_old[1:-1,-1] + U_old[2:,-1] + U_old[:-2,-1] + 2*U_old[1:-1,-2]) ) # top
        U_new[1:-1,0] = U_old[1:-1,0] + dt*( nu/(h**2)*(-4*U_old[1:-1,0] + U_old[2:,0] + U_old[:-2,0] + 2*U_old[1:-1,1]) ) # bottom
        U_new[0,1:-1] = U_old[0,1:-1] + dt*( nu/(h**2)*(-4*U_old[0,1:-1] + U_old[0,2:] + U_old[0,:-2] + 2*U_old[1,1:-1]) ) # left
        U_new[-1,1:-1] = U_old[-1,1:-1] + dt*( nu/(h**2)*(-4*U_old[-1,1:-1] + U_old[-1,2:] + U_old[-1,0:-2] + 2*U_old[-2,1:-1]) ) # right
        
        # Update corners of the unit square
        U_new[0,0] = U_old[0,0] + dt*( nu/(h**2)*(-4*U_old[0,0] + 2*U_old[0,1] + 2*U_old[1,0]) ) # bottom left
        U_new[-1,-1] = U_old[-1,-1] + dt*( nu/(h**2)*(-4*U_old[-1,-1] + 2*U_old[-2,-1] + 2*U_old[-1,-2]) ) # top right
        U_new[0,-1] = U_old[0,-1] + dt*( nu/(h**2)*(-4*U_old[0,-1] + 2*U_old[1,-1] + 2*U_old[0,-2]) ) # top left
        U_new[-1,0] = U_old[-1,0] + dt*( nu/(h**2)*(-4*U_old[-1,0] + 2*U_old[-1,1] + 2*U_old[-2,0]) ) # bottom right
        
        # Update solution data structure
        U_old = np.copy(U_new)
    
    # Solution at time 1
    return U_old # = a_smooth


def W_kernel(r, ep=1e-2, p=4):
    return np.max(0, 1 - r/ep)**p


@nb.njit
def sigma(r, sig_plus, sig_minus, denom_scale):
    '''
    Thresholded sigmoidal function
    '''
    output = (sig_plus - sig_minus)/(1 + np.exp(-(r/denom_scale))) + sig_minus
    return output


def rf(a_input, gr_field, f_num, sig_plus=1, sig_minus=-1, denom_scale=0.15):
    '''
    Generates a predictor--corrector Poisson solution random feature mapping with Dirichlet BCs
        -- Requires the functions ``solve_poisson_fast, sigma, sigmoid''
    Input:
        a_input: (K,K), high contrast coefficient function
        gr_field: (K,K,2) precomputed Gaussian random fields
        f_num: (K,K) RHS source term in original Darcy PDE
        sig_plus, sig_minus: (float), thresholds for sigmoids
        denom_scale: (float), scaling in sigmoid(r/denom_scale)
        sig_in_num: (boolean), choose to have sigmoid random nonlinearity in numerator of RHS source
    Output
        output: (K,K) matrix for varphi(a_input; gr_field) random feature
    '''

    # Extract GRFs
    gr_field1 = gr_field[:,:,0]
    gr_field2 = gr_field[:,:,1]

    # Predictor-Corrector
# =============================================================================
#     # Option 1
#     F0 = f_num/(a_input + sigma(gr_field1, sig_plus, sig_minus, denom_scale) )
#     gradX_U0, gradY_U0 = np.gradient(solve_poisson_fast(F0), 1/(a_input.shape[0] - 1), edge_order=2)
#     gradX_loga, gradY_loga = np.gradient(np.log(mollify(a_input)), 1/(a_input.shape[0] - 1), edge_order=2)
#     F = f_num/(a_input +  sigma(gr_field2, sig_plus, sig_minus, denom_scale) ) + gradX_loga*gradX_U0 + gradY_loga*gradY_U0
# =============================================================================
    
    # Option 2
    F0 = f_num/(a_input) + sigma(gr_field1, sig_plus, sig_minus, denom_scale)
    gradX_U0, gradY_U0 = np.gradient(solve_poisson_fast(F0), 1/(a_input.shape[0] - 1), edge_order=2)
    gradX_loga, gradY_loga = np.gradient(np.log(mollify(a_input)), 1/(a_input.shape[0] - 1), edge_order=2)
    F = f_num/(a_input) + sigma(gr_field2, sig_plus, sig_minus, denom_scale) + gradX_loga*gradX_U0 + gradY_loga*gradY_U0
    
# =============================================================================
#     # Option 3
#     F0 = f_num/a_input
#     gradX_U0, gradY_U0 = np.gradient(solve_poisson_fast(F0), 1/(a_input.shape[0] - 1), edge_order=2)
#     gradX_loga, gradY_loga = np.gradient(np.log(mollify(a_input)), 1/(a_input.shape[0] - 1), edge_order=2)
#     F = f_num/(a_input +  sigma(gr_field2, sig_plus, sig_minus, denom_scale) ) + gradX_loga*gradX_U0 + gradY_loga*gradY_U0 + sigma(gr_field1, sig_plus, sig_minus, denom_scale)
# #    F = f_num/(a_input +  sigma(gr_field2, -1, -2.9, denom_scale) ) + gradX_loga*gradX_U0 + gradY_loga*gradY_U0 + sigma(gr_field1, sig_plus, sig_minus, denom_scale) 
# =============================================================================
       
    # Fast Poisson solve for random feature output
    return solve_poisson_fast(F)

