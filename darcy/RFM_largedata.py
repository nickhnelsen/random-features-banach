"""
A module file with class for the Random Feature Model on Function Space (Darcy Flow Elliptic PDE Problem)
    -- Accelerated with Numba (@numba.njit decorators)
    -- For use only with single large (multiple GB), high resolution Darcy flow dataset
"""

import numpy as np
from scipy import linalg

# Custom imports to this file
from scipy.fft import dst
import scipy.io
import numba as nb
import sys
from utilities.activations import elu

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
    Class implementation of the random feature model. Requires ``params.mat, data.mat'' files in ``data'' directory.
    '''

    def __init__(self, K=65, n=100, m=256, ntest=512, lamreg=1e-9, sig_plus=1*1/12, sig_minus=-1*1/3, denom_scale=0.15, params_path='data/params.mat', data_path='data/data.mat', rf_choice=0):
        '''
        Arguments:
            K:              (int), number of mesh points in one spatial direction (one plus a power of two)
            
            n:              (int), number of data (max: 1024)
            
            m:              (int), number of random features (max: 1024)
            
            ntest:          (int), number of test points (max: 5000)
            
            lamreg:         (float), regularization/penalty hyperparameter strength
            
            sig_plus:       (float), upper sigmoidal threshold
            
            sig_minus:      (float), lower sigmoidal threshold
            
            denom_scale:    (float), denominator scaling inside the sigmoid function (controls bandwidth)
            
            params_path:    (str), file path name to parameter file
            
            data_path:      (str), file path name to data file
            
            rf_choice:        (int), 0 for PC, or 1 for flow-based random features


        Attributes:
            [arguments]:    (various), see ``Arguments'' above for description

            K_fine:         (int), number of high resolution mesh points in one spatial direction (one plus a power of two)
            
            grf_g:          (K, K, m, 2), precomputed Gaussian random fields 
            
            F_darcy:        (K, K), PDE forcing (scalar function)
            
            n_max:          (int), max number of data (max: 1024)
            
            m_max:          (int), max number of random features (max: 1024)
            
            ntest_max:      (int), max number of test points (max: 5000)
            
            rngseed:        (int), fixed seed for random number generator used in MATLAB
            
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
            
            rf_type:        (str), random feature type: ``pc'' or ``flow'' 
        '''
        
        # From input arguments 
        self.K = K
        self.n = n
        self.m = m
        self.ntest = ntest
        self.lamreg = lamreg # hyperparameter
        self.sig_plus = sig_plus # hyperparameter
        self.sig_minus = sig_minus # hyperparameter
        self.denom_scale = denom_scale # hyperparameter
        self.params_path = params_path
        self.data_path = data_path
        
        # Read from the parameter file
        P = scipy.io.loadmat(params_path)
        _, _, _, K_fine, n_max, m_max, ntest_max, rndseed, tau_a, al_a, tau_g, al_g, a_plus, a_minus = tuple(P.values())
        self.K_fine = K_fine.item()
        self.n_max = n_max.item()
        self.m_max = m_max.item()
        self.ntest_max = ntest_max.item()
        self.rndseed = rndseed.item()
        self.tau_a = tau_a.item()
        self.al_a = al_a.item()
        self.tau_g = tau_g.item()
        self.al_g = al_g.item()
        self.a_plus = a_plus.item()
        self.a_minus = a_minus.item()
        
        # Set random seed
        np.random.seed(self.rndseed)
        slice_grf = np.random.choice(self.m_max, self.m, replace=False)
        self.slice_n = np.random.choice(self.n_max, self.n, replace=False)
        self.slice_ntest = np.random.choice(self.ntest_max, self.ntest, replace=False)

        # Read from data file and downsample
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        self.grf_g = scipy.io.loadmat(data_path, variable_names='grf_g')['grf_g'][np.ix_(slice_subsample, slice_subsample, slice_grf)].astype(np.float64)
        self.F_darcy = scipy.io.loadmat(data_path, variable_names='F_d')['F_d'][np.ix_(slice_subsample, slice_subsample)].astype(np.float64)
       
        # Non-input argument attributes of RFM
        self.al_model = np.zeros(self.m)
        self.RF_train = 0 # init to zero now, update to 4-tensor in ``fit'' method
        self.AstarA = 0 # init to zero now
        self.AstarY = 0 # init to zero now

        # Make physical grid
        x = y = np.arange(0, 1 + 1/(K-1), 1/(K-1))
        self.X, self.Y = np.meshgrid(x, y)
        
        # Address choice of random feature mapping
        if rf_choice == 0: # pc rf
            self.rf_type = 'pc'
        else: # flow rf
            self.rf_type = 'flow'
            def rf_flow(a_input, gr_field):
                '''
                Shortcut function for flow-based random feature mapping using global function ``rff''.
                '''
                return rff(a_input, gr_field, sig_plus=sig_plus, sig_minus=sig_minus, denom_scale=denom_scale)
            self.rf_local = rf_flow
    
    
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
        input_train, output_train = self.get_trainpairs()
        
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
            self.al_model = linalg.pinv2(AstarA)@AstarY
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
    
    
    def get_trainpairs(self):
        '''
        Extract train input/output pairs as a 2-tuple.
        '''
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        input_train = scipy.io.loadmat(self.data_path, variable_names='input_train')['input_train'][np.ix_(slice_subsample, slice_subsample, self.slice_n)].astype(np.float64)
        output_train = scipy.io.loadmat(self.data_path, variable_names='output_train')['output_train'][np.ix_(slice_subsample, slice_subsample, self.slice_n)].astype(np.float64)
        return (input_train, output_train)
    
    def get_testpairs(self):
        '''
        Extract test input/output pairs as a 2-tuple.
        '''
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        input_test = scipy.io.loadmat(self.data_path, variable_names='input_test')['input_test'][np.ix_(slice_subsample, slice_subsample, self.slice_ntest)].astype(np.float64)
        output_test = scipy.io.loadmat(self.data_path, variable_names='output_test')['output_test'][np.ix_(slice_subsample, slice_subsample, self.slice_ntest)].astype(np.float64)
        return (input_test, output_test)
    
    def get_inputoutputpairs(self):
        '''
        Extract train and test input/output pairs as a 4-tuple.
        '''
        return self.get_trainpairs() + self.get_testpairs()
    
    
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
        input_tensor, output_tensor = self.get_testpairs() # test pairs
        ip = InnerProduct(1/(self.K - 1), order)
        er = 0
        boch_num = 0
        boch_den = 0
        num_samples = input_tensor.shape[-1]
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
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        output_tensor = scipy.io.loadmat(self.data_path, variable_names='output_train')['output_train'][np.ix_(slice_subsample, slice_subsample, self.slice_n)].astype(np.float64)
        ip = InnerProduct(1/(self.K - 1), order)
        er = 0
        boch_num = 0
        boch_den = 0
        num_samples = output_tensor.shape[-1]
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
        Shortcut function for PC random feature mapping using global function ``rf''. Cannot be numba-ized.
        '''
        return rf(a_input, gr_field, f_num=self.F_darcy, sig_plus=self.sig_plus, sig_minus=self.sig_minus, denom_scale=self.denom_scale)
    
    
    def regsweep(self, lambda_list=[1e-6, 1e-7, 1e-8, 1e-9, 1e-10]):
        '''
        Regularization hyperparameter sweep. Requires model to be fit first at least once. Updates model parameters to best performing ones.
        Input:
            lambda_list: (list), list of lambda values
        Output:
            er_store : (len(lambda_list), 5) numpy array, error storage
        '''
        if isinstance(self.AstarA, int):
            sys.exit("ERROR: Model must be trained at least once before calling ``regsweep''. ")
            return None
        
        al_list = [] # initialize list of learned coefficients
        al_list.append(self.al_model)
        er_store = np.zeros([len(lambda_list)+1, 5]) # lamreg, e_train, b_train, e_test, b_test
        er_store[0, 0] = self.lamreg
        print('Running \lambda =', er_store[0, 0])
        er_store[0, 1:3] = self.relative_error_train()
        er_store[0, 3:] = self.relative_error_test()
        print('Expected relative error (Train, Test):' , (er_store[0, 1], er_store[0, 3]))
        
        for loop in range(len(lambda_list)):
            reg = lambda_list[loop]
            er_store[loop + 1, 0] = reg 
            print('Running \lambda =', reg)
            
            # Solve linear system
            if reg == 0:
                self.al_model = linalg.pinv2(self.AstarA)@self.AstarY
            else:
                self.al_model = linalg.solve(self.AstarA + reg*np.eye(self.m), self.AstarY, assume_a='sym')
            al_list.append(self.al_model)
            
            # Training error
            er_store[loop + 1, 1:3] = self.relative_error_train()
            
            # Test error
            er_store[loop + 1, 3:] = self.relative_error_test()
            
            # Print
            print('Expected relative error (Train, Test):' , (er_store[loop + 1, 1], er_store[loop + 1, 3]))
            
        # Find lambda with smallest test error and update class regularization attribute
        ind_arr = np.argmin(er_store, axis=0)[3] # smallest test index
        self.lamreg = er_store[ind_arr, 0]
        
        # Update model parameter class attribute corresponding to best lambda
        self.al_model = al_list[ind_arr]
        
        return er_store
            
    
    def gridsweep(self, K_list=[9, 17, 33, 65, 129]):
        '''
        Grid sweep to show mesh invariant relative error. K_list must be 1 plus powers of two.
        '''
        er_store = np.zeros([len(K_list), 5]) # 5 columns: K, e_train, b_train, e_test, b_test
        almodel_store = np.zeros([len(K_list), self.m])
        for loop in range(len(K_list)):
            K = K_list[loop]
            er_store[loop,0] = K
            print('Running grid size s =', K)
            
            # Train
            rfm = RandomFeatureModel(K, self.n, self.m, self.ntest, self.lamreg, self.sig_plus, self.sig_minus, self.denom_scale, self.params_path, self.data_path)
            rfm.fit()
            almodel_store[loop, :] = rfm.al_model
            
            # Train error
            er_store[loop, 1:3] = rfm.relative_error_train()
            print('Training done...')
            del rfm.RF_train
            rfm.RF_train = 0
            
            # Test error
            er_store[loop, 3:] = rfm.relative_error_test()
            
            # Print
            print('Expected relative error (Train, Test):' , (er_store[loop, 1], er_store[loop, 3]))
        
        return er_store, almodel_store
    
            
class InnerProduct:
    '''
    Class implementation of L^2, H^1, and H_0^1 inner products from samples of data on [0,1]^2.
    Reference: https://stackoverflow.com/questions/50440592/is-there-any-good-way-to-optimize-the-speed-of-this-python-code
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
            def L2_simp(F, G):
                '''
                L^2 inner product
                '''
                return InnerProduct.simps_static(InnerProduct.simps_static(F*G, dx=h), dx=h)
            
            def H01_simp(F, G):
                '''
                H_0^1 inner product
                '''
                Fx, Fy = np.gradient(F, h, edge_order=2)
                Gx, Gy = np.gradient(G, h, edge_order=2)
                integrand = Fx*Gx + Fy*Gy
                return InnerProduct.simps_static(InnerProduct.simps_static(integrand, dx=h), dx=h)
            
            self.L2 = L2_simp
            self.H01 = H01_simp
            
            
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
        '''
        return InnerProduct.trapz2D_static(F*G, self.h)

    def H01(self, F, G):
        '''
        H_0^1 inner product
        '''
        Fx, Fy = np.gradient(F, self.h, edge_order=2)
        Gx, Gy = np.gradient(G, self.h, edge_order=2)
        integrand = Fx*Gx + Fy*Gy
        return InnerProduct.trapz2D_static(integrand, self.h)
    
    def H1(self, F, G):
        '''
        H^1 inner product
        '''
        return self.L2(F, G) + self.H01(F, G)


def dstnhn(x):
    '''
    Orthonormalized disrete sine transform type I (does NOT include zeros at boundaries in x), where dst(dst(x))=x
    Input:
        x: (n, m) numpy array
    Output:
        output: (n, m) numpy array
    '''
    return dst(x, 1, axis=0, norm='ortho')


def dst2(x):
    '''
    2D Orthonormalized discrete sine transform type I
    '''
    return dstnhn(dstnhn(x).T).T


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
#    eig = 4/(h**2)*(np.sin(K1/2*np.pi/(N-1))**2 + np.sin(K2/2*np.pi/(N-1))**2) # FD eigs
    eig = 2/(h**2)*(2 - np.cos(K1*np.pi/(N-1)) - np.cos(K2*np.pi/(N-1))) # FD eigs (faster)
#    eig = (np.pi*K1)**2+(np.pi*K2)**2 # Exact Laplacian eigs (spectral method)
    # Fst = dst2(F[1:-1,1:-1]) 
    Ust = dst2(F[1:-1,1:-1])/eig
    U = np.zeros((N,N))     
    U[1:-1,1:-1] = dst2(Ust)
    return U


@nb.njit(fastmath=True)
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
        
        # Update interior of spatial domain
        for i in range(1, K - 1):
            for j in range(1, K - 1):
                U_new[i,j] = U_old[i,j] + dt*( nu/(h**2)*(-4*U_old[i,j] + U_old[i+1,j] + U_old[i-1,j] + U_old[i,j+1] + U_old[i,j-1]) )
        
        # Update homogeneous Neumann boundaries (NOT including corners)
        for ind in range(1, K - 1):
            U_new[ind,-1] = U_old[ind,-1] + dt*( nu/(h**2)*(-4*U_old[ind,-1] + U_old[ind+1,-1] + U_old[ind-1,-1] + 2*U_old[ind,-2]) ) # top
            U_new[ind,0] = U_old[ind,0] + dt*( nu/(h**2)*(-4*U_old[ind,0] + U_old[ind+1,0] + U_old[ind-1,0] + 2*U_old[ind,1]) ) # bottom
            U_new[0,ind] = U_old[0,ind] + dt*( nu/(h**2)*(-4*U_old[0,ind] + U_old[0,ind+1] + U_old[0,ind-1] + 2*U_old[1,ind]) ) # left
            U_new[-1,ind] = U_old[-1,ind] + dt*( nu/(h**2)*(-4*U_old[-1,ind] + U_old[-1,ind+1] + U_old[-1,ind-1] + 2*U_old[-2,ind]) ) # right
        
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
    return (sig_plus - sig_minus)/(1 + np.exp(-(r/denom_scale))) + sig_minus


def rf(a_input, gr_field, f_num, sig_plus=1/12, sig_minus=-1/3, denom_scale=0.15):
    '''
    Generates a predictor--corrector Poisson solution random feature mapping with Dirichlet BCs
        -- Requires the functions ``solve_poisson_fast, sigma''
    Input:
        a_input: (K,K), high contrast coefficient function
        gr_field: (K,K,2) precomputed Gaussian random fields
        f_num: (K,K) RHS source term in original Darcy PDE
        sig_plus, sig_minus: (float), thresholds for sigmoids
        denom_scale: (float), scaling in sigmoid(r/denom_scale)
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
    
# =============================================================================
#     # Option 4
#     F0 = f_num/(a_input + sigma(gr_field1, -1, -2.8, denom_scale) )
#     gradX_U0, gradY_U0 = np.gradient(solve_poisson_fast(F0), 1/(a_input.shape[0] - 1), edge_order=2)
#     gradX_loga, gradY_loga = np.gradient(np.log(mollify(a_input)), 1/(a_input.shape[0] - 1), edge_order=2)
#     F = f_num/(a_input) + sigma(gr_field2, sig_plus, sig_minus, denom_scale) + gradX_loga*gradX_U0 + gradY_loga*gradY_U0
# =============================================================================
       
    # Fast Poisson solve for random feature output
    return solve_poisson_fast(F)


# =============================================================================
# @nb.njit(fastmath=True, parallel=False)
# def rff(a_input, gr_field, nu=10e-3, sig_plus=1/12, sig_minus=-1/3, denom_scale=0.15):
#     '''
#     Generates a flow--based diffusion random feature mapping with Dirichlet BCs
#         -- Requires the function ``sigma''
#     Input:
#         a_input: (K,K), high contrast coefficient function
#         gr_field: (K,K,2) precomputed Gaussian random fields
#         nu: (float), viscosity constant (default: 5e-2)
#         sig_plus, sig_minus: (float), thresholds for sigmoids
#         denom_scale: (float), scaling in sigmoid(r/denom_scale)
#     Output
#         output: (K,K) matrix for varphi(a_input; gr_field) random feature
#     '''
# 
#     # Store forcing
#     force = sigma(gr_field[:,:,1], sig_plus, sig_minus, denom_scale)
#     
#     # Extract mesh size
#     K = a_input.shape[0]
#     h = 1/(K - 1)
#     
#     # Form mesh in time with CFL constraint
#     Nt = 1//(0.95*(h**2)/(4*nu)) + 1
#     dt = 1/(Nt - 1)    
#     
#     # Set initial condition
#     U_temp = 1/a_input + sigma(gr_field[:,:,0], sig_plus, sig_minus, denom_scale) # option 1
#     # U_temp = 1/(a_input + sigma(gr_field[:,:,0], sig_plus, sig_minus, denom_scale)) # option 2
#     U_old = U_temp/U_temp.max()
#     U_new = np.zeros((K, K), dtype = np.float64) # homogeneous Dirichlet boundaries included
#     
#     # Finite difference scheme
#     for tstep in range(Nt - 1):
#         
#         # Update interior of spatial domain
#         for spacex in range(1, K - 1):
#             for spacey in range(1, K - 1):
#                 U_new[spacex,spacey] = U_old[spacex,spacey] + dt*np.tanh( (nu/(h**2))*(-4*U_old[spacex,spacey] + U_old[spacex + 1,spacey] + U_old[spacex - 1,spacey] + U_old[spacex,spacey + 1] + U_old[spacex,spacey - 1]) + force[spacex,spacey])
#                 # U_new[spacex,spacey] = U_old[spacex,spacey] + dt*np.tanh( (nu/(h**2))*(-4*U_old[spacex,spacey] + U_old[spacex + 1,spacey] + U_old[spacex - 1,spacey] + U_old[spacex,spacey + 1] + U_old[spacex,spacey - 1]) + force[spacex,spacey] + 1/a_input[spacex,spacey])
# 
#         # Update solution data structure
#         U_old = np.copy(U_new)
#     
#     # Solution at time t = 1
#     return U_old # == output
# =============================================================================


def act_filter(r, al_rf):
    return np.maximum(0, np.minimum(2*r, np.power(0.5 + r, -al_rf)))

def rff(a_input, gr_field, sig_plus=1/12, sig_minus=-1/3, denom_scale=0.15, nu_rf=5e-3, al_rf=4, K_fine=257):
    '''
    Generates a Fourier-space random feature mapping with Dirichlet BCs
        -- Requires the functions ``dst2, sigma''
    Input:
        a_input: (K,K), high contrast coefficient function
        gr_field: (K,K,2) precomputed Gaussian random fields (ideally with Dirichlet BCs)
        f_num: (K,K) RHS source term in original Darcy PDE
        sig_plus, sig_minus: (float), thresholds for sigmoids
        denom_scale: (float), scaling in sigmoid(r/denom_scale)
    Output
        output: (K,K) matrix for varphi(a_input; gr_field) random feature
    '''
    K = a_input.shape[0]
    K1, K2 = np.meshgrid(np.arange(1, K - 1), np.arange(1, K - 1)) # Fourier sine variable
    
    # Extract GRFs without boundaries
    gr_field1 = gr_field[1:-1,1:-1,0]
    # gr_field2 = gr_field[1:-1,1:-1,1]
    
    # Define mapping
    wave_func = act_filter(np.pi*nu_rf*np.sqrt(K1**2 + K2**2), al_rf) * 1 + 1   # TEST
    U = np.zeros((K,K))     
    # U[1:-1,1:-1] = elu( (K_fine - 2)/(K - 2)*dst2( wave_func*dst2(a_input[1:-1,1:-1])*dst2(gr_field1) ) )
    U[1:-1,1:-1] = elu( dst2( wave_func*dst2(a_input[1:-1,1:-1])*dst2(gr_field1)*sig_plus ) )
    return U
