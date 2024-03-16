import numpy as np
from scipy import linalg

# Custom imports to this file
from scipy.fft import fft, ifft
from utilities.activations import sawtooth, relu, elu, selu, softplus, lrelu, sigmoid
import numba as nb
import sys

# %% Functions and classes


@nb.njit
def temp_trapz1D_static(U, dx):
    '''
    * only to be used inside numba jitted functions inside RandomFeatureModel class
    '''
    return np.trapz(U, dx=dx)

    
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
    Class implementation of the random feature model for Burgers' equation solution map. Requires ``params.npz, data.npz'' files in ``datasets'' directory.
    '''

    def __init__(self, K=257, n=256, m=512, ntest=1024, lamreg=0, nu_rf=2.5e-3, al_rf=4., dir_path='datasets/4/', new_seed=None):
        '''
        Arguments:
            K:              (int), number of mesh points in one spatial direction (one plus a power of two)
            
            n:              (int), number of data (max: 1024)
            
            m:              (int), number of random features (max: 1024)
            
            ntest:          (int), number of test points (max: 5000)
            
            lamreg:         (float), regularization/penalty hyperparameter strength
            
            nu_rf:          (float), RF-Flow viscosity constant
            
            al_rf:          (float), fractional power of Laplacian term in RF map
                        
            dir_path:      (str), path to desired data directory
            
            new_seed:       (int), new rng seed for randomizing the selection of train/test pairs
            

        Attributes:
            [arguments]:    (various), see ``Arguments'' above for description

            K_fine:         (int), number of high resolution mesh points (one plus a power of two)
            
            nu:             (float), true viscosity for Burgers' PDE
            
            grf_g:          (K, m, 2), precomputed Gaussian random fields 
                        
            n_max:          (int), max number of data (max: 1024)
            
            m_max:          (int), max number of random features (max: 1024)
            
            ntest_max:      (int), max number of test points (max: 5000)
            
            rngseed:        (int), fixed seed for random number generator used in MATLAB
            
            tau_a:          (float), inverse length scale (20, 5, 3)
            
            al_a:           (float), regularity (3, 3.5, 2)
            
            tau_g:          (float), inverse length scale (default: 15, 7.5)
            
            al_g:           (float), regularity (default: 2, 2)
            
            al_model:       (m,) numpy array, random feature model expansion coefficents/parameters to be learned
            
            RF_train:       (K, n, m) numpy array, random feature map evaluated on all training points for each i=1,...,m
        
            AstarA:         (m, m) numpy array, normal equation matrix
            
            AstarY:         (m,), RHS in normal equations
            
            X:              (K), coordinate grids in physical space
        '''
        
        # From input arguments 
        self.K = K
        self.n = n
        self.m = m
        self.ntest = ntest
        self.lamreg = lamreg # hyperparameter
        self.nu_rf = nu_rf # hyperparameter
        self.al_rf = al_rf # hyperparameter
        self.dir_path = dir_path
        
        # Read from the parameter file
        P = np.load(dir_path + 'params.npy', allow_pickle=True) # load dictionary
        self.K_fine = P.item().get('K')
        self.nu_burg = P.item().get('nu')
        self.n_max = P.item().get('n')
        self.m_max = P.item().get('m')
        self.ntest_max = P.item().get('ntest')
        self.rndseed = P.item().get('seed')
        self.tau_a = P.item().get('tau_a')
        self.al_a = P.item().get('al_a')
        self.tau_g = P.item().get('tau_g')
        self.al_g = P.item().get('al_g')
        self.tmax = P.item().get('T')
        
        # Set random seed
        np.random.seed(self.rndseed)
        if new_seed is not None:
            np.random.seed(new_seed)
        slice_grf = np.random.choice(self.m_max, self.m, replace=False)
        self.slice_n = np.random.choice(self.n_max, self.n, replace=False)
        self.slice_ntest = np.random.choice(self.ntest_max, self.ntest, replace=False)
        
        # Read from data file and downsample
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        self.grf_g = np.load(dir_path + 'grftheta.npy')[np.ix_(slice_subsample, slice_grf)].astype(np.float64)
       
        # Non-input argument attributes of RFM
        self.al_model = np.zeros(self.m)
        self.RF_train = 0 # init to zero now, update to 4-tensor in ``fit'' method
        self.AstarA = 0 # init to zero now
        self.AstarY = 0 # init to zero now

        # Make physical grid
        self.X = np.arange(0, 1 + 1/(K-1), 1/(K-1))
    
    
    @staticmethod
    @nb.njit(fastmath=True)
    def formAstarA_trapz_static(ind, n, AstarA, RF_train, h):
        '''
        Output:
            Returns temporary AstarA matrix with partial entries filled in (trapezoid rule).
        '''
        for i in range(ind + 1):
            for k in range(n):
                AstarA[i,ind] += temp_trapz1D_static(RF_train[:,k,i]*RF_train[:,k,ind], h)
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
                AstarA[i,ind] += temp_simps_static(RF_train[:,k,i]*RF_train[:,k,ind], dx=h)
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
            input_train: (K, n), Burgers IC
            output_train: (K, n), Burgers' solution at time 1
        '''
        # Get data
        input_train, output_train = self.get_trainpairs()
        
        # Allocate data structures
        RF_train = np.zeros((self.K, self.n, self.m))
        AstarY = np.zeros(self.m)
        AstarA = np.zeros((self.m, self.m))
        
        # Set inner product order
        ip = InnerProduct1D(1/(self.K - 1), order)
                
        # Form b=(A*)Y and store random features
        for i in range(self.m):
            for k in range(self.n):
                rf_temp = self.rf_local(input_train[:,k], self.grf_g[:,i,:])
                RF_train[:,k,i] = np.copy(rf_temp)
                AstarY[i] += ip.L2(output_train[:,k], rf_temp)
        
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
            # self.al_model = linalg.pinv(AstarA)@AstarY
            self.al_model = linalg.pinv2(AstarA)@AstarY
        else:
            self.al_model = linalg.solve(AstarA + self.lamreg*np.eye(self.m), AstarY, assume_a='sym')
            
            
    def predict(self, a):
        '''
        Evaluate random feature model on a given coefficent function ``a''.
        '''
        output = np.zeros(self.K)
        for j in range(self.m):
            output = output + self.al_model[j]*self.rf_local(a, self.grf_g[:,j,:])
        return output/self.m
    
    
    def get_trainpairs(self):
        '''
        Extract train input/output pairs as a 2-tuple.
        '''
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        train_temp = np.load(self.dir_path + 'train.npy')
        input_train = train_temp[:,:,0][np.ix_(slice_subsample, self.slice_n)].astype(np.float64)
        output_train = train_temp[:,:,1][np.ix_(slice_subsample, self.slice_n)].astype(np.float64)
        del train_temp
        return (input_train, output_train)
    
    def get_testpairs(self):
        '''
        Extract test input/output pairs as a 2-tuple.
        '''
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        test_temp = np.load(self.dir_path + 'test.npy')
        input_test = test_temp[:,:,0][np.ix_(slice_subsample, self.slice_ntest)].astype(np.float64)
        output_test = test_temp[:,:,1][np.ix_(slice_subsample, self.slice_ntest)].astype(np.float64)
        del test_temp
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
        ip = InnerProduct1D(1/(self.K - 1), order)
        er = 0
        num_samples = input_tensor.shape[-1]
        for i in range(num_samples):
            y_i = output_tensor[:,i]
            resid = np.abs(y_i - self.predict(input_tensor[:,i]))
            er += np.sqrt(ip.L2(resid, resid)/ip.L2(y_i, y_i))
        er /= num_samples
        return er
    
    
    def relative_error_test(self, order=1):
        '''
        Compute the expected relative error and Bochner error on the test set.
        '''
        input_tensor, output_tensor = self.get_testpairs() # test pairs
        ip = InnerProduct1D(1/(self.K - 1), order)
        er = 0
        boch_num = 0
        boch_den = 0
        num_samples = input_tensor.shape[-1]
        for i in range(num_samples):
            y_i = output_tensor[:,i]
            resid = np.abs(y_i - self.predict(input_tensor[:,i]))
            num = ip.L2(resid, resid)
            den = ip.L2(y_i, y_i)
            er += np.sqrt(num/den)
            boch_num += num
            boch_den += den
        er /= num_samples
        er_bochner = np.sqrt(boch_num/boch_den)
        return er, er_bochner
    
    def compose(self, n_comp, input_func):
        y = self.predict(input_func)
        for comp in range(n_comp):
            y = self.predict(y)
        return y
        
    def relative_error_timeupscale_test(self, n_comp, order=1):
        '''
        Compute the expected relative error from timeupscaling on the test set.
        '''
        input_tensor, output_tensor = self.get_testpairs() # test pairs
        ip = InnerProduct1D(1/(self.K - 1), order)
        er = 0
        num_samples = input_tensor.shape[-1]
        for i in range(num_samples):
            y_i = output_tensor[:,i]
            resid = np.abs(y_i - self.compose(n_comp, input_tensor[:,i]))
            num = ip.L2(resid, resid)
            den = ip.L2(y_i, y_i)
            er += np.sqrt(num/den)
        er /= num_samples
        return er
    
    
    @staticmethod
    @nb.njit(fastmath=True)
    def predict_train(ind_a, K, m, al_model, RF_train):
        '''
        Evaluate random feature model on a given coefficent training sample ``a_{ind_a}'' (precomputed).
        '''
        output = np.zeros(K)
        for j in range(m):
            output = output + al_model[j]*RF_train[:,ind_a,j]
        return output/m
    
    def relative_error_train(self, order=1):
        '''
        Compute the expected relative error and Bochner error on the training set (using pre-computed random features).
        '''
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        output_tensor = np.load(self.dir_path + 'train.npy')[:,:,1][np.ix_(slice_subsample, self.slice_n)].astype(np.float64)
        ip = InnerProduct1D(1/(self.K - 1), order)
        er = 0
        boch_num = 0
        boch_den = 0
        num_samples = output_tensor.shape[-1]
        for i in range(num_samples):
            y_i = output_tensor[:,i]
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
        return rf_fourier(a=a_input, w=gr_field, nu_rf=self.nu_rf, al_rf=self.al_rf, K_fine=self.K_fine)
    

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
            

    def gridsweep(self, K_list=[17, 33, 65, 129, 257, 513, 1025], quad_order=1):
        '''
        Grid sweep to show mesh invariant relative error. K_list must be 1 plus powers of two.
        '''
        er_store = np.zeros([len(K_list), 5]) # 5 columns: K, e_train, b_train, e_test, b_test
        almodel_store = np.zeros([len(K_list), self.m])
        for loop in range(len(K_list)):
            K = K_list[loop]
            er_store[loop,0] = K
            print('Running grid size K =', K)
            
            # Train
            rfm = RandomFeatureModel(K, self.n, self.m, self.ntest, self.lamreg, self.nu_rf, self.al_rf, self.dir_path)
            rfm.fit(quad_order)
            almodel_store[loop, :] = rfm.al_model
            
            # Train error
            er_store[loop, 1:3] = rfm.relative_error_train(quad_order)
            print('Training done...')
            
            # Test error
            er_store[loop, 3:] = rfm.relative_error_test(quad_order)
            
            # Print
            print('Expected relative error (Train, Test):' , (er_store[loop, 1], er_store[loop, 3]))
        
        return er_store, almodel_store
    
            
class InnerProduct1D:
    '''
    Class implementation of L^2, H^1, and H_0^1 inner products from samples of data on [0,1].
    Reference: https://stackoverflow.com/questions/50440592/is-there-any-good-way-to-optimize-the-speed-of-this-python-code
    '''

    def __init__(self, h, ORDER=1):
        '''
        Initializes the class. 
        Arguments:
            h:          Mesh size in x direction.
            
            ORDER:      The order of accuracy of the numerical quadrature: [1] for trapezoid rule, [2] for Simpson's rule.

        Parameters:
            h:          (float), Mesh size in x direction.

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
                return InnerProduct1D.simps_static(F*G, dx=h)
            
            def H01_simp(F, G):
                '''
                H_0^1 inner product
                '''
                Fx = np.gradient(F, h, edge_order=2)
                Gx = np.gradient(G, h, edge_order=2)
                return InnerProduct1D.simps_static(Fx*Gx, dx=h)
            
            self.L2 = L2_simp
            self.H01 = H01_simp
            
            
    @staticmethod
    @nb.njit
    def trapz1D_static(U, h):
        '''
        1D trapezoid rule 
        '''
        return np.trapz(U, dx=h)
        
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
        return InnerProduct1D.trapz1D_static(F*G, self.h)

    def H01(self, F, G):
        '''
        H_0^1 inner product
        '''
        Fx = np.gradient(F, self.h, edge_order=2)
        Gx = np.gradient(G, self.h, edge_order=2)
        return InnerProduct1D.trapz1D_static(Fx*Gx, self.h)
    
    def H1(self, F, G):
        '''
        H^1 inner product
        '''
        return self.L2(F, G) + self.H01(F, G)


@nb.njit
def sigma(r, sig_plus, sig_minus, denom_scale):
    '''
    Thresholded sigmoidal function
    '''
    return (sig_plus - sig_minus)/(1 + np.exp(-(r/denom_scale))) + sig_minus


def act_filter(r, al_rf):
    return np.maximum(0, np.minimum(2*r, np.power(0.5 + r, -al_rf)))
    
def rf_fourier(a, w, nu_rf=2.5e-3, al_rf=4, K_fine=1025):
    ''' 
    K_fine = 1 + 1024 # fine grid the input data is sampled from.
    Activations: id, sin, relu, sawtooth, elu, selu
    '''    
    # Derived
    w = w[:-1,:]
    w1 = w[:,0]
    # w2 = w[:,1] # second GRF not needed
    N = a.shape[0] - 1
    k = 2*np.pi*np.array([i for i in range(N//2)] + [N//2] + [ii for ii in range(-N//2 + 1,0)])
    
    # # TEST
    # inds = np.array([i for i in range(N//2)] + [N//2] + [ii for ii in range(-N//2 + 1,0)])
    # k = 2*np.pi*inds
    # k[np.abs(inds) > N//3 ] = 0
 
    # Define mapping
    # wave_func = (nu_rf*(np.abs(k)**al_rf)) * -1j*np.sign(k)   # add sawtooth activation filter maybe
    # wave_func = sawtooth(nu_rf*np.abs(k)**al_rf)   # TEST
    wave_func = act_filter(nu_rf*np.abs(k), al_rf)   # TEST
    # wave_func = sawtooth(nu_rf*(15**2+np.abs(k)**2)**(al_rf/2)) # TEST
    # aa = 1.2
    # bb = 5e-3
    # cc = 1
    # wave_func = cc*sawtooth(bb*np.abs(k))**aa  # TEST

    U = elu( (K_fine - 1)/N*np.real(ifft( wave_func*fft(w1)*fft(a[:-1]) )) )
    return np.append(U, U[0])

