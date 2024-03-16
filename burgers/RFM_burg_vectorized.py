import numpy as np
from scipy import linalg

# Custom imports to this file
from scipy.fft import fft, ifft
from utilities.activations import elu
from timeit import default_timer
import sys

# %% Functions and classes

class RandomFeatureModel:
    '''
    Class implementation of the random feature model for Burgers' equation solution map. Requires ``params.npz, data.npz'' files in ``datasets'' directory.
    '''

    def __init__(self, K=257, n=256, m=512, ntest=1024, lamreg=0, nu_rf=2.5e-3, al_rf=4., dir_path='datasets/nhn_data_burg/', new_seed=None, bsize_train=None, bsize_test=None):
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
            
            bsize_train:    (int), batch size for training
            
            bsize_test:     (int), batch size for testing
            

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
        self.new_seed = new_seed
        self.bsize_train = bsize_train
        self.bsize_test = bsize_test
        if bsize_test is None:
            self.bsize_test = self.ntest
        
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
        self.RF_train = 0 # init to zero now, update to 3-tensor in ``fit'' method
        self.AstarA = 0 # init to zero now
        self.AstarY = 0 # init to zero now

        # Make physical grid
        self.h = 1/(self.K - 1)
        self.X = np.arange(0, 1 + self.h, self.h)
    
    
    def rf_local_batch(self, a_input, gr_field):
        '''
        Batched RF map, shortcut function
        '''
        return rf_fourier_batch(a_batch=a_input, w_batch=gr_field, nu_rf=self.nu_rf, al_rf=self.al_rf, K_fine=self.K_fine)

    # TODO: minibatch in j=1,...,m RF loop
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
        
        if self.bsize_train is None:
            # Allocate data structure
            AstarA = np.zeros((self.m, self.m))
            
            # Form feature tensor (high memory allotment, be careful; can mini-batch this step in data space)
            RF_train = self.rf_local_batch(input_train, self.grf_g)
            
            # Form b=A*Y (trapezoid rule)
            AstarY = np.sum(np.trapz(RF_train * output_train[...,None], dx=self.h, axis=0), axis=0)
            
            # Form A*A symmetric positive semi-definite matrix
            #(DONE, passed simple check) TODO: check if this is correct multiplication
            for j in range(self.m):
                AstarA[j,:] = np.sum(np.trapz(RF_train*RF_train[...,j][...,None], dx=self.h, axis=0), axis=0)
            AstarA /= self.m
    
            # Update class attributes
            self.RF_train = RF_train 
            self.AstarY = AstarY
            self.AstarA = AstarA
        else:
            self.AstarY = np.zeros(self.m)
            self.AstarA = np.zeros((self.m, self.m))
            c = 0
            t0 = default_timer()
            for btch in range(self.n//self.bsize_train):
                # Input and Outputs for this batch
                y = output_train[...,c:(c+self.bsize_train)]
                a = input_train[...,c:(c+self.bsize_train)]
            
                # Form RF-based tensors
                RF = self.rf_local_batch(a, self.grf_g)
                self.AstarY += np.sum(np.trapz(RF * y[...,None], dx=self.h, axis=0), axis=0)
                # for j in range(self.m):
                #     self.AstarA[j,:] += np.sum(np.trapz(RF*RF[...,j][...,None], dx=self.h, axis=0), axis=0)
                
                cc = 0
                mb = 32
                for j in range(self.m//mb):
                    self.AstarA[cc:(cc+mb),:] += np.sum(np.trapz(RF[...,cc:(cc+mb)][...,None]*RF[:,:,None,:], dx=self.h, axis=0), axis=0)
                    cc += mb
       
                # Update
                c += self.bsize_train
                t1 = default_timer()
                print("(Training) Batch, Samples, Time Elapsed:", (btch+1, c, t1-t0))
            self.AstarA /= self.m
        
        # Solve linear system
        if self.lamreg == 0:
            self.al_model = linalg.pinv2(self.AstarA)@self.AstarY
            self.AstarAnug = self.AstarA
        else:
            self.AstarAnug = self.AstarA + self.lamreg*np.eye(self.m)
            self.al_model = linalg.solve(self.AstarAnug, self.AstarY, assume_a='sym')
    
    def get_trainpairs(self):
        '''
        Extract train input/output pairs as a 2-tuple.
        '''
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        train_temp = np.load(self.dir_path + 'train.npy')
        input_train = train_temp[...,0][np.ix_(slice_subsample, self.slice_n)].astype(np.float64)
        output_train = train_temp[...,1][np.ix_(slice_subsample, self.slice_n)].astype(np.float64)
        del train_temp
        return (input_train, output_train)
    
    def get_testpairs(self):
        '''
        Extract test input/output pairs as a 2-tuple.
        '''
        width_subsample = round((self.K_fine - 1)/(self.K - 1))
        slice_subsample = np.arange(0, self.K_fine, width_subsample)
        test_temp = np.load(self.dir_path + 'test.npy')
        input_test = test_temp[...,0][np.ix_(slice_subsample, self.slice_ntest)].astype(np.float64)
        output_test = test_temp[...,1][np.ix_(slice_subsample, self.slice_ntest)].astype(np.float64)
        del test_temp
        return (input_test, output_test)
    
    def get_inputoutputpairs(self):
        '''
        Extract train and test input/output pairs as a 4-tuple.
        '''
        return self.get_trainpairs() + self.get_testpairs()
    
    def predict(self, a):
        '''
        Evaluate random feature model on a given batch of coefficent functions ``a''.
        Inputs:
            a: (K,nbatch) array or (K,) array
        Output:
            Returns (K,nbatch) array or (K,)
        '''
        if a.ndim==1:
            a = a[:,None] # size (K, 1)
            features = self.rf_local_batch(a, self.grf_g) # size (K,nbatch,m)
            output = np.sum(self.al_model*features, axis=-1)/self.m
            return np.squeeze(output, axis=-1)
        else:
            features = self.rf_local_batch(a, self.grf_g) # size (K,nbatch,m)
            return np.sum(self.al_model*features, axis=-1)/self.m
    
    def relative_error(self, input_tensor, output_tensor, order=1):
        '''
        Compute the expected relative error on an arbitrary set of ``num_samples'' input/output pairs.
        ''num_samples'' should be a small enough batch size so as to avoid memory overflow errors.
        Inputs:
            input_tensor: (K,num_samples) array
            output_tensor: (K,num_samples) array
        Output:
            er: () array (numpy float)
        '''
        ip = InnerProduct1D(1/(self.K - 1), order)
        resid = np.abs(output_tensor - self.predict(input_tensor)) # (K, num_samples)
        er = np.sum(np.sqrt(ip.L2(resid, resid)/ip.L2(output_tensor, output_tensor)))/input_tensor.shape[-1]
        return er
    
    def relative_error_test(self, order=1):
        '''
        Compute the expected relative error and Bochner error on the test set.
        '''
        input_tensor, output_tensor = self.get_testpairs() # test pairs
        ip = InnerProduct1D(1/(self.K - 1), order)
        Nt = output_tensor.shape[-1]
        c = 0
        er = 0
        boch_num = 0
        boch_den = 0
        t0 = default_timer()
        for btch in range(Nt//self.bsize_test):
            # Input and Outputs for this batch
            y = output_tensor[...,c:(c+self.bsize_test)]
            a = input_tensor[...,c:(c+self.bsize_test)]
        
            # Unscaled error for this batch
            resid = np.abs(y - self.predict(a)) # (K, self.bsize_test)
            boch_num_vec = ip.L2(resid, resid)
            boch_den_vec = ip.L2(y, y)
            er += np.sum(np.sqrt(boch_num_vec/boch_den_vec))
            boch_num += np.sum(boch_num_vec)
            boch_den += np.sum(boch_den_vec)
        
            # Update
            c += self.bsize_test
            t1 = default_timer()
            print("(Testing Set) Batch, Samples, Time Elapsed:", (btch+1, c, t1-t0))
        return er/Nt, np.sqrt(boch_num/boch_den) # or maybe er/c if c does not equal Nt
    
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
        Nt = output_tensor.shape[-1]
        c = 0
        er = 0
        t0 = default_timer()
        for btch in range(Nt//self.bsize_test):
            # Input and Outputs for this batch
            y = output_tensor[...,c:(c+self.bsize_test)]
            a = input_tensor[...,c:(c+self.bsize_test)]
        
            # Unscaled error for this batch
            resid = np.abs(y - self.compose(n_comp, a)) # (K, self.bsize_test)
            boch_num_vec = ip.L2(resid, resid)
            boch_den_vec = ip.L2(y, y)
            er += np.sum(np.sqrt(boch_num_vec/boch_den_vec))
        
            # Update
            c += self.bsize_test
            t1 = default_timer()
            print("(Testing Set) Batch, Samples, Time Elapsed:", (btch+1, c, t1-t0))
        return er/Nt    # or maybe er/c if c does not equal Nt
    
    def predict_train(self, ind_a):
        '''
        Evaluate random feature model on a given coefficent training sample ``a_{ind_a}'' (precomputed)
        --- Only used if bsize_train is None ---
        Input:
            ind_a is a tuple or numpy array of slice locations
        '''
        return np.sum(self.al_model*self.RF_train[:,ind_a,:], axis=-1)/self.m
    
    def relative_error_train(self, order=1):
        '''
        Compute the expected relative error and Bochner error on the training set (using pre-computed random features).
        '''
        if self.bsize_train is None:
            width_subsample = round((self.K_fine - 1)/(self.K - 1))
            slice_subsample = np.arange(0, self.K_fine, width_subsample)
            output_tensor = np.load(self.dir_path + 'train.npy')[...,1][np.ix_(slice_subsample, self.slice_n)].astype(np.float64)
            ip = InnerProduct1D(1/(self.K - 1), order)
            num_samples = output_tensor.shape[-1]
            resid = np.abs(output_tensor - self.predict_train(np.array(range(num_samples)))) # (K, num_samples)
            boch_num = ip.L2(resid, resid)
            boch_den = ip.L2(output_tensor, output_tensor)
            er = np.sum(np.sqrt(boch_num/boch_den))/num_samples
            er_bochner = np.sqrt(np.sum(boch_num)/np.sum(boch_den))
            return er, er_bochner
        else:
            input_tensor, output_tensor = self.get_trainpairs() # train pairs
            ip = InnerProduct1D(1/(self.K - 1), order)
            Nt = output_tensor.shape[-1]
            c = 0
            er = 0
            boch_num = 0
            boch_den = 0
            t0 = default_timer()
            for btch in range(Nt//self.bsize_train):
                # Input and Outputs for this batch
                y = output_tensor[...,c:(c+self.bsize_train)]
                a = input_tensor[...,c:(c+self.bsize_train)]
            
                # Unscaled error for this batch
                resid = np.abs(y - self.predict(a)) # (K, self.bsize_test)
                boch_num_vec = ip.L2(resid, resid)
                boch_den_vec = ip.L2(y, y)
                er += np.sum(np.sqrt(boch_num_vec/boch_den_vec))
                boch_num += np.sum(boch_num_vec)
                boch_den += np.sum(boch_den_vec)
            
                # Update
                c += self.bsize_train
                t1 = default_timer()
                print("(Training Set) Batch, Samples, Time Elapsed:", (btch+1, c, t1-t0))
            return er/Nt, np.sqrt(boch_num/boch_den) # or maybe er/c if c does not equal Nt          

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
        self.quad_type = 'trapezoid'
            
    def L2(self, F, G):
        '''
        L^2 inner product
        Inputs: F, G are numpy arrays of size (K, d1, d2, ...), where F*G multiplciation must be broadcastable
        '''
        return np.trapz(F*G, dx=self.h, axis=0)


def sigma(r, sig_plus, sig_minus, denom_scale):
    '''
    Thresholded sigmoidal function
    '''
    return (sig_plus - sig_minus)/(1 + np.exp(-(r/denom_scale))) + sig_minus

def act_filter(r, al_rf):
    return np.maximum(0, np.minimum(2*r, np.power(0.5 + r, -al_rf)))

def rf_fourier_batch(a_batch, w_batch, nu_rf=2.5e-3, al_rf=4, K_fine=1025):
    ''' 
    Inputs:
        a_batch: (K,n) array
        w_batch: (K,m,2) array, total of 2*m GRFs
        K_fine = 1 + 1024 # fine grid the input data is sampled from.
    Output:
        Returns (K,n,m) array
    '''    
    # Derived
    N = a_batch.shape[0] - 1
    k = 2*np.pi*np.array([i for i in range(N//2)] + [N//2] + [ii for ii in range(-N//2 + 1,0)])
 
    # Define filter mapping
    wave_func = act_filter(nu_rf*np.abs(k), al_rf)
    a_fft = fft(a_batch[:-1,:], axis=0)[...,None]
    grf_fft = fft(w_batch[:-1,:,0], axis=0)[:,None,:]

    # Compute features
    U = elu( (K_fine - 1)/N*np.real(ifft(wave_func[:,None,None]*a_fft*grf_fft, axis=0)) )
    U = np.append(U, U[0,...][None,...], axis=0)    # hard code periodicity into output array
    return U
