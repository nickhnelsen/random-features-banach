import numpy as np

# Custom imports to this file
from scipy.fft import idct, dst
from scipy.interpolate import RectBivariateSpline



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


class GaussianRandomField:
    '''
    Return a sample of a Gaussian random field on [0,1]^2 with: 
        -- mean function m = 0
        -- covariance operator C = (-Delta + tau^2)^(-alpha),
    where Delta is the Laplacian with zero Dirichlet or Neumann boundary conditions.
    Requires the functions: ``idct, idct2, dst, dstnhn, dst2, RectBivariateSpline''
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
            K1, K2 = np.meshgrid(np.arange(1,N-1), np.arange(1,N-1)) # k= (K1, K2) (does not include first or last wavenumbers)
            coef = (self.tau**(self.alpha - 1))*(np.pi**2*(K1**2 + K2**2) + self.tau**2)**(-self.alpha/2) # (alpha-d/2) scaling
            
            # Construct the KL (discrete sine transform) coefficients
            B = (N-1)*coef*theta[1:-1, 1:-1] # multiply by (N-1) to satisfy the DST Type I definition
            
            # Inverse (fast FFT--based) 2D discrete sine transform
            U_noBC = dst2(B) # sums B*2/(N-1)*sin(k1*pi*x)*sin(k2*pi*y) over all k1,k2=1,...,N-2
            
            # Impose zero boundary data on the output
            grf = np.zeros((N,N))     
            grf[1:-1,1:-1] = np.copy(U_noBC)
        
        return grf
    
    
    @staticmethod
    def make_coeff(grf, a_plus=12, a_minus=3):
        '''
        Given a (Gaussian random) field in 2D, return the high contrast coefficent obtained by thresholding the field about the zero level set.
        Input:
            grf: (K, K) numpy array
            a_plus, a_minus: (float), upper and lower thresholds, respectively
        Output:
            _ : (K, K) high contrast interface field
        '''
        return a_plus*(grf > 0).astype(np.float64) + a_minus*(grf <= 0).astype(np.float64)
