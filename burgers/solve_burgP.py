"""
A module file with function to solve periodic viscous Burgers' equation in one space dimension. Also solves periodic 1D heat equation.
"""

import numpy as np

# Custom imports to this file
from scipy.fft import fft, ifft


def solve_burgP(IC, nu=5e-4, tmax=1, fudge1=0.6*5, fudge2=0.6*5):
    """
    Add forcing term ability. Evaluate at any t by using interp1d function in time axis of f_input.

    Parameters
    ----------
    IC : TYPE
        DESCRIPTION.
    nu : TYPE, optional
        DESCRIPTION. The default is 5e-3.
    tmax : TYPE, optional
        DESCRIPTION. The default is 2.
    fudge1 : TYPE, optional
        DESCRIPTION. The default is 0.6.
    fudge2 : TYPE, optional
        DESCRIPTION. The default is 0.6.

    Returns
    -------
    None.

    """
    # Derived
    K = IC.shape[0]
    h = 1/(K - 1)
    k = 2*np.pi*np.array([i for i in range((K - 1)//2)] + [(K - 1)//2] + [ii for ii in range(-(K - 1)//2 + 1,0)])
    
    # Form mesh in time with CFL constraint
    Nt = int(tmax//min(fudge1*nu/(np.abs(IC).max())**2, fudge2*(h**2)/(2*nu))) + 1
    dt = tmax/(Nt - 1)    
    
    # Precomputed arrays
    g = -1j*dt*k/2
    g[(K - 1)//2] = 0 # odd derivative, set N//2 mode to zero
    nuk2 = nu*(k**2)
    Emh = np.exp(-nuk2*dt)
    Emh2 = np.exp(-nuk2*dt/2)
    
    # Initialize
    uhat_old = fft(IC[:-1])
    U_hat = np.zeros(((K - 1), Nt), dtype=np.complex_)
    U_hat[:,0] = uhat_old
    
    # RK4
    for tstep in range(Nt - 1):
        a = g*fft(np.real(ifft(uhat_old))**2)
        b = g*fft(np.real(ifft(Emh2*(uhat_old + a/2)))**2)
        c = g*fft(np.real(ifft(Emh2*uhat_old + b/2))**2)
        d = g*fft(np.real(ifft(Emh*uhat_old + Emh2*c))**2)
        uhat_new = Emh*uhat_old + 1/6*(Emh*a + 2*Emh2*(b + c) + d)
        U_hat[:,tstep + 1] = uhat_new
        uhat_old = uhat_new
     
# =============================================================================
#     # SSP-Rk3 
#     Eph2 = np.exp(nuk2*dt/2)
#     for tstep in range(Nt - 1):
#         a = g*fft(np.real(ifft(uhat_old))**2)
#         b = g*fft(np.real(ifft(Emh*(uhat_old + a)))**2)
#         c = g*fft(np.real(ifft(Emh2*(uhat_old + a/4) + Eph2*b/4))**2)
#         uhat_new = Emh*uhat_old + 1/6*(Emh*a + b) + 2/3*Emh2*c
#         U_hat[:, tstep + 1] = uhat_new
#         uhat_old = uhat_new
# =============================================================================
    
    # Recover solution in physical space
    U = np.real(ifft(U_hat, axis=0))
    return np.vstack((U, U[0,:]))

# nnt = 20 # TODO: testing 4th order convergence of scheme
# TODO: Make quicker using a smaller time step (see Trefethen SISC paper) 
def solnmap_burgP(IC, nu=5e-4, tmax=1, fudge1=0.6*5, fudge2=0.6*5):
    """
    Fast evaluation of basic IC to u(1) solution map for Burgers' equation. No forcing for this function.
    """
    # Derived
    K = IC.shape[0]
    k = 2*np.pi*np.array([i for i in range((K - 1)//2)] + [(K - 1)//2] + [ii for ii in range(-(K - 1)//2 + 1,0)])
    
    # Form mesh in time with CFL constraint
    Nt = int(tmax//min(fudge1*nu/(np.abs(IC).max())**2, fudge2*((1/(K - 1))**2)/(2*nu))) + 1
    # Nt = nnt + 1
    dt = tmax/(Nt - 1)    
    
    # Precomputed arrays
    g = -1j*dt*k/2
    g[(K - 1)//2] = 0 # odd derivative, set N//2 mode to zero
    nuk2 = nu*(k**2)
    Emh = np.exp(-nuk2*dt)
    Emh2 = np.exp(-nuk2*dt/2)
        
    # RK4 IF
    uhat = fft(IC[:-1])
    for tstep in range(Nt - 1):
        a = g*fft(np.real(ifft(uhat))**2)
        b = g*fft(np.real(ifft(Emh2*(uhat + a/2)))**2)
        c = g*fft(np.real(ifft(Emh2*uhat + b/2))**2)
        uhat = Emh*uhat + 1/6*(Emh*a + 2*Emh2*(b + c) + g*fft(np.real(ifft(Emh*uhat + Emh2*c))**2))
    
    # Recover solution in physical space
    uT1 = np.real(ifft(uhat))
    return np.append(uT1, uT1[0])


def solnmap_heatP(IC, nu=1e-2, tmax=1, c=1.75):
    """
    Fast evaluation of basic IC to u(1) solution map for linear constant speed advection-diffusion equation on [0,1). No forcing for this function.
    """
    K = IC.shape[0]
    k = 2*np.pi*np.array([i for i in range((K - 1)//2)] + [(K - 1)//2] + [ii for ii in range(-(K - 1)//2 + 1,0)])
    k_odd = np.copy(k)
    k_odd[(K - 1)//2] = 0
    uT = np.real(ifft(np.exp(-(nu*k**2 + 1j*k_odd*c)*tmax)*fft(IC[:-1])))
    return np.append(uT, uT[0])

