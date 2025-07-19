### Filtering ###

import numba
from numba import jit, njit, float64
from numba.experimental import jitclass
import numpy as np
import scipy

# Generate root-raise cosine filter coefficients
def rrc_coef(n_taps=101, beta=0.35, Ts=1.0):

    # initialize vectors
    h = np.empty(n_taps, dtype=complex)
    t_vec = np.arange(n_taps)  - (n_taps-1)//2 # -50, -49, ..., 49, 50
   
    for i, t in enumerate(t_vec):  
        # Piecewise definition from https://en.wikipedia.org/wiki/Root-raised-cosine_filter
       
        # t = 0:
        if t == 0:
            h[i] = 1/Ts * (1 + beta*(4/np.pi - 1))
            continue
   
        # t = Ts/(4*beta):
        if abs(t) == Ts/(4*beta):
            h[i] = beta/(Ts*np.sqrt(2)) * ( (1 + 2/np.pi)*np.sin(np.pi/(4*beta)) + \
                                        (1 - 2/np.pi)*np.cos(np.pi/(4*beta)) )
            continue
   
        # otherwise
        h[i] = 1/Ts * (np.sin(np.pi*(t/Ts)*(1-beta)) + 4*beta*(t/Ts)*np.cos(np.pi*(t/Ts)*(1+beta))) / \
                    (np.pi*(t/Ts)*(1 - (4*beta*(t/Ts))**2))
       
    return h
       
def upsample(signal, factor):
    sig_upsampled = np.zeros(signal.size*factor, dtype=signal.dtype)
    sig_upsampled[::factor] = signal
    return sig_upsampled

@njit
def iir_lowpass(x, y_prev, alpha):
    # 0 < alpha < 1
    # lower alpha -> smoother output
    return (1 - alpha) * y_prev + alpha * x

