### Filtering ###

from numba import jit, njit
import numpy as np
import scipy
from collections import deque

# TODO: 
# - maybe add rrc specific interpolation for frame detection
# - Add Farrow structures

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
        

@njit
def iir_lowpass(x, y_prev, alpha):
    # 0 < alpha < 1
    # lower alpha -> smoother output
    return (1 - alpha) * y_prev + alpha * x


def decimate(signal, factor):
    return signal[::factor]


class CubicFarrowStructure:

    def __init__(self):
        self.order = 3
        self.buffer = deque([0.0] * (self.order + 1), maxlen=self.order + 1)
        self.coeffs = np.array([
            np.array([0, 0, 1, 0]),
            np.array([-1/6, 1, -1/2, -1/3]),
            np.array([0, 1/2, -1, 1/2]),
            np.array([1/6, -1/2, 1/2, -1/6]),
        ])

        pass

    def update(self, x):
        self.buffer.appendleft(x)

    def interpolate(self, mu):
        c_k = []
        sample_segment = np.array(list(self.buffer))[:len(self.coeffs[0])]
        for k in range(self.order + 1):
            c_k.append(np.dot(self.coeffs[k], sample_segment))
        output = sum(c * mu**k for k, c in enumerate(c_k))
        return output

    def process_sample(self, sample, mu):
        self.update(sample)
        return self.interpolate(mu)
    
    def process_batch(self, samples, mu):
        out = np.array([self.process_sample(samp, mu) for samp in samples])
        return out