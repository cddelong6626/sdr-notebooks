### Filtering ###

import numba
from numba import jit, njit, float64
from numba.experimental import jitclass
import numpy as np
import scipy
from collections import deque

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


class CubicFarrowStructure:
    def __init__(self):        
        self.ORDER = 3
        self.COEFFS = np.array([
            np.array([0, 0, 1, 0]),
            np.array([-1/6, 1, -1/2, -1/3]),
            np.array([0, 1/2, -1, 1/2]),
            np.array([1/6, -1/2, 1/2, -1/6]),
        ])

        self.buffer = deque([0.0] * (self.ORDER + 1), maxlen=self.ORDER + 1)

    def update(self, x):
        if isinstance(x, np.ndarray):
            for value in x[::-1]:  # Reverse to keep newest elements at the front
                self.buffer.appendleft(float(value))
        else:
            self.buffer.appendleft(float(x))

    def interpolate(self, mu):
        c_k = []
        sample_segment = np.array(list(self.buffer))[:len(self.COEFFS[0])]
        for k in range(self.ORDER + 1):
            c_k.append(np.dot(self.COEFFS[k], sample_segment))
        output = sum(c * mu**k for k, c in enumerate(c_k))
        return output

    def process_sample(self, sample, mu):
        self.update(sample)
        return self.interpolate(mu)
   
    def process_batch(self, samples, mu):
        out = np.array([self.process_sample(samp, mu) for samp in samples])
        return out
    
    def process_batch_with_pad(self, samples, mu):
        signal_with_padding = np.concatenate((samples, np.full(2, samples[-1], dtype=samples.dtype)))
        return self.process_batch(signal_with_padding, mu)[2:]



spec = [
    ('K_p', float64),
    ('K_i', float64),
    ('K_d', float64),
    ('sum_e', float64),
    ('prev_e', float64),
]
@jitclass(spec)
class PIDFeedback:
    def __init__(self, K_p=0.0, K_i=0.0, K_d=0.0):
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

        self.sum_e = 0.0
        self.prev_e = 0.0

    def update(self, e):
        self.sum_e += e
        d = self.prev_e - e
        x = (self.K_i * self.sum_e) + (self.K_p * e) + (self.K_d * d)

        self.prev_e = e

        return x