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


# Simple linear interpolator class, similar buffer style to Farrow
class LinearInterpolator:
    def __init__(self):
        self.buffer = deque([0j, 0j], maxlen=2)  # Need 2 samples for linear interp
    
    def update(self, sample):
        self.buffer.append(sample)
    
    def interpolate(self, mu):
        """
        Linear interp between buffer[-2] and buffer[-1]:
        y(mu) = (1-mu)*x[n-1] + mu*x[n]
        mu in [0,1)
        """
        x0, x1 = self.buffer[0], self.buffer[1]
        return (1 - mu)*x0 + mu*x1

class CubicFarrowInterpolator:
    """
    Cubic Lagrange interpolator using Farrow structure.
    Supports real and complex input (e.g., np.complex128).
    """
    def __init__(self):
        self.ORDER = 3
        self.NUM_TAPS = self.ORDER + 1

        # Flipped coefficients for Lagrange basis (to match [oldest -> newest] buffer)
        raw_coeffs = np.array([
            [0, 0, 1, 0],
            [-1/6, 1, -1/2, -1/3],
            [0, 1/2, -1, 1/2],
            [1/6, -1/2, 1/2, -1/6],
        ], dtype=np.float64)

        # Flip to align with buffer: buffer[0] = oldest, buffer[-1] = newest
        self.COEFFS = np.fliplr(raw_coeffs).astype(np.complex128)

        # Complex-valued buffer: oldest first, newest last
        self.buffer = deque([0.0j] * self.NUM_TAPS, maxlen=self.NUM_TAPS)

    def reset(self):
        """Clear the buffer back to zeros."""
        self.buffer = deque([0.0j] * self.NUM_TAPS, maxlen=self.NUM_TAPS)

    def load(self, x):
        """Append a single sample or iterable of samples (complex or real)."""
        if isinstance(x, (np.ndarray, list, tuple)):
            for sample in x:
                self.buffer.append(np.complex128(sample))
        else:
            self.buffer.append(np.complex128(x))

    def interpolate(self, mu, integer_offset=0):
        """
        Interpolate sample at (integer_offset + mu) samples before newest sample.
        integer_offset: 0 means interpolate between newest and second newest samples
        mu: fractional delay between 0 and 1
        """
        # assert -1 <= integer_offset+mu < 3

        # Build sample segment starting at position integer_offset
        mu -= integer_offset
        segment = np.array(list(self.buffer))
        
        # Use FIR filters to calculate the polynomial coefficients
        c_k = self.COEFFS @ segment
        powers = mu ** np.arange(self.ORDER + 1)
        return np.dot(c_k, powers)
    
    def process_sample(self, sample, mu, integer_offset):
        """Add a new sample and immediately interpolate at fractional delay mu."""
        self.load(sample)
        return self.interpolate(mu, integer_offset)

    def process_batch(self, samples, mu, integer_offset):
        """Interpolate a batch of complex samples using a fixed mu."""
        return np.array([self.process_sample(s, mu, integer_offset) for s in samples], dtype=np.complex128)

    def process_batch_with_tail_padding(self, samples, mu, integer_offset=0):
        """Add 2-sample tail padding and interpolate the batch at fixed mu."""
        last_val = samples[-1] if len(samples) > 0 else 0.0j
        padded = np.concatenate([samples, np.full(2, last_val, dtype=samples.dtype)])
        return self.process_batch(padded, mu, integer_offset)[2:]



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