
from collections import deque
import numpy as np


# Simple linear interpolator class
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
    Supports real and complex input (e.g., np.complex64).
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
        self.COEFFS = np.fliplr(raw_coeffs).astype(np.complex64)

        # Complex-valued buffer: oldest first, newest last
        self.buffer = deque([0.0j] * self.NUM_TAPS, maxlen=self.NUM_TAPS)

    def reset(self):
        """Clear the buffer back to zeros."""
        self.buffer = deque([0.0j] * self.NUM_TAPS, maxlen=self.NUM_TAPS)

    def load(self, x):
        """Append a single sample or iterable of samples (complex or real)."""
        if isinstance(x, (np.ndarray, list, tuple)):
            for sample in x:
                self.buffer.append(np.complex64(sample))
        else:
            self.buffer.append(np.complex64(x))

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
        return np.array([self.process_sample(s, mu, integer_offset) for s in samples], dtype=np.complex64)

    def process_batch_with_tail_padding(self, samples, mu, integer_offset=0):
        """Add 2-sample tail padding and interpolate the batch at fixed mu."""
        last_val = samples[-1] if len(samples) > 0 else 0.0j
        padded = np.concatenate([samples, np.full(2, last_val, dtype=samples.dtype)])
        return self.process_batch(padded, mu, integer_offset)[2:]
