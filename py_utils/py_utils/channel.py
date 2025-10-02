import numpy as np
from .interpolators import CubicFarrowInterpolator

def apply_awgn(signal, snr_db):
    """Apply AWGN (additive Gaussian white noise) to signal"""
    signal_power = np.mean(abs(signal) ** 2)
    noise_power = signal_power / (10**(snr_db / 10))
   
    awgn = np.sqrt(noise_power / 2) * (np.random.normal(0, 1, len(signal)) + 1j * np.random.normal(0, 1, len(signal)))
    sig_noisy = signal + awgn

    return sig_noisy


def apply_cfo(signal, pct_offset=0.03, w_offset=None):
    """Apply carrier frequency offset to signal"""
    # testing/realistic: 1-5%, aggressive: 10%

    if w_offset is None:
        w_offset = pct_offset*(2*np.pi)  # radians/sample
       
    n = np.arange(len(signal))
    sig_offset = signal * np.exp(1j*w_offset*n)

    return sig_offset

def apply_sto(signal, mu, integer_offset=0):
    """Apply symbol timing offset to signal"""

    # Interpolate efficiently using a cubic farrow structure and a lagrange polynomial
    farrow = CubicFarrowInterpolator()
    sig_offset = farrow.process_batch_with_tail_padding(signal, mu, integer_offset)

    return sig_offset

def apply_fto(frames: np.ndarray, max_delay):
    """Apply frame timing offset to frames to simulate bursty transmission"""
    sig_out = np.empty(0, frames[0][0].dtype)
    for frame in frames:
        offset = np.zeros(np.random.randint(1, max_delay))
        sig_out = np.concatenate([sig_out, offset, frame])
    
    return sig_out