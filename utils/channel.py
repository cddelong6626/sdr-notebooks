
import numpy as np


def apply_awgn(signal, snr_db):

    signal_power = np.mean(abs(signal) ** 2)
    noise_power = signal_power / (10**(snr_db / 10))
    
    awgn = np.sqrt(noise_power / 2) * (np.random.normal(0, 1, len(signal)) + 1j * np.random.normal(0, 1, len(signal)))
    noisy_signal = signal + awgn

    return noisy_signal


def apply_cpo(signal, phase_offset=None):

    if phase_offset is None: 
        phase_offset = 2*np.pi*np.random.rand()  # [radians]
    
    signal_offset = signal * np.exp(1j*phase_offset)

    return signal_offset


def apply_cfo(signal, pct_offset=0.03, w_offset=None): 
    # testing/realistic: 1-5%, aggressive: 10%

    if w_offset is None:
        w_offset = pct_offset*(2*np.pi)  # radians/sample
        
    n = np.arange(len(signal))
    signal_offset = signal * np.exp(1j*w_offset*n)

    return signal_offset
