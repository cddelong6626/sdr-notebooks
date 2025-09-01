
import numpy as np
from numba import njit

@njit
def diff_encode_psk_symbols(symbols):
    encoded = np.zeros(len(symbols) + 1, dtype=np.complex128)
    
    encoded[0] = np.sqrt(1j)
    for i in range(len(encoded[1:])):
        encoded[i+1] = encoded[i] * symbols[i] / np.abs(symbols[i]) / np.sqrt(1j)

    return encoded

@njit
def diff_decode_psk_symbols(symbols):
    return symbols[1:]/symbols[:-1] * np.sqrt(1j)