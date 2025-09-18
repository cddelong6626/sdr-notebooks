
from numba import jit, njit
import numpy as np
import numpy.typing as npt


def modulate_qpsk(bits):
    """
    Gray code mapping:
    [0, 0]: +1 +1j
    [0, 1]: +1 -1j
    [1, 1]: -1 -1j
    [1, 0]: -1 +1j
    """

    # Map each pair of bits to a vector using gray coding
    re = -2*np.array(bits[::2]) + 1
    im = -2j*np.array(bits[1::2]) + 1j

    symbols = (re + im)/np.sqrt(2)
    return symbols


def demodulate_qpsk(symbols):
    """
    Gray code demapping:
    +1 +1j: [0, 0]
    +1 -1j: [0, 1]
    -1 -1j: [1, 1]
    -1 +1j: [1, 0]
    """

    # De-map each symbol, making optimum decision given AWGN
    bits = np.zeros(shape=(len(symbols), 2), dtype=np.uint8)
    bits[:, 0] = symbols.real < 0
    bits[:, 1] = symbols.imag < 0
    bits = bits.ravel()

    return bits


def optimum_decider_qpsk(symbols):
    return np.sign(symbols.real) + 1j*np.sign(symbols.imag)