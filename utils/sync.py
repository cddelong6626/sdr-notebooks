
from numba import njit
import numpy as np


# TODO:
#   - Add Gardner
#   - Add M&M
#   - Figure out costas_loop function signature
#   - Add aquisition + tracking stages? or maybe just keep that in notebook 
#   - Add phase lock detection 

@njit
def costas_inner_loop(symbol, K_p, K_i, theta_arr, sum_e_arr, s_rot_out_arr):
    # Rotate signal by current VCO phase
    s_rot_out_arr[0] = symbol * np.exp(-1j * theta_arr[0])
    
    # Decision-directed error signal
    I = s_rot_out_arr[0].real
    Q = s_rot_out_arr[0].imag
    ref = np.sign(I) + 1j*np.sign(Q)
    e = np.angle(s_rot_out_arr[0] * np.conj(ref))  # Phase error

    # Update VCO input
    sum_e_arr[0] += K_i*e
    theta_arr[0] += K_p*e + sum_e_arr[0]

# njit sped up function from ~6s to ~0.6s
@njit
def costas_loop(symbols, K_p, K_i, theta=None):
    
    if theta is None: theta = 0.0
    theta = np.array([theta])

    sym_rot = np.empty(len(symbols), dtype=np.complex128)
    theta = np.array([0.0])
    sum_e = np.array([0.0])
    s_rot = np.empty(1, dtype=np.complex128)

    for i, s in enumerate(symbols):
        costas_inner_loop(s, K_p, K_i, theta, sum_e, s_rot)

        symbols[i] = s_rot[0]
    return s_rot