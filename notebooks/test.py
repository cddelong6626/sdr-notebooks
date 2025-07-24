# Imports
import scipy
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '.')
# import importlib
# import utils
# importlib.reload(utils)
from utils import *


# # Constants
# SPS = int(2)
# N_BITS = 10**6
# N_SYMBOLS = int(N_BITS/2) + 1       # qpsk: 2 bits/sample, differential coding: +1 symbol
# N_RRC_TAPS = SPS*10 + 1
# SNR_DB = 20


# fig, axs = plt.subplots(1, 3, figsize=(15, 4))


# SPS = 2
# N_RRC_TAPS = SPS*10 + 1
# N_SAMPS = 70

# # TX
# h_rrc = rrc(Ts=SPS, n_taps=N_RRC_TAPS)
# s = np.convolve(upsample(modulate_qpsk(np.random.randint(2, size=N_SAMPS)), SPS), h_rrc)
# samples = np.array([s[i]*(i % SPS == 0) for i in np.arange(s.size)], dtype=s.dtype)

# plot_signal(s, samples, title="TX", ax=axs[0])


# # Channel
# mu = 0.2
# s = apply_symbol_timing_offset(s, mu)

# samples = np.array([s[i]*(i % SPS == 0) for i in np.arange(s.size)], dtype=s.dtype)

# plot_signal(s, samples, title="Post-Channel", ax=axs[1])


# # RX
# s = np.convolve(s, h_rrc)[20:-20]
# stc = SymbolTimingCorrector()
# s = stc.correct_batch(batch=s)
# samples = np.array([s[i]*(i % SPS == 0) for i in np.arange(s.size)], dtype=s.dtype)

# plot_signal(s, samples, title="RX", ax=axs[2])

# apply_frame_timing_offset(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3], [4, 5, 6, 7]]), 5)

# print(zadoff_chu(5, 1))
# j = complex(0, 1)
# # complex(0, 1) * np.arange(5) 
# np.exp(-j*2*np.pi/5)


preamble = zadoff_chu(5)
payload = np.random.randint(2, size=20)
# print(payload)
# print(to_frames(preamble, payload, 5))
frames = to_frames(preamble, payload, 5)

signal = apply_frame_timing_offset(frames, 10)
matched = scipy.signal.lfilter(preamble[::-1].conj(), 1, signal)[(len(preamble)-1):]
metric = np.abs(matched) / (np.linalg.norm(signal) * np.linalg.norm(preamble))
plot_signal(metric)

print(frames)

fsm = FramingStateMachine(preamble, 10, 0.3)
detected_frames = fsm.update(signal)

print(frames)
