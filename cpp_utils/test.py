import py_sdrlib as sdr
import numpy as np

a = np.array([1, 1, 1, 1, 1], dtype=np.complex64)
b = np.empty(5, dtype=np.complex64)
pct_offset = 0.25
w_offset = pct_offset * np.pi * 2
sdr.apply_cfo(a, b, 5, w_offset)

print(b)
