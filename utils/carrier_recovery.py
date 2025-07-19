
import numba
from numba import njit
from numba.experimental import jitclass
import numpy as np
from .interpolators import CubicFarrowInterpolator
from .control import PIDFeedback


spec = [
    ('upper_threshold', numba.float64),
    ('lower_threshold', numba.float64),
    ('is_locked', numba.boolean)
]
@jitclass(spec)
class PhaseLockDetector:
    # TODO: ADD N REQUIREMENT
    def __init__(self, upper_threshold=0.2, lower_threshold=0.3):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.is_locked = False        

    def update(self, phase_error):
        if self.is_locked:
            if phase_error > self.upper_threshold:
                self.is_locked = False
        else:
            if phase_error < self.lower_threshold:
                self.is_locked = True
        return self.is_locked

# njit sped up function from ~6s to ~0.6s
@njit
def costas_loop(symbols, control, lock_detector=None, debug=None, theta=None):
    if theta is None: theta = 0.0
    sym_rot = np.empty(len(symbols), dtype=np.complex128)

    for i, s in enumerate(symbols):
        # Rotate signal by current VCO phase
        sym_rot[i] = s * np.exp(-1j*theta)

        # Decision directed error signal
        I = sym_rot[i].real
        Q = sym_rot[i].imag
        ref = np.sign(I) + 1j*np.sign(Q)
        e = np.angle(sym_rot[i] * np.conj(ref))

        # Update VCO input
        theta += control.update(e)
        debug[i] = e

        if lock_detector is None: continue
        # debug[i] = lock_detector.update(e)
        # if not lock_detector.update(e):
        #     theta += control.update(e)

    return sym_rot

