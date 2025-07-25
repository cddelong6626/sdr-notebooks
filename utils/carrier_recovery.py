
import numba
from numba import njit
from numba.experimental import jitclass
import numpy as np
from .interpolators import CubicFarrowInterpolator
from .control import PIDFeedback
from .framing import FramingStateMachine


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

class CoarseCFOCorrector:
    def __init__(self, preamble, detection_threshold=0.5):
        self.preamble = preamble
        self.detection_threshold = detection_threshold
        
        self.w_est = None
        self.fsm = FramingStateMachine(preamble, len(self.preamble), self.detection_threshold)

    def guess(self, new_samples):
        """Guess frequency offset based on the first preamble detected in an array of samples"""

        # Detect preamble
        preambles = self.fsm.update(new_samples)

        # No preambles detected: no guess made
        if len(preambles) == 0:
            return False
        
        # Preamble detected: estimate CFO based on first preamble
        self.estimate_cfo(preambles[0])
        return True
        
    def estimate_cfo(self, preamble_in):
        """Estimate CFO (rads/sample) based on phase drift of preamble over time"""
        phase_off = np.angle(preamble_in / self.preamble)
        rel_phase_off = phase_off[1:] - phase_off[:-1]
        self.w_est = np.mean(rel_phase_off)
        
        self.test = (phase_off, preamble_in, self.preamble)

    def correct(self, signal):
        """Correct the CFO of a signal assuming CFO has been estimated using estimate_w()"""
        if self.w_est is None:
            raise AttributeError("CFO must be estimated before it can be corrected")

        n = np.arange(len(signal))
        sig_offset = signal * np.exp(1j*self.w_est*n)

        return sig_offset


