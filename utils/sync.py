import numba
from numba import njit
from numba.experimental import jitclass
import numpy as np
from .filters import CubicFarrowStructure, PIDFeedback

# TODO:
#   - Add M&M
#   - Figure out costas_loop function signature
#   - Add aquisition + tracking stages? or maybe just keep that in notebook


spec = [
    ('upper_threshold', numba.float64),
    ('lower_threshold', numba.float64),
    ('is_locked', numba.boolean)
]
@jitclass(spec)
class PhaseLockDetector:
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

        # if lock_detector is None: continue
        # # debug[i] = lock_detector.update(e)
        # if lock_detector.update(e):
        #     control.K_i = 0.005
        # else:
        #     control.K_i = 0.05



    return sym_rot

def gardner_ted(signal, i, mu):

    farrow = CubicFarrowStructure()
    farrow.update(signal[i-2 : i])
   
    last = farrow.update(i, mu)
    middle = farrow.update(i, mu+0.5)
    curr = farrow.update(i+1, mu)

    e = middle * (last - curr)

    return e

class SymbolTimingCorrector:
    def __init__(self, ted_func, control, signal=np.empty(1)):
        self.load_signal(signal)
        self._farrow = CubicFarrowStructure()
        self.ted_func = ted_func
        self.control = control
        self.idx = 0
        self.mu = 0.0
   
    def load_signal(self, signal):
        self._signal = signal.append(signal[-1])

    def correct_sample(self):
        # find error from current offset and correct
        self._farrow.update(self._signal[self.idx])
        e = self.ted_func(self._signal, self.idx, self.mu)
        self.mu = self.control.update(e)

        # farrow stucture loading assumes 3 samples previously loaded in
        samp_resamp = self._farrow.process_sample(self._signal[self.idx], self.mu)
        return samp_resamp

    def correct_batch(self, n, batch=None):
        if batch:
            self.load_signal(batch)

        return np.array([self.correct_sample() for _ in range(n)])
