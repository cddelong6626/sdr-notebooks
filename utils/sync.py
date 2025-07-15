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

def gardner_ted(signal, i, mu, farrow):
    last = farrow.interpolate(mu)
    middle = farrow.interpolate(mu+0.5)
    curr = farrow.process_sample(signal[i], mu)

    e = middle * (last - curr)

    return e

class SymbolTimingCorrector:
    def __init__(self, ted_func=gardner_ted, control=PIDFeedback(K_p=0.1), signal=np.empty(1)):
        # Create farrow structure for interpolation
        self._farrow = CubicFarrowStructure()

        # Store args as attributes
        self.load_signal(signal)
        self.ted_func = ted_func
        self.control = control
        
        # Current index within _signal and current fractional delay
        self.idx = 0
        self.mu = 0.0
   
    def load_signal(self, signal):
        self.SIG_SIZE = signal.size
        self._signal = np.concat((signal, [signal[-1]]*2))
        self._farrow.update(self._signal[:2])

    def correct_sample(self):
        if self.idx >= self.SIG_SIZE:
            raise Exception("SymbolTimingCorrector: correct_sample: Index out of range")

        # find error from current offset and correct
        e = self.ted_func(self._signal, self.idx, self.mu, self._farrow)

        # update fractional delay in PID feedback loop, keeping it between 0 and 1
        self.mu = self.control.update(e)
        print(e)
        # if self.mu < 0: self.mu += 1
        # if self.mu > 1: self.mu -= 1

        # farrow stucture loading assumes 3 samples previously loaded in
        samp_resamp = self._farrow.interpolate(self.mu)

        self.idx += 1
        return samp_resamp

    def correct_batch(self, batch=None, n=None):
        if batch is not None:
            self.load_signal(batch)
        elif self._signal is None:
            raise Exception("No batch provided")
        if n is None:
            n = self.SIG_SIZE

        self._signal = np.concat((self._signal, [self._signal[-1]]*2))
        return np.array([self.correct_sample() for _ in range(n)])[2:]
