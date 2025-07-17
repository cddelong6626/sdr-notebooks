import numba
from numba import njit
from numba.experimental import jitclass
import numpy as np
from .filters import CubicFarrowInterpolator, PIDFeedback

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


def gardner_ted(mu, farrow):
    """
    Gardner TED for SPS=2 with single-sample buffer update.
    Assume buffer holds samples up to index i, and mu in [0,1].
    """
    # Check if mu is close to 0 or 1
    # Shift base index and adjust mu accordingly to keep mu ± 1 in [0, 1)

    # Interpolate using safe range
    prev = farrow.interpolate(mu - 1)
    curr = farrow.interpolate(mu)
    next = farrow.interpolate(mu + 1)

    # print(f"prev {prev}\t\tcurr {curr}\t\tnext {next}")

    e = np.real((prev - next) * np.conj(curr))
    return e

class SymbolTimingCorrector:
    def __init__(self, ted_func=gardner_ted, control=None, signal=None):
        self._farrow = CubicFarrowInterpolator()
        self.ted_func = ted_func
        self.control = control if control else PIDFeedback(K_p=0.05)
        self.signal = None
        self.i = 0
        self.mu = 0.5

        self.mu_log = []
        self.e_log = []

        if signal is not None:
            self.load_signal(signal)

    def load_signal(self, signal):
        self.signal = np.asarray(signal, dtype=np.complex128)
        self.SIG_SIZE = len(self.signal)
        self._farrow.load(signal[:2])
        self.i = 0

    def correct_sample(self):
        # Add next sample to Farrow buffer first
        self._farrow.load(self.signal[self.i])

        # Compute TED error at current mu using Farrow buffer
        e = self.ted_func(self.mu, self._farrow)

        # Update fractional delay using PID or loop filter
        self.mu += self.control.update(e)
        # self.mu %= 1.0
        # self.mu = 0.

        self.mu_log.append(self.mu)
        self.e_log.append(e)

        # Interpolate at current timing phase
        sample_out = self._farrow.interpolate(self.mu, integer_offset=0)

        self.i += 1
        return sample_out

    def correct_batch(self, batch=None, n=None):
        if batch is not None:
            self.load_signal(batch)
        elif self.signal is None:
            raise ValueError("No input signal provided.")

        if n is None:
            n = self.SIG_SIZE

        out = []
        for _ in range(n):
            out.append(self.correct_sample())
        return np.array(out, dtype=np.complex128)
