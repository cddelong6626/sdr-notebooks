import numba
from numba import njit
from numba.experimental import jitclass
import numpy as np
from .interpolators import CubicFarrowInterpolator
from .control import PIDFeedback

# TODO:
#   - Add M&M
#   - Figure out costas_loop function signature
#   - Add aquisition + tracking stages? or maybe just keep that in notebook



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

    e = np.real((prev - next) * np.conj(curr))
    return e

class SymbolTimingCorrector:
    def __init__(self, ted_func=gardner_ted, control=None, signal=None):
        self._farrow = CubicFarrowInterpolator()
        self.ted_func = ted_func
        self.control = control if control else PIDFeedback(K_p=0.1)
        self.signal = None
        self.i = 1
        self.mu = 0.05
        self._offset = 0

        self.mu_log = []
        self.e_log = []

        if signal is not None:
            self.load_signal(signal)

    def load_signal(self, signal):
        self.signal = np.asarray(signal, dtype=np.complex128)
        self.SIG_SIZE = len(self.signal)
        self._farrow.load(signal[:2])
        self.i = 0

    def correct_symbol(self):
        
        # Add next sample to Farrow buffer first
        if self.mu > 1:
            # self._increment(2)
            self.mu = 0.05
            self._offset = not self._offset
        elif self.mu < 0:
            self.mu = 0.95
            self._offset = not self._offset
        # else:
            # self._increment()
        self._increment()

        # Update fractional delay using PID or loop filter
        if self.i % 2 == self._offset:
            e = self.ted_func(self.mu, self._farrow)
            self.mu += self.control.update(e) 

            self.mu_log.append(self.mu)
            self.e_log.append(e)

            return self.correct_symbol()

        # Interpolate at current timing phase
        sample_out = self._farrow.interpolate(self.mu)
        return sample_out
        
    def _increment(self, n=1):
        for i in range(n):
            self._farrow.load(self.signal[self.i])
            self.i += 1

    def correct_batch(self, batch=None, n=None):
        if batch is not None:
            self.load_signal(batch)
        elif self.signal is None:
            raise ValueError("No input signal provided.")

        if n is None:
            n = self.SIG_SIZE

        out = []
        while self.i < n:
            out.append(self.correct_symbol())
        return np.array(out, dtype=np.complex128)
