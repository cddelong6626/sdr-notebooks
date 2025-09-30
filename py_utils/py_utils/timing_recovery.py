
import numpy as np
from abc import ABC, abstractmethod

from .interpolators import CubicFarrowInterpolator
from .control import PIDFeedback

# TODO:
#   - Add M&M


class SymbolTimingCorrector(ABC):
    @staticmethod
    def ted(mu, farrow):
        raise NotImplementedError

    @abstractmethod
    def process(self, signal=None):
        raise NotImplementedError


class GardnerSymbolTimingCorrector(SymbolTimingCorrector):
    def ted(mu, farrow):
        """
        Gardner TED for SPS=2 with single-sample buffer update.
        Assume buffer holds samples up to index i, and mu in [0,1].
        """
        # Check if mu is close to 0 or 1

        # Interpolate using safe range
        prev = farrow.interpolate(mu - 1)
        curr = farrow.interpolate(mu)
        next = farrow.interpolate(mu + 1)

        e = np.real((prev - next) * np.conj(curr))
        return e

    def __init__(self, signal=None, control=None):
        self._farrow = CubicFarrowInterpolator()
        self.control = control if control else PIDFeedback(K_p=0.1)
        self.signal = None
        self.i = 0
        self.mu = 0.5
        self._offset = 0
        self._pad_len = 2

        self.mu_log = []
        self.e_log = []

        if signal is not None:
            self.load_signal(signal)

    def reset(self):
        # Reset internal state
        self._farrow.reset()
        self.control.reset()
        self.signal = None
        self.mu = 0.5
        self.i = self._pad_len
        self._offset = 0

        self.mu_log = []
        self.e_log = []

    def load_signal(self, signal):
        # Load signal and initialize Farrow buffer
        sig = np.asarray(signal, dtype=np.complex64)
        pad = np.repeat(sig[-1], self._pad_len)
        self.signal = np.concatenate((sig, pad))
        self.SIG_SIZE = len(self.signal)
        self._farrow.load(self.signal[:2])
        self.i = self._pad_len

    def process_symbol_pair(self):
        # Process pairs of samples. After two samples, record one output sample. 
        # This ensures that the number of output samples is half the number of input samples.
        if self.signal is None:
            raise ValueError("No input signal provided.")
        if self.i + 2 > self.SIG_SIZE:
            raise ValueError("End of input signal reached.")
        
        sample_out = None
        e = 0.0

        for i in range(2):
            # Interpolation is best when mu is close to 0.5 (i.e., center of buffer)
            # If mu drifts too far from 0.5, reset it and flip offset.
            # Use hysteresis to avoid rapid flipping around thresholds.
            H = 0.1
            if self.mu > 0.5 + H:
                self.mu = -0.5
                self._offset = not self._offset
            elif self.mu < -0.5 - H:
                self.mu = 0.5
                self._offset = not self._offset

            # Add next sample to Farrow buffer 
            self._increment()

            # Update fractional delay using PID controller every other sample.
            if self.i % 2 == self._offset:
                e = GardnerSymbolTimingCorrector.ted(self.mu, self._farrow)
                self.mu += self.control.update(e) 

            else:
                # Allow sample_out to be overwritten by second iteration if offset swaps
                sample_out = self._farrow.interpolate(self.mu)

        # Log values after both samples processed. This ensures one log entry per output sample.
        self.mu_log.append(self.mu)
        self.e_log.append(e)

        if sample_out is None:
            # defensive fallback
            sample_out = self._farrow.interpolate(self.mu)

        # Interpolate at current timing phase
        return sample_out

    def process(self, signal=None):
        if signal is not None:
            self.load_signal(signal)
        elif self.signal is None:
            raise ValueError("No input signal provided.")

        out = []
        while self.i + 2 <= self.SIG_SIZE:
            out.append(self.process_symbol_pair())
        return np.array(out, dtype=np.complex64)
    
    def _increment(self, n=1):
        for _ in range(n):
            self._farrow.load(self.signal[self.i])
            self.i += 1