
import numba
from numba import njit
from numba.experimental import jitclass
import numpy as np
from .framing import DifferentialCorrelationFrameDetector
from .control import PIDFeedback
from abc import ABC
import math


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
def costas_loop(symbols, controller, error_history=None, theta=None):
    if theta is None: theta = 0.0
    sym_rot = np.empty(len(symbols), dtype=np.complex64)

    for i, s in enumerate(symbols):
        # Rotate signal by current VCO phase
        sym_rot[i] = s * np.exp(-1j*theta)

        # Decision directed error signal
        I = sym_rot[i].real
        Q = sym_rot[i].imag
        ref = np.sign(I) + 1j*np.sign(Q)
        e = np.angle(sym_rot[i] * np.conj(ref))

        # Update VCO input
        theta += controller.update(e)
        if error_history is not None: error_history[i] = e

    return sym_rot

class CostasLoopQPSK:
    def __init__(self, loop_bw: float):
        # Recommended loop bandwidth: R/20 to R/200 where R = sample rate
        # Found at: https://john-gentile.com/kb/dsp/PI_filter.html
        self.loop_bw = loop_bw

        self.error_history = None
        self.correction = 0.0

    @property
    def loop_bw(self):
        return self._loop_bw

    @loop_bw.setter
    def loop_bw(self, value):
        # Below equations derived at: https://john-gentile.com/kb/dsp/PI_filter.html
        self._loop_bw = value

        damping_factor = 0.707
        alpha = 1 - 2 * damping_factor**2
        scaled_bw = self._loop_bw / math.sqrt(alpha + math.sqrt(alpha**2 + 1))
        K_d = 1
        K_p = 2*damping_factor*scaled_bw/K_d
        K_i = scaled_bw**2 / K_d

        self.controller = PIDFeedback(K_p=K_p, K_i=K_i)

    def reset(self):
        self.error_history = None
        self.correction = 0.0

    def process(self, symbols_in, symbols_out):
        # Ensure error history is allocated
        if self.error_history is None or len(self.error_history) != len(symbols_in):
            self.error_history = np.empty(len(symbols_in), dtype=np.float32)

        # Costas Loop algorithm
        if len(symbols_out) != len(symbols_in):
            raise ValueError("symbols_out must be the same length as symbols_in")

        for i, s in enumerate(symbols_in):
            # Rotate signal by current VCO phase
            symbols_out[i] = s * np.exp(-1j*self.correction)

            # Decision directed error signal
            I = symbols_out[i].real
            Q = symbols_out[i].imag
            ref = np.sign(I) + 1j*np.sign(Q)
            e = np.angle(symbols_out[i] * np.conj(ref))
            self.error_history[i] = e 

            # Update VCO input
            self.correction += self.controller.update(e)
        

class CoarseCFOCorrector(ABC):
    def __init__(self, preamble: np.ndarray, detection_threshold: float=0.6, detector_cls=DifferentialCorrelationFrameDetector):
        self._preamble = None
        self._detection_threshold = None
        self._w_est = None
        self._fd = detector_cls(
            preamble=preamble, 
            expected_frame_length=len(preamble), 
            detection_threshold=detection_threshold
        )

        self.preamble = preamble
        self.detection_threshold = detection_threshold

        self.debug = None
    
    @property
    def preamble(self):
        return self._preamble
    
    @preamble.setter
    def preamble(self, value: np.ndarray):
        self._preamble = value
        
        self._fd.preamble = value
        self._fd.expected_frame_length = len(value)

    @property
    def detection_threshold(self):
        return self._detection_threshold
    
    @detection_threshold.setter
    def detection_threshold(self, value: float):
        assert 0 <= value and value <= 1, "Detection threshold must be bounded between 0 and 1"

        self._detection_threshold = value
        self._fd.detection_threshold = value

    def process(self, new_samples):
        """Estimate frequency offset based on the first preamble detected in an array of samples"""
        # Detect preamble
        preambles = [detected.frame for detected in self._fd.process(new_samples)]
        self.debug = preambles

        # No preambles detected: no guess made
        if len(preambles) == 0:
            return False
        
        # Preamble detected: estimate CFO based on first preamble
        self.estimate_cfo(preambles[0])
        return True
    
    def get_estimate(self):
        """Return estimate of CFO in [radians/sample]"""
        return self._w_est

    def correct(self, signal):
        """Correct the CFO of a signal assuming CFO has been estimated using estimate_w()"""
        if self.w_est is None:
            raise AttributeError("CFO must be estimated before it can be corrected")

        n = np.arange(len(signal))
        sig_offset = signal * np.exp(-1j*self.w_est*n)

        return sig_offset

    def estimate_cfo(self, rx_preamble: np.ndarray):
        raise NotImplementedError()
    

class SCCoarseCFOCorrector(CoarseCFOCorrector): 
    """Coarsely estimate CFO using Schmidl-Cox algorithm"""
    def __init__(self, preamble: np.ndarray, detection_threshold: float=0.6, detector_cls=DifferentialCorrelationFrameDetector):
        super().__init__(preamble, detection_threshold, detector_cls)

    @property
    def preamble(self):
        return self._preamble
    
    @preamble.setter
    def preamble(self, value: np.ndarray):
        assert len(value) % 2 == 0, "Schmidl-Cox preamble must be even in length"
        assert np.all(value[:int(len(value)/2)] == value[int(len(value)/2):]), "Schmidl-Cox preamble must be two identical halves"

        self._preamble = value
        self._T = len(value)//2
        self._fd.preamble = value
        self._fd.expected_frame_length = len(value)
        
    def estimate_cfo(self, rx_preamble: np.ndarray):
        """
        Coarsely estimate CFO (rads/sample) using Schmidl-Cox algorithm
        
        See:
            T. M. Schmidl and D. C. Cox, “Robust frequency and timing synchronization for OFDM,” IEEE Transactions 
            on Communications, vol. 45, no. 12, pp. 1613-1621, 1997, doi: 10.1109/26.650240.
            https://doi.org/10.1109/26.650240
        """
        assert rx_preamble.dtype == np.complex64

        # Calculate CFO estimate for each pair
        P = rx_preamble[:self._T].conj() * rx_preamble[self._T:]    # Eq. 5
        phi_hat = np.angle(P)                                       # Eq. 38
        w_hat_i = phi_hat/self._T                                   # Eq. 39

        # Use Median Absolute Devaition (MAD) to filter out outliers. This makes detection robust to frame sync 
        # errors (to some degree)  
        k = 2.5
        med = np.median(w_hat_i)
        abs_dev_i = np.abs(w_hat_i - med)
        mad = np.median(abs_dev_i)

        # Estimate CFO as mean of inliers
        if mad == 0:
            mad = med * 0.03
        inliers = w_hat_i[abs_dev_i < k*mad]
        self._w_est = np.mean(inliers)
        
        return self._w_est

class PhaseDriftCFOCorrector(CoarseCFOCorrector):
    """Coarsely estimate CFO based on phase drift of preamble over time"""
    def __init__(self, preamble: np.ndarray, detection_threshold: float=0.5, detector_cls=DifferentialCorrelationFrameDetector):
        super().__init__(preamble, detection_threshold, detector_cls)

    def estimate_cfo(self, rx_preamble: np.ndarray):
        """
        Coarsely estimate CFO (rads/sample) based on phase drift of preamble over time
        
        TODO: How does this perform at 2 SPS when many samples are near 0 and noise is more prominent? 
        """
        assert rx_preamble.dtype == np.complex64

        # The difference in phase between the transmitted preamble and received preamble changes at rate of CFO
        phase_off = np.angle(rx_preamble * self.preamble.conj())
        w_hat_i = phase_off[1:] - phase_off[:-1]

        # Estimate CFO as mean of estimates
        self._w_est = np.mean(w_hat_i)

        return self._w_est