
import numba
from numba import njit
from numba.experimental import jitclass
import numpy as np
from .framing import CorrelationFrameDetector
from abc import ABC


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


class CoarseCFOCorrector(ABC):
    def __init__(self, preamble: np.ndarray, detection_threshold: float=0.8, detector_cls=CorrelationFrameDetector):
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
        preambles = self._fd.process(new_samples)

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
    def __init__(self, preamble: np.ndarray, detection_threshold: float=0.8, detector_cls=CorrelationFrameDetector):
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
        assert rx_preamble.dtype == np.complex128

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
    def __init__(self, preamble: np.ndarray, detection_threshold: float=0.8, detector_cls=CorrelationFrameDetector):
        super().__init__(preamble, detection_threshold, detector_cls)

    def estimate_cfo(self, rx_preamble: np.ndarray):
        """
        Coarsely estimate CFO (rads/sample) based on phase drift of preamble over time
        
        TODO: How does this perform at 2 SPS when many samples are near 0 and noise is more prominent? 
        """
        assert rx_preamble.dtype == np.complex128

        # The difference in phase between the transmitted preamble and received preamble changes at rate of CFO
        phase_off = np.angle(rx_preamble * self.preamble.conj())
        w_hat_i = phase_off[1:] - phase_off[:-1]

        # Estimate CFO as mean of estimates
        self._w_est = np.mean(w_hat_i)

        return self._w_est