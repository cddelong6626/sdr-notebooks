
from dataclasses import dataclass
import numpy as np
from scipy import signal

from .channel import apply_cfo


### PREAMBLES ###

def zadoff_chu(N: int, q: int=1):
    """
    Generate a Zadoff-Chu sequence with sequence length N_zc and root index q

    See:
        Andrews, J.G. (2022). A Primer on Zadoff Chu Sequences. arXiv preprint arXiv:2211.05702.
        https://arxiv.org/abs/2211.05702
    """

    if N % 2 == 0:
        raise ValueError("Zadoff-Chu sequence length N_zc must be an odd number.")
    if q < 0 or q > (N-1):
        raise ValueError("Zadoff-Chu root index q must be between an odd number from 1 to (N_zc-1).")
    
    n = np.arange(N)
    j = complex(0, 1)
    return np.exp(-j*np.pi*q*n*(n+1)/N)


def pn(N: int):
    """Generate a Pseudo-random Number (PN) sequence of length N"""
    return np.random.randint(2, size=N)


def to_frames(preamble: np.ndarray, payload: np.ndarray, n: int):
    """Convert payload and preamble to an array of frames"""

    # Spilt payload into n-sized chunks. One per frame
    if len(payload) % n != 0:
        raise ValueError("Payload not n-divisble.")
    payload = payload.reshape((-1, n))

    # Create frames: preamble + payload
    frames = []
    for chunk in payload:
        frame = np.concatenate([preamble, chunk])
        frames.append(frame)
    return frames



### FRAME DETECTION ###

class FrameDetector:
    """
    Detect frames within a signal using a preamble
    
    This parent class contains the FSM and buffering logic. Child classes must implement a _detect_preamble() method
    """
    def __init__(self, preamble: np.ndarray, expected_frame_length: int, detection_threshold: float=0.8):
        self.detection_threshold = detection_threshold
        self.expected_frame_length = expected_frame_length
        self.preamble = preamble

        self._state = "SEARCH"
        self.buffer = np.empty(0, dtype=np.complex128)

        self.debug = None
        

    def process(self, new_samples: np.ndarray):
        """Update the buffer and check if there are any frames detected"""
        self.buffer = np.concatenate((self.buffer, new_samples))
        results = []
        idx = 0

        while True:
            # Pause state machine when buffer can't hold a frame
            if len(self.buffer) < self.expected_frame_length:
                break

            # SEARCH state: Search for preamble within buffer
            if self._state == "SEARCH":
                res = self._detect_preamble()
                if res is not None:
                    results.append(res)
                    self.buffer = self.buffer[res.idx:]
                    self._state = "ACQUIRE"
                    continue
                else:   # If no preamble is detected: pause search
                    self.buffer = self.buffer[-len(self.preamble):]
                    break

            # ACQUIRE state: Add result to list of results
            if self._state == "ACQUIRE":
                frame = np.array(self.buffer[:self.expected_frame_length])
                results[idx].frame = frame
                idx += 1

                self.buffer = self.buffer[self.expected_frame_length:]
                self._state = "SEARCH"
                continue

        return results

    def _detect_preamble(self):
        """Return the index of the first detected preamble or None if no preamble is detected"""
        raise NotImplementedError()
    

@dataclass
class DetectionResult:
    frame: np.ndarray = None
    metric: float = None
    idx: int = None
    cfo: float = None


class CorrelationFrameDetector(FrameDetector):
    """Detect frames using correlation with the complex conjugate of the preamble"""
    def __init__(self, preamble: np.ndarray, expected_frame_length: int, detection_threshold: float=0.8, mode='first'):
        self._preamble = None
        self._preamble_norm = None
        self._matched_filter = None
        super().__init__(preamble, expected_frame_length, detection_threshold)

        self.mode = mode

    @property
    def preamble(self):
        return self._preamble
    
    @preamble.setter
    def preamble(self, value: np.ndarray):
        """Set preamble and related values"""
        self._preamble = value
        self._preamble_norm = np.sum(np.abs(value) ** 2)
        self._matched_filter = value[::-1].conj()
    
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, value: str):
        if value not in ['first', 'max']:
            raise ValueError("Frame detection mode must either be 'first' or 'max'")
        self._mode = value

    def _detect_preamble(self):
        """Apply matched filter to signal and report the index of the first spike"""
        matched = signal.convolve(self._matched_filter, self.buffer, mode='valid')

        # Compute energy of sliding window for normalization
        window_norm = signal.convolve(np.ones_like(self._matched_filter), np.abs(self.buffer)**2, mode='valid')
        normalization = (self._preamble_norm * window_norm).astype(np.float32)

        # Avoid division by zero
        normalization[normalization == 0] = 1e-12

        # Metric = Energy of matched / (product of energies of filter and window)
        # This makes detection more robust to noise and varying SNRs
        metric = (np.abs(matched) ** 2) / normalization

        if self.mode == 'first':
            peaks = np.where(metric > self.detection_threshold)
            if len(peaks[0]) == 0:
                return None
            idx = peaks[0][0]
        elif self.mode == 'max':
            idx = np.argmax(metric)

        m_ret = metric[idx]
        res = DetectionResult(idx=idx, metric=m_ret)


        if self.debug is None: # TODO: REMOVE
            self.debug = (metric, self._preamble, self.buffer[idx:idx+len(self._preamble)])


        return res
        


class DifferentialCorrelationFrameDetector(FrameDetector):
    """Detect frames using differential correlation with the complex conjugate of the preamble"""
    def __init__(self, preamble: np.ndarray, expected_frame_length: int, detection_threshold: float=0.8):
        self._preamble = None
        self._preamble_norm = None
        self._matched_filter = None

        dpreamble = preamble[1:] - preamble[:-1]
        super().__init__(dpreamble, expected_frame_length, detection_threshold)

    @property
    def preamble(self):
        return self._preamble

    @preamble.setter
    def preamble(self, value: np.ndarray):
        """Set preamble and related values"""
        self._preamble = value[1:] - value[:-1]
        self._preamble_norm = np.sum(np.abs(self._preamble) ** 2)
        self._matched_filter = self._preamble[::-1].conj()
    
    def _detect_preamble(self):
        """Apply matched filter to signal and report the index of the first spike"""
        dbuffer = self.buffer[1:] - self.buffer[:-1]
        matched = signal.convolve(self._matched_filter, dbuffer, mode='valid')

        # Compute energy of sliding window for normalization
        window_norm = signal.convolve(np.ones_like(self._matched_filter), np.abs(dbuffer)**2, mode='valid')
        normalization = (self._preamble_norm * window_norm).astype(np.float32)

        # Avoid division by zero
        normalization[normalization == 0] = 1e-12

        # Metric = Energy of matched / (product of energies of filter and window)
        # This makes detection more robust to noise and varying SNRs
        metric = (np.abs(matched) ** 2) / normalization
        peaks = np.where(metric > self.detection_threshold)

        if self.debug is None: # TODO: REMOVE
            self.debug = (metric, self._preamble, dbuffer[peaks[0][0]:peaks[0][0]+len(self._preamble)])

        if len(peaks[0]) == 0:
            return None

        idx = peaks[0][0]
        m_ret = metric[idx]
        res = DetectionResult(idx=idx, metric=m_ret)

        return res
    

class AcquisitionFrameDetector(FrameDetector):
    """Uses multiple correlation detectors with different CFO hypotheses to acquire the best initial frame based on maximum correlation metric"""
    def __init__(self, preamble: np.ndarray, expected_frame_length: int, detection_threshold: float=0.5, cfo_vector: np.ndarray=np.linspace(0, 0.05*(2*np.pi/100), 6)):
        self._preamble = None
        self._preamble_norm = None
        self._cfo_vector = cfo_vector
        self._preambles = []
        self._corr_detectors = []

        self.cfo_vector = cfo_vector
        super().__init__(preamble, expected_frame_length, detection_threshold)

    @property
    def preamble(self):
        return self._preamble
    
    @preamble.setter
    def preamble(self, value: np.ndarray):
        self._preamble = value
        self._preamble_norm = np.sum(np.abs(value) ** 2)
        if self.cfo_vector is not None:
            self._generate_preamble_hypotheses()
            self._generate_corr_detectors()
        
    @property
    def cfo_vector(self):
        return self._cfo_vector
    
    @cfo_vector.setter
    def cfo_vector(self, value: np.ndarray):
        self._cfo_vector = value
        if self.preamble is not None:
            self._generate_preamble_hypotheses()
            self._generate_corr_detectors()

    def _generate_preamble_hypotheses(self):
        self._preambles = []
        for cfo in self.cfo_vector:
            preamble_off = apply_cfo(self.preamble, w_offset=cfo)
            self._preambles.append(preamble_off)

    def _generate_corr_detectors(self):
        # Create new CorrelationFrameDetector objects
        self._corr_detectors = [CorrelationFrameDetector(filt, self.expected_frame_length, self.detection_threshold, mode='max') for filt in self._preambles]

    def _detect_preamble(self):
        # Process buffer using all detectors
        results = []
        metrics = []
        for i, det in enumerate(self._corr_detectors):
            res = det.process(self.buffer)[0]
            if res is not None:
                res.cfo = self.cfo_vector[i]
                results.append(res)
                metrics.append(res.metric)

        if len(metrics) == 0:
            # No frame detected
            return None

        # Select result with the highest metric
        best_idx = np.argmax(metrics)
        best_res = results[best_idx]
        return best_res
        


class SCFrameDetector(FrameDetector):
    """
    Detect frames using the Schmidl and Cox algorithm. TODO
    
    See:
        T. M. Schmidl and D. C. Cox, “Robust frequency and timing synchronization for OFDM,” IEEE Transactions 
        on Communications, vol. 45, no. 12, pp. 1613-1621, 1997, doi: 10.1109/26.650240.
        https://doi.org/10.1109/26.650240
    """
    def __init__(self):
        return NotImplementedError()