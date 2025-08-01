
import numpy as np
from scipy import signal



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
    
    This parent class contains the FSM and buffering logic. Child classes must implement a detect_preamble() method
    """
    def __init__(self, preamble: np.ndarray, expected_frame_length: int, detection_threshold: float=0.8):
        self.preamble = preamble
        self.expected_frame_length = expected_frame_length
        self.detection_threshold = detection_threshold

        self.state = "SEARCH"
        self.buffer = np.empty(0, dtype=np.complex128)

        self.debug = None
        
    def process(self, new_samples: np.ndarray):
        """Update the buffer and check if there are any frames detected"""
        self.buffer = np.concatenate((self.buffer, new_samples))
        frames = []

        while True:
            # Pause state machine when buffer can't hold a frame
            if len(self.buffer) < self.expected_frame_length:
                break

            # SEARCH state: Search for preamble within buffer
            if self.state == "SEARCH":
                idx = self.detect_preamble()
                if idx is not None:
                    self.buffer = self.buffer[idx:]
                    self.state = "ACQUIRE"
                    continue
                else:   # If no preamble is detected: pause search
                    self.buffer = self.buffer[-len(self.preamble):]
                    break

            # ACQUIRE state: Add found frame to frames array
            if self.state == "ACQUIRE":
                frame = np.array(self.buffer[:self.expected_frame_length])
                frames.append(frame)
                self.buffer = self.buffer[self.expected_frame_length:]
                self.state = "SEARCH"
                continue

        return frames

    def detect_preamble(self):
        """Return the index of the first detected preamble or None if no preamble is detected"""
        raise NotImplementedError()
    

class CorrelationFrameDetector(FrameDetector):
    """Detect frames using correlation with the complex conjugate of the preamble"""
    def __init__(self, preamble: np.ndarray, expected_frame_length: int, detection_threshold: float=0.8):
        self._preamble = None
        self._preamble_norm = None
        self._matched_filter = None
        super().__init__(preamble, expected_frame_length, detection_threshold)

    @property
    def preamble(self):
        return self._preamble
    
    @preamble.setter
    def preamble(self, value: np.ndarray):
        """Set preamble and related values"""
        self._preamble = value
        self._preamble_norm = np.sum(np.abs(value) ** 2)
        self._matched_filter = value[::-1].conj()
    
    def detect_preamble(self):
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
        peaks = np.where(metric > self.detection_threshold)

        if self.debug is None: # TODO: REMOVE
            self.debug = metric

        return peaks[0][0] if len(peaks[0]) > 0 else None
    
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
    
    def detect_preamble(self):
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

        return peaks[0][0] if len(peaks[0]) > 0 else None
    

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