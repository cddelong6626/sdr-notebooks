
import numpy as np
from scipy import signal


def zadoff_chu(N_zc: int, q: int=1):
    """
    Generate a Zadoff-Chu sequence with sequence length N_zc and root index q

    See https://arxiv.org/pdf/2211.05702
    """

    if N_zc % 2 == 0:
        raise ValueError("Zadoff-Chu sequence length N_zc must be an odd number.")
    if q < 0 or q > (N_zc-1):
        raise ValueError("Zadoff-Chu root index q must be between an odd number from 1 to (N_zc-1).")
    
    n = np.arange(N_zc)
    j = complex(0, 1)
    return np.exp(-j*np.pi*q*n*(n+1)/N_zc)


def to_frames(preamble: np.ndarray, payload: np.ndarray, n: int):
    """Convert payload and preamble to an array of frames"""

    # Spilt payload into n-sized chunks. One per frame
    if len(payload) % n != 0:
        raise ValueError("Payload not n-divisble.")
    payload = payload.reshape((-1, n))

    # Create frames: preamble + payload
    frames = []
    for p in payload:
        frame = np.concatenate([preamble, p])
        frames.append(frame)
    return frames


class FramingStateMachine:
    def __init__(self, preamble: np.ndarray, expected_frame_length: int, detection_threshold: float=0.8):
        self.preamble = preamble
        self.preamble_norm = np.sum(np.abs(self.preamble) ** 2)
        self.matched_filter = self.preamble[::-1].conj()
 
        self.expected_frame_length = expected_frame_length
        self.detection_threshold = detection_threshold

        self.state = "SEARCH"
        self.buffer = []
        
        self.test = None
            
    def update(self, new_samples: np.ndarray):
        self.buffer.extend(new_samples)
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
                frame = self.buffer[:self.expected_frame_length]
                frames.append(frame)
                self.buffer = self.buffer[self.expected_frame_length:]
                self.state = "SEARCH"
                continue

        return frames
    
    def detect_preamble(self):
        """Apply matched filter to signal and report the index of the first spike"""
        matched = signal.convolve(self.matched_filter, self.buffer, mode='valid')

        # Compute L2 norm of sliding window for normalization
        window_norm = signal.convolve(np.ones_like(self.matched_filter), np.abs(self.buffer)**2, mode='valid')
        normalization = (self.preamble_norm * window_norm).astype(np.float32)

        # Avoid division by zero
        normalization[normalization == 0] = 1e-12

        # Metric = Energy of matched / (product of energies of filter and window)
        # This makes detection more robust to noise and varying SNRs
        metric = (np.abs(matched) ** 2) / normalization

        peaks = np.where(metric > self.detection_threshold)

        if self.test is None:
            self.test = metric

        return peaks[0][0] if len(peaks[0]) > 0 else None
