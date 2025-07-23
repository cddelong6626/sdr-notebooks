
import numpy as np


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
    pass