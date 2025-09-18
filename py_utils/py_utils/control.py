
import numba
from numba import jit, njit, float64
from numba.experimental import jitclass


spec = [
    ('K_p', float64),
    ('K_i', float64),
    ('K_d', float64),
    ('sum_e', float64),
    ('prev_e', float64),
]
@jitclass(spec)
class PIDFeedback:
    def __init__(self, K_p=0.0, K_i=0.0, K_d=0.0):
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d

        self.sum_e = 0.0
        self.prev_e = 0.0

    def update(self, e):
        self.sum_e += e
        d = e - self.prev_e
        x = (self.K_i * self.sum_e) + (self.K_p * e) + (self.K_d * d)

        self.prev_e = e

        return x
    
    def reset(self):
        self.sum_e = 0.0
        self.prev_e = 0.0