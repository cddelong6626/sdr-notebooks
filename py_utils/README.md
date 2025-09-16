# py_utils — Python SDR/DSP Library

This module contains modular, pure-Python implementations of SDR and DSP algorithms for prototyping and validation.

## Features

- Carrier and timing recovery (Costas, Gardner, Mueller-Muller, etc.)
- Channel models (fading, noise, multipath)
- Modulation/demodulation (QPSK, OFDM)
- Equalization, coding, metrics, visualization, and more

## Usage

```python
from py_utils import modulation, channel, timing_recovery

# Example: Generate QPSK symbols, pass through channel, recover timing
```

## Purpose

- Rapid prototyping and validation
- Reference for C++ ports in `cpp_utils`