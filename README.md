# SDR Notebooks

![WIP](https://img.shields.io/badge/status-WIP-orange)

> **Work In Progress**  
> This project is under active development. Core features are incomplete, and the implementation and interface are subject to change.  
> Contributions and issue reports are welcome, but please note that stability is not guaranteed.

This repo contains a collection of Jupyter notebooks for exploring signal processing topics in digital communications and software-defined radio (SDR). It supports algorithm development and design decisions for larger projects like `qpsk-pluto2rtl-pipeline` and `ofdm-pluto2rtl-pipeline`.

## Topics Covered

- Carrier recovery (Costas loop)
- Symbol timing recovery (Gardner, Mueller-Muller, etc.)
- Interpolation techniques (Farrow structure, Linear)
- Modulation/demodulation (QPSK, OFDM)
- Constellation shaping and filtering


## Structure

```
sdr-notebooks/
├── notebooks/                 # Interactive experiments, simulations, and visualizations
│   ├── qpsk_sim.ipynb         # QPSK modulation + timing/sync pipeline simulation
│   ├── ofdm_sim.ipynb         # Combined QPSK/OFDM pipeline simulation
...
│
├── utils/                     # Pure Python DSP + SDR utilities
│   ├── channel.py             # Fading, noise, and multipath models
│   ├── coding.py              # Forward error correction
│   ├── equalization.py        # Linear/ZF/MMSE equalizers
│   ├── filters.py             # Matched filtering, resampling
│   ├── interpolators.py       # Linear/Farrow (cubic Lagrange) interpolation
│   ├── modulation.py          # QPSK/OFDM mod/demod
│   ├── metrics.py             # BER, EVM, etc.
│   ├── carrier_recovery.py    # Costas loop
│   ├── timing_recovery.py     # Gardner, M&M, correlation methods, etc.
│   ├── visualization.py       # Frequency spectrums, constellations, etc.
...
│
├── pybind11_bindings/         # C++ algorithms and Python bindings (via pybind11)
│   ├── dsp/                   # C++ implementations
│   └── python/                # Pybind11 interface code
│
├── CMakeLists.txt             # CMake build file for binding C++ to Python
├── setup.py                   # For pip-installable development
└── README.md                  # This file
```

## Goals

- Use this pipeline as a learning and demonstration tool for SDR systems
- Explore trade-offs in synchronization and equalization to use on real hardware
- Eventually extend to support adaptive equalization and ML-based blocks

## Screenshots

TODO