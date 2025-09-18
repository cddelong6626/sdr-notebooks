# SDR Notebooks

This directory contains Jupyter notebooks for exploring, simulating, and visualizing SDR and digital communications algorithms.

---
## Usage

### Install Required Packages

#### Python packages
```sh
cd notebooks
pip install -r requirements.txt
```

#### C++ SDR Library
```sh
cd cpp_utils
./run_build
```

### Open Notebooks
```sh
cd notebooks
python -m notebook
```

---
## Notebooks

- `qpsk_sim.ipynb` — QPSK modulation/demodulation, timing and carrier recovery
- `ofdm_sim.ipynb` — End-to-end OFDM pipeline simulation
- `*_dev.ipynb` - Development notebook

---
## Purpose

- Rapid prototyping and visualization
- Reference for Python and C++ implementations