# cpp_utils — C++ SDR/DSP Library

This project contains high-performance C++ implementations of SDR algorithms, with unit tests and Python bindings.

---
## Structure

- `sdrlib/` — Core C++ library (`src/`, `include/`, `tests/`)
- `py_bindings/` — pybind11 bindings for Python interoperability

---
## Features

- C++ ports of Python algorithms for real-time use
- Unit tests (Google Test)
- Python bindings for seamless integration with Python workflows

---
## Requirements

- **Compiler:** clang
- **Build system:** ninja, cmake
- **Libraries:** KFR DSP library, Google Test, pybind11

### Install dependencies (Ubuntu/Debian example)

```sh
sudo apt update
sudo apt install -y clang ninja-build cmake libgtest-dev libeigen3-dev pybind11-dev
# For KFR DSP library, follow instructions at https://www.kfrlib.com/
```

**Note:** Before building, set the `KFR_PATH` environment variable to the location of your KFR DSP library installation's 'KFRConfig.cmake' file, as required by `./run_build`. For example:

```sh
export KFR_PATH=/path/to/kfr/targets/lib/cmake/kfr
```

After building, Python bindings are automatically added to the python site-packages of your current python env as the module `py_sdrlib` by the build script.

---
## Purpose

- High-performance, real-time SDR algorithms
- Integration with GNU Radio as custom blocks (see pipeline repos)