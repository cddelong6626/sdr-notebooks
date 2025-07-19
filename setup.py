# setup.py
from setuptools import setup, find_packages

setup(
    name="sdr-notebooks",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "numba", "pybind11"],
    author="Cole Delong",
    description="Sandbox to develop algorithms using Jupyter notebooks for a software defined radio",
)


