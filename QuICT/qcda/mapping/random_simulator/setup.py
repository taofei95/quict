from setuptools import setup, Extension
from Cython.Build import cythonize

ext_module = Extension(
    "random_simulator",
    ["random_simulator.pyx"],
)

setup(ext_modules = cythonize(ext_module))

