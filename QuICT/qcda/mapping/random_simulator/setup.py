import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

ext_module = Extension(
    "random_simulator",
    ["random_simulator.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(ext_modules=cythonize(ext_module))
