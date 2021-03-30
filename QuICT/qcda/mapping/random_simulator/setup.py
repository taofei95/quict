import numpy
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_module = Extension(
    "random_simulator",
    ["random_simulator.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(ext_modules = cythonize(ext_module))

