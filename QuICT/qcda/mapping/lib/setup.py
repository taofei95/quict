from setuptools import setup, Extension
from Cython.Build import cythonize

ext_module = Extension(
    "qubit_mapping",
    ["qubit_mapping.pyx"],
    language="c++",
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"]
)

setup(ext_modules = cythonize(ext_module))