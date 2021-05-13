import os

import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

file_path = os.path.realpath(__file__)
dir, _ = os.path.split(file_path)

ext_module = Extension(
    "mcts_wrapper",
    ["mcts_wrapper.pyx"],
    extra_compile_args = ["-std=c++14",
                           "-I{}/lib/include/".format(dir)
                         # "-I /home/shoulifu/libtorch/include/torch/csrc/api/include",
                         # "-I /home/shoulifu/libtorch/include"
                         ],
    extra_link_args = ["-L{}/lib/build/".format(dir)],
    include_dirs=[numpy.get_include()],
    libraries=["mcts"],
    runtime_library_dirs=["{}/lib/build".format(dir)]
)

setup(ext_modules=cythonize(ext_module))
