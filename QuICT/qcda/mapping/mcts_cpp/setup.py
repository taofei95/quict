from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

#from skbuild import setup

# setup(
#     name="mcts_cpp",
#     description="a minimal example package (cython version)",
#     author='shou lifu',
#     license="MIT",
#     packages=['mcts_cpp'],
#     # The extra '/' was *only* added to check that scikit-build can handle it.
#     package_dir={'mcts_cpp': '/'},
# )

ext_module = Extension(
    "mcts_wrapper",
    ["mcts_wrapper.pyx"],
    include_dirs=[numpy.get_include()],
    libraries = ["mcts"],
)

setup(ext_modules = cythonize(ext_module))