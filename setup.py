"""
Some codes below are from pybind11 cmake example:
https://github.com/pybind/cmake_example/blob/0baee7e073a9b3738052f543e6bed412aaa22750/setup.py
"""

import os
import sys
import subprocess
from os import path, getcwd, system
from Cython.Build import cythonize
from setuptools import setup
from setuptools import find_packages, Extension
# from setuptools.command.build_ext import build_ext
from Cython.Distutils.build_ext import build_ext
import platform

import numpy as np

from typing import *


# print helpers
def print_segment():
    print("\033[92m", "=" * 80, "\033[39m", sep="")


def print_cyan(s):
    print(f"\033[36m{s}\033[39m")


def print_magenta(s):
    print(f"\033[95m{s}\033[39m")


def print_yellow(s):
    print(f"\033[33m{s}\033[39m")


def print_if_not_none(s):
    if s:
        print(s)


def print_with_wrapper(header, out_obj):
    if header[0] != '\033':
        if len(header) > 12:
            header = header[:9] + "..."
        if len(header) < 12:
            for i in range(12 - len(header)):
                header += "."

        header = f"\033[36m[{header}]\033[39m "

    if out_obj is None:
        return
    if isinstance(out_obj, str):
        print(header, out_obj)
    else:
        for line in iter(out_obj.readline, b""):
            print(header, line.decode("unicode_escape"), sep="", end="")


def run_with_output_wrapper(header, args, cwd):
    if len(header) > 12:
        header = header[:9] + "..."
    if len(header) < 12:
        for i in range(12 - len(header)):
            header += "."

    header = f"\033[36m[{header}]\033[39m "

    try:
        with subprocess.Popen(
                args=args,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # universal_newlines=True,
        ) as proc:
            print_with_wrapper(header, proc.stdout)
            ret_code = proc.wait()
        if ret_code:
            raise subprocess.CalledProcessError(ret_code, args)
    except:
        proc.kill()
        raise


# Detect if I'm in `root` or `root/build`
py_file_path = path.dirname(path.abspath(__file__))

prj_root_relative = "." if getcwd() == py_file_path else ".."
prj_root = path.abspath(prj_root_relative)

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
class CMakeExtension(Extension):
    def __init__(self, name, source_dir, extra_cmake_macro=None):
        Extension.__init__(self, name, sources=[])
        self.source_dir = os.path.abspath(source_dir)
        self.extra_cmake_macro = extra_cmake_macro


class CythonExtension(Extension):
    def __init__(self, name, cython_sources, extra_compile_args,
                 extra_link_args, libraries,
                 runtime_library_dirs, cmake_dep):
        # self.name = name
        # self.sources = sources

        Extension.__init__(self, name, sources=[])
        self.cython_sources = cython_sources
        self.extra_compile_args = extra_compile_args
        self.extra_link_args = extra_link_args
        # self.include_dirs = include_dirs,
        self.libraries = libraries
        self.runtime_library_dirs = runtime_library_dirs
        self.cmake_dep = cmake_dep


class ExtensionBuild(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            self.cmake_build_extension(ext)
        elif isinstance(ext, CythonExtension):
            self.cython_build_extension(ext)

    def cython_build_extension(self, ext):
        if ext.cmake_dep:
            self.cmake_build_extension(ext.cmake_dep)
        cython_ext = cythonize(Extension(
            ext.name,
            ext.cython_sources,
            extra_compile_args=ext.extra_compile_args,
            extra_link_args=ext.extra_link_args,
            libraries=ext.libraries,
            runtime_library_dirs=ext.runtime_library_dirs,
        ))
        build_ext.build_extension(self, cython_ext[0])

    def cmake_build_extension(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # required for auto-detection of auxiliary "native" libs
        if not ext_dir.endswith(os.path.sep):
            ext_dir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(ext_dir),
            "-DPYTHON_EXECUTABLE={}".format(sys.executable),
            "-DBUILD_VERSION_INFO={}".format(self.distribution.get_version()),
            "-DCMAKE_BUILD_TYPE={}".format(cfg),  # not used on MSVC, but no harm
        ]
        build_args = []

        if self.compiler.compiler_type != "msvc":
            if not cmake_generator:
                cmake_args += ["-GUnix Makefiles"]

        else:

            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}".format(cfg.upper(), ext_dir)
                ]
                build_args += ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += ["-j{}".format(self.parallel)]

        ext_name = ext.name
        if ext_name[-1] == ".":
            ext_name = ext_name[:-1]
        build_temp = f"{self.build_temp}.{ext_name}"
        ext_name_split = ext_name.split(".")
        ext_name = ext_name_split[-1]


        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        print_cyan(f"[{ext_name}]")
        print_with_wrapper(ext_name, " ".join(["cmake", ext.source_dir] + cmake_args))
        if hasattr(self, "parallel") and self.parallel:
            print_yellow("Extensions are built in parallel. Shell output might be messed up.")
        print_with_wrapper(ext_name, "Configuring...")
        run_with_output_wrapper(
            header=ext_name,
            args=["cmake", ext.source_dir] + cmake_args,
            cwd=build_temp,
        )
        print_with_wrapper(ext_name, "Building...")
        run_with_output_wrapper(
            header=ext_name,
            args=["cmake", "--build", "."] + build_args,
            cwd=build_temp,
        )
        print_with_wrapper(ext_name, "Copying back...")
        libs = []
        for f in os.listdir(ext_dir):
            if f.endswith(".so"):
                # print_with_wrapper(ext_name, f"!!!!!!!!!!!{f}")
                libs.append(f"{ext_dir}{f}")
        run_with_output_wrapper(
            header=ext_name,
            args=["cp", " ".join(libs), ext.source_dir],
            cwd=build_temp,
        )


print_segment()
print_cyan("[Project Root]")
print(f"Project root: {prj_root}")

packages = find_packages(where=prj_root_relative)

print_segment()
print_cyan("[Packages]")

print(f"Found packages: {packages}")

print_segment()

packages = find_packages(where=prj_root_relative)

print(f"Found packages: {packages}")

# if platform.system() == 'Linux':
#     lib1 = f"{prj_root_relative}/QuICT/qcda/mapping/mcts/mcts_core/mcts_wrapper.cpython-38-x86_64-linux-gnu.so"
#     lib2 = f"{prj_root_relative}/QuICT/qcda/mapping/mcts/mcts_core/lib/build/libmcts.so"
# else:
#     lib1 = f"{prj_root_relative}/QuICT/qcda/mapping/mcts/mcts_core/mcts_wrapper.cpython-38-darwin.so"
#     lib2 = f"{prj_root_relative}/QuICT/qcda/mapping/mcts/mcts_core/lib/build/libmcts.dylib"
#     system(f"install_name_tool -add_rpath {path.dirname(lib2)} {lib1}")

# static file
file_data = [
    ("QuICT/lib/qasm/libs", [f"{prj_root_relative}/QuICT/lib/qasm/libs/qelib1.inc"]),
    # ("QuICT/qcda/synthesis/initial_state_preparation",
    #  [f"{prj_root_relative}/QuICT/qcda/synthesis/initial_state_preparation/initial_state_preparation_cdll.so"],
    #  ),
    # ("QuICT/qcda/mapping/mcts/mcts_core",
    #  [lib1]
    #  ),
    # ("QuICT/qcda/mapping/mcts/mcts_core/lib/build",
    #  [lib2]
    #  )
]

# 3rd party library
requires = [
    'pytest>=6.2.3',
    'numpy>=1.20.1',
    'networkx>=2.5.1',
    'matplotlib>=3.3.4',
    'cython>=0.29.23',
    'ply>=3.11',
    'scipy',
    'ujson',
]

# version information
about = {}

with open(f"{prj_root_relative}/QuICT/__version__.py", 'r') as f:
    exec(f.read(), about)

print_cyan("[Build Python]")

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    author=about["__author__"],
    author_email=about["__email__"],
    url=about["__url__"],
    package_dir={"QuICT": f"{prj_root_relative}/QuICT/"},
    ext_modules=[
        CMakeExtension("QuICT.utility.graph_structure.", f"{prj_root}/QuICT/utility/graph_structure"),
        CMakeExtension("QuICT.backends.", f"{prj_root}/QuICT/backends"),
        CMakeExtension("QuICT.qcda.synthesis.initial_state_preparation.",
                       f"{prj_root}/QuICT/qcda/synthesis/initial_state_preparation/"),
        CythonExtension(
            "QuICT.qcda.mapping.mcts.mcts_core.mcts_wrapper",
            [f"{prj_root}/QuICT/qcda/mapping/mcts/mcts_core/mcts_wrapper.pyx"],
            extra_compile_args=["-std=c++14",
                                f"-I{prj_root}/QuICT/qcda/mapping/mcts/mcts_core/lib/include/",
                                f"-I{np.get_include()}"
                                ],
            extra_link_args=[f"-L{prj_root}/QuICT/qcda/mapping/mcts/mcts_core/lib/"],
            libraries=["mcts"],
            runtime_library_dirs=[f"{prj_root}/QuICT/qcda/mapping/mcts/mcts_core/lib/"],
            cmake_dep=CMakeExtension(
                "QuICT.qcda.mapping.mcts.mcts_core.lib.",
                f"{prj_root}/QuICT/qcda/mapping/mcts/mcts_core/lib/"
            ),
        ),
    ],
    cmdclass={"build_ext": ExtensionBuild},
    packages=packages,
    data_files=file_data,
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requires,
    zip_safe=False,
)
print_segment()
