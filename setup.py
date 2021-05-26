#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2019/12/13 1:32 下午
# @Author  : Han Yu
# @File    : setup.py.py

import argparse

from os import path, getcwd, system
from datetime import datetime
from setuptools import setup
from setuptools import find_packages
import platform

py_file_path = path.dirname(path.abspath(__file__))

cwd = getcwd()

prj_root_relative = "." if cwd == py_file_path else ".."

print(f"Project root: {prj_root_relative}")

packages = find_packages(where=prj_root_relative)

print(f"Found packages: {packages}")

if platform.system() == 'Linux':
    lib1 = f"{prj_root_relative}/QuICT/qcda/mapping/mcts/mcts_core/mcts_wrapper.cpython-38-x86_64-linux-gnu.so"
    lib2 = f"{prj_root_relative}/QuICT/qcda/mapping/mcts/mcts_core/lib/build/libmcts.so"
else:
    lib1 = f"{prj_root_relative}/QuICT/qcda/mapping/mcts/mcts_core/mcts_wrapper.cpython-38-darwin.so"
    lib2 = f"{prj_root_relative}/QuICT/qcda/mapping/mcts/mcts_core/lib/build/libmcts.dylib"
    system(f"install_name_tool -add_rpath {path.dirname(lib2)} {lib1}")

# static file
file_data = [
    ("QuICT/backends", [f"{prj_root_relative}/QuICT/backends/quick_operator_cdll.so"]),
    ("QuICT/lib/qasm/libs", [f"{prj_root_relative}/QuICT/lib/qasm/libs/qelib1.inc"]),
    ("QuICT/qcda/synthesis/initial_state_preparation",
     [f"{prj_root_relative}/QuICT/qcda/synthesis/initial_state_preparation/initial_state_preparation_cdll.so"],
     ),
    ("QuICT/qcda/mapping/mcts/mcts_core",
     [lib1]
    ),
    ("QuICT/qcda/mapping/mcts/mcts_core/lib/build",
     [lib2]
    )
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

cur_time_str = datetime.now().strftime("%y%m%d_%H%M%S")
cur_build_str = f"_{cur_time_str}"
if path.isfile(f"{prj_root_relative}/.test_build"):
    about["__version__"] += cur_build_str

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    author=about["__author__"],
    author_email=about["__email__"],
    url=about["__url__"],
    packages=packages,
    data_files=file_data,
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=requires,
    zip_safe=False,
    package_dir={"QuICT": f"{prj_root_relative}/QuICT/"}
)
