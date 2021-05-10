#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2019/12/13 1:32 下午
# @Author  : Han Yu
# @File    : setup.py.py
import os
from os import path, getcwd, listdir
from setuptools import setup
from setuptools import find_packages

py_file_path = path.dirname(path.abspath(__file__))

cwd = getcwd()

prj_root_relative = "." if cwd == py_file_path else ".."

print("="*40)
print(f"Project root: {prj_root_relative}")

packages = find_packages(where=prj_root_relative)

print("="*40)
print(f"Found packages: {packages}")

print("="*40)
print("Searching dynamic libraries")

graph_structure_build_dir = f"{prj_root_relative}/build/graph_structure"
graph_structure_build_dir_files = listdir(graph_structure_build_dir)
graph_lib_name = ""
for file in graph_structure_build_dir_files:
    if file.startswith("graph_structure"):
        # print(file)
        graph_lib_name = file
        print(graph_lib_name)
        break

print("="*40)

# static file
file_data = [
    ("QuICT/utility/graph_structure", [f"{graph_structure_build_dir}/{graph_lib_name}"]),
    ("QuICT/backends", [f"{prj_root_relative}/QuICT/backends/quick_operator_cdll.so"]),
    ("QuICT/lib/qasm/libs", [f"{prj_root_relative}/QuICT/lib/qasm/libs/qelib1.inc"]),
    ("QuICT/qcda/synthesis/initial_state_preparation",
     [f"{prj_root_relative}/QuICT/qcda/synthesis/initial_state_preparation/initial_state_preparation_cdll.so"],
     ),
]

# 3rd party library
requires = ['scipy']

# version information
about = {}

with open(f"{prj_root_relative}/QuICT/__version__.py", 'r') as f:
    exec(f.read(), about)

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
    python_requires=">=3.0",
    install_requires=requires,
    zip_safe=False,
    package_dir={"QuICT": f"{prj_root_relative}/QuICT/"}
)
print("="*40)
