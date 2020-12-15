#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2019/12/13 1:32 下午
# @Author  : Han Yu
# @File    : setup.py.py

from os import path
from setuptools import setup
from setuptools import find_packages


py_file_path = path.dirname(path.abspath(__file__))

packages = find_packages()

# static file
file_data = [
    ("QuICT/backends", [f"{py_file_path}/QuICT/backends/quick_operator_cdll.so"]),
    ("QuICT/lib/qasm/libs", [f"{py_file_path}/QuICT/lib/qasm/libs/qelib1.inc"]),
    ("QuICT/QCDA/synthesis/initial_state_preparation",
     [f"{py_file_path}/QuICT/QCDA/synthesis/initial_state_preparation/initial_state_preparation_cdll.so"],
     ),
]

# 3rd party library
requires = ['scipy']

# version information
about = {}

with open(f"{py_file_path}/QuICT/__version__.py", 'r') as f:
    exec(f.read(), about)

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    author=about["__author__"],
    author_email=about["__email__"],
    url=about["__url__"],
    packages=find_packages(),
    data_files=file_data,
    include_package_data=True,
    python_requires=">=3.0",
    install_requires=requires,
    zip_safe=False,
)
