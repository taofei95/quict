#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2019/12/13 1:32 下午
# @Author  : Han Yu
# @File    : setup.py.py

from setuptools import setup
from setuptools import find_packages

packages = find_packages()

# 静态文件
file_data = [
    ("QuICT/backends", ["QuICT/backends/quick_operator_cdll.so"]),
    ("QuICT/lib/qasm/libs", ["QuICT/lib/qasm/libs/qelib1.inc"]),
    ("QuICT/synthesis/initial_state_preparation", ["QuICT/synthesis/initial_state_preparation/initial_state_preparation_cdll.so"]),
]

# 第三方库依赖
requires = ['scipy']

# version信息
about = {}
with open('./QuICT/__version__.py', 'r') as f:
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
