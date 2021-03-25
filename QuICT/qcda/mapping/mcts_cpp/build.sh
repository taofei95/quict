#!/bin/bash

cd ./lib/build
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
cmake --build .
cd ../..
python setup.py build_ext --inplace