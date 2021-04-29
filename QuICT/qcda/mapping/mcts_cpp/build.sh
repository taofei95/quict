#!/bin/bash
cd ./lib/build

cmake \
-DTEST_MCTS=on \
-DTEST_RL_MCTS=off \
-DTEST_DATA_GENERATOR=off \
-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
-DCMAKE_BUILD_TYPE=Release .. 

cmake --build . --config Release
cd ../..
python setup.py build_ext --inplace
