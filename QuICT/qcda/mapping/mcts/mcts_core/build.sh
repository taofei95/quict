#!/bin/bash
cd ./lib/
if [ ! -d build ]; then
    mkdir build
fi
cd ./build

cmake \
-DTEST_MCTS=off \
-DTEST_RL_MCTS=off \
-DTEST_DATA_GENERATOR=off \
-USE_OpenMP=on\
-DCMAKE_BUILD_TYPE=Release .. 
#-DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \

cmake --build . --config Release
cd ../..
python setup.py build_ext --inplace
SHELL_FOLDER=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
export LD_LIBRARY_PATH=${SHELL_FOLDER}/lib/build${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
