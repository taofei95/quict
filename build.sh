#!/bin/bash

print_segment () {
   printf "%0.s=" {1..60}
   echo ""
}

# Set C++ compiler

print_segment

echo "Selecting C++ compiler"

print_segment

CXX=g++
CXX=$(which $CXX)

echo "Selecting $CXX as C++ compiler"

# Build TBB

print_segment

echo "Building TBB"

print_segment

make -C oneTBB

tbb_build_dir=""

for dir in ./oneTBB/build/*; do
  if [[ -d $dir ]] && [[ $dir == *"_release" ]]; then
    tbb_build_dir=$dir
  fi
done

echo "TBB built in $tbb_build_dir"

# Build C++ sources in tree

print_segment

echo "Build C++ sources in tree"

print_segment

cd ./QuICT/backends && \
$CXX \
 -o quick_operator_cdll.so dll.cpp \
 -std=c++11  -fPIC \
 -shared  -I"../../oneTBB/include" -ltbb -L"../../$tbb_build_dir"  || exit 1

cd ../QCDA/synthesis/initial_state_preparation && \
$CXX \
  -o initial_state_preparation_cdll.so initial_state_preparation.cpp \
  -std=c++11  -fPIC -shared  -I"../../../../oneTBB/include" -ltbb -L"../../../../$tbb_build_dir"  || exit 1


print_segment

echo "Building python egg"

print_segment

cd ../../../../ && \
python3 setup.py build
