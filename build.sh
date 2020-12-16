#!/bin/bash

prj_root=$(pwd)

prj_build_dir="$prj_root/build"

OS=$(uname -a)

PYTHON3=$(which python3)

print_segment () {
   printf "%0.s=" {1..60}
   echo ""
}

# Set building directory

print_segment

echo "Prepare building directory"

print_segment

[[ -d $prj_build_dir ]] || mkdir $prj_build_dir

echo "Selecting $prj_build_dir as building directory"

# Set C++ compiler

print_segment

echo "Selecting C++ compiler"

print_segment

[[ $CXX == "" ]] &&  CXX=g++
CXX=$(which $CXX)

echo "Selecting $CXX as C++ compiler"

# Build C++ sources in tree

print_segment

echo "Build C++ sources in tree"

print_segment

cd ./QuICT/backends && \
$CXX \
 -o quick_operator_cdll.so dll.cpp \
 -std=c++11  -fPIC \
 -shared -ltbb && 
 cd ..  || exit 1

cd ./qcda/synthesis/initial_state_preparation && \
$CXX \
  -o initial_state_preparation_cdll.so initial_state_preparation.cpp \
  -std=c++11  -fPIC -shared -ltbb || exit 1


print_segment

echo "Building python egg"

print_segment

cd $prj_build_dir && \
$PYTHON3 ../setup.py build