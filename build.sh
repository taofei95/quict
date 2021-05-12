#!/bin/bash

prj_root=$(pwd)

prj_build_dir="$prj_root/build"

OS=$(uname -a)

PYTHON3=$(which python3)

print_segment () {
   echo -e "\033[92m================================================================================\033[39m"
}

print_cyan() {
  echo -e "\033[36m$1\033[39m"
}

# Set building directory

print_segment
print_cyan "[Temporary Building Directory]"

echo "Prepare building directory"

[[ -d $prj_build_dir ]] || mkdir $prj_build_dir

echo "Selected $prj_build_dir as building directory"

# Set C++ compiler

print_segment
print_cyan "[C/C++ Compiler]"

echo "Selecting C compiler"

[[ $CC == "" ]] && CC=gcc
CC=$(command -v $CC)
export CC=$CC

echo "Selected $CC as C compiler"

echo "Selecting C++ compiler"

[[ $CXX == "" ]] && CXX=g++
CXX=$(command -v $CXX)
export CXX=$CXX

echo "Selected $CXX as C++ compiler"

print_segment

print_cyan "[CMake Generator]"

echo "Selecting CMake generator"

if [[ $CMAKE_GENERATOR == "" ]];then
  [[ -x $(command -v make) ]] && cmake_gen="Unix Makefiles"
#  [[ -x $(command -v ninja) ]] && cmake_gen="Ninja"
  CMAKE_GENERATOR=$cmake_gen
  export CMAKE_GENERATOR=$CMAKE_GENERATOR
fi

echo "Selected $cmake_gen as CMake generator"

print_segment

NPROC=4

if [[ $OS =~ "Linux" ]]; then
  NPROC=$(grep -c ^processor /proc/cpuinfo)
elif [[ $OS =~ "Darwin" ]]; then
  NPROC=$(sysctl hw.ncpu | awk '{print $$2}')
fi

NPROC=$($PYTHON3 -c "print(int($NPROC/2))")

export CMAKE_BUILD_PARALLEL_LEVEL=$NPROC

echo "Building with parallel parameter: $NPROC"

# Build TBB

if [[ $OS =~ "Darwin" ]];then
  print_segment

  echo "Building TBB from source"

  print_segment

  tbb_src_dir="$prj_build_dir/oneTBB"

  if ! [[ -d $tbb_src_dir ]]; then
    mkdir -p $tbb_src_dir
    git clone -b tbb_2020 https://github.com/oneapi-src/oneTBB.git $tbb_src_dir
  fi

  echo "Detecting protential parallel"

  # possible build failure
  cd $tbb_src_dir && \
    make -j$NPROC && \
    cd $prj_root || exit 1

  tbb_build_dir=""

  for dir in "$tbb_src_dir/build/"*; do
    if [[ -d $dir ]] && [[ $dir == *"_release" ]]; then
      tbb_build_dir=$dir
    fi
  done

  [[ $tbb_build_dir == "" ]] && echo "TBB build directory error! Exit." && exit 1

  echo "TBB built in $tbb_build_dir"
fi

# Build C++ sources in tree

print_segment

print_cyan "[Python Egg]"

echo "Building python egg"

print_segment

cd $prj_build_dir && \
$PYTHON3 ../setup.py build --parallel $NPROC