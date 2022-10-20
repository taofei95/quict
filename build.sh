#!/usr/bin/env bash


print_segment () {
   echo -e "\033[92m================================================================================\033[39m"
}

print_cyan() {
  echo -e "\033[36m$1\033[39m"
}

print_magenta() {
  echo -e "\033[95m$1\033[39m"
}

prj_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

prj_build_dir="$prj_root/build"

OS=$(uname -a)

PYTHON3=$(command -v python3)

# Set root & building directory

print_segment

print_cyan "[Project Root Directory]"

echo "Detected $prj_root as project root directory"

print_segment

print_cyan "[Temporary Building Directory]"

echo "Prepare building directory"

[[ -d $prj_build_dir ]] || mkdir $prj_build_dir

echo "Selected $prj_build_dir as building directory"

# Initialize git submodule if needed

print_segment

print_cyan "[Git Submodule]"

echo "git submodule update --init --recursive"

git submodule update --init --recursive

# Clear older version build.sh remnants

# Set C++ compiler

print_segment
print_cyan "[C/C++ Compiler]"

echo "Selecting C compiler"

[[ $CC == "" ]] && CC=cc
CC=$(command -v $CC)
export CC=$CC

echo "Selected $CC as C compiler"

echo "Selecting C++ compiler"

[[ $CXX == "" ]] && CXX=c++
CXX=$(command -v $CXX)
export CXX=$CXX

echo "Selected $CXX as C++ compiler"

print_segment

print_cyan "[CMake Generator]"

echo "Selecting CMake generator"

if [[ $CMAKE_GENERATOR == "" ]];then
  [[ -x $(command -v make) ]] && cmake_gen="Unix Makefiles"
  [[ -x $(command -v ninja) ]] && cmake_gen="Ninja"
  CMAKE_GENERATOR=$cmake_gen
  export CMAKE_GENERATOR=$CMAKE_GENERATOR
fi

echo "Selected $cmake_gen as CMake generator"

# Build C++ sources in tree

print_segment

print_cyan "[Python Wheel]"

echo "Building python wheel"


print_segment

cd $prj_root && \
$PYTHON3 ./setup.py bdist_wheel "$@" || exit 1

print_magenta "Done."

