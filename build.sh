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

  NPROC=4

  if [[ $OS =~ "Linux" ]]; then
    NPROC=$(grep -c ^processor /proc/cpuinfo)
  elif [[ $OS =~ "Darwin" ]]; then
    NPROC=$(sysctl hw.ncpu | awk '{print $$2}')
  fi

  NPROC=$($PYTHON3 -c "print(int($NPROC/2))")

  echo "Building with parallel parameter: $NPROC"

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

echo "Build C++ sources in tree"

print_segment

if [[ $OS =~ "Darwin" ]];then
  tbb_include_dir="$tbb_src_dir/include"
  cd ./QuICT/backends && \
  $CXX \
  -o quick_operator_cdll.so dll.cpp \
  -std=c++11  -fPIC \
  -shared  -I$tbb_include_dir -ltbb -L$tbb_build_dir &&
  install_name_tool -add_rpath $tbb_build_dir quick_operator_cdll.so
  cd ..  || exit 1

  cd ./qcda/synthesis/initial_state_preparation && \
  $CXX \
    -o initial_state_preparation_cdll.so initial_state_preparation.cpp \
    -std=c++11  -fPIC -shared  -I$tbb_include_dir -ltbb -L$tbb_build_dir  || exit 1
  install_name_tool -add_rpath $tbb_build_dir initial_state_preparation_cdll.so

  cd $prj_root/QuICT/qcda/mapping/mcts/mcts_core && ./build.sh  || exit 1
else
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

  cd $prj_root/QuICT/qcda/mapping/mcts/mcts_core && chmod u+x ./build.sh && ./build.sh  || exit 1
fi


print_segment

echo "Building python egg"

print_segment

# test_build file indicator
test_build_file="$prj_root/.test_build"
cmd_test_arg=$1
[[ $cmd_test_arg == "--test" ]] && [[ ! -f "$test_build_file" ]] && echo "build a test version" > "$test_build_file"
[[ $cmd_test_arg == "" ]] && [[ -f "$test_build_file" ]] && rm "$test_build_file"

cd $prj_build_dir && \
$PYTHON3 ../setup.py build
