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

#print_segment
#
#print_cyan "[Clear Remnants]"
#
#echo "Deleting useless files in source code tree created by older version of build.sh"
#
#[[ -f "$prj_root/QuICT/backends/quick_operator_cdll.so" ]] && \
#echo "Deleting $prj_root/QuICT/backends/quick_operator_cdll.so" && \
#rm "$prj_root/QuICT/backends/quick_operator_cdll.so"
#
#[[ -f "$prj_root/QuICT/qcda/synthesis/initial_state_preparation/initial_state_preparation_cdll.so" ]] && \
#echo "Deleting $prj_root/QuICT/qcda/synthesis/initial_state_preparation/initial_state_preparation_cdll.so" && \
#rm "$prj_root/QuICT/qcda/synthesis/initial_state_preparation/initial_state_preparation_cdll.so"

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
  [[ -x $(command -v ninja) ]] && cmake_gen="Ninja"
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

#if [[ $OS =~ "Darwin" ]];then
#  cd "$prj_root"/QuICT/qcda/mapping/mcts/mcts_core && ./build.sh  || exit 1
#else
#  cd "$prj_root"/QuICT/qcda/mapping/mcts/mcts_core && chmod u+x ./build.sh && ./build.sh  || exit 1
#fi

cd $prj_root && \
$PYTHON3 ./setup.py build "$@" || exit 1

#print_cyan "[Copying Back]"
#
#echo -e "Copying built libraries back into source code tree to help run pytest\n"
#
#find "$prj_root/build/" -type f -name "*.so" | while read file
#do
#    dest=$(echo "$file" | grep -oE "build.*" | grep -oE "QuICT.*" ) || exit 1
#    dest="$prj_root/$dest"
#    dest=$(echo $dest | sed -E "s/[^/]*\.so//")
#    echo -e "cp $file $dest\n"
#    cp "$file" "$dest" || exit 1
#done

print_magenta "Done."

