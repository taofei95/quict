#!/bin/bash

prj_root=$(pwd)

prj_build_dir="$prj_root/build"

OS=$(uname -a)

PYTHON3=$(which python3)

tbb_build_dir=""

for dir in ./build/oneTBB/build/*; do
  if [[ -d $dir ]] && [[ $dir == *"_release" ]]; then
    tbb_build_dir=$dir
  fi
done

[[ tbb_build_dir == "" ]] && echo "No tbb built!" && exit 1

echo "Installing TBB"

if [[ $OS =~ "Darwin" ]];then
  cp $tbb_build_dir/libtbb.dylib /usr/local/lib
elif [[ $OS =~ "Linux" ]]; then
  cp $tbb_build_dir/*.so /usr/lib
  cp $tbb_build_dir/*.so.2 /usr/lib
fi

cd $prj_build_dir && \
  $PYTHON3 ../setup.py install
