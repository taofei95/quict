#!/bin/bash

prj_root=$(pwd)

prj_build_dir="$prj_root/build"

OS=$(uname -a)

PYTHON3=$(which python3)

if [[ $OS =~ "Darwin" ]];then
  echo "Installing TBB"
  
  tbb_build_dir=""

  for dir in ./build/oneTBB/build/*; do
    if [[ -d $dir ]] && [[ $dir == *"_release" ]]; then
      tbb_build_dir=$dir
    fi
  done

  [[ tbb_build_dir == "" ]] && echo "No tbb built!" && exit 1
  cp $tbb_build_dir/libtbb.dylib /usr/local/lib
fi

cd $prj_build_dir && \
  $PYTHON3 ../setup.py install "$@"
