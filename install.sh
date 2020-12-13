#!/bin/bash

tbb_build_dir=""

for dir in ./oneTBB/build/*; do
  if [[ -d $dir ]] && [[ $dir == *"_release" ]]; then
    tbb_build_dir=$dir
  fi
done

[[ tbb_build_dir == "" ]] && echo "No tbb built!" && exit 1

echo "Installing TBB"

a=`uname  -a`
b="Darwin"
if [[ $a =~ $b ]];then
    sudo cp $tbb_build_dir/libtbb.dylib /usr/local/lib
else
    sudo cp $tbb_build_dir/*.so /usr/lib
    sudo cp $tbb_build_dir/*.so.2 /usr/lib
fi

sudo python3 setup.py install
