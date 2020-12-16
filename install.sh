#!/bin/bash
<<<<<<< HEAD
cd ./tbb-2020.1
sudo rm release_path.sh
make
. ./release_path.sh
a=`uname  -a`
b="Darwin"
if [[ $a =~ $b ]];then
    sudo cp libtbb.dylib /usr/local/lib
else
    sudo cp *.so /usr/lib
    sudo cp *.so.2 /usr/lib
fi
cd ../../../QuICT/backends
g++ -std=c++11 dll.cpp -fPIC -shared -o quick_operator_cdll.so -I . -ltbb
cd ../synthesis/initial_state_preparation
g++ -std=c++11 _initial_state_preparation.cpp -fPIC -shared -o initial_state_preparation_cdll.so -I ../../backends -ltbb
cd ../../../

cd ./QuICT/mapping/lib
sudo python setup.py build_ext --inplace 
if [ -e ~/.bash_profile ]
then
    echo 
else
    touch ~/.bash_profile
fi

echo "export PYTHONPATH=$(pwd)/build/lib.linux-x86_64-3.7/" >> ~/.bash_profile
export PYTHONPATH=$(pwd)/build/lib.linux-x86_64-3.7/
cd ../../../

sudo python3 setup.py install
=======

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
  $PYTHON3 ../setup.py install
>>>>>>> origin/main
