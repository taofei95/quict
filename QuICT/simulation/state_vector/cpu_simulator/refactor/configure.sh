#!/bin/bash

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

[ "$script_dir" != $(pwd) ] && echo "Changing directory into ${script_dir}" && cd $script_dir

[ -d build ] || mkdir build 

generator="Unix Makefiles"
[[ -x $(command -v ninja) ]] && generator="Ninja"

[[ $CC = "" ]] && [[ -x $(command -v clang) ]] && export CC="clang"

[[ $CXX = "" ]] && [[ -x $(command -v clang++) ]] && export CXX="clang++"

cd build && cmake -G$generator -DCMAKE_EXPORT_COMPILE_COMMANDS=1 $@ ..
