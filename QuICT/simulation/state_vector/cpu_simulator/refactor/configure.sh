#!/bin/bash

script_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

[ "$script_dir" != $(pwd) ] && echo "Changing directory into ${script_dir}" && cd $script_dir

[ -d build ] || mkdir build 

cd build && cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ..
