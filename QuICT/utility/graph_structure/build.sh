#!/usr/bin/env bash

graph_src_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

prj_root=$(pwd)

prj_build_dir="$prj_root/build"

graph_build_dir="$prj_build_dir/graph_structure"

[[ -d $graph_build_dir ]] && echo "Building dir detected"
[[ -d $graph_build_dir ]] || (echo "No building dir, making one..." && mkdir -p "$graph_build_dir")

cd "$graph_build_dir" || exit 1
cmake --version || (echo "No CMake! Exit." && exit 1)
cmake "$graph_src_dir" -DCMAKE_BUILD_TYPE=Release
cmake --build .