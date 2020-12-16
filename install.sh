#!/bin/bash

prj_root=$(pwd)

prj_build_dir="$prj_root/build"

OS=$(uname -a)

PYTHON3=$(which python3)

cd $prj_build_dir && \
  $PYTHON3 ../setup.py install
