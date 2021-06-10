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

echo "rm -rf $prj_build_dir"

rm -rf "$prj_build_dir"

egg_dir="$prj_root/QuICT.egg-info"

[[ -d $egg_dir ]] && echo "rm -rf $egg_dir" && rm -rf $egg_dir

print_magenta "Cleaned."