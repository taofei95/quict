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

[ -d $prj_build_dir ] && echo "rm -rf $prj_build_dir" && rm -rf "$prj_build_dir"

egg_dir="$prj_root/quict.egg-info"

[[ -d $egg_dir ]] && echo "rm -rf $egg_dir" && rm -rf $egg_dir

dist_dir="$prj_root/dist"

[[ -d $dist_dir ]] && echo "rm -rf $dist_dir" && rm -rf $dist_dir

echo "Remove all built .so files"

find . -name "*.so" -type f -delete || exit 1

print_magenta "Cleaned."
