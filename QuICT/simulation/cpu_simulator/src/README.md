Because of a GCC bug, you should compile current simulator using llvm & clang.

Firstly, install llvm & clang.

```
# Assuming you are using ubuntu
sudo apt update 
sudo apt install llvm clang
```

Then specify llvm for cmake.

```
export CC=clang && export CXX=clang++
mkdir cmake-build
cd cmake-build
cmake <source_code_directory>
# Do whatever you want
```