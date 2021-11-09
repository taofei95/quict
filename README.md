# QuICT

## How to use

### In case that you have a favor over Docker

Though releasing images on Docker Hub is in our agenda, currently
docker users might need to build docker image from sources.
With the docker file we provide in the repository, one can easily
build a docker image with only a little performance loss. 

```
docker build -t quict .
```

### Install Dependency

You can try `sudo ./dependency.sh` to install dependencies 
automatically(script only supports Ubuntu currently).
If you prefer install python packages using `pip`, just skip 
setuptools, numpy and scipy in following commands.

To install dependencies on Ubuntu:

```
sudo apt install build-essential libtbb2 libtbb-dev clang llvm \
  python3 python3-setuptools python3-numpy python3-scipy
```

> Or just install `sudo apt install build-essential libtbb2 libtbb-dev`
> if you handle python parts in another way.

> Our helper scripts would use `command`, `uname`, `grep` and `sed`. Install them if they are not equipped.

### Build & Install QuICT

**Remove old version before install a new one!**

Following commands would build QuICT and install it system-wide.
You might need "sudo" privileges to install QuICT into system python package path.

> Due to a gcc issue, current version has to be built with clang++ and llvm.
> In future versions, gcc will be supported

```
export CC=clang && export CXX=clang++ && ./build.sh && ./install.sh
```

If you want to install QuICT without root access, try

```
export CC=clang && export CXX=clang++ && ./build.sh && ./install.sh --user
```
