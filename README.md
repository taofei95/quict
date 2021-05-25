# QuICT

## Project Structure

Goto [detailed explanations](./doc/project_structure.md)

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
automatically(only Ubuntu and Fedora are supported currently).
If you prefer install python packages using `pip`, just skip 
setuptools, numpy and scipy in following commands.

To install dependencies on Ubuntu:

```
sudo apt install build-essential libtbb2 libtbb-dev \ 
  python3 python3-setuptools python3-numpy python3-scipy
```

> Or just install `sudo apt install build-essential libtbb2 libtbb-dev`
> if you handle python parts in another way.

To install dependencies on Fedora:

```
sudo dnf install make gcc gcc-c++ kernel-devel linux-headers tbb tbb-devel \
  python3 python3-setuptools python3-numpy python3-scipy
```

> Or just `sudo dnf install make gcc gcc-c++ kernel-devel linux-headers tbb tbb-devel` 
> if you handle python parts in another way.

> Our helper scripts would use `which`, `uname`, `grep` and `sed`. Install them if they are not equipped.

### Build & Install QuICT

Following commands would build QuICT and install it system-wide(in user directory).

```
./build.sh && ./install.sh
```

You can also specify `--test` flag for `build.sh` to add special post fix for version number.
(This is only useful when testing.)

```
./build.sh --test && ./install.sh
```
