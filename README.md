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

> Our helper scripts would use `command`, `uname`, `grep` and `sed`. Install them if they are not equipped.

### Build & Install QuICT

Following commands would build QuICT and install it system-wide.
You might need "sudo" privileges to install QuICT into system python package path.

```
./build.sh && ./install.sh
```

If you want to install QuICT without root access, try

```
./build.sh && ./install.sh --user
```
