# QuICT

### to review the framework, you can check

#### cores

​	the main part of the framework, contains:

- qubit.py
  - implement the quantum bit and quantum register
  - implement the tangle, which is the basic calculation unit  for the amplitude
- circuit.py
  - implement the quantum circuit
- gate.py
  - implement some basic quantum gate 

# How to use
## Install Dependency

You can try `sudo ./dependency.sh` to install dependencies automatically(only Ubuntu and Fedora are supported currently).
If you prefer install python packages using `pip`, just skip setuptools, numpy and scipy in following commands.

To install dependencies on Ubuntu:

```
sudo apt install build-essential libtbb2 libtbb-dev python3 python3-setuptools python3-numpy python3-scipy
```

To install dependencies on Fedora:

```
sudo dnf install make gcc gcc-c++ kernel-devel linux-headers tbb tbb-devel python3 python3-setuptools python3-numpy python3-scipy
```

> Our helper scripts use `which` and `uname` command.

## Build & Install QuICT

Following commands would build QuICT and install it system-wide. If you are going to install it into some virtual python environment, do it without any `sudo`. 

```
./build.sh && \
sudo ./install.sh
```
