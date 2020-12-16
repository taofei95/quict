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

You need oneTBB to build and/or run QuICT. To install it on Ubuntu:

```
sudo apt install build-essential libtbb2 libtbb-dev
```

> Our helper scripts use `which` and `uname` command.

## Build & Install QuICT

Following commands would build QuICT and install it system-wide. If you are going to install it into some virtual python environment, do it without any `sudo`. 

```
./build.sh && \
sudo ./install.sh
```
