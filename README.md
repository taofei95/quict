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

## Clone

You should clone this repo with this command:

```
git clone --recurse-submodules <repo_url>
```

If you have already clone this repo, you need to execute:

```
git submodule update --init
```

## Build & Install

```
./build.sh && \
sudo ./install.sh
```
