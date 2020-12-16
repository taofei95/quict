# QuICT

<<<<<<< HEAD
## 安装

需要Python 3.7的版本支持, 依赖**numpy**库，**目前只支持Linux和OS X系统运行**

#### 安装方式

运行QuICT根目录下的**install.sh**脚本进行安装

#### 注意事项

**install.sh**安装脚本中的最后一个语句如下:

```shell
sudo python3 setup.py install
```

该语句默认python3链接到了python3.7，若提示python3不存在，可通过以下命令查看依赖

```shell
python --version
```

在python指向python3.7的情形下，将**install.sh**中最后语句改为

```shell
sudo python setup.py install
```

即可解决此问题，否则请先手动配置python3.7环境

若提示不存在**numpy**模块，请手动安装模块
=======
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
>>>>>>> refactoring
