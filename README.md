[![](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](./LICENSE) ![](https://img.shields.io/badge/platform-windows_|_linux-lightgrey) ![](https://img.shields.io/badge/Python-3.7_|_3.8_|_3.9-blue) ![](https://img.shields.io/badge/version-v1.0.0-blue) ![](https://img.shields.io/badge/Docs-failed-red) [![](https://img.shields.io/badge/UI-Ready-gree)](http://49.235.108.172:8080/) ![](https://img.shields.io/badge/UnitTest-pass-gree) ![](https://img.shields.io/badge/Pypi-v1.0.0-blue) ![](https://img.shields.io/badge/Build-Clang++-orange) ![](https://img.shields.io/badge/Docker-CPU_|_GPU-orange)
<div align=center><img src="./docs/zh/docs/assets/images/IMG_1986.PNG"></div>

## QuICT 平台简介
QuICT (Quantum Computer of Institute of Computing Technology)是一个开源量子计算操作平台。目前QuICT已能支持6种常见指令集以及20余种量子门操作，已实现3种不同类型的量子电路模拟器，并且都有对CPU/GPU的支持和加速，可进行含噪声量子电路仿真模拟。设计并实现了QCDA(Quantum Circuit Design Automation)量子电路辅助设计模块，包括量子初态制备、指令集转换、酉矩阵合成、量子电路优化和映射等功能。算法方面，实现了shor、grover、qae等常见量子算法，可进行因数分解、SAT问题求解等；也实现了QML领域内的QAOA 和 QNN算法，支持图求解最大割问题和MNIST手写数字图片二分类。 QuICT同时也构建了量子算法电路库和针对量子计算机的性能基准测试，通过设计不同的量子电路赛道，来实现针对不同量子机特性的基准测试。

<div align=center><img src="./docs/zh/docs/assets/images/quictv1.drawio.png" width="400" height="400"></div>

主要模块
- QuICT.algorithm: 包含多种常见量子算法，例如shor, grover, qaoa, vqe等。
- QuICT.core: 包含构建电路所需的组件，Circuit, Gates, Qubits等。
- QuICT.qcda: 量子电路生成、优化和映射。
- QuICT.simulation: 量子电路模拟器，支持 Unitary、StateVector、DensityMatrix。
- QuICT.tools: 辅助模块，包括画图、QASM转换、量子电路库以及Benchmark等。

相关链接
- 代码库：https://gitee.com/quictucas/quict
- 文档：https://pypi.org/project/quict/
- Pypi：https://pypi.org/project/quict/
- UI：http://49.235.108.172:8080/

## 安装说明
### 预先准备
- C++ Compiler
    - Windows: [Installing Clang/LLVM for use with Visual Studio](https://devblogs.microsoft.com/cppblog/clang-llvm-support-in-visual-studio/)
    - Linux: `clang/LLVM`
        ```sh
        sudo apt install build-essential clang llvm
        ```
- GPU required
    - Cupy: [Installing Cupy](https://docs.cupy.dev/en/stable/install.html)
        ```sh
        nvcc -V     # 获得cuda版本号

        pip install cupy-cuda{version}      # 根据cuda版本号进行安装
        ```

- Quantum Machine Learning required
    - PyTorch: [Installing PyTorch](https://pytorch.org/get-started/locally/)


### 从 pypi 安装
```
pip install quict
```

### 从Gitee处安装
- 克隆 QuICT 仓库
    ```sh
    # git clone
    git clone https://gitee.com/quictucas/quict.git
    ```

- Linux 系统 \
推荐使用 Python venv。在系统范围内安装软件包可能会导致权限错误。以下命令将构建 QuICT 并安装它。如果您在安装时遇到权限错误，请尝试使用 venv 或为 install.sh 附加 --user 标志。
    > 由于低版本 GCC (<=11) 中缺少一些功能，建议使用 clang 构建当前的 QuICT。在未来的版本中，将支持 GCC。
    ```sh
    export CC=clang && export CXX=clang++ && ./build.sh && ./install.sh

    # If you are encountered with permission issues during installing, try
    export CC=clang && export CXX=clang++ && ./build.sh && ./install.sh --user
    ```

- Windows 系统 \
推荐使用 clang-cl.exe，它是带有 MSVC CLI 的 clang 编译器。其他编译器可能工作但未经测试。打开“PowerShell"，将工作目录更改为 QuICT 存储库根目录。然后使用以下命令构建：

    ```powershell
    .\build.ps1
    ```

- Docker 构建指令
    ```sh
    # Build QuICT docker for target device [cpu/gpu]
    sudo docker build -t quict/{device} -f dockerfile/{device}.quict.df .
    ```

### QuICT 命令行界面

```sh
quict --help
```

## 使用示例
在 quict/example 下，有关于QuICT各个主要模块的用例，另外在[Tutorial](https://gitee.com/quictucas/quict)中会有更详细的教程说明。

这里是一个简单例子关于QuICT的电路构建。

```python
from QuICT.core import Circuit

# Construct the circuit with 5 qubits and 20 random gates.
circuit = Circuit(5)
circuit.random_append(20)

print(circuit.qasm())
```

## 作者及引用
作者为量子计算和理论计算机科学实验室, 中国科学院计算技术研究所。如果您使用QuICT，请按照[此文件](./citation)进行引用

## 开源协议
Copyright (c) Institute of Computing Technology, Chinese Academy of Sciences. All rights reserved.

Licensed under the Apache 2.0 License.
