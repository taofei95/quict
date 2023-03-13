---
hide:
  - navigation
---

## QuICT 安装说明

### 预先准备

- PYTHON VERSION >= 3.7
- GPU环境要求
  
    - Cupy: [Installing Cupy](https://docs.cupy.dev/en/stable/install.html)
        
        ```sh
        nvcc -V     # 获得cuda版本号
        pip install cupy-cuda{version}      # 根据cuda版本号进行安装
        ```

### 从 pypi 安装

``` shell
pip install quict
```

### 从Gitee处安装

- 克隆 QuICT 仓库

    ``` sh
    # git clone
    git clone https://gitee.com/quictucas/quict.git
    ```

- QuICT 安装
    
    ``` sh
    # 在quict仓库根目录下
    python setup.py install
    ```

- Docker 构建指令
    
    ``` sh
    # Build QuICT docker for target device [cpu/gpu]
    sudo docker build -t quict/{device} -f dockerfile/{device}.quict.df .
    ```

## QuICT-ml 安装说明

### 预先准备

- PYTHON VERSION >= 3.7

- GPU环境要求

    - Cupy: [Installing Cupy](https://docs.cupy.dev/en/stable/install.html)
        
        ```sh
        nvcc -V     # 获得cuda版本号
        pip install cupy-cuda{version}      # 根据cuda版本号进行安装
        ```
    
    - Pytorch: [Installing Pytorch](https://pytorch.org/get-started/locally/)

- RL mapping额外要求

    - pytorch_geometric: [Installing pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
    - torch_sparse: [Installing torch-sparse](https://pypi.org/project/torch-sparse/)
    - tensorboard:

        ``` shell
        pip install tensorboard
        ```

### 从 pypi 安装

``` shell
pip install quict-ml
```

### 从Gitee处安装

- 克隆 QuICT 仓库

    ``` sh
    # git clone
    git clone https://gitee.com/quictucas/quict-ml.git
    ```

- QuICT 安装
    
    ``` sh
    # 在quict仓库根目录下
    python setup.py install
    ```

## QuICT-sim 安装说明

### 预先准备
- C++ Compiler
    - Windows: [Installing Clang/LLVM for use with Visual Studio](https://devblogs.microsoft.com/cppblog/clang-llvm-support-in-visual-studio/)

    - Linux: `clang/LLVM`
    ```sh
    sudo apt install build-essential clang llvm
    ```

### 从 pypi 安装

``` sh
pip install quict-sim
```

### 从 Gitee 安装
- 克隆 QuICT_smi 仓库
    ```sh
    # git clone
    git clone https://gitee.com/quictucas/quict-sim.git
    ```

- Linux 系统 \
推荐使用 Python venv。在系统范围内安装软件包可能会导致权限错误。以下命令将构建 QuICT 并安装它。如果您在安装时遇到权限错误，请尝试使用 venv 或为 install.sh 附加 --user 标志。

    > 由于低版本 GCC (<=11) 中缺少一些功能，建议使用 clang 构建当前的 QuICT。在未来的版本中，将支持 GCC。
    ```sh
    export CC=clang && export CXX=clang++ && ./build.sh && ./install.sh

    # If you encounter with permission issues during installing, try
    export CC=clang && export CXX=clang++ && ./build.sh && ./install.sh --user
    ```

- Windows 系统 \
推荐使用 clang-cl.exe，它是带有 MSVC CLI 的 clang 编译器。其他编译器可能工作但未经测试。打开“PowerShell"，将工作目录更改为 QuICT 存储库根目录。然后使用以下命令构建：

    ```powershell
    .\build.ps1
    ```
