---
hide:
  - navigation
---

## 安装说明
### 从 pypi 安装
```
pip install quict
```

### 从Gitee处安装
- 预先准备
  - C++ Compiler
    - Windows: [Installing Clang/LLVM for use with Visual Studio](https://devblogs.microsoft.com/cppblog/clang-llvm-support-in-visual-studio/)

    - Linux: `clang/LLVM`
    ```sh
    sudo apt install build-essential libtbb2 libtbb-dev clang llvm python3 python3-setuptools python3-numpy python3-scipy
    # if you handle python parts in another way, just install
    sudo apt install build-essential libtbb2 libtbb-dev clang llvm.
    ```

- 克隆 QuICT 仓库
    ```sh
    # git clone
    git clone https://gitee.com/quictucas/quict.git
    ```

- Linux 系统 \
以下命令将构建 QuICT 并在系统范围内安装它。您可能需要“sudo”权限才能将 QuICT 安装到系统 python 包路径中。
    > 由于低版本 GCC (<=11) 中缺少一些功能，建议使用 clang 构建当前的 QuICT。在未来的版本中，将支持 GCC。
    ```sh
    export CC=clang && export CXX=clang++ && ./build.sh && ./install.sh

    # If you are encountered with permission issues during installing, try
    export CC=clang && export CXX=clang++ && ./build.sh && ./install.sh --user
    ```

- Windows 系统 \
推荐使用 clang-cl.exe，它是带有 MSVC CLI 的 clang 编译器。其他编译器可能工作但未经测试。打开“Developer PowerShell for VS”，将工作目录更改为 QuICT 存储库根目录。然后使用以下命令构建：

    ```powershell
    $ENV:CC="clang-cl.exe"
    $ENV:CXX="clang-cl.exe"
    $ENV:ComSpec="powershell.exe"
    python3 .\setup.py bdist_wheel
    ```

- Docker 构建指令
    ```sh
    # Build QuICT docker for target device [cpu/gpu]
    sudo docker build -t quict/{device} -f dockerfile/{device}.quict.df .
    ```
