---
hide:
  - navigation
---

## 安装说明
### 预先准备
- PYTHON VERSION >= 3.7
- GPU环境要求
    - Cupy: [Installing Cupy](https://docs.cupy.dev/en/stable/install.html)
        ```sh
        nvcc -V     # 获得cuda版本号

        pip install cupy-cuda{version}      # 根据cuda版本号进行安装
        ```


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

- QuICT 安装
    ```sh
    # 在quict仓库根目录下
    python setup.py install
    ```

- Docker 构建指令
    ```sh
    # Build QuICT docker for target device [cpu/gpu]
    sudo docker build -t quict/{device} -f dockerfile/{device}.quict.df .
    ```
