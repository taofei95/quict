# QuICT

## 安装

需要Python 3.7的版本支持, 依赖 **numpy** 库，**目前只支持Linux和OS X系统运行**

### 安装方式

1. 通过 **Makefile**, 运行如下命令
```
make
sudo make install
```
2. 运行QuICT根目录下的 **install.sh** 脚本进行安装

### 注意事项

#### Makefile

不可以不执行 `make` 而直接执行 `sudo make install`, 这会导致某些子项目的依赖无法正常安装.

对于使用 virtualenv 的用户, 应当在 shell 中激活相应的 virtualenv, 并且在该 shell 下执行 `make` 命令.

可以通过 `sudo make uninstall` 删除当前版本的 QuICT, 这同时会移除在安装过程中被拷贝至系统目录的一些动态库.

在任何情况下, 如果需要重新编译 QuICT, 必须先执行 `make clean`, 再执行 `make`.

#### install.sh

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

