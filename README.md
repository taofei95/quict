# QuICT

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
