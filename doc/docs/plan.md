## 总体规划

### 框架更新

1、提高框架效率:

 + 框架本身计算效率
     + 在向量运算上，使用GPU并行
     + 改进模拟算法
 + 框架对算法开发者的支持
    + 改进检测算法
    + 添加一些向量运算的API，供算法开发者调用

2、框架拓展

+ 本地的图形化展示
+ 通用化处理，暂定输出HiQ、IBMQ的代码，以及与OPENQASM的相互转化
+ **效率测试模块！**（在随机电路上的效果，某些算法最多能跑多少位，作为比较的依据）

### 算法开发	

+ 量子算法（Shor等）
+ 量子电路设计、优化（经典算法，如CNOT电路的化简等）
+ 量子门分解（酉矩阵的分解等）

### 服务器部署

1、页面前端开发

2、数据管理

3、**框架面向服务器的优化**，包括存储、线程等等



## 2020年3月15日计划

**服务器部署**

1、页面前端开发在2～3周之内初步完成，初步完成的工作包括图形化的门拖拽与网页端的代码编写，即IBMQ网页上的当前两个功能

**框架更新（2周计划）**

1、完成代码与OPENQASM的转化

2、对 IBMQ 支持的电路，添加 API，使得可以直接连接到 IBMQ 服务器进行运算(通过用户名和密码)，返回结果

3、完成本地图形化工作 (本地绘制电路图等)

4、准备与服务器部署的前端对接

## 2020年3月29日计划（3～4周）
**服务器部署**

1、完成服务器部署对接，部署在雁栖湖服务器上

**框架更新**

1、单振幅模块算法调研、编写，效率测试

[1]Simulation of low-depth quantum circuits as complex undirected graphical models, S. Boixo, S. V. Isakov, V. N. Smelyanskiy, and H. Neven, arXiv e-prints (2017), arXiv:1712.05384 [quant-ph] .

[2]Classical Simulation of Intermediate-Size Quantum Circuits, Jianxin Chen, Fang Zhang, Cupjin Huang, Michael Newman, Yaoyun Shi, arXiv:1805.01450 [quant-ph]

[3]A Complete Anytime Algorithm for Treewidth, Vibhav Gogate and Rina Dechter, arXiv:1207.4109

2、酉矩阵分解算法调研、编写

[1]The Solovay-Kitaev algorithm, Christopher M. Dawson, Michael A. Nielsen, arXiv:0505030
[quant-ph].
[2]Optimising the Solovay-Kitaev algorithm, Pham Tien Trung, Rodney Van Meter, Clare
Horsman, arXiv:1209.4139 [quant-ph].
