## 手册目的

开发者手册的编写目的是为简要介绍算法编写者常用的算法构造接口，并简要介绍计算上的一些实现技术，在查看开发者手册之前，需要先阅读用户手册

## 算法开发概述

由QuICT的开发目的，QuICT框架将算法分为四类：

+ 算法（Algorithm），包括经典的Shor算法等
+ 电路合成（Synthesis），包括一些oracle在门级上的实现
+ 电路优化（Optimization），即在电路层面上，优化电路深度、规模、总代价的算法
+ 比特映射（Mapping），将电路映射到某种拓扑结构上的算法，优化目标可能是门数等

### 开发原则

+ 当开发者进行开发时，应先明确算法所属类型，并编写一个基于这四个基类之一的子类，在输入和输出类型混合了电路和参数的情况下，根据哪部分为主体进行分类。
+ 开发者应对算法做一个简要的介绍，包括算法**引用文章**，算法的**参数**，算法的**效果或目的**，算法的**时间复杂度**和简要的实现介绍，同时，开发者应给出相应的单元测试模块，要求尽可能覆盖极端情况并选取较多的随机情况。

### 算法基类与需要用到的类

1、基类介绍

算法（Algorithm）

电路合成（Synthesis）

电路优化（Optimization）

比特映射（Mapping）

2、基类方法run和\_\_run\_\_，开发者创建的子类一般重写\_\_run\_\_，或者同时重写run和\_\_run\_\_

+ 对于输入变量只有参数的情况，run即是对\_\_run\_\_的调用，但在输入参数有电路的情况，run函数原则上先对传入电路进行上锁（不允许在算法进行过程中改变），再调用\_\_run\_\_

+ 对于输出变量只有参数的情况，一般只需重写 \_\_run\_\_，run会返回\_\_run\_\_的返回值，对输出变量主体为电路的情况，\_\_run\_\_返回电路，run返回电路的门，是否重写\_\_run\_\_由算法本身决定，可以直接看这四个基类的代码

3、电路构造中需要用到的类GateBuilder和枚举类型GateType

+ 从电路的gates数组中获取的门gate有基类BasicGate，通过调用函数type()会返回枚举类型GateType，更多相关接口可以查看models中的_gate.py文件

```python
@staticmethod
def type():
	"""
	:return: 返回H
	"""
	return GateType.H

class GateType(Enum):
    Error = -1
    H = 0
    S = 1
    S_dagger = 2
    X = 3
    Y = 4
    Z = 5
    ID = 6
    U1 = 7
    U2 = 8
    U3 = 9
    Rx = 10
    Ry = 11
    Rz = 12
    T = 13
    T_dagger = 14
    CZ = 15
    CX = 16
    CH = 17
    CRz = 18
    CCX = 19
    Measure = 20
    Swap = 21
    Perm = 22
    Custom = 23
```


+ 门的构造器GateBuilder。在生成电路的门的时候，尽管可以生成一个电路，再用｜语法生成门，再从电路中取出来，但这种方法可能在开发的时候并不明确，所以对于开发者的电路生成，提供了一个构造器GateBuilder，它有以下几个重要的函数:

  + **setGateType(type)**，给予一个接下来生成门的类型(GateType)

  + **setPargs(pargs)**，给出参数(list或者int)

  + **setTargs(targs)**，给出作用位索引(list或者int)

  + **setCargs(cargs)**，给出控制位索引(list或者int)

  + **getCargsNumber**，对于控制位数明确的门，获取控制位数

  + **getTargsNumber**，对于作用位数明确的门，获取作用位数

  + **getParamsNumber**，对于参数个数明确的门，获取参数数量

  + **getGate**，通过给予GateBuilder的参数获取对应的一个门
  
+ 使用GateBuilder构建门的一个实例可见QuICT.algorithm中_alter_depth_decomposition.py文件的\_\_run\_\_部分

```python
@staticmethod
def __run__(circuit : Circuit):
	matrix = read(circuit)
	solve(matrix)
	gates = []
	GateBuilder.setGateType(GateType.CX)
	for cnot in CNOT:
		GateBuilder.setCargs(cnot[0])
		GateBuilder.setTargs(cnot[1])
		gates.append(GateBuilder.getGate())
	return gates
```



4、算法检查中可能用到的类ETChecker

+ ETChecker可以检验小规模电路的等价变换是否正确，它会随机生成电路，运行算法，并比较算法生成的电路与原电路是否等级啊。当电路位数小于等于7时，它会综合电路的门，比较生成的酉矩阵，当电路位数大于7时，它会生成一个随机单位复向量，进行电路模拟，并比较最后的全振幅(这个过程会重复多次)，它有以下几个重要的函数

  + **setAlgorithm(algorithm)**，设置即需要测试的函数
  + **setDepthNumber(min, max)**，设置电路深度在[min, max]间随机生成
  + **setQubitNumber(min, max)**，设置电路比特数在[min, max]间随机生成
  + **setRoundNumber(round)**，设置随机测试轮数
  + **setTypeList(type_list)**，传递一个GateType的数组，设置可以生成的门的类型

### 后端C++运算原理

为了加速运算，这里做了两个简单的处理

1、最直接的一点，对于向量和矩阵的运算，使用了tbb进行了CPU并行(未来预计使用CUDA C，在N卡上进行GPU并行加速)

2、在python代码中隐含了一个Tangle类，意为纠缠块。一个纠缠块对应一个Qureg和其振幅，初始状态下，一个纠缠块对应一个Qubit，而当Qubit相互作用的时候，纠缠块才会进行合并。当对纠缠块中的一个Qubit作用Measure门后，它又会从纠缠块中移除。通过这种手段，某些电路在运算可以有较大的速度提升。同时，在物理设备上，经过测量的Qubit往往在同一次运算中无法再使用，但模拟的时候进行再利用也往往能获取一些好处。例如在Shor算法中，需要作用IQFT的Qureg可以在一个Qubit上重复进行(IQFT有单向控制的特性)，从而将所需的比特数减少(n - 1)位而不改变运算结果，可以参见Example中的Shor.py，并与QIQC中的电路进行比较
