# 量子物理机基准测试

## 介绍

量子计算机是一种基于量子理论而工作的计算机。追根溯源，是对可逆机的不断探索促进了量子计算机的发展，它是不断迭代并不断受到评估的，QuICT物理机benchmark是这样一个依据量子电路的属性、物理机模拟振幅以及测试环境等因素通过设计科学的测试方法和测试系统对执行各项任务的能力进行排名，实现对量子物理设备的基准测试。

## 基准测试框架

通过下图基准测试框架流程设计可以清晰的观察到基准测试每一步的操作。

![benchmark framework](../assets/images/functions/Machinebenchmark/benchmark_framework.png)

## 基准测试流程说明

### 1. 获得电路

一改仅使用完全随机量子门电路进行基准测试衡量性能，QuICT基准测试采用三种电路组供选择，其中包含特殊基准测试电路、量子算法电路、随机指令集电路，通过不同的方向提供对真实量子物理机性能进行更广泛更准确的测试方法。此电路选择亦可大大提高效率节省时间，提供一站式提供物理机属性输出电路的方法。

#### 电路组选择

电路组选择根据待测量的真实的量子物理机属性，例如：量子物理机量子比特数、拓扑结构以及物理机特定指令集，在选择电路组后会根据比特数承载量提供范围内的电路组，也可以选择每一个量子电路是否通过QuICT平台特有的量子电路设计自动化，即根据待测量物理机拓扑结构实现映射以及带测量物理机特定指令集实现量子门转换。

- 第一种

    - 四种特殊benchmark电路【高度并行化电路、高度串行化电路、高度纠缠电路、中间态测量电路】

    - 若干种随机电路【aspen-4, ourense, rochester, sycamore, tokyo, ctrl_unitary, diag, single_bit, ctrl_diag, google, ibmq, ionq, ustc, nam, origin】

- 第二种

    - 四种特殊benchmark电路【高度并行化电路、高度串行化电路、高度纠缠电路、中间态测量电路】

    - 若干种随机电路【aspen-4, ourense, rochester, sycamore, tokyo, ctrl_unitary, diag, single_bit, ctrl_diag, google, ibmq, ionq, ustc, nam, origin】

    - 部分算法电路【adder, clifford, qft】

- 第三种

    - 四种特殊benchmark电路【高度并行化电路、高度串行化电路、高度纠缠电路、中间态测量电路】
    
    - 五种随机电路【aspen-4, ourense, rochester, sycamore, tokyo, ctrl_unitary, diag, single_bit, ctrl_diag, google, ibmq, ionq, ustc, nam, origin】
    
    - 所有算法电路【adder, clifford, qft, grover, cnf, maxcut, qnn, quantum_walk, vqe】

### 2. 运行测试

测试流程包含两种方法：

第一种方法：获得电路组后，在量子物理机模拟出结果后进入评分系统。

第二种方法：此基准测试提供物理机接口，从QuICT电路库中得到电路后实时进行物理机模拟，进而进入评分系统进行基准测试（该流程大大的节省电路输出以及物理机结果输入时间，减少系统错误率）。

### 3. 评分系统

#### 对所有电路进行熵值的评分

熵值（相对熵函数、交叉熵函数、回归损失函数等）基准测试是一种通过量子电路并测量观察到的模拟振幅值与真实物理机振幅值预期概率之间分布的差异来评估物理机性能的方法，并用函数值的平均值的百分制作为电路熵值的评分。

#### 筛选出有效电路

在量子计算机上运行的每个电路的成功与否是根据每个电路的平均熵值是否大于阙值决定的，评判电路是否属于有效电路，继而才能进行后续测试。

#### 对特殊基准电路进行指标值分析

!!!  
    **特殊基准电路介绍**
    - 高度并行化电路
        - 不同量子算法的结构允许不同程度的并行化，通过比较量子门数量，门数和电路深度的比率高度并行的应用将大量运算放入相对较小的电路深度中。
        - P = （ng / nd -1）/ (nw - 1) 其中ng表示门的总数，nd表示电路深度，nw表示电路宽度，P越接近1的电路并行化程度越高。

    - 高度串行化电路
        - 设置电路深度的最长路径上两个量子位相互作用的数量接近总的双比特数量。
        - S = 1 - ns / ng 其中ng表示门的总数，ns表示不在最长路径上的双比特门数，S越接近1的电路串行化程度越高。

    - 高度纠缠电路
        - 通过计算两个量子位相互作用的所有门操作的比例，测试电路种两个量子位相互作用程度。
        - E = 1 - ne / ng 其中ng表示门的总数，ne表示电路完全纠缠下多余的双比特门数，E越接近1的电路纠缠程度越高。

    - 中间态测量电路
        - 对于多个连续层的门操作组成的电路，测量门在不同层数为程序执行期间和之后提取信息
        - M = md / nd 其中md表示电路中测量门所在的层数，nd表示电路深度，E越接近1的电路测量性能越完整。

#### 对算法电路进行量子体积分析

量子体积（Quantum Volume），是一个用于评估量子计算系统性能的协议。量子计算系统的操作保真度越高，关联性越高，有越大的校准过的门操作集合便会有越高的量子体积。量子体积与系统的整体性能相关联，即与系统的整体错误率，潜在的物理比特关联和门操作并行度相联系。总的来说，量子体积是一个用于近期整体评估量子计算系统的一个实用方法，数值越高，系统整体错误率就越低，性能就越好。量子体积被定义为指数形式：

\begin{equation}
V_{Q} = 2^n
\end{equation}

其中n表示在给定比特数目m（m大于n）和完成计算任务的条件下，电路的最大逻辑深度，如果电路的最大逻辑深度n大于电路比特数m，那么系统的量子体积就是：

\begin{equation}
2^m
\end{equation}

当然为了数据耦合度高，我们结构分析展示的是去除二次方的结果。基准测试对每一个电路都进行量子体积的分析。

### 4. 结果分析展示

结果分析分为三个展示方向，分别是Radar chart雷达图、TxT文本文件以及Excel表格。

文本文件和表格展示对截止于有效电路筛选之前对所有电路组的熵值以及量子体积的评判结果。雷达图（左）展示有效电路中特殊基准电路指标值最优指标值×该电路的量子体积值，随机电路量子体积的最优值，雷达图（右）展示有效电路中各个算法赛道中算法电路的量子体积最优值。

#### 结果展示示例
**雷达图**

![radar graph](../assets/images/functions/Machinebenchmark/benchmark_radar_chart_show.jpg)

**TXT文本文件**

![TXT](../assets/images/functions/Machinebenchmark/benchmark_txt.png)

## 基准测试基本用法

### 通过物理机接口实时进行基准测试

首先，我们需要选择如下参数：

- output_path: 基准测试结果展示文件存储地址。

- show_type: 基准测试结果展示形式。

- simulator_interface: 待测物理机接口，使后端串联。

- quantum_machine_info：包含待测物理机量子比特数、拓扑结构、特定指令集。

- mapping: 是否根据物理机的拓扑结构对每一个电路执行映射。

- gate_transform: 是否根据物理机特定指令集对每一个电路执行门转换。

!!! tip
    执行之前，请选择结果分析的存储路径，选择结构分析的类型（雷达图是默认生成的，需要选择Txt文本文件或者Excel表格），如果按照下述构建步骤操作，此库将位于当前目录下的benchmark文件夹下。

具体代码实现如下：

``` python
#初始化
benchmark = QuICTBenchmark(output_path="./benchmark", show_type="txt")
# 传入拓扑结构
layout_file = Layout.load_file("./layout/grid_3x3.json")
#传入指令集
Inset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])
#传入物理机接口, 直接进入评分系统
results = benchmark.run(simulator_interface=machine_interface, quantum_machine_info={"qubits_number":5, "layout_file":layout_file, "Instruction_Set":Inset}, mapping=True, gate_transform=True)
```

### 用电路进行基准测试

从QuICT电路库中得到电路后，拿到物理机模拟后，提供电路组和模拟振幅组进入评分系统进行基准测试。

首先，我们需要选择如下参数：

- output_path: 基准测试结果展示文件存储地址。

- show_type: 基准测试结果展示形式。

- quantum_machine_info：包含待测物理机量子比特数、拓扑结构、特定指令集。

- mapping: 是否根据物理机的拓扑结构对每一个电路执行映射。

- gate_transform: 是否根据物理机特定指令集对每一个电路执行门转换。

- circuits_list：电路组。

- amp_results_list：物理机模拟振幅。

具体代码实现如下：

``` python
#初始化
benchmark = QuICTBenchmark(output_path="./benchmark", show_type="txt")
# 传入拓扑结构
layout_file = Layout.load_file("./layout/grid_3x3.json")
#传入指令集
Inset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])
#获得电路
circuits = benchmark.get_circuits(quantum_machine_info={"qubits_number":5, "layout_file":layout_file, "Instruction_Set":Inset}, mapping=True, gate_transform=True)
#传入电路组以及物理机模拟结果，进入评分系统
results = benchmark.evaluate(circuits_list, amp_results_list)
```
