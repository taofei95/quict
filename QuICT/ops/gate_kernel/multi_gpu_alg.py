import cupy as cp
import numpy as np

import random


"""
Matrix based gate algorithm for multi-gpu/cpu; data swap

devices' index: 0, 1, ..., n
total qubit = MAX device index + log2(n)
device t_index(t) = target index = MAX device index

1-qreg:
based matrix (inner product):
    [[c1, c2],    *   [v0,      = [c1*v0 + c2*v1,
     [c3, c4]]         v1]         c3*v0 + c4*v1]
    
    Data transfer (when target index > MAX device index):
        switch data between the devices t_index = 0 and the devices t_index = 1.
        t=0 device: [v0, v1, ... v_n/2, t_0, ... t_n/2]
        t=1 device: [v_n/2+1, ..., v_n, t_n/2+1, ... t_n]
    
    Do Inner product with target index = MAX device index

    Switch data back.
diagonal matrix (multiply):
    [[c1, 0],   *   [v0,    =   [c1*v0,  
     [0, c2]]        v1]         c2*v1]

    If target index > MAX device index:
        t=0 device: [v0, v1, ..., vn] * c1
        t=1 device: [t0, t1, ..., tn] * c2
Controlled matrix (multiply):
    [[1, 0],    *   [v0,    =   [v0,
     [0, c]]         v1]         c*v1]

    if target index > MAX device index:
        t=0 device: do nothing
        t=1 device: [t0, t1, ..., tn] * c
RDiagonal matrix (swap):
     [[0, 1],   *   [v0,    =   [v1,
      [1, 0]]        v1]         v0]

    if target index > MAX device index:
        Way 1: swap whole data between t=0 devices and t=1 devices
        Way 2: re-index devices (t=0 <==> t=1)
        Way 3: hold a dictionary to represent the swap relationship between devices
RDiagonal matrix (Multiplyswap):
    [[0, c1],   *   [v0,    =   [c1*v1,  
     [c2, 0]]        v1]         c2*v0]
    
    if target index > MAX device index:
      Step 1:
        t=0 device: [v0, v1, ..., vn] * c2
        t=1 device: [t0, t1, ..., tn] * c1
      Step 2:
        Way 1: swap whole data between t=0 devices and t=1 devices
        Way 2: re-index devices (t=0 <==> t=1)
        Way 3: hold a dictionary to represent the swap relationship between devices

2 t-qregs:
t-args [t_0, t_1] (t_0 < t_1)
diagonal matrix (multiply):
    [[c1, 0, 0, 0],     *   [v0,    = [c1*v0,
     [0, c2, 0, 0],          v1,       c2*v1,
     [0, 0, c3, 0],          v2,       c3*v2,
     [0, 0, 0, c4]]          v3]       c4*v3]

    if t_1 > MAX device index and t_0 < MAX device index:
        t_1=0 devices:
            vec * c1 if t_0 = 0
            vec * c2 if t_0 = 1
        t_1=1 devices:
            vec * c3 if t_0 = 0
            vec * c4 if t_0 = 1
    if t_1 and t_0 > MAX device index
        t_1=0, t_0=0 devices: [v00, v01, ..., v0n] * c1
        t_1=0, t_0=1 devices: [t00, t01, ..., t0n] * c2
        t_1=1, t_0=0 devices: [v10, v11, ..., v1n] * c3
        t_1=1, t_0=1 devices: [t10, t11, ..., t1n] * c4

completed matrix (M*IP):
    [[A, 0, 0, 0],    *     [v0,    =   [A*v0,
     [0, c, d, 0],           v1,         c*v1 + d*v2,
     [0, e, f, 0],           v2,         e*v1 + f*v2,
     [0, 0, 0, B]]           v3]         B*v3]

    if t_1 > MAX device index and t_0 < MAX device index:
      Step 1:
        Swap v0 and v2. (switch data by t_0 index)
        for t_1 = 0 devices, choice index with t_0 = 1
        for t_1 = 1 devices, choice index with t_0 = 1
      Step 2:
        for t_1 = 0 devices, do 1 qreg inner product with index = t_0
        for t_1 = 1 devices, do 1 qreg multiply with index = t_0
      Step 3:
        Swap back
    if t_1 and t_0 > MAX device index
      t_1=0, t_0=0 devices: [v00, v01, ..., v0n] * A
      t_1=1, t_0=1 devices: [t10, t11, ..., t1n] * B

      For (t_1=0, t_0=1) and (t_1=1, t_0=0) devices: (Same as 1 qreg based matrix)
        Step 1:
          switch data
            (t_1=0, t_0=1) device: [t00, t01, ... t_0n/2, v_10, ... v_1n/2]
            (t_1=1, t_0=0) device: [t_0n/2+1, ..., t_0n, v_1n/2+1, ... v_1n]
        Step 2:
          Do Inner product with target index = MAX device index
        Step 3:
          Swap Back

completed matrix (IP*IP):
    [[A, 0, 0, B],    *     [v0,    =   [A*v0 + B*v3,
     [0, c, d, 0],           v1,         c*v1 + d*v2,
     [0, e, f, 0],           v2,         e*v1 + f*v2,
     [C, 0, 0, D]]           v3]         C*v0 + D*v3]

    if t_1 > MAX device index and t_0 < MAX device index:
      Step 1:
        Swap v0 and v2. (switch data by t_0 index)
        for t_1 = 0 devices, choice index with t_0 = 1
        for t_1 = 1 devices, choice index with t_0 = 1
      Step 2:
        do Inner Product with index = t_0
      Step 3:
        Swap back
    if t_1 and t_0 > MAX device index:
        (t_1=0, t_0=0) and (t_1=1, t_0=1) devices: do 1 qreg based matrix
        (t_1=0, t_0=1) and (t_1=1, t_0=0) devices: do 1 qreg based matrix

controlled matrix (swap):
    [[1, 0, 0, 0],    *     [v0,    =   [v0,
     [0, 0, 1, 0],           v1,         v2,
     [0, 1, 0, 0],           v2,         v1,
     [0, 0, 0, 1]]           v3]         v3]

    if t_1 > MAX device index and t_0 < MAX device index:
      Swap v1 and v2
        For t_1=0 devices: choice data with index t_0 = 1
        For t_1=1 devices: choice data with index t_0 = 0
    if t_1 and t_0 > MAX device index:
      For (t_1=0, t_0=0) and (t_1=1, t_0=1) devices: do nothing
      For (t_1=0, t_0=1) and (t_1=1, t_0=0) devices: Same as RDiagonal Swap (1 qreg)

2 ct-qregs:
qregs: [t_index, c_index] (v0 = 0, v1 = v0 + t_index, v2 = v0 + c_index, v3 = 1)
controlled matrix (multiply):
    [[1, 0, 0, 0],    *     [v0,    =   [v0,
     [0, 1, 0, 0],           v1,         v1,
     [0, 0, a, 0],           v2,         v2*a,
     [0, 0, 0, b]]           v3]         v3*b]

    if c_index > MAX device index and t_index < MAX device index:
        For c=0 devices: do nothing
        For c=1 devices: do Multiply with t_index
    if t_index > MAX device index and c_index < MAX device index:
        For t=0 devices: do Controlled matrix (M) with c_index and a
        For t=1 devices: do Controlled matrix (M) with c_index and b
    if c_index and t_index > MAX device index:
        For (c=0, t=0) and (c=0, t=1) devices: do nothing
        (c=1, t=0) devices: vec * a
        (c=1, t=1) devices: vec * b

controlled matrix (Inner product):
    [[1, 0, 0, 0],    *     [v0,    =   [v0,
     [0, 1, 0, 0],           v1,         v1,
     [0, 0, a, b],           v2,         a*v2 + b*v3,
     [0, 0, c, d]]           v3]         c*v2 + d*v3]

    if c_index > MAX device index:
        For c=0 devices: do nothing
        For c=1 devices: do Inner product with t_index
    if t_index > MAX device index:
      Step 1:
        Swap data between t=0 devices and t=1 devices by c_index
        For t=0 devices, choice index with c_index = 1
        For t=1 devices, choice index with c_index = 0
      Step 2:
        For t=0 devices, do nothing
        For t=1 devices, do Inner product with c_index
      Step 3:
        Swap back
    if t_index and c_index > MAX device index:
        For (c=0, t=0) and (c=0, t=1) devices: do nothing
        For (c=1, t=0) and (c=1, t=1) devices:
          Step 1: (idea!! Do multiply splited then use scatter add, repeat 2 times)
            Swap data as (H Gate), do inner product with index = MAX device index

controlled matrix (multiplySwap):
    [[1, 0, 0, 0],    *     [v0,    =   [v0,
     [0, 1, 0, 0],           v1,         v1,
     [0, 0, 0, a],           v2,         v3*a,
     [0, 0, b, 0]]           v3]         v2*b]

    if c_index > MAX device index:
        For c=0 devices: do nothing
        For c=1 devices: do MultiplySwap with t_index
    if t_index > MAX device index:
      Step 1:
        For t=0 devices: do Controlled matrix (Multi) with c_index and b
        For t=1 devices: do Controlled matrix (Multi) with c_index and a
      Step 2:
        Swap data
        For t=0 devices, choice data with index c_index = 1
        For t=1 devices, choice data with index c_index = 1
    if c_index and t_index > MAX device index:
      Step 1:
        For (c=0, t=0) and (c=0, t=1) devices: do nothing
        (c=1, t=0) devices: vec * b
        (c=1, t=1) devices: vec * a
      Step 2:
        Swap data between (c=1, t=0) and (c=1, t=1) devices

3+ qregs
2 c_indexes and 1 t_index   [t, c0, c1], c0 < c1

controlled matrix (multiply)
       [[1, 0, 0, 0, 0, 0, 0, 0],       *       vec
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, a, 0],
        [0, 0, 0, 0, 0, 0, 0, b]]
    if t_index > MAX device index:
        For t=0 devices: do multiply (a) when index (c1=1, c0=1)
        For t=1 devices: do multiply (b) when index (c1=1, c0=1)
    if c1 > MAX device index:
        For c1=0 devices: do nothing
        For c1=1 devices: do controlled matrix multiply ctargs (c0, t)
    if c0, c1 > MAX device index:
        For c1=1, c0=1 devices: do Diagonal Multiply with t_index
        otherwise: do nothing
    if t_index, c1 > MAX device index:
        For c1=0 devices: do nothing
        For c1=1, t=0 devices: do controlled matrix multiply targ (c0 and a)
        For c1=1, t=1 devices: do controlled matrix multiply targ (c0 and b)
    if t_index, c0, c1 > MAX device index:
        For c0=1, c1=1, t=0 devices: do Multiply (a)
        For c0=1, c1=1, t=1 devices: do Multiply (b)
        otherwise do nothing

controlled matrix (swap)
       [[1, 0, 0, 0, 0, 0, 0, 0],       *       vec
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]]
    if t_index > MAX device index:
        Swap t=0 devices and t=1 devices with index (c1=1, c0=1)
    if c1 > MAX device index:
        For c1=0 devices: do nothing
        For c1=1 devices: do controlled matrix swap ctargs (c0=1, t)
    if c0, c1 > MAX device index:
        For c1=1, c0=1 devices: do RDiagonal Swap with t_index
        otherwise: do nothing
    if t_index, c1 > MAX device index:
        For c1=0 devices: do nothing
        For c1=1, t=0 devices and c1=1, t=1 devices:
          Swap data by c0
    if t_index, c0, c1 > MAX device index:
        Swap data between c0=1, c1=1, t=0 devices and c0=1, c1=1, t=1 devices
        otherwise do nothing

2 t_indexes and 1 c_index   [t0, t1, c] (t0 < t1)
       
controlled matrix (swap)
       [[1, 0, 0, 0, 0, 0, 0, 0],       *       vec
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]]
    if c > MAX device index:
        For c=0 devices: do nothing
        For c=1 devices: do controlled matrix (Swap) [t0, t1]
    if t1 > MAX device index:
        For t_1=0 devices: swap data with index (c=1, t0=1)
        For t_1=1 devices: swap data with index (c=1, t0=0)
    if t1, t0 > MAX device index:
        For (t0=0, t1=0) and (t0=1, t1=1) devices: do nothing
        For (t0=1, t1=0) devices: swap data with index (c=1)
        For (t0=0, t1=1) devices: swap data with index (c=1)
    if c, t1 > MAX device index:
        For (c=0) devices: do nothing
        For (c=1, t1=0) devices: swap data with index (t0=1)
        For (c=1, t1=1) devices: swap data with index (t0=0)
    if c, t0, t1 > MAX device index:
        Swap data between (c=1, t0=1, t1=0) and (c=1, t0=0, t1=1) devices
"""

"""
4 ways of data switch
    1. half-to-half (target index = MAX device index)
    2. switch by target index
    3. switch by 2 target indexes
    4. Swap whole data

loop:
    1. check indexes exceed MAX device index
    2. data switch (if need)
    3. gate alg by different case
    4. data switch back (if need)
"""

"""
UI
    object: circuit/np.ndarray/compliex_gate/file_path
    ===QCDA==========
    InstructionSet: set (class)
    layout: class (tuopu)
    optimization: bool
    mapping: bool
    =================
    backend: str (==> set(Enum class)) (need interface (jiayao?))
"""
