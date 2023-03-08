from enum import Enum


class GateType(Enum):
    h = "H gate"
    hy = "Self-inverse gate"
    s = "S gate"
    sdg = "The conjugate transpose of Phase gate"
    x = "Pauli-X gate"
    y = "Pauli-Y gate"
    z = "Pauli-Z gate"
    sx = "sqrt(X) gate"
    sy = "sqrt(Y) gate"
    sw = "sqrt(W) gate"
    id = "Identity gate"
    u1 = "U1 gate"
    u2 = "U2 gate"
    u3 = "U3 gate"
    rx = "Rx gate"
    ry = "Ry gate"
    rz = "Rz gate"
    ri = "Ri gate"
    t = "T gate"
    tdg = "The conjugate transpose of T gate"
    phase = "Phase gate"
    gphase = "Global Phase gate"
    cz = "controlled-Z gate"
    cx = "controlled-X gate"
    cy = "controlled-Y gate"
    ch = "controlled-Hadamard gate"
    crz = "controlled-Rz gate"
    cu1 = "controlled-U1 gate"
    cu3 = "controlled-U3 gate"
    fsim = "fSim gate"
    rxx = "Rxx gate"
    ryy = "Ryy gate"
    rzz = "Rzz gate"
    rzx = "Rzx gate"
    swap = "Swap gate"
    cswap = "cswap gate"
    iswap = "iswap gate"
    iswapdg = "The conjugate transpose of iswap gate"
    sqiswap = "square root of iswap gate"
    ccx = "Toffoli gate"
    ccz = "Multi-Control Z Gate"
    ccrz = "CCRz gate"

    # Special gate below
    measure = "Measure gate"
    reset = "Reset gate"
    barrier = "Barrier gate"
    unitary = "Unitary gate"

    # no qasm represent below
    perm = "Permutation gate"
    perm_fx = "Perm-Fx gate"

    # Composite gate
    qft = "QFT gate"
    iqft = "IQFT gate"


class MatrixType(Enum):
    """ Different Type of quantum gates' matrix

    normal: based type of matrix
        1-bits: [[a,b], [c,d]]
        2-bits(control): [[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, a, b],
                          [0, 0, c, d]]
    diagonal: diagonal matrix
        1-bits: [a, 0], [0, b]
        2-bits(control): [[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, a, 0],
                          [0, 0, 0, b]]
        2-bits(targets): [[a, 0, 0, 0],
                          [0, b, 0, 0],
                          [0, 0, c, 0],
                          [0, 0, 0, d]]
        3-bits (control, target); [1, 1, 1, 1, 1, 1, a, b] -- diagonal values
    control: control diagonal matrix
        1-bits: [[1, 0], [0, a]]
        2-bits(control): [[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, a]]
    swap: swap quantum gates' matrix
        1-bit [x]: [[0, 1], [1, 0]]
        2-bit [swap]: [[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]]
        3-bit [cswap]: [[ID(4)],
                        ,   [1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]]
    reverse; reverse matrix
        1-bit: [0, a], [b, 0]
        2-bit: [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, a],
                [0, 0, b, 0]]
        3-bit: [ID(4)]
                    [1, 0, 0, 0]
                    [0, 1, 0, 0]
                    [0, 0, 0, a]
                    [0, 0, b, 0]
    special: no matrix [Measure, Reset, Barrier, Perm]
    diag_diag: 2-qubits diagonal matrix
        2-bits [Rzz]:  [[a, 0, 0, 0],
                        [0, b, 0, 0],
                        [0, 0, c, 0],
                        [0, 0, 0, d]]
    ctrl_normal: control-normal mixed quantum gate's matrix
        2-bits [FSim]: [[1, 0, 0, 0],
                        [0, a, b, 0],
                        [0, c, d, 0],
                        [0, 0, 0, A]]
    normal-normal: normal-normal mixed quantum gate's matrix
        2-bits [Rxx, Ryy]: [[A, 0, 0, B],
                            [0, a, b, 0],
                            [0, c, d, 0],
                            [C, 0, 0, D]]
    diagonal-normal: diagonal-normal mixed quantum gate's matrix
        2-bits [Rzx]: [[A, B, 0, 0],
                       [C, D, 0, 0],
                       [0, 0, a, b],
                       [0, 0, c, d]]
    """
    identity = "identity matrix"
    normal = "normal matrix"
    diagonal = "diagonal matrix"
    control = "control matrix"
    swap = "swap matrix"
    reverse = "reverse matrix"
    special = "special matrix"
    diag_diag = "diagonal * diagonal"
    ctrl_normal = "control * matrix"
    normal_normal = "normal * normal"
    diag_normal = "diagonal * normal"


GATEINFO_MAP = {
    GateType.h: (0, 1, 0, MatrixType.normal),
    GateType.hy: (0, 1, 0, MatrixType.normal),
    GateType.s: (0, 1, 0, MatrixType.control),
    GateType.sdg: (0, 1, 0, MatrixType.control),
    GateType.x: (0, 1, 0, MatrixType.swap),
    GateType.y: (0, 1, 0, MatrixType.reverse),
    GateType.z: (0, 1, 0, MatrixType.control),
    GateType.sx: (0, 1, 0, MatrixType.normal),
    GateType.sy: (0, 1, 0, MatrixType.normal),
    GateType.sw: (0, 1, 0, MatrixType.normal),
    GateType.id: (0, 1, 0, MatrixType.identity),
    GateType.u1: (0, 1, 1, MatrixType.control),
    GateType.u2: (0, 1, 2, MatrixType.normal),
    GateType.u3: (0, 1, 3, MatrixType.normal),
    GateType.rx: (0, 1, 1, MatrixType.normal),
    GateType.ry: (0, 1, 1, MatrixType.normal),
    GateType.rz: (0, 1, 1, MatrixType.diagonal),
    GateType.t: (0, 1, 0, MatrixType.control),
    GateType.tdg: (0, 1, 0, MatrixType.control),
    GateType.phase: (0, 1, 1, MatrixType.control),
    GateType.gphase: (0, 1, 1, MatrixType.diagonal),
    GateType.cz: (1, 1, 0, MatrixType.control),
    GateType.cx: (1, 1, 0, MatrixType.reverse),
    GateType.cy: (1, 1, 0, MatrixType.reverse),
    GateType.ch: (1, 1, 0, MatrixType.normal),
    GateType.crz: (1, 1, 1, MatrixType.diagonal),
    GateType.cu1: (1, 1, 1, MatrixType.control),
    GateType.cu3: (1, 1, 3, MatrixType.diagonal),
    GateType.fsim: (0, 2, 2, MatrixType.ctrl_normal),
    GateType.rxx: (0, 2, 1, MatrixType.normal_normal),
    GateType.ryy: (0, 2, 1, MatrixType.normal_normal),
    GateType.rzz: (0, 2, 1, MatrixType.diag_diag),
    GateType.rzx: (0, 2, 1, MatrixType.diag_normal),
    GateType.measure: (0, 1, 0, MatrixType.special),
    GateType.reset: (0, 1, 0, MatrixType.special),
    GateType.barrier: (0, 1, 0, MatrixType.special),
    GateType.swap: (0, 2, 0, MatrixType.swap),
    GateType.iswapdg: (0, 2, 0, MatrixType.swap),
    GateType.sqiswap: (0, 2, 0, MatrixType.swap),
    GateType.ccx: (2, 1, 0, MatrixType.reverse),
    GateType.ccz: (2, 1, 0, MatrixType.control),
    GateType.ccrz: (2, 1, 1, MatrixType.diagonal),
    GateType.cswap: (1, 2, 0, MatrixType.swap),
}


PAULI_GATE_SET = [GateType.x, GateType.y, GateType.z, GateType.id]


CLIFFORD_GATE_SET = [
    GateType.x,
    GateType.y,
    GateType.z,
    GateType.h,
    GateType.s,
    GateType.sdg,
    GateType.cx,
]
