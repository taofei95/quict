from enum import Enum


class GateType(Enum):
    h = "H gate"
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
    t = "T gate"
    tdg = "The conjugate transpose of T gate"
    phase = "Phase gate"
    cz = "controlled-Z gate"
    cx = "controlled-X gate"
    cy = "controlled-Y gate"
    ch = "controlled-Hadamard gate"
    crz = "controlled-Rz gate"
    cu1 = "controlled-U1 gate"
    cu3 = "controlled-U3 gate"
    fsim = "fSim gate"
    Rxx = "Rxx gate"
    Ryy = "Ryy gate"
    Rzz = "Rzz gate"
    swap = "Swap gate"
    cswap = "cswap gate"
    iswap = "iswap gate"
    sqiswap = "square root of iswap gate"
    ccx = "Toffoli gate"
    CCRz = "CCRz gate"

    # Special gate below
    measure = "Measure gate"
    reset = "Reset gate"
    barrier = "Barrier gate"
    unitary = "Unitary gate"

    # no qasm represent below
    perm = "Perm gate"
    control_perm_detail = "Control-Perm_Detail gate"
    perm_shift = "Perm-Shift gate"
    control_perm_shift = "Control-Perm-Shift gate"
    perm_mul = "Perm-Mul gate"
    control_perm_mul = "Control-Perm-Mul gate"
    perm_fx = "Perm-Fx gate"
    shor_init = "Shor Initial gate"

    # Composite gate
    qft = "QFT gate"
    iqft = "IQFT gate"
    mgr = "Modified Givens Rotation gate"


SPECIAL_GATE_SET = [
    GateType.measure,
    GateType.reset,
    GateType.barrier,
    GateType.unitary,
    GateType.perm,
    GateType.control_perm_detail,
    GateType.perm_shift,
    GateType.control_perm_shift,
    GateType.perm_mul,
    GateType.control_perm_mul,
    GateType.perm_fx,
    GateType.shor_init,
    GateType.qft,
    GateType.iqft
]


DIAGONAL_GATE_SET = [
    GateType.s,
    GateType.sdg,
    GateType.z,
    GateType.id,
    GateType.u1,
    GateType.rz,
    GateType.t,
    GateType.tdg,
    GateType.phase,
    GateType.cz,
    GateType.crz,
    GateType.cu1,
    GateType.Rzz,
    GateType.CCRz
]


SUPREMACY_GATE_SET = [
    GateType.sx,
    GateType.sy,
    GateType.sw
]
