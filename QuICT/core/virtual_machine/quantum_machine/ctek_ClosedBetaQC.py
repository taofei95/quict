""" Generate The Virtual Quantum Machine Model for Quantum Computing's CTEK ClosedBetaQC. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 66
iset = InstructionSet(GateType.cz, [GateType.x, GateType.y, GateType.z])
unreachable_nodes = [2, 18, 22, 27, 37, 42, 54, 60, 61, 62, 66]
layout = Layout.grid_layout(qubit_number, unreachable_nodes)
qubit_fidelity = [
    (0.988, 0.9425), (0.996, 0.9755), (0.9645, 0.9335), (0.996, 0.9055), (0.961, 0.812), (0.909, 0.956), (0.9915, 0.9145),
    (0.9655, 0.933), (0.9985, 0.977), (0.995, 0.957), (0.986, 0.8985), (0.9885, 0.935), (0.9945, 0.9515), (0.9945, 0.932),
    (0.99, 0.967), (0.991, 0.941), (0.96, 0.952), (0.9635, 0.9335), (0.924, 0.9125), (0.986, 0.8885), (0.981, 0.9235),
    (0.978, 0.936), (0.997, 0.955), (0.991, 0.951), (0.9775, 0.9545), (0.965, 0.967), (0.964, 0.913), (0.9785, 0.9795),
    (0.9845, 0.954), (0.9795, 0.9445), (0.989, 0.9235), (0.991, 0.955), (0.9605, 0.9425), (0.9915, 0.9685), (0.984, 0.9605),
    (0.965, 0.8965), (0.9625, 0.9395), (0.9855, 0.967), (0.965, 0.9505), (0.975, 0.9265), (0.9405, 0.8865), (0.989, 0.933),
    (0.9975, 0.935), (0.978, 0.9615), (0.994, 0.9625), (0.9915, 0.97), (0.9705, 0.951), (0.982, 0.956), (0.9925, 0.9255),
    (0.9795, 0.9355), (0.9955, 0.9635), (0.975, 0.9285), (0.9715, 0.9525), (0.9795, 0.9425)
    ]
gate_fidelity = [
    0.0034, 0.0016, 0.001394, 0.00956, 0.0035, 0.0015, 0.0065, 0.0025, 0.0012, 0.004, 0.0044, 0.0018, 0.0028,
    0.003, 0.0021, 0.0019, 0.0012, 0.0012, 0.0034, 0.0045, 0.0026, 0.0011, 0.0015, 0.0016, 0.0019, 0.0009, 0.0014,
    0.0013, 0.001, 0.0011, 0.0036, 0.0016, 0.0018, 0.001, 0.0012, 0.0034, 0.0039, 0.0015, 0.0012, 0.0013, 0.0225,
    0.0012, 0.0026, 0.0012, 0.0018, 0.0008, 0.0014, 0.0016, 0.0023, 0.0017, 0.0024, 0.0037, 0.0011, 0.0018
    ]
coupling_strength = [
    (7, 1, 2.3), (8, 1, 3.53), (9, 3, 1.46), (10, 3, 1.09), (10, 4, 53.78), (11, 4, 23.0), (11, 5, 30.81), (12, 5, 42.53),
    (12, 6, 31.76), (13, 7, 2.17), (13, 8, 5.85), (14, 8, 4.91), (14, 9, 2.53), (15, 9, 1.22), (15, 10, 1.34), (16, 10, 1.74),
    (16, 11, 2.9), (17, 11, 2.32), (17, 12, 2.2), (19, 13, 2.58), (20, 13, 1.58), (20, 14, 2.3), (21, 14, 2.08), (21, 15, 2.25),
    (23, 16, 2.9), (23, 17, 2.59), (24, 17, 2.88), (25, 19, 1.71), (25, 20, 2.39), (26, 20, 1.66), (26, 21, 3.14), (28, 23, 2.67),
    (29, 23, 3.13), (29, 24, 1.76), (31, 25, 1.68), (32, 25, 2.0), (32, 26, 2.59), (33, 26, 3.2), (34, 28, 2.64), (35, 28, 2.26),
    (35, 29, 1.84), (36, 29, 1.94), (38, 32, 2.41), (38, 33, 1.79), (39, 33, 2.41), (39, 34, 1.61), (40, 34, 1.93), (40, 35, 3.52),
    (41, 35, 2.21), (41, 36, 2.65), (44, 38, 2.94), (45, 38, 1.79), (45, 39, 3.01), (46, 39, 2.22), (46, 40, 1.68), (47, 40, 2.09),
    (47, 41, 2.89), (48, 41, 13.7), (49, 43, 1.69), (49, 44, 4.18), (50, 44, 5.28), (50, 45, 1.76), (51, 45, 2.66), (51, 46, 2.92),
    (52, 46, 1.6), (52, 47, 3.89), (53, 47, 2.61), (53, 48, 16.67), (55, 49, 1.3), (56, 49, 2.31), (56, 50, 2.37), (57, 50, 1.63),
    (57, 51, 1.66), (58, 51, 2.16), (58, 52, 2.09), (59, 52, 1.88), (59, 53, 2.54), (63, 57, 3.23), (63, 58, 3.13), (64, 58, 2.16),
    (64, 59, 1.92), (65, 59, 2.1)
    ]
readout_frequency = [
    5.4885, 5.4353, 5.2714, 5.5567, 5.3664, 5.6098, 5.452, 5.5592, 5.3349, 5.4424, 5.2904, 5.3032, 5.3748, 5.4622, 5.477, 5.1586,
    5.3825, 5.4732, 5.4104, 5.2724, 5.2302, 5.2831, 5.3514, 5.1655, 5.3969, 5.4042, 5.4903, 5.4652, 5.2091, 5.3168, 5.2633, 5.3966,
    5.33, 5.2522, 5.3762, 5.2248, 5.4436, 5.5397, 5.3966, 5.0994, 5.1845, 5.2869, 5.5731, 5.4173, 5.2986, 5.2576, 5.3769, 5.4692,
    5.3347, 5.3658, 5.4647, 5.5026, 5.4085, 5.578
    ]
T1_times = [
    12.71, 22.04, 29.78, 25.32, 28.33, 24.59, 32.18, 19.31, 28.37, 10.29, 18.95, 14.87, 15.67, 27.12, 31.46, 18.9,
    16.86, 25.72, 19.35, 13.49, 16.07, 23.94, 15.98, 30.09, 35.05, 21.01, 21.6, 20.35, 27.16, 19.32, 18.79, 20.26,
    34.21, 24.41, 25.61, 5.29, 21.06, 16.0, 33.75, 13.81, 19.68, 16.77, 11.89, 21.37, 28.32, 34.09, 25.07, 31.35,
    23.28, 28.51, 20.49, 16.01, 24.11, 30.54
    ]
T2_times = [
    2.7, 12.13, 2.91, 3.34, 15.66, 4.88, 11.32, 10.8, 7.62, 1.78, 1.88, 1.78, 7.59, 6.62, 8.82, 1.21, 5.67, 8.01, 4.08,
    1.35, 4.0, 9.41, 2.11, 2.41, 3.62, 3.17, 12.74, 3.4, 3.8, 4.28, 0.8, 6.32, 2.34, 3.49, 2.4, 7.21, 1.41, 7.82, 7.68,
    2.69, 1.69, 3.76, 4.44, 3.7, 8.98, 3.92, 12.61, 10.78, 7.92, 1.52, 3.18, 20.03, 8.24, 11.82
    ]

CTEKOneD12 = VirtualQuantumMachine(
    qubits=qubit_number,
    instruction_set=iset,
    qubit_fidelity=qubit_fidelity,
    gate_fidelity=gate_fidelity,
    coupling_strength=coupling_strength,
    layout=layout,
    readout_frequency=readout_frequency,
    t1_coherence_time=T1_times,
    t2_coherence_time=T2_times
)
