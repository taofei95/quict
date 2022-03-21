# import numpy as np
#
# from scipy.stats import unitary_group
# from ..unitary_transform import UnitaryTransform
# from time import time
#
# from ...uniformly_gate import UniformlyRy, UniformlyRz
#
#
# def test_uniformly_rotation_cnot():
#     print("\n")
#     for controlled_cnt in range(1, 5):
#         qubit_num = controlled_cnt + 1
#         print(f"Qubit={qubit_num}, controlled={controlled_cnt}")
#         angle_cnt = 1 << controlled_cnt
#         angle_list = [np.random.uniform(low=0, high=np.pi) for _ in range(angle_cnt)]
#         gates = UniformlyRy.execute(angle_list).unitary_transform()
#         cnt = 0
#         for gate in gates:
#             if gate.controls + gate.targets == 2:
#                 cnt += 1
#         print(f"Uniformly Ry CNOT count: {cnt}")
#
#         gates = UniformlyRz.execute(angle_list).unitary_transform()
#         cnt = 0
#         for gate in gates:
#             if gate.controls + gate.targets == 2:
#                 cnt += 1
#         print(f"Uniformly Rz CNOT count: {cnt}")
#
#
# def test_basis_diff():
#     for qubit_num in range(1, 7):
#         mat = unitary_group.rvs(1 << qubit_num)
#         print(f"[Qubit={qubit_num}]")
#         for basis in [1, 2]:
#             print(f"Basis={basis}")
#             start_time = time()
#             gates = UnitaryTransform.execute(mat, recursive_basis=basis).unitary_transform()
#             end_time = time()
#             print(f"Time elapsed: {end_time - start_time:.4f} s")
#             two_bit_gate_cnt = 0
#             for gate in gates:
#                 if gate.controls + gate.targets == 2:
#                     two_bit_gate_cnt += 1
#             print(f"Two-bit gate: {two_bit_gate_cnt}")
#         print("=" * 40)
