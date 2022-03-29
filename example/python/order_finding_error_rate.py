from QuICT.algorithm.quantum_algorithm.shor import BEA_zip_run, BEA_run

import logging
# logging.root.setLevel(logging.INFO)

import QuICT
print(QuICT.__file__)
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
simulator = ConstantStateVectorSimulator(
    precision="double",
    gpu_device_id=0,
    sync=True
)

def naive_order_finding(a, N):
    for i in range(1,N):
        if (a**i)%N == 1:
            return i
    return 0

error = 0
all = 0
for method in {BEA_run, BEA_zip_run}:
    for N in (7, 8, 9, 11, 12, 13):
        for a in range(2, N):
            ref_order = naive_order_finding(a, N)
            if ref_order == 0:
                continue
            print(f"[{method.__name__}]testing [{a:3},{N:3}]...", end="")
            order = method(a, N, simulator=simulator)
            all += 1
            if order != ref_order:
                error += 1
                print(f"failed: {order:2}, ref: {ref_order:2}")
            else:
                print(f"succeed:{order:2}")
    print(f"[{method.__name__}]error rate:{error/all:.4f}")

# error = 0
# all = 0
# for N in (7, 11, 13, 17):
#     for a in range(2, N):
#         print(f"[HRS]testing [{a:3},{N:3}]...", end="")
#         order = HRS_order_finding.run(a, N, simulator=simulator)
#         all += 1
#         if order == 0:
#             error += 1
#             print("failed")
#         else:
#             print(f"succeed:{order}")
# print(f"[HRS]error rate:{error/all:.4f}")