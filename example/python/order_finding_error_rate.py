from QuICT.algorithm.quantum_algorithm import HRS_order_finding, BEA_order_finding
import QuICT
print(QuICT.__file__)

error = 0
all = 0
for N in (7, 11, 13, 17):
    for a in range(2, N):
        print(f"[BEA]testing [{a:3},{N:3}]...", end="")
        order = BEA_order_finding.run(a, N)
        all += 1
        if order == 0:
            error += 1
            print("failed")
        else:
            print(f"succeed:{order}")
print(f"[BEA]error rate:{error/all:.4f}")

# error = 0
# all = 0
# for N in (7, 11, 13, 17):
#     for a in range(2, N):
#         print(f"[HRS]testing [{a:3},{N:3}]...", end="")
#         order = HRS_order_finding.run(a, N)
#         all += 1
#         if order == 0:
#             error += 1
#             print("failed")
#         else:
#             print(f"succeed:{order}")
# print(f"[HRS]error rate:{error/all:.4f}")
