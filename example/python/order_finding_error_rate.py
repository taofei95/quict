from QuICT.algorithm.quantum_algorithm.shor.BEA_zip import order_finding as BEA_zip_run
from QuICT.algorithm.quantum_algorithm.shor.BEA import construct_circuit as BEA_circuit
from QuICT.algorithm.quantum_algorithm.shor.BEA import order_finding as BEA_run

from QuICT.algorithm.quantum_algorithm.shor.HRS_zip import order_finding as HRS_zip_run
from QuICT.algorithm.quantum_algorithm.shor.HRS import (
    construct_circuit as HRS_circuit,
)  # TODO
from QuICT.algorithm.quantum_algorithm.shor.HRS import order_finding as HRS_run  # TODO

from QuICT.algorithm.quantum_algorithm.shor.utility import (
    reinforced_order_finding_constructor,
)

HRS_zip_run = reinforced_order_finding_constructor(HRS_zip_run)
HRS_run = reinforced_order_finding_constructor(HRS_run)

import logging

# logging.root.setLevel(logging.INFO)

import QuICT

print(QuICT.__file__)


def naive_order_finding(a, N):
    for i in range(1, N):
        if (a ** i) % N == 1:
            return i
    return 0


methods = {
    "BEA_zip_run": BEA_zip_run,
    "BEA_run": BEA_run,
    "HRS_zip_run": HRS_zip_run,
    "HRS_run": HRS_run,
}
for mode in methods.keys():
    error = 0
    all = 0
    for N in (7, 8, 9, 11, 12, 13):
        for a in range(2, N):
            ref_order = naive_order_finding(a, N)
            if ref_order == 0:
                continue
            print(f"[{mode}]testing [{a:3},{N:3}]...", end="")
            order = methods[mode](a=a, N=N)
            all += 1
            if order != ref_order:
                error += 1
                print(f"failed: {order:2}, ref: {ref_order:2}")
            else:
                print(f"succeed:{order:2}")
    print(f"[{mode}]error rate:{error/all:.4f}")
