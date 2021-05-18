
from scipy.stats import unitary_group
    
from QuICT.core import *
from QuICT.qcda.synthesis.unitary_transform import *

if __name__ == '__main__':
    U = unitary_group.rvs(2 ** 3)
    compositeGate, _ = UTrans(U)

    circuit = Circuit(3)
    circuit.set_exec_gates(compositeGate)
    circuit.draw_photo(show_depth=False)
