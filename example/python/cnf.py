import os
from QuICT.core import Circuit
from QuICT.algorithm.quantum_algorithm.CNF.cnf import CNFSATOracle
import pathlib


cnf_test = """p cnf 3 3
-1 2 0
1 3 0
-2 0"""
filename_test = "./test.cnf"
n_var = 3
n_aux = 3
path = pathlib.Path(filename_test)
if path.exists():
    print(f"remove {filename_test} and try again")
else:
    with open(filename_test, "w") as f:
        f.write(cnf_test)
    cnf = CNFSATOracle()
    cnf.run(filename_test, n_aux, 1)
    circ = Circuit(n_var+n_aux+1)
    cnf.circuit() | circ    
    circ.draw(method="matp_file")
    os.remove(filename_test)