import os
from QuICT.core import Circuit
from QuICT.algorithm.quantum_algorithm.CNF.cnf import CNFSATOracle
import pathlib


filename_test = os.path.join(os.path.dirname(__file__), "test.cnf")
n_var = 3
n_aux = 3

cnf = CNFSATOracle()
cnf.run(filename_test, n_aux, 1)
circ = Circuit(n_var + n_aux + 1)
cnf.circuit() | circ
circ.draw(method="matp_file")
