import numpy as np

from QuICT.algorithm.quantum_algorithm.hhl import * 
from QuICT.algorithm.quantum_algorithm.hhl.matrices import *


n = 2 ** 3
A = RS(n, 1).matrix()
b = np.ones(n, dtype=np.complex128)

test = LinearSolver(A, b) 
slt = test.solution()
hhl_u, _ = test.hhl(measure=True, method='unitary')
slt /= np.linalg.norm(slt)
if hhl_u is not None:
      hhl_u /= np.linalg.norm(hhl_u.real)
      print(A.real)
      print(f"solution     = {slt.real}\n" +
            f"hhl(unitary) = {hhl_u.real}")
else:
      print("Failed.")