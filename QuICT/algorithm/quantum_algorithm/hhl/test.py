import numpy as np

from QuICT.algorithm.quantum_algorithm.hhl.linear_solver import LinearSolver
from QuICT.algorithm.quantum_algorithm.hhl.matrices import *


n = 2 ** 1
A = RS(n, 1).matrix()
b = np.ones(n, dtype=np.complex128)

test = LinearSolver(A, b)
slt = test.solution()
slt /= np.linalg.norm(slt)
hhl_u = test.hhl()
if hhl_u is not None:
      # print(hhl_u)
      hhl_u /= np.linalg.norm(hhl_u.real)
      print(f"solution     = {slt.real}\n" +
            f"hhl(unitary) = {hhl_u.real}")
else: 
      print("Failed.")
      
