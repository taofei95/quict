HHL
==================================

The quantum algorithm for linear systems of equations, 
also called HHL algorithm, designed by Aram Harrow, Avinatan Hassidim, 
and Seth Lloyd, is a quantum algorithm formulated in 2009 for solving linear systems. 
The algorithm estimates the result of a scalar measurement on the solution vector 
to a given linear system of equations.

HHL algorithm in QuICT takes an :math:`N \times N` Hermitian matrix :math:`A` 
and an unit vector :math:`b` as input. At the end of the quantum computation,
the target qureg would be in a state representing vector :math:`x = A^{-1}b`,
that is, the solution for linear equation :math:`Ax = b`. 

When HHL is done in simulation way, QuICT can get the quantum state of the circuit, 
so can output :math:`x` directly, while on real quantum devices, the superposition
state cannot be known without multiple measurements. So HHL is often followed by
other procedures to get some feature of :math:`x`, rather than :math:`x` itself explicitly. 