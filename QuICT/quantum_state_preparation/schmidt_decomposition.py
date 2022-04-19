import numpy as np

class SchmidtDecomposition(object):
    """
    |𝜓⟩ = ∑ 𝜆_i |i_A⟩ |i_B⟩, where 𝜆_i are non-negative real numbers, ∑ 𝜆_i^2 = 1, and
    |i_A⟩ |i_B⟩ are orthonormal states.
    """
