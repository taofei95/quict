import numpy as np
from typing import List, Tuple

import unittest
from QuICT.core import *
from QuICT.core.gate import H
from QuICT.simulation.state_vector import StateVectorSimulator

from QuICT.algorithm.submodule.multiplier import RCOutOfPlaceModMultiplier

class TestRCModMulti(unittest.TestCase):

    def test_out_of_place_mod_multi(self):
        
        n = 5
        modulus = 11
        multiple = 7

        counts = self.__run_out_of_place_mod_multiplier(
            qreg_size = n,
            modulus = modulus,
            multiple = multiple
        )

        for z, Xz_mod, ancilla in self.__decode_mod_multiplier(counts, n):
            # check the gate outputs modular multiplication for all inputs
            self.assertEqual(Xz_mod, (multiple * z)%modulus)
            # check ancilla bits are properly uncomputed
            self.assertEqual(ancilla, 0)
        
        return
        

    def __run_out_of_place_mod_multiplier(self, qreg_size, modulus, multiple):
        n = qreg_size
        m = int(np.ceil(np.log2(n)))

        mm_circ = Circuit(2*n + m + 1)

        for i in range(n):
            H | mm_circ([i])
        
        RCOutOfPlaceModMultiplier(
            modulus = modulus,
            multiple = multiple,
            qreg_size = n
        ) | mm_circ

        sv_sim = StateVectorSimulator()

        sv_sim.run(circuit = mm_circ)
        counts = sv_sim.sample(shots = 2**(n+4))

        return counts
        

    def __decode_mod_multiplier(self, counts, n) -> List[Tuple[int, int, int]]:
        m = int(np.ceil(np.log2(n)))
        decoded_res = []
        
        for idx, count in enumerate(counts):
            if count != 0:
                total_str = np.binary_repr(idx, 2*n+m+1)
                
                # n bit input 
                z_bin = total_str[:n]
                # n bit output
                Xz_mod_bin = total_str[n:2*n]
                # (m + 1) bit ancilla
                ancilla_bin = total_str[2*n:]
                
                # convert to decimal
                z = int(z_bin, base=2)
                Xz_mod = int(Xz_mod_bin, base= 2)
                ancilla = int(ancilla_bin, base= 2)

                decoded_res.append((z, Xz_mod, ancilla))
        
        return decoded_res

if __name__ == '__main__':
    unittest.main()