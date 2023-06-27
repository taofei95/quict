import numpy as np
from typing import List, Tuple

import unittest
from QuICT.tools.exception.core.gate_exception import *
from QuICT.core import *
from QuICT.core.gate import H, X
from QuICT.simulation.state_vector import StateVectorSimulator

from QuICT.algorithm.submodule.multiplier import RCOutOfPlaceModMultiplier

class TestRCModMulti(unittest.TestCase):

    def test_out_of_place_mod_multi(self):
        ##### test for correctness of the gate #####
        n = 5
        modulus = 11
        multiple = 7

        counts = self.__run_out_of_place_mod_multiplier(
            qreg_size = n,
            modulus = modulus,
            multiple = multiple
        )

        ## confirm modular multiplication is properly calculated for all input states
        for z, Xz_mod, ancilla in self.__decode_mod_multiplier(counts, n):
            # check the gate outputs modular multiplication for all inputs
            self.assertEqual(Xz_mod, (multiple * z)%modulus)
            # check ancilla bits are properly uncomputed
            self.assertEqual(ancilla, 0)
        
        ## test the output state only contains the single desired state 
        input_val = 6
        decoded_result = self.__decode_mod_multiplier(
            counts = self.__run_out_of_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple,
                init_val = input_val
            ),
            n = n
        )
        # single output state for a single input state
        self.assertEqual(len(decoded_result), 1)
        z, Xz_mod, ancilla = decoded_result[0]
        # correctness of the output
        self.assertEqual(Xz_mod, (multiple * z)%modulus)
        # uncompute successfully
        self.assertEqual(ancilla, 0)

        ##### test input errors #####
        
        ## modulus can not be larger than the maximum for the register size
        n = 5
        modulus = 37 # larger than 2^n - 1
        multiple = 7

        with self.assertRaises(GateParametersAssignedError):
            self.__run_out_of_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple
            )

        ## modulus can not be even numbers
        n = 5
        modulus = 14
        multiple = 3
        input_val = 5

        with self.assertRaises(ValueError):
            self.__run_out_of_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple,
                init_val = input_val
            )
        
        return

    # def test_ctl_in_place_mod_multi(self):
        
    #     return
        

    def __run_out_of_place_mod_multiplier(
        self, 
        qreg_size, 
        modulus, 
        multiple, 
        init_val: int = 0
    ):
        n = qreg_size
        m = int(np.ceil(np.log2(n)))

        mm_circ = Circuit(2*n + m + 1)

        if 0 == init_val:
            for i in range(n):
                H | mm_circ([i])
        else:
            for i, bit in enumerate(np.binary_repr(init_val, n)):
                if '1' == bit:
                    X | mm_circ([i])
        
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