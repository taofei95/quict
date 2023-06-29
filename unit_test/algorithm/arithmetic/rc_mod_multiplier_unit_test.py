import numpy as np
from typing import List, Tuple

import unittest
from QuICT.tools.exception.core.gate_exception import *
from QuICT.core import *
from QuICT.core.gate import H, X
from QuICT.simulation.state_vector import StateVectorSimulator

from QuICT.algorithm.arithmetic import *

class TestRCModMulti(unittest.TestCase):

    def test_out_of_place_mod_multi_correctness(self):
        ##### test for correctness of the gate #####
        n = 5
        modulus = 11
        multiple = 7

        counts = self.__run_out_of_place_mod_multiplier(
            qreg_size = n,
            modulus = modulus,
            multiple = multiple
        )

        # confirm modular multiplication is properly calculated for all input states
        for z, Xz_mod, ancilla in self.__decode_mod_multiplier(counts, n):
            # check the gate outputs modular multiplication for all inputs
            self.assertEqual(Xz_mod, (multiple * z)%modulus)
            # check ancilla bits are properly uncomputed
            self.assertEqual(ancilla, 0)
        
        # test the output state only contains the single desired state 
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
        
        return
    

    def test_out_of_place_mod_multi_raised_exception(self):
        ##### test input errors #####
        
        # modulus can not be larger than the maximum for the register size
        n = 5
        modulus = 37 # larger than 2^n - 1
        multiple = 7

        with self.assertRaises(GateParametersAssignedError):
            self.__run_out_of_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple
            )

        # modulus can not be even numbers
        n = 5
        modulus = 14
        multiple = 3
        input_val = 5

        with self.assertRaises(GateParametersAssignedError):
            self.__run_out_of_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple,
                init_val = input_val
            )
        
        return


    def test_in_place_mod_multi_correctness(self):
        ##### test for correctness of the gate #####
        n = 5
        modulus = 11
        multiple = 7

        # randomly choose an input val smaller than modulus
        input_val = np.random.randint(modulus)
        decoded_result = self.__decode_mod_multiplier(
            self.__run_in_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple,
                init_val = input_val
            ),
            n = n
        )
        # single output state for a single input state
        self.assertEqual(len(decoded_result), 1)
        
        Xz_mod, ancilla_n, ancilla_mp1 = decoded_result[0]
        # correctness
        self.assertEqual(Xz_mod, (multiple * input_val)%modulus)
        # check uncompute on ancilla
        self.assertEqual(ancilla_n, 0)
        self.assertEqual(ancilla_mp1, 0)        

        return
    

    def test_in_place_mod_multi_inappropriate(self):
        ##### Test for apply the gate on inappropriate register #####
        n = 5
        modulus = 11
        multiple = 7
        
        # value in the input quantum register is >= modulus
        input_val = 15
        decoded_result = self.__decode_mod_multiplier(
            self.__run_in_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple,
                init_val = input_val
            ),
            n = n
        )

        for single_meas in decoded_result:
            Xz_mod, ancilla_n, ancilla_mp1 = single_meas
            # calculation is correct
            self.assertEqual(Xz_mod, (multiple * input_val)%modulus)
            # but higher n bits in the ancilla cannnot be uncomputed as desired
            self.assertNotEqual(ancilla_n, 0)
            self.assertEqual(ancilla_mp1, 0)

        return

    
    def test_in_place_mod_multi_raised_exception(self):
        ##### Test input errors #####
        # modulus can not be larger than the maximum for the register size
        n = 5
        modulus = 37 # larger than 2^n - 1
        multiple = 7

        with self.assertRaises(GateParametersAssignedError):
            self.__run_in_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple
            )

        # modulus can not be even numbers
        n = 5
        modulus = 14
        multiple = 3

        with self.assertRaises(GateParametersAssignedError):
            self.__run_in_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple
            )

        # modulus and multiple must be coprime
        n = 5
        modulus = 15
        multiple = 3

        with self.assertRaises(GateParametersAssignedError):
            self.__run_in_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple
            )

        return
    

    def test_ctl_in_place_mod_multi_correctness(self):
        ##### test for correctness of the gate #####
        n = 5
        modulus = 13
        multiple = 4

        # randomly choose an input val smaller than modulus
        input_val = np.random.randint(modulus)
        decoded_result = self.__decode_ctl_mod_multiplier(
            self.__run_ctl_in_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple,
                init_val = input_val
            ),
            n = n
        )
        # 2 outputs for control bit in |+>
        self.assertEqual(len(decoded_result), 2)
        
        for i in range(2):
            ctl, output, ancilla_n, ancilla_mp1 = decoded_result[i]
            # correctness based on the control bit 
            if ctl == 0:
                self.assertEqual(output, input_val)
            if ctl == 1:
                self.assertEqual(output, (multiple * input_val)%modulus)
            
            # check uncompute on ancilla
            self.assertEqual(ancilla_n, 0)
            self.assertEqual(ancilla_mp1, 0)        

        return


    def test_ctl_in_place_mod_multi_inappropriate(self):
        n = 5
        modulus = 13
        multiple = 4

        # value in the input quantum register is >= modulus
        input_val = 15
        decoded_result = self.__decode_ctl_mod_multiplier(
            self.__run_ctl_in_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple,
                init_val = input_val
            ),
            n = n
        )
        
        for single_meas in decoded_result:
            ctl, output, ancilla_n, ancilla_mp1 = single_meas
            # output in the register is correct
            if ctl == 0:
                self.assertEqual(output, input_val)
                self.assertEqual(ancilla_n, 0)
            if ctl == 1:
                self.assertEqual(output, (multiple * input_val)%modulus)
                # higher n bits of the ancilla are not properly uncomputed as desired
                self.assertNotEqual(ancilla_n, 0)
            self.assertEqual(ancilla_mp1, 0)


        return


    def test_ctl_in_place_mod_multi_raised_exception(self):
        ##### Test input errors #####
        # modulus can not be larger than the maximum for the register size
        n = 5
        modulus = 43 # larger than 2^n - 1
        multiple = 4

        with self.assertRaises(GateParametersAssignedError):
            self.__run_ctl_in_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple
            )

        # modulus can not be even numbers
        n = 5
        modulus = 18
        multiple = 7

        with self.assertRaises(GateParametersAssignedError):
            self.__run_ctl_in_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple
            )

        # modulus and multiple must be coprime
        n = 5
        modulus = 21
        multiple = 7

        with self.assertRaises(GateParametersAssignedError):
            self.__run_ctl_in_place_mod_multiplier(
                qreg_size = n,
                modulus = modulus,
                multiple = multiple
            )

        return
        

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
    
    def __run_in_place_mod_multiplier(
        self,
        qreg_size, 
        modulus, 
        multiple, 
        init_val: int = -1
    ):
        n = qreg_size
        m = int(np.ceil(np.log2(n)))

        mm_circ = Circuit(2*n + m + 1)

        if init_val == -1:
            init_val = np.random.randint(1, modulus)

        for i, bit in enumerate(np.binary_repr(init_val, n)):
            if '1' == bit:
                X | mm_circ([i])

        RCModMultiplier(
            modulus = modulus,
            multiple = multiple,
            qreg_size = n
        ) | mm_circ

        sv_sim = StateVectorSimulator()

        sv_sim.run(circuit = mm_circ)
        counts = sv_sim.sample(shots = 2**(n+4))

        return counts
    
    def __run_ctl_in_place_mod_multiplier(
        self,
        qreg_size, 
        modulus, 
        multiple, 
        init_val: int = -1,
        ctl_bit: str = ''
    ):
        n = qreg_size
        m = int(np.ceil(np.log2(n)))

        mm_circ = Circuit(1 + 2*n + m + 1) # ctl + input/output + ancilla

        if len(ctl_bit) == 0:
            H | mm_circ([0])
        elif ctl_bit == '1':
            X | mm_circ([0])

        if init_val == -1:
            init_val = np.random.randint(1, modulus)

        for i, bit in enumerate(np.binary_repr(init_val, n)):
            if '1' == bit:
                X | mm_circ([i+1])

        RCModMultiplierCtl(
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
                in_bin = total_str[:n]
                # n bit output
                out_bin = total_str[n:2*n]
                # (m + 1) bit ancilla
                ancilla_bin = total_str[2*n:]
                
                # convert to decimal
                in_val = int(in_bin, base=2)
                out_val = int(out_bin, base= 2)
                ancilla = int(ancilla_bin, base= 2)

                decoded_res.append((in_val, out_val, ancilla))
        
        return decoded_res
    
    def __decode_ctl_mod_multiplier(self, counts, n) -> List[Tuple[int, int, int, int]]:
        m = int(np.ceil(np.log2(n)))
        decoded_res = []

        for idx, count in enumerate(counts):
            if count != 0:
                total_str = np.binary_repr(idx, 1+2*n+m+1)
                
                # 1 bit ctl
                ctl_bin = total_str[0]
                # n bit io register 
                out_bin = total_str[1:n+1]
                # n bit ancilla
                ancilla_n_bin = total_str[n+1:2*n+1]
                # (m + 1) bit ancilla
                ancilla_mp1_bin = total_str[2*n+1:]
                
                # convert to decimal
                ctl = int(ctl_bin, base=2)
                out_val = int(out_bin, base=2)
                ancilla_n = int(ancilla_n_bin, base= 2)
                ancilla_mp1 = int(ancilla_mp1_bin, base= 2)

                decoded_res.append((ctl, out_val, ancilla_n, ancilla_mp1))
        
        return decoded_res

if __name__ == '__main__':
    unittest.main()