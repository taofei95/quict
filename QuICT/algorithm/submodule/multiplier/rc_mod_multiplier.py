import numpy as np

from typing import List

from QuICT.core.gate import BasicGate, Swap, CX
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.algorithm.submodule.adder import RCFourierAdderWired
from QuICT.algorithm.submodule.qft import ry_QFT

class RCOutOfPlaceModMultiplier(CompositeGate):
    """
        An out-of-place modular multiplier using Fourier-basis arithmetic and 
        Montgomery Reduction. Uses in total "2n + logn + 1" qubits which includes 
        "logn + 1" ancilla qubits.

        Based on paper "High Performance Quantum Modular Multipliers" 
        by Rich Rines, Isaac Chuang: https://arxiv.org/abs/1801.01081
    """

    def __init__(
        self, 
        modulus: int,
        multiple: int,
        qreg_size: int,
        inverse_multiple: bool = False,
        name: str = None
    ):
        """
            Construct the out-of-place modular multiplier that does:

            |z(n)>|b(n)>|0(logn+1)> ------> |z(n)>|b + M*z mod N(n)>|0(logn+1)>

            the (logn + 1) 0s are ancilla qubits.

            Args:
                modulus (int): 

                multiple (int): 

                qreg_size (int): the number of qubits to hold the input register and 
                store the result, two sizes are the same.

                inverse_multiple (bool): if True, will calculate "M^(-1)*z mod N".

        """
        assert int(np.ceil(np.log2(modulus+1))) <= qreg_size, "Not enough register size for modulus"
        
        # TODO: is gcd(M,N) != 1 condition for all cases or only when M^(-1) is calculated

        self._modulus  = modulus
        self._multiple = multiple
        
        self._register_size = qreg_size
        self._ancilla_size = int(np.ceil(np.log2(qreg_size))) + 1
        self._total_size = 2 * qreg_size + self._ancilla_size

        self._input_qubit_list = [i for i in range(qreg_size)]
        self._output_qubit_list = [i for i in range(qreg_size, 2 * qreg_size)]
        self._ancilla_qubit_list = [i for i in range(2 * qreg_size, self._total_size)]

        super().__init__(name)

        # requires multiple and modulus are co-prime
        if inverse_multiple:
            multiple = pow(multiple, -1, modulus)

        multi_prime = (multiple * pow(2, self._ancilla_size - 1, modulus)) % modulus

        ##### Multiplication stage #####        
        self.phi_MAC_mod(
            in_reg_size = self._register_size,
            out_reg_size = self._register_size + self._ancilla_size,
            modulus = modulus,
            multiple = multi_prime
        ) | self

        ##### Reduction stage ######

        ### estimate 
        for i in range(self._ancilla_size - 1):
            ctl_idx = self._total_size - 1 - i
            reduced_reg = self._output_qubit_list + self._ancilla_qubit_list[:-i-1]
            RCFourierAdderWired(
                qreg_size = self._register_size + self._ancilla_size - 1 - i,
                addend = - modulus / 2,
                controlled = True,
                in_fourier = True,
                out_fourier = True
            ) | self([ctl_idx] + reduced_reg)

        ### correction
        correction_block = self._output_qubit_list + self._ancilla_qubit_list[:1]
        ry_QFT(
            targets = self._register_size + 1,
            inverse = True
        ) | self(correction_block)

        ry_QFT(
            targets = self._register_size,
            inverse = False
        ) | self(correction_block[1:])

        # Consider output register + first bit of the ancilla as a block,
        # inside the block, permute one position upward such that 
        # the highest bit goes to the bottom of the block
        for i in range(len(correction_block) - 1):
            Swap | self(correction_block[i:i+2])

        # # NOTE: inverse ry_qft before u_tilde correction or after
        ctl_idx = correction_block[-1]
        RCFourierAdderWired(
            qreg_size = self._register_size,
            addend = modulus,
            controlled = True,
            in_fourier = True,
            out_fourier = True
        ) | self([ctl_idx] + self._output_qubit_list)

        # correct u_tilde_(m+1)'s highest bit
        CX | self(correction_block[-2:])

        ##### Get modular multiplication result #####
        ry_QFT(
            targets = self._register_size,
            inverse = True
        ) | self(self._output_qubit_list)

        ##### Uncomputation stage #####
        ry_QFT(
            targets = self._ancilla_size,
            inverse = False
        ) | self(self._ancilla_qubit_list)

        # # modulus and 2 have to be co-prime
        comp_mod = pow(2, self._ancilla_size)
        for i in range(self._register_size):
            uncompute_addend = ((multi_prime * pow(2, i, modulus)) % modulus) * pow(modulus, -1, comp_mod)
            ctl_idx = self._input_qubit_list[self._register_size - 1 - i]
            RCFourierAdderWired(
                qreg_size = self._ancilla_size,
                addend = - uncompute_addend,
                controlled = True,
                in_fourier = True,
                out_fourier = True
            ) | self([ctl_idx] + self._ancilla_qubit_list)

        return


    def phi_MAC_mod(
        self, 
        in_reg_size: int,
        out_reg_size: int,
        modulus: int,
        multiple: int
    ) -> CompositeGate:
        """
            Quantum Montgomery Multiplication's multiplication stage. 
            Refer to section 3.1 in the original paper for detail.

            a quantum multiply accumulator specially for calculating an
            approximation t' of t = X*z mod N, namely:
            
            t' = \sum_{k=0}^{n-1}{X * z_k *(2^{k} \mod{N})}
            
            where "X" is the "multiple", "z" is in the input_reg and "N" is 
            the "modulus"

            t' = t (mod N) and t' < nN, "n" is "input_reg_size"
        """

        phi_mac_gate = CompositeGate(name = "mac")
        accumulator_reg = [j for j in range(in_reg_size, in_reg_size + out_reg_size)]

        # start with (X * 2^0) mod N
        addend = multiple % modulus

        for i in range(in_reg_size):
            # apply each adder gate controlled by bits on the input register
            # from the lowest to the highest bit
            ctl_idx = in_reg_size - 1 - i
            RCFourierAdderWired(
                qreg_size = out_reg_size,
                addend = addend,
                controlled = True,
                in_fourier = True,
                out_fourier = True
            ) | phi_mac_gate([ctl_idx] + accumulator_reg)
            # update addend
            addend = (addend * 2) % modulus

        return phi_mac_gate
    
    

class RCModMultiplier(CompositeGate):
    pass

class RCModMultiplierCtl(CompositeGate):
    pass