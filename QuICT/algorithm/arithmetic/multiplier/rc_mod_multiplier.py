import numpy as np

from typing import List

from QuICT.tools.exception.core.gate_exception import *
from QuICT.core.gate import X, CX, CU3, Swap, CSwap
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.algorithm.arithmetic.adder import RCFourierAdderWired
from QuICT.algorithm.qft import ry_QFT

class RCOutOfPlaceModMultiplier(CompositeGate):
    """
        An out-of-place modular multiplier using Fourier-basis arithmetic and 
        Montgomery Reduction. Uses in total "2n + logn + 1" qubits which includes 
        "logn + 1" clean ancilla qubits.

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

            |z(n)>|0(n)>|0(logn+1)> ------> |z(n)>|M*z mod N(n)>|0(logn+1)>

            the (logn + 1) 0s are ancilla qubits.

            Args:
                modulus (int): modulus in the modular multiplication.

                multiple (int): multiple in the modular multiplication.

                qreg_size (int): the number of qubits to hold the input register and 
                store the result, two sizes are the same.

                inverse_multiple (bool): if True, will calculate "M^(-1)*z mod N". 
                Requires M to be coprime with modulus N.

        """
        if int(np.ceil(np.log2(modulus+1))) > qreg_size:
            raise GateParametersAssignedError("Not enough register size for modulus.")
        if modulus%2 == 0:
            raise GateParametersAssignedError("Modulus cannot be an even number.")

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

        ##### Preprocess #####

        # ry_QFT(self._register_size + self._ancilla_size) | self(self._output_qubit_list + self._ancilla_qubit_list)

        ##### Multiplication stage #####        
        self.__phi_MAC_mod(
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

        # correct the output
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
        # parallel to the inverse qft above
        ry_QFT(
            targets = self._ancilla_size,
            inverse = False
        ) | self(self._ancilla_qubit_list)

        # modulus and 2 have to be co-prime
        comp_mod = pow(2, self._ancilla_size)
        uncompute_addend_list = []
        for i in range(self._register_size):
            uncompute_addend = ((multi_prime * pow(2, i, modulus)) % modulus) * pow(modulus, -1, comp_mod)
            uncompute_addend_list.append(-uncompute_addend)

        self.__phi_MAC_list(
            in_reg_size = self._register_size,
            out_reg_size = self._ancilla_size,
            addend_list = uncompute_addend_list
        ) | self(self._input_qubit_list + self._ancilla_qubit_list)

        return


    def __phi_MAC_mod(
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

        # start with (X * 2^0) mod N
        addend = multiple % modulus
        addend_list = [addend]

        for _ in range(in_reg_size - 1):
            addend = (addend * 2) % modulus
            addend_list.append(addend)

        return self.__phi_MAC_list(in_reg_size, out_reg_size, addend_list)
    
    def __phi_MAC_list(
        self,
        in_reg_size: int,
        out_reg_size: int,
        addend_list: List[int]
    ) -> CompositeGate:
        """
            multiplication accumulator, the addends controlled by each qubit
            in in_reg is given by the addend list.
        """
        assert len(addend_list) == in_reg_size, f"addend list length: {len(addend_list)} not agree with register size: {in_reg_size}"

        total_size = in_reg_size + out_reg_size

        phi_mac_gate = CompositeGate()
        accumulator_reg = [j for j in range(in_reg_size, in_reg_size + out_reg_size)]

        # When input reg size is less than output reg size, can further optimize the depth
        # by paralleling all the CRys that are commute with each other.
        if in_reg_size <= out_reg_size:
            for i in range(out_reg_size):
                j_bound = min(i+1, in_reg_size)
                for j in range(j_bound):
                    theta = np.pi * addend_list[j] / (2**(i - j))
                    ctl_idx = in_reg_size - 1 - j
                    target_idx = total_size - 1 - i + j
                    CU3(theta, 0, 0) | phi_mac_gate([ctl_idx, target_idx])

                for j in range(in_reg_size - j_bound):
                    theta = np.pi * addend_list[j + j_bound] / (2**(out_reg_size - 1 - j))
                    ctl_idx = in_reg_size - 1 - j_bound - j
                    target_idx = in_reg_size + j
                    CU3(theta, 0, 0) | phi_mac_gate([ctl_idx, target_idx])
        # general cases, no depth optimization
        else:
            for i in range(in_reg_size):
                # apply each adder gate controlled by bits on the input register
                # from the lowest to the highest bit
                ctl_idx = in_reg_size - 1 - i
                RCFourierAdderWired(
                    qreg_size = out_reg_size,
                    addend = addend_list[i],
                    controlled = True,
                    in_fourier = True,
                    out_fourier = True
                ) | phi_mac_gate([ctl_idx] + accumulator_reg)
        
        return phi_mac_gate
            
    

class RCModMultiplier(CompositeGate):
    """
        An in-place modular multiplier using Fourier-basis arithmetic and 
        Montgomery Reduction. Uses in total "2n + logn + 1" qubits which includes 
        "n + logn + 1" clean ancilla qubits.

        Based on paper "High Performance Quantum Modular Multipliers" 
        by Rich Rines, Isaac Chuang: https://arxiv.org/abs/1801.01081
    """

    def __init__(
        self, 
        modulus: int,
        multiple: int,
        qreg_size: int,
        name: str = None
    ):
        """
            Construct the in-place modular multiplier that does:

            |z(n)>|0(n + logn + 1)> ----> |Mz mod N(n)>|0(n + logn + 1)>

            Args:
                modulus (int): modulus in the modular multiplication.

                multiple (int): multiple in the modular multiplication.

                qreg_size (int): the number of qubits to hold the input register and 
                store the result.

            NOTE: currently the quantum data inside |z(n)> has to be smaller than modulus
            for the circuit to properly set all the ancilla qubits back to 0s. This is due to
            the use of out-of-place modular multiplier with inversed multiple. When running the
            circuit backwards, it cannot cancel z >= modulus (it is because when running forward 
            the gate obviously cannot generate value greater than modulus on any register by design).
        """
        
        if int(np.ceil(np.log2(modulus+1))) > qreg_size:
            raise GateParametersAssignedError("Not enough register size for modulus.")
        if modulus%2 == 0:
            raise GateParametersAssignedError("Modulus cannot be an even number.")
        if np.gcd(modulus, multiple) != 1:
            raise GateParametersAssignedError("Modulus and multiple have to be co-prime.")

        self._modulus  = modulus
        self._multiple = multiple
        
        self._register_size = qreg_size
        self._ancilla_n     = qreg_size
        self._ancilla_mp1   = int(np.ceil(np.log2(qreg_size))) + 1

        self._total_size = self._register_size + self._ancilla_n + self._ancilla_mp1

        self._register_list = [i for i in range(qreg_size)]
        self._ancilla_n_list = [i for i in range(qreg_size, 2 * qreg_size)]
        self._ancilla_mp1_list = [i for i in range(2 * qreg_size, self._total_size)]

        super().__init__(name)

        # *multiple % modulus, forward
        RCOutOfPlaceModMultiplier(
            modulus = modulus,
            multiple = multiple,
            qreg_size = qreg_size,
            inverse_multiple = False
        ) | self
        
        # replace input with output and prepare for uncompute 
        for i in range(self._register_size):
            Swap | self([self._register_list[i], self._ancilla_n_list[i]])
        
        # *multiple^(-1) % modulus, backwards
        RCOutOfPlaceModMultiplier(
            modulus = modulus,
            multiple = multiple,
            qreg_size = qreg_size,
            inverse_multiple = True
        ).inverse() | self

        return


class RCModMultiplierCtl(CompositeGate):
    """
        A controlled in-place modular multiplier using Fourier-basis arithmetic and 
        Montgomery Reduction. Uses in total "2n + logn + 1" qubits which includes 
        "n + logn + 1" clean ancilla qubits.

        Based on paper "High Performance Quantum Modular Multipliers" 
        by Rich Rines, Isaac Chuang: https://arxiv.org/abs/1801.01081
    """

    def __init__(
        self, 
        modulus: int,
        multiple: int,
        qreg_size: int,
        name: str = None
    ):
        """
            Construct the in-place modular multiplier that does:

            |c(1)>|z(n)>|0(n + logn + 1)> ----> |c(1)>|c * (Mz mod N) (n)>|0(n + logn + 1)>

            Args:
                modulus (int): modulus in the modular multiplication.

                multiple (int): multiple in the modular multiplication.

                qreg_size (int): the number of qubits to hold the input register and 
                store the result.

            NOTE: For the same reason as the simple in-place modular multiplication,
            The ancilla qubits can be properly set back to 0s only when z < modulus.
        """

        if int(np.ceil(np.log2(modulus+1))) > qreg_size:
            raise GateParametersAssignedError("Not enough register size for modulus.")
        if modulus%2 == 0:
            raise GateParametersAssignedError("Modulus cannot be an even number.")
        if np.gcd(modulus, multiple) != 1:
            raise GateParametersAssignedError("Modulus and multiple have to be co-prime.")
        
        self._modulus  = modulus
        self._multiple = multiple
        
        self._register_size = qreg_size
        self._ancilla_n     = qreg_size
        self._ancilla_mp1   = int(np.ceil(np.log2(qreg_size))) + 1

        self._total_size = 1 + self._register_size + self._ancilla_n + self._ancilla_mp1

        self._control_list = [0]
        self._register_list = [i for i in range(1, qreg_size + 1)]
        self._ancilla_n_list = [i for i in range(qreg_size + 1, 2 * qreg_size + 1)]
        self._ancilla_mp1_list = [i for i in range(2 * qreg_size + 1, self._total_size)]

        super().__init__(name)

        swap_list = [i for i in range(self._total_size - qreg_size, self._total_size) ]
        # 0-ctl cswap two registers
        X | self(self._control_list)
        for i in range(self._register_size):
            CSwap | self(self._control_list + [self._register_list[i], swap_list[i]])
        X | self(self._control_list)

        # *multiple % modulus, forward
        RCOutOfPlaceModMultiplier(
            modulus = modulus,
            multiple = multiple,
            qreg_size = qreg_size,
            inverse_multiple = False
        ) | self(self._register_list + self._ancilla_n_list + self._ancilla_mp1_list)

        # replace input with output and prepare for uncompute (conditioned)
        for i in range(self._register_size):
            CSwap | self(self._control_list + [self._register_list[i], self._ancilla_n_list[i]])
        
        RCOutOfPlaceModMultiplier(
            modulus = modulus,
            multiple = multiple,
            qreg_size = qreg_size,
            inverse_multiple = True
        ).inverse() | self(self._register_list + self._ancilla_n_list + self._ancilla_mp1_list)

        # 0-ctl cswap two registers
        X | self(self._control_list)
        for i in range(self._register_size):
            CSwap | self(self._control_list + [self._register_list[i], swap_list[i]])
        X | self(self._control_list)

        return

    