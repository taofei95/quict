import random
import logging
from math import pi
import numpy as np
from fractions import Fraction
from typing import List, Tuple

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.bea import *
from .utility import *

from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.simulation import Simulator
from QuICT.algorithm.quantum_algorithm.shor import BEA_zip_run, BEA_circuit, HRS_zip_run, HRS_circuit
class ShorFactor:

    allowed_modes = {"BEA", "HRS", "BEA_zip", "HRS_zip"}
    run_method_of_mode = {"BEA":None, "HRS":None, "BEA_zip":BEA_zip_run, "HRS_zip":HRS_zip_run}
    #TODO: circuit_method_of_mode["BEA"]/["HRS"], without one-bit trick
    circuit_method_of_mode = {"BEA":BEA_circuit, "HRS":HRS_circuit, "BEA_zip":None, "HRS_zip":None}

    def __init__(self, mode: str, N: int, eps: float = 1 / 10, max_rd: int = 2) -> None:
        if mode not in ShorFactor.allowed_modes:
            raise ValueError(f"{mode} mode is not valid. Consider {ShorFactor.allowed_modes}")
        self.mode = mode
        self.N = N
        self.eps = eps
        self.max_rd = max_rd

    def circuit(self) -> Tuple[Circuit, List[int]]:
        """construct the quantum part of Shor algorithm, i.e. order finding circuit

        Returns:
            Circuit: order finding circuit that can be passed to ShorFactor::run method
            List[int]: the indices to be measured to get ~phi
        """
        if ShorFactor.circuit_method_of_mode[self.mode] == None:
            raise ValueError(f"{self.mode} mode has no circuit() method.")
        return ShorFactor.circuit_method_of_mode[self.mode]() #TODO: add construction params

    def run(self, circuit: Circuit = None, indices: List[int] = None, simulator: Simulator = CircuitSimulator()) -> int:
        # check if input is prime (using MillerRabin in klog(N), k is the number of rounds to run MillerRabin)
        if miller_rabin(self.N):
            logging.info("N does not pass miller rabin test, may be a prime number")
            return 0
        # 1. If n is even, return the factor 2
        if self.N % 2 == 0:
            logging.info("Shor succeed: N is even, found factor 2 classically")
            return 2
        # 2. Classically determine if N = p^q
        y, L = np.log2(self.N), int(np.ceil(np.log2(self.N)))
        for b in range(2, L):
            squeeze = np.power(2, y / b)
            u1, u2 = int(np.floor(squeeze)), int(np.ceil(squeeze))
            if pow(u1, b) == self.N:
                logging.info(f"Shor succeed: N is exponential, found the only factor {u1} classically")
                return u1
            if pow(u2, b) == self.N:
                logging.info(f"Shor succeed: N is exponential, found the only factor {u2} classically")
                return u2
        rd = 0
        while rd < self.max_rd:
            # 3. Choose a random number a (1<a<N)
            a = random.randint(2, self.N - 1)
            gcd = np.gcd(a, self.N)
            if gcd > 1:
                logging.info(f'Shor succeed: randomly chosen a = {a}, who has common factor {gcd} with N classically')
                return gcd
            logging.info(f'round = {rd}')
            rd += 1
            # 4. Use quantum order-finding algorithm to find the order of a
            logging.info(f'Quantumly determine the order of the randomly chosen a = {a}')
            # check if any input circuit. if no, run according to `mode`; else run the input circuit
            if circuit == None:
                r = ShorFactor.run_method_of_mode[self.mode]() #TODO: add run() params
            else:
                simulator.run(circuit) # TODO: run the circuit with fresh start
                phi = int(circuit[indices])<<len(indices) # TODO: break to see if it is ~phi
                r = Fraction(phi).limit_denominator(self.N - 1).denominator # TODO: break to see if it works

            if r == 0: # no order found
                logging.info(f'Shor failed: did not find the order of a = {a}')
            elif r % 2 == 1: # odd order
                logging.info(f'Shor failed: found odd order r = {r} of a = {a}')
            else: # N | h^2 - 1, h = a^(r/2)
                h = pow(a, int(r / 2), self.N)
                if h == self.N - 1:  # N | h + 1
                    logging.info(f'Shor failed: found order r = {r} of a = {a} with negative square root')
                else: # N !| h + 1, therefore N | h - 1
                    f1,f2 = np.gcd(h - 1, self.N), np.gcd(h + 1, self.N)
                    if f1 > 1 and f1 < self.N:
                        logging.info(f'Shor succeed: found factor {f1}, with the help of a = {a}, r = {r}')
                        return f1
                    elif f2 > 1 and f2 < self.N:
                        logging.info(f'Shor succeed: found factor {f2}, with the help of a = {a}, r = {r}')
                        return f2
                    else:
                        logging.info(f'Shor failed: can not find a factor with a = {a}')
        return 0
