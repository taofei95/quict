import random
import logging
from math import pi, gcd
import numpy as np
from fractions import Fraction
from typing import List, Tuple

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.operator import Trigger
from .utility import *

from QuICT.simulation.cpu_simulator import CircuitSimulator

from .BEA_zip import construct_circuit as BEA_zip_circuit
from .BEA_zip import order_finding as BEA_zip_run
from .BEA import construct_circuit as BEA_circuit
from .BEA import order_finding as BEA_run

from .HRS_zip import construct_circuit as HRS_zip_circuit
from .HRS_zip import order_finding as HRS_zip_run
from .HRS import construct_circuit as HRS_circuit
from .HRS import order_finding as HRS_run


class ShorFactor:

    _ALLOWED_MODES = {"BEA", "HRS", "BEA_zip", "HRS_zip"}
    _RUN_METHOD_OF_MODE = {
        "BEA": reinforced_order_finding_constructor(BEA_run),
        "HRS": reinforced_order_finding_constructor(HRS_run),
        "BEA_zip": reinforced_order_finding_constructor(BEA_zip_run),
        "HRS_zip": reinforced_order_finding_constructor(HRS_zip_run),
    }
    _CIRCUIT_METHOD_OF_MODE = {
        "BEA": BEA_circuit,
        "HRS": HRS_circuit,
        "BEA_zip": BEA_zip_circuit,
        "HRS_zip": HRS_zip_circuit,
    }

    # add a, N here
    def __init__(self, mode: str, N: int, eps: float = 1 / 10, max_rd: int = 2) -> None:
        random.seed(2022)
        a = N
        while gcd(a, N) != 1:
            a = random.randrange(0, N)
        self._CIRCUIT_CACHE = ShorFactor._CIRCUIT_METHOD_OF_MODE[mode](a, N, eps)
        if mode not in ShorFactor._ALLOWED_MODES:
            raise ValueError(
                f"{mode} mode is not valid. Consider {ShorFactor._ALLOWED_MODES}"
            )
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
        return self._CIRCUIT_CACHE

    def run(
        self,
        simulator=CircuitSimulator(),
        circuit: Circuit = None,
        indices: List = None,
    ) -> int:
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
                logging.info(
                    f"Shor succeed: N is exponential, found the only factor {u1} classically"
                )
                return u1
            if pow(u2, b) == self.N:
                logging.info(
                    f"Shor succeed: N is exponential, found the only factor {u2} classically"
                )
                return u2
        rd = 0
        while rd < self.max_rd:
            # 3. Choose a random number a (1<a<N)
            a = random.randint(2, self.N - 1)
            gcd = np.gcd(a, self.N)
            if gcd > 1:
                logging.info(
                    f"Shor succeed: randomly chosen a = {a}, who has common factor {gcd} with N classically"
                )
                return gcd
            logging.info(f"round = {rd}")
            rd += 1
            # 4. Use quantum order-finding algorithm to find the order of a
            logging.info(
                f"Quantumly determine the order of the randomly chosen a = {a}"
            )
            # check if any input circuit. if no, run according to `mode`; else run the input circuit
            if circuit is None:
                r = ShorFactor._RUN_METHOD_OF_MODE[self.mode](
                    a=a, N=self.N, simulator=simulator
                )
            else:
                simulator.run(circuit)
                if len(indices) > 0 and type(indices[0]) == int:
                    phi = int(circuit[indices]) / (1 << len(indices))
                elif len(indices) > 0 and type(indices[0]) == Trigger:
                    phi = eval(
                        "0b" + "".join([str(trig.measured[0]) for trig in indices])
                    ) / (1 << len(indices))
                else:
                    raise ValueError("wrong indices")
                logging.info(f"phi: {phi:4.3f}")
                r = Fraction(phi).limit_denominator(self.N - 1).denominator

            if r == 0:  # no order found
                logging.info(
                    f"Shor failed: did not find the order of a = {a}"
                )  # add mode info here
            elif r % 2 == 1:  # odd order
                logging.info(f"Shor failed: found odd order r = {r} of a = {a}")
            else:  # N | h^2 - 1, h = a^(r/2)
                h = pow(a, int(r / 2), self.N)
                if h == self.N - 1:  # N | h + 1
                    logging.info(
                        f"Shor failed: found order r = {r} of a = {a} with negative square root"
                    )
                else:  # N !| h + 1, therefore N | h - 1
                    f1, f2 = np.gcd(h - 1, self.N), np.gcd(h + 1, self.N)
                    if f1 > 1 and f1 < self.N:
                        logging.info(
                            f"Shor succeed: found factor {f1}, with the help of a = {a}, r = {r}"
                        )
                        return f1
                    elif f2 > 1 and f2 < self.N:
                        logging.info(
                            f"Shor succeed: found factor {f2}, with the help of a = {a}, r = {r}"
                        )
                        return f2
                    else:
                        logging.info(f"Shor failed: can not find a factor with a = {a}")
        return 0
