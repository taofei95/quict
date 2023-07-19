import numpy as np

from QuICT.core.gate import *
from QuICT.core import Circuit

from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.differentiator.adjoint import (
    AdjointDifferentiator,
)


class Differentiator:
    """ The high-level differentiator class, including all QuICT differentiator mode. """

    __DEVICE = ["CPU", "GPU"]
    __BACKEND = ["adjoint"]
    __PRECISION = ["single", "double"]
    __OPTIONS_DICT = {
        "adjoint": ["gpu_device_id"],
        "parameter_shift": ["gpu_device_id"],
    }

    def __init__(
        self,
        device: str = "GPU",
        backend: str = "adjoint",
        precision: str = "double",
        **options,
    ):
        """Initialize a differentiator.

        Args:
            device (str, optional): The device of the simulator. One of [CPU, GPU]. Defaults to "GPU".
            backend (str, optional): The backend for the simulator. One of ["adjoint"]. Defaults to "adjoint".
            precision (str, optional): The precision of simulator, one of ["single", "double"]. Defaults to "double".
            **options (dict): other optional parameters for the simulator.
                adjoint: [gpu_device_id] (only for gpu)
        """
        assert device in Differentiator.__DEVICE, ValueError(
            "Differentiator.device", "[CPU, GPU]", device
        )
        self._device = device
        assert backend in Differentiator.__BACKEND, ValueError(
            "Differentiator.backend", "[adjoint]", backend
        )
        self._backend = backend
        assert precision in Differentiator.__PRECISION, ValueError(
            "Differentiator.precision", "[single, double]", precision
        )
        self._precision = precision

        if options:
            if not self._options_validation(options):
                raise ValueError

        self._options = options

        # load differentiator
        self._differentiator = self._load_differentiator()

    def _options_validation(self, options: dict) -> bool:
        default_option_list = Differentiator.__OPTIONS_DICT[self._backend]
        option_keys = list(options.keys())

        for option_key in option_keys:
            if option_key not in default_option_list:
                return False

        return True

    def _load_differentiator(self):
        if self._backend == "adjoint":
            differentiator = AdjointDifferentiator(
                self._device, self._precision, **self._options
            )
        else:
            raise ValueError

        return differentiator

    def run(
        self,
        circuit: Circuit,
        variables: Variable,
        state_vector: np.ndarray,
        expectation_op: Hamiltonian,
    ):
        """Calculate the gradients and expectation of a Parameterized Quantum Circuit (PQC).

        Args:
            circuit (Circuit): PQC that needs to calculate gradients.
            variables (Variable): The parameters of the circuit.
            state_vector (np.ndarray): The state vector output from forward propagation.
            expectation_op (Hamiltonian): The hamiltonian that need to get expectation.

        Returns:
            Variable: The parameters with gradients.
            np.float: The expectation.
        """
        return self._differentiator.run(
            circuit, variables, state_vector, expectation_op
        )

    def run_batch(
        self,
        circuit: Circuit,
        variables: Variable,
        state_vector_list: list,
        expectation_op: Hamiltonian,
    ):
        """Calculate the gradients and expectations of a batch of PQCs.

        Args:
            circuit (Circuit): PQC that needs to calculate gradients.
            variables (Variable): The parameters of the circuit.
            state_vector_list (list): The state vectors output from multiple FP process.
            expectation_op (Hamiltonian): The hamiltonian that need to get expectation.

        Returns:
            list: The list of parameters with gradients.
            np.ndarray: The expectations.
        """
        return self._differentiator.run_batch(
            circuit, variables, state_vector_list, expectation_op
        )

    def get_expectation(
        self, state_vector: np.ndarray, expectation_op: Hamiltonian,
    ):
        """Calculate the expectation of a PQC.

        Args:
            state_vector (np.ndarray): The state vector output from forward propagation.
            expectation_op (Hamiltonian): The hamiltonian that need to get expectation.

        Returns:
            np.float: The expectation.
        """
        return self._differentiator.get_expectation(state_vector, expectation_op)

    def get_expectations_batch(
        self, state_vector_list: list, expectation_op: Hamiltonian,
    ):
        """Calculate the expectations of a batch of PQCs.

        Args:
            state_vector_list (list): The state vectors output from multiple FP process.
            expectation_op (Hamiltonian): The hamiltonian that need to get expectation.

        Returns:
            np.ndarray: The expectations.
        """
        return self._differentiator.get_expectations_batch(
            state_vector_list, expectation_op
        )

