# !/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 2:02 下午
# @Author  : Han Yu
# @File    : _simulator
import numpy as np
import torch
from typing import Union

from QuICT.core import Circuit
from QuICT.core.noise import NoiseModel
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.simulation.unitary import UnitarySimulator
from QuICT.simulation.density_matrix import DensityMatrixSimulator
from QuICT.simulation.utils import Result
from QuICT.tools.exception.core import ValueError
from QuICT.tools.exception.simulation import SimulatorOptionsUnmatchedError
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian


class Simulator:
    """ The high-level simulation class, including all QuICT simulator mode.

    Args:
        device (str): The device of the simulator. One of [CPU, GPU]
        backend (str): The backend for the simulator. One of [unitary, state_vector, density_matrix]
        shots (int): The running times; must be a positive integer, default to 1.
        precision (str): The precision of simulator, one of [single, double], default to double.
        circuit_record (bool): whether record circuit's qasm in output, default to False.
        amplitude_record (bool): whether record the amplitude of qubits, default to False.
        **options (dict): other optional parameters for the simulator.
            state_vector: [gpu_device_id] (only for gpu)
            density_matrix: [accumulated_mode]
            unitary: None
    """

    __DEVICE = ["CPU", "GPU"]
    __BACKEND = ["unitary", "state_vector", "density_matrix"]
    __PRECISION = ["single", "double"]
    __OPTIONS_DICT = {
        "state_vector": ["gpu_device_id"],
        "density_matrix": ["accumulated_mode"]
    }

    def __init__(
        self,
        device: str = "CPU",
        backend: str = "state_vector",
        precision: str = "double",
        circuit_record: bool = False,
        amplitude_record: bool = False,
        output_path: str = None,
        **options
    ):
        assert device in Simulator.__DEVICE, ValueError("Simulator.device", "[CPU, GPU]", device)
        self._device = device
        assert backend in Simulator.__BACKEND, \
            ValueError("Simulator.backend", "[unitary, state_vector, density_matrix]", backend)
        self._backend = backend
        assert precision in Simulator.__PRECISION, ValueError("Simulator.precision", "[single, double]", precision)
        self._precision = precision

        if options:
            if not self._options_validation(options):
                raise SimulatorOptionsUnmatchedError(
                    f"Unmatched options arguments depending on {self._device} and {self._backend}."
                )

        self._options = options

        # load simulator
        self._simulator = self._load_simulator()

        # Result's arguments
        self._result_recorder = Result(
            device, backend, precision, circuit_record, amplitude_record, self._options, output_path
        )
        self.isRun=False

    def _options_validation(self, options: dict) -> bool:
        if self._backend == "unitary" or (self._backend == "state_vector" and self._device == "CPU"):
            return False

        default_option_list = Simulator.__OPTIONS_DICT[self._backend]
        option_keys = list(options.keys())

        for option_key in option_keys:
            if option_key not in default_option_list:
                return False

        return True

    def _load_simulator(self):
        """ Initial simulator. """
        if self._backend == "state_vector":
            simulator = StateVectorSimulator(self._device, precision=self._precision, **self._options)
        elif self._backend == "unitary":
            simulator = UnitarySimulator(device=self._device, precision=self._precision)
        else:
            simulator = DensityMatrixSimulator(device=self._device, precision=self._precision, **self._options)

        return simulator

    def run(
        self,
        circuit: Union[Circuit, np.ndarray],
        shots: int = 1,
        state_vector: np.ndarray = None,
        density_matrix: np.ndarray = None,
        noise_model: NoiseModel = None,
        use_previous: bool = False
    ):
        """ start simulator with given circuit

        Args:
            circuit Union[Circuit, np.ndarray]: The quantum circuits or unitary matrix.
            state_vector (ndarray): The initial state vector.
            density_matrix (ndarray): The initial density matrix.
            noise_model (NoiseModel, optional): The NoiseModel only for density_matrix simulator. Defaults to None.
            use_previous (bool, optional): Using the previous state vector. Defaults to False.

        Yields:
            [dict]: The Result Dict.
        """
        if isinstance(circuit, np.ndarray) and self._backend != "unitary":
            raise SimulatorOptionsUnmatchedError(
                f"The unitary matrix input only allows in the unitary backend, not {self._backend}."
            )

        if (density_matrix is not None or noise_model is not None) and self._backend != "density_matrix":
            raise SimulatorOptionsUnmatchedError(
                "The density matrix and noise model input only allows in the density_matrix backend, " +
                f"not {self._backend}."
            )

        if state_vector is not None and self._backend == "density_matrix":
            raise SimulatorOptionsUnmatchedError(
                "The state vector input is not allowed in the density matrix backend."
            )

        if isinstance(circuit, Circuit):
            self._result_recorder.record_circuit(circuit)

        if self._backend == "density_matrix":
            amplitude = self._simulator.run(circuit, density_matrix, noise_model, use_previous)
        else:
            amplitude = self._simulator.run(circuit, state_vector, use_previous)

        self._result_recorder.record_amplitude(amplitude)

        sample_result = self._simulator.sample(shots)
        self._result_recorder.record_sample(sample_result)
        self.isRun = True

        return self._result_recorder.__dict__()
<<<<<<< HEAD
    
=======

>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
    def get_expectation(self,state_vector,Hamiltonian:Hamiltonian,n_qubits:int ):
        """ a toy method that get the expectation acted on targetr qubit of a qcircuit with hermitian obersered operator 

        Args:
            state_vector(tensor): 
            Hamiltonian(): a  hermitian obersered operator 
            n_qubits(int):the number of the circuit Hamiltonian acting on 
        Returns:
            the expectation of the circuit
        """
        e_val = 0
        if not self.isRun:
            raise TypeError("StateVectorSimulation.get_expetation should be executed after StateVectorSimulation.run!")
        bra=state_vector.copy().conj()
        ket=state_vector.copy()
        e_val = np.dot(bra,Hamiltonian.get_hamiton_matrix(n_qubits=n_qubits))
        e_val = np.dot(e_val,ket).real

        return e_val
    def forward(self,cir:Circuit, ham:Hamiltonian=Hamiltonian([[0.5, 'Y0', 'X4', 'Z2', 'I6'], [0.0]])):
        state_vector=self.run(cir)
        state_vector= state_vector['data']['state_vector']
        #ham = Hamiltonian([[0.5, 'Y0', 'X4', 'Z2', 'I6'], [0.0]])
        e_val=self.get_expectation(state_vector,ham,len(cir._qubits))
        return e_val,state_vector
    def backward(self,cir:Circuit,idx_gate:int,idx_param:int,grad,lr:float):
        gate =cir.gates[idx_gate]
        param_list = gate.pargs.copy()
        param_list[idx_param] +=lr*grad
        gate.pargs=param_list.copy()
        cir.replace_gate(idx_gate,gate)
        return cir
    def backward(self,cir:Circuit,grad:list,lr):
        index = 0
        for gate in cir.gates:
            if not gate.is_requires_grad():
                continue
            params = [gate.pargs]
            if len(params)!=1:
                params= params[0]
            new_params=gate.pargs.copy()
            for i in range(len(new_params)):
                new_params[i]+=lr*grad[index]
                index+=1
            idx= cir.gates.index(gate)
            gate.pargs=new_params
            cir.replace_gate(idx,gate)
        return cir
<<<<<<< HEAD
=======

>>>>>>> 3f5539fac7f58b5765c00c227eb2da8bfa11b3dd
