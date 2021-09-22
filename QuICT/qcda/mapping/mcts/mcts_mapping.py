
from typing import List

from QuICT.core.circuit import *
from QuICT.core.exception import *
from QuICT.core.gate import *
from QuICT.core.layout import *
from QuICT.qcda.mapping.utility import *
from QuICT.qcda.mapping._mapping import Mapping
from .mcts import *


class MCTSMapping(Mapping):
    @classmethod
    def execute(
        cls,
        circuit: Circuit,
        layout: Layout,
        init_mapping: List[int] = None,
        init_mapping_method: str = "naive",
        inplace: bool = False,
        Nsim: int = 2,
        Nsch: int = 5000,
        num_of_process: int = 4
    ) -> Circuit:
        """Mapping the logical circuit to a NISQ device.
        Args:
            circuit: The input circuit that needs to be mapped to a NISQ device.
            layout: The physical layout of the NISQ devices.
            init_mapping: Initial position of logical qubits on physical qubits.
                The argument is optional. If not given, it will be determined by init_mapping method.
                A simple Layout instance is shown as follow:
                    index: logical qubit -> List[index]:physical qubit
                        4-qubit device init_mapping: [ 3, 2, 0, 1 ]
                            logical qubit -> physical qubit
                            0         3
                            1         2
                            2         0
                            3         1
            init_mapping_method: The method used to dertermine the initial mapping.
                "naive": Using identity mapping, i.e., [0,1,...,n] as the initial mapping.
                "anneal": Using simmulated annealing method[1] to generate the initial mapping.
            inplace: Indicate wether the algorithm returns a new circuit or modifies the original circuit in place.
            Nsim: The repeated times of the simulation module.
            Nsch: Number of search times in MCTS.
            num_of_process: Number of threads used in tree parallel MCTS.
        Return:
            the hardware-compliant circuit after mapping.

        [1]Zhou, X., Li, S., & Feng, Y. (2020). Quantum Circuit Transformation Based on Simulated Annealing and
        Heuristic Search. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 39, 4683-4694.
        """
        circuit.const_lock = True
        num = layout.qubit_number

        circuit_dag = DAG(circuit=circuit, mode=Mode.TWO_QUBIT_CIRCUIT)
        coupling_graph = CouplingGraph(coupling_graph=layout)

        if init_mapping is None:
            num_of_qubits = coupling_graph.size
            if init_mapping_method == "anneal":
                cost_f = Cost(circuit=circuit_dag, coupling_graph=coupling_graph)
                init_mapping = np.random.permutation(num_of_qubits)
                _, best_mapping = simulated_annealing(
                    init_mapping=init_mapping,
                    cost=cost_f,
                    method="nnc",
                    param={"T_max": 100, "T_min": 1, "alpha": 0.99, "iterations": 1000}
                )
                init_mapping = list(best_mapping)
            elif init_mapping_method == "naive":
                init_mapping = [i for i in range(num_of_qubits)]
            else:
                raise Exception("No such initial mapping method")

        if not isinstance(init_mapping, list):
            raise Exception("Layout should be a list of integers")
        mcts_tree = MCTS(coupling_graph=coupling_graph, Nsim=Nsim, selection_times=Nsch, num_of_process=num_of_process)
        mcts_tree.search(logical_circuit=circuit, init_mapping=init_mapping)

        gates = mcts_tree.physical_circuit

        circuit.const_lock = False
        if inplace:
            circuit.set_flush_gates(gates)
        else:
            new_circuit = Circuit(num)
            new_circuit.extend(gates)
            return new_circuit
