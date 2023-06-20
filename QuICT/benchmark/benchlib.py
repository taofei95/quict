import numpy as np
import re

from QuICT.core.circuit.circuit import Circuit


class BenchLib:
    """ A data structure for storing benchmark information. """
    @property
    def circuits(self) -> list:
        """ Return the circuits of QuantumMachinebenchmark. """
        return self._circuits

    @circuits.setter
    def circuits(self, circuits:list):
        self._circuits = circuits

    @property
    def machine_amp(self) -> list:
        """ Return the quantum machine sample result of circuits. """
        return self._machine_amp

    @machine_amp.setter
    def machine_amp(self, machine_amp:list):
        self._machine_amp = machine_amp

    @property
    def field(self) -> list:
        """ Return the field of circuits. """
        field_list = []
        for circuit in self._circuits:
            name_split = circuit.name.split("+")[:-1]
            if name_split[0] != "random":
                field_list.append(name_split[0])
            else:
                field_list.append(name_split[1])
        return field_list

    @property
    def level(self) -> list:
        """ Return the level of circuits. """
        level_list = []
        for circuit in self._circuits:
            name_split = int(circuit.name[-1])
            level_list.append(name_split)
        return level_list

    @property
    def simulation_amp(self) -> list:
        """ Return the simulation sample result of circuits. """
        simulation_amp = []
        for i in range(len(self._circuits)):
            based_name = self._circuits[i].name
            based_field = self._circuits[i].name.split("+")[:-1]
            if based_field[0] == "algorithm":
                amp_result = np.load(f"QuICT/benchmark/cir_amp_result/{str(based_field[1])}/{str(based_name[:-7])}.npy")
                cir_sim_amp = bin(np.where(amp_result==np.max(amp_result))[0][0])[2:]
                simulation_amp.append(cir_sim_amp)

        return simulation_amp

    @property
    def fidelity(self) -> list:
        """ Return the fidelity of each circuit. """
        qv_list, fidelity_list, evaluate_list = [], [], []
        machine_amp, simulation_amp = self.machine_amp.copy(), self.simulation_amp.copy()
        for i in range(len(self.field)):
            # quantum volumn
            cir_attribute = re.findall(r"\d+", self.circuits[i].name)
            QV = min(int(cir_attribute[0]), int(cir_attribute[2]))
            qv_list.append(QV)
            # fidelity
            if self.field[i] != 'algorithm':
                fidelity_list.append(machine_amp[0][0])
                machine_amp.remove(machine_amp[0])
            else:
                index = int(simulation_amp[0])
                fidelity_list.append(machine_amp[0][index])
                simulation_amp.remove(simulation_amp[0])
                machine_amp.remove(machine_amp[0])
        level_list = self.level
        for i in range(len(qv_list)):
            evaluate_list.append(qv_list[i] * fidelity_list[i] * level_list[i])

        return evaluate_list

    def __init__(
        self,
        circuits:list = None,
        machine_amp:list = None,
    ):
        """
        Args:
            circuits (list, optional): The list of circuit which from QuantumMachinebenchmark.
            field (List[str], optional): The field of each circuit in circuits.
            level (List[int], optional): The level of each circuit in circuits.
            machine_amp (List[np.array], optional): The list of the quantum machine amplitude of the input circuit. Defaults to None.
            fidelity (List[float], optional): The fidelity of each circuit.
        """
        self._circuits = circuits
        self._machine_amp = machine_amp

        if circuits is not None:
            assert isinstance(circuits, list)
            for cir in circuits:
                assert isinstance(cir, Circuit)
            self.circuits = circuits
        if machine_amp is not None:
            self.machine_amp = machine_amp