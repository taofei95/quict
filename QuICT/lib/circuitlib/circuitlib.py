import os
import shutil
from typing import Union, List

from QuICT.core import Circuit
from QuICT.lib.qasm.exceptions import QasmError
from QuICT.tools import Logger
from QuICT.tools.interface import OPENQASMInterface
from .circuit_lib_sql import CircuitLibDB


logger = Logger("QuICT_Circuit_Library")


class CircuitLib:
    """
    A class handling QuICT circuit library.

    Args:
        output_type (str, optional): one of [circuit, qasm, file]. Defaults to "circuit".
        output_path (str, optional): The path to store qasm file if output type is file.
            Defaults to the current working path.
    """

    __DEFAULT_TYPE = ["template", "random", "algorithm", "benchmark", "instructionset"]
    __DEFAULT_CLASSIFY = {
        "random": [
            "aspen-4", "ourense", "rochester", "sycamore", "ctrl_diag",
            "ctrl_unitary", "diag", "single_bits", "unitary", "tokyo"
        ],
        "algorithm": ["adder", "clifford", "grover", "qft", "supremacy", "vqe"],
        "benchmark": ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"],
        "instructionset": ["google", "ibmq", "ionq", "ustc", "quafu"]
    }
    __LIB_PATH = os.path.join(os.path.dirname(__file__), 'circuit_qasm')

    def __init__(
        self,
        output_type: str = "circuit",
        output_path: str = '.'
    ):
        self._output_type = output_type
        self._output_path = output_path
        logger.debug(f"Initial Circuit Library with output {self._output_type}.")

        self._db = CircuitLibDB()

    @property
    def size(self) -> int:
        return self._db.size()

    def _get_circuit_from_qasm(self, file_path: str) -> Circuit:
        """ Load Circuit from a qasm file. """
        qasm = OPENQASMInterface.load_file(file_path)
        if not qasm.valid_circuit:
            logger.warn("Failure to load circuit from qasm.")
            raise QasmError("Missing input file")

        split_path = file_path.split('/')
        cir_name, cir_classify, cir_type  = split_path[-1:-4:-1]
        cir_name = os.path.splitext(cir_name)[0]
        qasm.circuit.name = "+".join([cir_type, cir_classify, cir_name])
        return qasm.circuit

    def _get_string_from_qasm(self, file_path: str) -> str:
        """ Return qasm string from qasm file. """
        return open(file_path, 'r').read()

    def _copy_qasm_file(self, file_path: str):
        """ Copy target qasm file to the given output path. """
        filename = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(self._output_path, filename))        

    def _get_all(self, folder: str, files: list) -> Union[List, None]:
        """ Load all qasm files in the list of files.

        Args:
            folder (str): The path of the qasm folder
            files (list): the list of file names

        Returns:
            List[(Circuit|string)]: _description_
        """
        circuit_all = []
        for file in files:
            file_path = os.path.join(folder, file[0])
            if self._output_type == "circuit":
                circuit_all.append(self._get_circuit_from_qasm(file_path))
            elif self._output_type == "qasm":
                circuit_all.append(self._get_string_from_qasm(file_path))
            else:
                self._copy_qasm_file(file_path)

        return circuit_all

    def get_template_circuit(
        self,
        max_width: int = None,
        max_size: int = None,
        max_depth: int = None
    ) -> Union[List[Union[Circuit, str]], None]:
        """
        Get template circuits in QuICT circuit library. A template will be loaded if
        it satisfies the following restrictions:
            1. its number of qubits <= `max_width`,
            2. its number of gates <= `max_size`,
            3. its depth <= `max_depth`.

        Restrictions will be ignored if not specified.

        Args:
            max_width(int): max number of qubits
            max_size(int): max number of gates
            max_depth(int): max depth

        Returns:
            (List[Circuit | String] | None): Return the list of output circuit order by output_type.
        """
        path = os.path.join(self.__LIB_PATH, "template")
        files = self._db.circuit_filter("template", "template", max_width, max_size, max_depth)

        return self._get_all(path, files)

    def get_random_circuit(
        self,
        classify: str,
        max_width: int = None,
        max_size: int = None,
        max_depth: int = None
    ) -> Union[List[Union[Circuit, str]], None]:
        """Get random circuits in QuICT circuit library. A template will be loaded if
        it satisfies the following restrictions:
            1. the circuit in the given classify.
            2. its number of qubits <= `max_width`,
            3. its number of gates <= `max_size`,
            4. its depth <= `max_depth`.

        Restrictions will be ignored if not specified.

        Args:
            classify (str): one of [ctrl_diag, ctrl_unitary, diag, qft, single_bit, unitary, ...]
            max_width (int, optional): _description_. Defaults to float('inf').

        Returns:
            (List[Circuit | String] | None): Return the list of output circuit order by output_type.
        """
        assert classify in self.__DEFAULT_CLASSIFY['random'], "error classify."
        path = os.path.join(self.__LIB_PATH, 'random', classify)
        files = self._db.circuit_filter("random", classify, max_width, max_size, max_depth)

        return self._get_all(path, files)

    def get_algorithm_circuit(
        self,
        classify: str,
        max_width: int = None,
        max_size: int = None,
        max_depth: int = None
    ) -> Union[List[Union[Circuit, str]], None]:
        """ Get algorithm circuits in QuICT circuit library. A template will be loaded if
        it satisfies the following restrictions:
            1. the circuit in the given classify.
            2. its number of qubits <= `max_width`,
            3. its number of gates <= `max_size`,
            4. its depth <= `max_depth`.

        Restrictions will be ignored if not specified.

        Args:
            classify (str): one of [clifford, grover, qft, supremacy, vqe, ...]
            max_width (int, optional): _description_. Defaults to float('inf').

        Returns:
            (List[Circuit | String] | None): Return the list of output circuit order by output_type.
        """
        assert classify in self.__DEFAULT_CLASSIFY['algorithm'], "error classify."
        path = os.path.join(self.__LIB_PATH, 'algorithm', classify)
        files = self._db.circuit_filter("algorithm", classify, max_width, max_size, max_depth)

        return self._get_all(path, files)

    def get_benchmark_circuit(
        self,
        classify: str,
        max_width: int = None,
        max_size: int = None,
        max_depth: int = None
    ) -> Union[List[Union[Circuit, str]], None]:
        """ Get experiment circuits in QuICT circuit library. A template will be loaded if
        it satisfies the following restrictions:
            1. the circuit in the given classify.
            2. its number of qubits <= `max_width`,
            3. its number of gates <= `max_size`,
            4. its depth <= `max_depth`.

        Restrictions will be ignored if not specified.

        Args:
            classify (str): one of [adder, mapping]
            max_width (int, optional): _description_. Defaults to float('inf').

        Returns:
            (List[Circuit | String] | None): Return the list of output circuit order by output_type.
        """
        assert classify in self.__DEFAULT_CLASSIFY['benchmark'], "error experiment classify."
        path = os.path.join(self.__LIB_PATH, 'benchmark', classify)
        files = self._db.circuit_filter("benchmark", classify, max_width, max_size, max_depth)

        return self._get_all(path, files)

    def get_instructionset_circuit(
        self,
        classify: str,
        max_width: int = None,
        max_size: int = None,
        max_depth: int = None
    ) -> Union[List[Union[Circuit, str]], None]:
        """ Get experiment circuits in QuICT circuit library. A template will be loaded if
        it satisfies the following restrictions:
            1. the circuit in the given classify.
            2. its number of qubits <= `max_width`,
            3. its number of gates <= `max_size`,
            4. its depth <= `max_depth`.

        Restrictions will be ignored if not specified.

        Args:
            classify (str): one of [adder, mapping]
            max_width (int, optional): _description_. Defaults to float('inf').

        Returns:
            (List[Circuit | String] | None): Return the list of output circuit order by output_type.
        """
        assert classify in self.__DEFAULT_CLASSIFY['instructionset'], "error experiment classify."
        path = os.path.join(self.__LIB_PATH, 'instructionset', classify)
        files = self._db.circuit_filter("instructionset", classify, max_width, max_size, max_depth)

        return self._get_all(path, files)

    def get_circuit(
        self,
        type: str,
        classify: str = "template",
        max_width: int = None,
        max_size: int = None,
        max_depth: int = None
    ) -> Union[List[Union[Circuit, str]], None]:
        """Get the target circuits from QuICT Circuit Library.

        Args:
            type (str): The type of circuits, one of [template, random, algorithm, experiment].
            classify (str, optional): The classify of selected circuit's type.
                WARNING: Only in template mode, the classify can be None. \n
                For template circuit's type, classify is None; \n
                For random circuit's type, classify is one of
                    [ctrl_diag, ctrl_unitary, diag, qft, single_bit, unitary, ...]
                For algorithm circuit's type, classify is one of
                    [clifford, grover, qft, supremacy, vqe, ...]
                For experiment circuit's type, classify is one of [adder, mapping]

            qubit_num (int, optional): upper bound of qubit number, default to None.
            size (int, optional): upper bound of size of templates, default to None.
            depth (int, optional): upper bound of depth of templates, default to None.

        Returns:
            (List[Circuit | String] | None): Return the list of output circuit order by output_type.
        """
        if type not in self.__DEFAULT_TYPE:
            raise KeyError("error_type")

        if classify not in self.__DEFAULT_CLASSIFY[type]:
            raise KeyError("error matched")

        files = self._db.circuit_filter(type, classify, max_width, max_size, max_depth)
        folder_path = os.path.join(self.__LIB_PATH, type, classify) if type != "template" else os.path.join(self.__LIB_PATH, type)

        return self._get_all(folder_path, files)
