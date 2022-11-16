import os
import yaml

from QuICT.core.layout import Layout
from QuICT.tools.interface import OPENQASMInterface


class JobValidation:
    _SIMULATION_PARAMETERS = {
        "precision": ["single", "double"],
        "backend": ["state_vector", "density_matrix", "unitary"],
        "device": ["CPU", "GPU"]
    }

    _BASED_METHODS = ["GateTransform", "Clifford", "Auto", "Commutative", "SymbolicClifford", "Template", "CNOT"]

    _QCDA_PARAMETERS = {
        "instruction_set": ["USTC", "Google", "IBMQ", "IonQ", "Quafu"],
        "auto_mode": ["light", "heavy"]
    }

    def __init__(self):
        self._job_template_path = os.path.join(
            os.path.dirname(__file__),
            "../template"
        )

    @staticmethod
    def get_circuit_info(qasm_file: str) -> dict:
        """ whether the given qasm file is valid or not. If not valid, raise ValueError. """
        try:
            circuit = OPENQASMInterface.load_file(qasm_file).circuit
        except:
            raise ValueError(f"Failure to load circuit from given file. {qasm_file}.")

        return {
            "qasm": circuit.qasm(),
            "width": circuit.width(),
            "size": circuit.size(),
            "depth": circuit.depth()
        }

    @staticmethod
    def output_path_prepare(job_name: str, output_path: str):
        # output-path preparation
        output_path = os.path.join(output_path, job_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    @staticmethod
    def load_yaml_file(file_path: str) -> dict:
        with open(os.path.abspath(file_path), encoding='utf-8') as yml:
            job_info = yaml.load(yml)

        return job_info

    def _based_info(self, job_dict):
        # Necessary feature
        name = job_dict["job_name"]
        assert isinstance(name, str), f"Job's name shoule be a string, not {type(name)}."
        _type = job_dict["type"]
        assert _type in ["qcda", "simulation"], f"Job's type should be one of [qcda, simulation], not {_type}."

        # circuit's qasm file validation
        circuit_info = JobValidation.get_circuit_info(job_dict["circuit"])
        job_dict['circuit_info'] = circuit_info

    def _simulation_validation(self, simulation_dict: dict):
        # validation job's simulation
        assert simulation_dict["shots"] >= 1, "The number of shots >= 1."
        assert simulation_dict["precision"] in self._SIMULATION_PARAMETERS['precision'], \
            "precesion should be one of [single, double]."
        assert simulation_dict["backend"] in self._SIMULATION_PARAMETERS['backend'], \
            "backend should be one of [state_vector, density_matrix, unitary]."
        assert simulation_dict["device"] in self._SIMULATION_PARAMETERS['device'], \
            "Job's resource device should be one of [CPU, GPU]."

    def _qcda_validation(self, qcda_dict: dict):
        # QCDA methods validation
        methods = qcda_dict["methods"]
        for method in methods:
            assert method in self._BASED_METHODS, \
                f"Unrecognized QCDA method, please use one of {self._BASED_METHODS}."

        # Extra Args Validation
        # TODO: qcda templates need add check exists in CircuitLib.
        if "GateTransform" in methods:
            assert qcda_dict["instruction_set"] in self._QCDA_PARAMETERS["instruction_set"], \
                "Wrong Instruction Set."

        if "Auto" in methods:
            assert qcda_dict["auto_mode"] in self._QCDA_PARAMETERS["auto_mode"], \
                "auto_mode should be one of [light, heavy]."

        if "Commutative" in methods:
            assert isinstance(qcda_dict["para"], bool) and isinstance(qcda_dict["depara"], bool), \
                "The para and depara should be bool."

        if "Template" in methods:
            pass

        if qcda_dict["mapping"]["enable"]:
            print(qcda_dict["mapping"]["layout_path"])
            try:
                _ = Layout(1).load_file(qcda_dict["mapping"]["layout_path"])
            except Exception as e:
                raise KeyError(f"Failure to load layout from file as {e}.")

    def _job_complement(self, info: dict, job_type: str):
        default_job_path = os.path.join(self._job_template_path, f"job_{job_type}.yml")
        default_job_dict = JobValidation.load_yaml_file(default_job_path)
        core_part = default_job_dict[job_type]

        for key, value in info.items():
            if key in list(core_part.keys()):
                core_part[key] = value

        return core_part

    def job_validation(self, job_file_path: str):
        # Step 1: Load yaml file from given file path
        job_info = JobValidation.load_yaml_file(job_file_path)

        # Step 2: check based information of job_yaml
        self._based_info(job_info)

        # Step 3: complement and validate job's simulation/qcda information
        if job_info["type"] == "simulation":
            required_info = self._job_complement(job_info["simulation"], job_info["type"])
            self._simulation_validation(required_info)
            job_info["simulation"] = required_info
        else:
            required_info = self._job_complement(job_info["qcda"], job_info["type"])
            self._qcda_validation(required_info)
            job_info["qcda"] = required_info

        # Step 4: Prepare output path
        JobValidation.output_path_prepare(job_info["job_name"], job_info["output_path"])
        job_info["output_path"] = os.path.join(job_info["output_path"], job_info["job_name"])

        return job_info
