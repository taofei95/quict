# two kinds of decorator: path check and yml decomposition
import os
import yaml

from QuICT.tools.interface import OPENQASMInterface


def qasm_validation(qasm_file):
    try:
        qasm = OPENQASMInterface.load_file(qasm_file)
        assert qasm.valid_circuit
    except:
        raise ValueError(f"Failure to load circuit from given file. {qasm_file}.")


def _job_validation(job_dict: dict):
    # Necessary feature
    name = job_dict["name"]
    assert isinstance(name, str), f"Job's name shoule be a string, not {type(name)}."
    _type = job_dict["type"]
    assert _type in ["qcda", "simulation"], f"Job's type should be one of [qcda, simulation], not {_type}."

    # circuit's qasm file validation
    circuit = job_dict["circuit"]
    qasm_validation(circuit)

    # Runtime parameters' validation
    if _type == "simulation":   # validation for simulation job
        # validation job's simulation
        simulation_dict: dict = job_dict["simulation"]
        assert simulation_dict["shots"] >= 1, "The number of shots >= 1."
        assert simulation_dict["precision"] in ["single", "double"], "precesion should be one of [single, double]."
        assert simulation_dict["backend"] in ["state_vector", "density_matrix", "unitary"], \
            "backend should be one of [state_vector, density_matrix, unitary]."

        # validation job's resource
        resource_dict: dict = simulation_dict["resource"]
        assert resource_dict["device"] in ["CPU", "GPU"], "Job's resource device should be one of [CPU, GPU]."
        assert resource_dict["num"] >= 1, "The number of resource device should >= 1."
    else:   # validation for qcda job
        qcda_dict: dict = job_dict["qcda"]
        # validation qcda-synthesis
        syn_dict: dict = qcda_dict["synthesis"]
        assert isinstance(syn_dict["enable"], bool)
        assert syn_dict["instruction_set"] in ["USTC", "IBMQ", "Google", "Ionq"]

        # validation qcda-optimization
        opt_dict: dict = qcda_dict["optimization"]
        assert opt_dict["mode"] in ["auto", "customized"]
        for key in ["enable", "template_matching", "commutative_opt", "CNOT", "Clifford"]:
            assert isinstance(opt_dict[key], bool)

        # validation qcda-mapping
        map_dict: dict = qcda_dict["mapping"]
        assert isinstance(map_dict["enable"], bool)
        assert map_dict["basic_type"] in ["line", "grid", "customized"]
        if map_dict["basic_type"] == "customized":
            from QuICT.core import Layout

            try:
                _ = Layout.load_file(map_dict["layout_path"])
            except Exception as e:
                raise ValueError(f"Failure to load Layout from layout_path, due to {e}.")

    # output-path preparation
    output_path = os.path.join(job_dict["output_path"], name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    job_dict["output_path"] = output_path


def path_check(func):
    """ create the output path, if not exist. """
    def wraps(*args, **kwargs):
        output_path = args[-1] if args else kwargs["output_path"]
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        func(*args, **kwargs)

    return wraps


def yaml_decompostion(func):
    def wraps(*args, **kwargs):
        yaml_file = args[0] if args else kwargs["file"]
        # step 1: load yaml file
        with open(os.path.abspath(yaml_file), encoding='utf-8') as yml:
            yaml_dict = yaml.load(yml)

        # step 2: validation
        _job_validation(yaml_dict)

        # step 3: run
        func(file=yaml_dict)

    return wraps
