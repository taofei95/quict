# two kinds of decorator: path check and yml decomposition
import os
import yaml

from QuICT.tools.interface import OPENQASMInterface


def qasm_validation(qasm_file):
    """ whether the given qasm file is valid or not. If not valid, raise ValueError. """
    try:
        qasm = OPENQASMInterface.load_file(qasm_file)
        assert qasm.valid_circuit
        return qasm.circuit
    except:
        raise ValueError(f"Failure to load circuit from given file. {qasm_file}.")


def _job_validation(job_dict: dict):
    """
    job_dict = {
        job_name(str),
        type(str): one of [qcda, simulation],
        circuit(str): circuit's qasm file path,
        circuit_string(str): circuit's qasm, 
        number_of_qubits(int): the number of qubits in circuit,
        
        ### only for simulation
        simulation(dict):{
            shots(int),
            precision(str), one of [single, double],
            backend(str), one of [state_vector, density_matrix, unitary]
        },
        ###
        ### only for qcda
        qcda(dict):{
            ...
        },
        ###
        
        resource(dict):{
            device(str): one of [CPU, GPU],
            num(int): the number of devices.
        },
        output_path(str)
    }
    """
    # Necessary feature
    name = job_dict["job_name"]
    assert isinstance(name, str), f"Job's name shoule be a string, not {type(name)}."
    _type = job_dict["type"]
    assert _type in ["qcda", "simulation"], f"Job's type should be one of [qcda, simulation], not {_type}."

    # circuit's qasm file validation
    circuit = qasm_validation(job_dict["circuit"])
    job_dict['circuit_string'] = circuit.qasm()

    # Create Resource Dict
    resource_dict = {}
    resource_dict['number_of_qubits'] = circuit.width()

    # Runtime parameters' validation
    if _type == "simulation":   # validation for simulation job
        # validation job's simulation
        simulation_dict: dict = job_dict["simulation"]
        assert simulation_dict["shots"] >= 1, "The number of shots >= 1."
        assert simulation_dict["precision"] in ["single", "double"], "precesion should be one of [single, double]."
        assert simulation_dict["backend"] in ["state_vector", "density_matrix", "unitary"], \
            "backend should be one of [state_vector, density_matrix, unitary]."

        # validation job's resource
        job_resource: dict = simulation_dict["resource"]
        assert len(job_resource.keys()) == 2, "Resource only accept input device and num."
        assert job_resource["device"] in ["CPU", "GPU"], "Job's resource device should be one of [CPU, GPU]."
        assert job_resource["num"] >= 1, "The number of resource device should >= 1."
        resource_dict.update(job_resource)
        del simulation_dict["resource"]
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

        # Resource update
        resource_dict['device'] = "CPU"

    # Combined Resouce dict into job dict
    job_dict['resource'] = resource_dict

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
    """ The decorator for normalized job's yaml file. """
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
