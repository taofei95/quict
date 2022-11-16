import os

from QuICT.cloud.cli.utils.validation import JobValidation


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

    def wraps(*args, **kwargs):
        job_file_path = args[0] if args else kwargs["file"]
        # Validation job file
        job_info = JobValidation().job_validation(job_file_path)

        # step 3: run
        func(file=job_info)

    return wraps
