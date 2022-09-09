# two kinds of decorator: path check and yml decomposition
import os
import yaml

from QuICT.tools.interface import OPENQASMInterface

 
def path_check(func):
    """ create the output path, if not exist. """
    def wraps(*args, **kwargs):
        output_path = args[-1] if args else kwargs["output_path"]
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        func(*args, **kwargs)

    return wraps


def validation_qasm(func):
    """ create the output path, if not exist. """
    def wraps(*args, **kwargs):
        qasm_file = args[-1] if args else kwargs["file"]
        try:
            qasm = OPENQASMInterface.load_file(qasm_file)
            assert qasm.valid_circuit
        except:
            raise ValueError(f"Failure to load circuit from given file. {qasm_file}.")

        func(*args, **kwargs)

    return wraps


def yaml_decompostion(func):
    def wraps(*args, **kwargs):
        yaml_file = args[0] if args else kwargs["file"]
        # step 1: load yaml file
        with open(os.path.abspath(yaml_file), encoding='utf-8') as yml:
            yaml_dict = yaml.load(yml)

        # step 2: validation
        func(file=yaml_dict)

    return wraps
