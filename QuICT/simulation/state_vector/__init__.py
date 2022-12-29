try:
    from .cpu_simulator import CircuitSimulator
except AttributeError:
    CircuitSimulator = None

try:
    from .gpu_simulator import ConstantStateVectorSimulator
<<<<<<< HEAD
<<<<<<< HEAD
except ModuleNotFoundError:
    ConstantStateVectorSimulator = None
=======
except Exception as e:
    pass
>>>>>>> ceb3be5e076f8251ddfc3e14dd65c38088e75607
=======
except Exception as e:
    pass
>>>>>>> dev_patch
