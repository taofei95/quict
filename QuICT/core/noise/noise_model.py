from ctypes import Union
import numpy as np
from typing import Union, List

from QuICT.core.utils import GateType


class NoiseModel:
    def __init__(self):
        pass
    
    def add(self, noise, qubit: Union[int, List[int]] = None, gates: Union[GateType, List[GateType]] = None):
        pass
