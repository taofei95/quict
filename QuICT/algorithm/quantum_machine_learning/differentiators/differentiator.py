import numpy as np
from QuICT.core import Circuit
from QuICT.simulation.simulator import Simulator
class Differentiator:
    def __init__(self,cir:Circuit,sim :Simulator) -> None:
        """
        Args:
        cir(Circuit): the circuit which is needed to calculate gradient
        """
        self._cir =cir
        self._sim =sim
    def get_param_shift(self,idx_gate:int,idx_param:int,shift_type:int):
        
        return 

    def get_grad(self,idx_gate:int,idx_param:int):
        return 