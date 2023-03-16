import numpy as np
from QuICT.core import Circuit
from QuICT.simulation.simulator import Simulator
class ParamterShift:
    def __init__(self,cir:Circuit,sim :Simulator) -> None:
        """
        Args:
        cir(Circuit): the circuit which is needed to calculate gradient
        """
        self._cir =cir
        self._sim =sim
    def get_param_shift(self,idx_gate:int,idx_param:int,shift_type:int):
        """
        Args:
        shift_type(int): 0 for left shift and 1 for right shift
        """
        if shift_type !=0 & shift_type != 1:
            raise  ValueError("shift_type must be 0 or 1")
        i =idx_param
        gate = self._cir.gates[idx_gate]
        sim = self._sim
        shift = 0.5
        shift_type = 1 if shift_type==1 else -1
        param_list = gate.pargs.copy()
        shift_param = param_list[i]+shift*shift_type

        param_list[i] = shift_param
        gate.pargs = param_list.copy()
        self._cir.replace_gate(idx_gate, gate)
        e_val = sim.forward(self._cir)
        return e_val

    def get_grad(self,idx_gate:int,idx_param:int):
        e_shift = np.array( [0.0 ,0.0]  )
        for i in range(2):
            e_shift[i]=self.get_param_shift(idx_gate,idx_param,i)
        grad = 1.0*(e_shift[1]-e_shift[0])
   
        return grad

    print('')