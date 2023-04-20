import numpy as np
from QuICT.core.gate import BasicGate,CSwap,H,Measure
from QuICT.core.circuit import Circuit
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.simulation.state_vector.statevector_simulator import StateVectorSimulator
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_algorithm.hhl.hhl import HHL
import math
def measure_gate_apply(
    index: int,
    vec: np.array
):
    """ Measured the state vector for target qubit

    Args:
        index (int): The index of target qubit
        vec (np.array): The state vector of qubits

    Raises:
        TypeError: The state vector should be np.ndarray.

    Returns:
        float: The measured prob result 0 
    """
    target_index = 1 << index
    vec_idx_0 = [idx for idx in range(len(vec)) if not idx & target_index]
    vec_idx_0 = np.array(vec_idx_0, dtype=np.int32)
    vec_idx_1 = [idx for idx in range(len(vec)) if idx & target_index]
    vec_idx_1 = np.array(vec_idx_1, dtype=np.int32)
    prob = np.sum(np.square(np.abs(vec[vec_idx_0])))
    random = np.random.rand(1000)
    prob= len(np.where(random<=prob)[0])/1000

    return prob
def modulus(x):
    return np.sqrt(np.sum(x*x))
class QSVM:
    '''
    reference:
    Quantum support vector machine for big data classification
    arXiv:1307.0471v3 [quant-ph] 10 Jul 2014
    '''
    def __init__(
        self,
        train_data,
        gama = 1,
        HHL_t = 9
        
    ):
        self._data = train_data
        self.gama = gama
        self._t = HHL_t
        self._X = train_data[0]
        self._quantum_X ,self._size= self.quantum_norm_data()
        self._K_hat = self.enact_K_hat()
        self._F = self.get_F()

    def initial_circuit(self,)
    def solve(self,):
        y_true = np.zeros((self._size))
        y_true[1:self._X.shape[0]+1] = self._data[1]
        hhl = HHL(simulator=Simulator())
        return hhl.run(self._F,y_true)
    def quantum_norm_data(self,):  
        X = self._X
        sum_col = np.sqrt(np.sum(X*X,axis= 1))
        len_sv = int(2**math.ceil(np.log2(X.shape[1])))
        quantum_X = np.zeros((X.shape[0],len_sv))
        for  i in range(len(sum_col)):
            quantum_X[i][:X.shape[1]] = X[i] / sum_col[i]
        return quantum_X,len_sv
    def enact_K_hat(self,): # it shouldn't be using numerical calculate here!
        X = self._X
        N_= np.sum(np.sum(X*X,axis= 1))
        K_hat = np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K_hat[i,j] = np.dot(self._quantum_X[i],self._quantum_X[j])*modulus(X[i])*modulus(X[j])
        K_hat = K_hat /N_
    def kernel(self,x1,x2):
        n_qubit = int(np.log2(len(x1)))
        state_vec = np.kron(np.array([1,0]),x1)
        state_vec = np.kron(state_vec,x2)
        swap_test_cir = Circuit(2*n_qubit+1)
        H|swap_test_cir(0)
        for i in range(1,n_qubit+1):
            CSwap & [0,i,i+n_qubit] | swap_test_cir
        H|swap_test_cir(0)
        #sim = Simulator()
        sim = StateVectorSimulator()
        sv = sim.run(swap_test_cir,state_vector=state_vec)
        if isinstance(sim,Simulator):
            sv = sv['data']['state_vector']
        prob = measure_gate_apply(0,sv)
        kernel_value = np.sqrt(abs(2*prob-1))

        return kernel_value
   
    def get_F(self,):
        n = self._X.shape[0]
        size = self._size
        F = np.zeros((size,size))
        F[0,1:] = 1
        F[1:,0] = 1
        F[1:n+1,1:n+1] =self._K_hat
        return F
    
    

if __name__ == '__main__':  
    data_x = np.random.random((10,128))
    data_y = np.random.random(10)
    qsvm = QSVM([data_x,data_y])
    ans = qsvm.solve()
    print(ans)