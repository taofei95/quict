import numpy as np
from QuICT.core.gate import BasicGate,CSwap,H,Measure
from QuICT.core.circuit import Circuit
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.simulation.state_vector.statevector_simulator import StateVectorSimulator
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
    def __init__(
        self,
        train_data,
        gama = 1
        
    ):
        self._data = train_data
        self.gama = gama
        self._X = self.norm_data(train_data[0])
        self._K_hat = self.get_K_hat()
        self._F = self.get_F()
    def norm_data(self,X:np.ndarray):  # X: M*N
        sum_col = np.sqrt(np.sum(X*X,axis= 1))
        for  i in range(len(sum_col)):
            X[i] = X[i] / sum_col[i]
        return X
    def kernel(self,x1,x2):
        n_qubit = int(np.log2(len(x1)))
        state_vec = np.kron(np.array([1,0]),x1)
        state_vec = np.kron(state_vec,x2)
        swap_test_cir = Circuit(2*n_qubit+1)
        H|swap_test_cir(0)
        for i in range(1,n_qubit+1):
            CSwap & [0,i,i+n_qubit] | swap_test_cir
        H|swap_test_cir(0)
        sim = StateVectorSimulator()
        sv = sim.run(swap_test_cir,state_vector=state_vec)
        prob = measure_gate_apply(0,sv)
        kernel_value = np.sqrt(abs(2*prob-1))
        if kernel_value != kernel_value:
            see = kernel_value

        return kernel_value
    def get_K_hat(self,):
        n = self._X.shape[0]  # number of sample
        K_hat = np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                xi= self._X[i]
                xj = self._X[j]
                #K_hat[i,j]= self.kernel(xi,xj)*modulus(xi)*modulus(xj)
                K_hat[i,j]= self.kernel(xi,xj)
                
                K_hat[j,i] = K_hat[i,j]
        '''
        sum =0
        for i in range(n):
            sum += modulus(self._X[i])^2
        K_hat= K_hat / sum
        '''
        penalty = 1/self.gama *np.eye(n,n)
        K_hat += penalty
        return K_hat
    def get_F(self,):
        n = self._X.shape[0]
        F = np.zeros((n+1,n+1))
        F[0,:] = 1
        F[:,0] = 1
        F[1:,1:] =self._K_hat
        return F
if __name__ == '__main__':  
    data = np.random.random((2,10,128))
    qsvm = QSVM(data)