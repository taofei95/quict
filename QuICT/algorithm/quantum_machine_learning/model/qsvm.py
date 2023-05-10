import numpy as np
from QuICT.core.gate import BasicGate,CSwap,H,Measure,IQFT,QFT
from QuICT.core.circuit import Circuit
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.simulation.state_vector.statevector_simulator import StateVectorSimulator
from QuICT.simulation.density_matrix.density_matrix_simulator import DensityMatrixSimulator
from QuICT.simulation.simulator import Simulator
from QuICT.algorithm.quantum_algorithm.hhl.hhl import HHL,c_rotation
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
    return np.sqrt(np.sum(abs(x)*abs(x)))

def quantum_norm_data(X):
    X_ = X.copy()
    if X_.ndim == 1:
        X = np.zeros((1,len(X_)),dtype=complex)
        X[:] = X_
    sum_col = np.sqrt(np.sum(X*X,axis= 1))
    len_sv = int(2**math.ceil(np.log2(X.shape[1])))
    quantum_X = np.zeros((X.shape[0],len_sv),dtype=complex)
    for  i in range(len(sum_col)):
        quantum_X[i][:X.shape[1]] = X[i] / sum_col[i]
    return quantum_X,len_sv

def train_data_oracle(train_data,psi):
    """
    an oracle ,
    reference:
    Quantum support vector machine for big data classification
    arXiv:1307.0471v3 [quant-ph] 10 Jul 2014
    
    """
    epsilo = 1e-5
    psi += epsilo
    quantum_data, size_train = quantum_norm_data(train_data)
    size_psi = len(psi)
    output = np.zeros((size_psi, size_train),dtype=complex)
    output[0, 0] = psi[0]
    absolute = psi[0]**2
    for i in range(1, quantum_data.shape[0]):
        output[i, :] = quantum_data[i-1]*psi[i]*modulus(train_data[i-1])
        absolute += abs(modulus(train_data[i-1])*psi[i])**2
    output = output.flatten()/np.sqrt(absolute)
    return output

class QSVM:
    '''
    the most of variable in this class is named by article in reference
    reference:
    Quantum support vector machine for big data classification
    arXiv:1307.0471v3 [quant-ph] 10 Jul 2014
    '''
    def __init__(
        self,
        train_data,
        gama = 1,
        HHL_t = 2
        
    ):
        self._data = train_data
        self.gama = gama
        
        self._X = train_data[0]
        self._quantum_X ,self._size= quantum_norm_data(self._X)
        self._K_hat,self._expend_M = self.enact_K_hat()
        self._register_C = HHL_t
        self._register_I = int(np.log2(self._expend_M))
        self._F = self.get_F()
        
        self._Y = np.zeros((self._expend_M),dtype=complex)
        self._Y [1:self._X.shape[0]+1] = self._data[1]
        self._SVM_param = None
        
    def solve(self,):  # use ControledUnitaoryDecomposition to exponent controled e^(-iF*t)
        hhl = HHL(simulator=StateVectorSimulator())
        self._SVM_param = hhl.run(self._F,self._Y,phase_qubits=3)
        if not np.where(self._SVM_param != 0):
            print('hhl give bad solution')
       
    def solve1(self,n_copies): 
        '''
        use DensityMatrixSimulator to exponent controled e^(-iF*t)
        this method cannot run in fact
        '''
        n_copies = 0
        for i in range(self._register_C):
            n_copies += 2**(i)
       
        hhl_cir = self.init_circuit(n_copies)
        ini_dens_mat = self.init_density_mat(n_copies)
        sim = DensityMatrixSimulator()
        dens_mat = sim.run(hhl_cir,ini_dens_mat)
        return dens_mat
    def classify(self,vec_x):
        ket_u = train_data_oracle(self._X,self._SVM_param)
        ket_x = train_data_oracle(vec_x,np.ones(self._expend_M))
        ket_psi = 1/np.sqrt(2)*(np.hstack((ket_u,ket_x)))
        swap_test_ini_sv = np.hstack((ket_psi,np.zeros((len(ket_psi)))))
        n_qubit = int(np.log2(len(swap_test_ini_sv)))
        swap_test_cir = Circuit(n_qubit)
        H& 0|swap_test_cir
        for i in range(1,int(np.log2(len(ket_u)))):
            CSwap&[0,i,2*i]|swap_test_cir
        H& 0|swap_test_cir
        sim = StateVectorSimulator()
        sv = sim.run(swap_test_cir,state_vector=swap_test_ini_sv)
        prob = measure_gate_apply(0,sv)
        y_predic = 1 if prob <1/2 else -1
        return y_predic

    def init_circuit(self,n_copies):
        n_width = 1+self._register_C+self._register_I+self._register_I*n_copies*2
        hhl_cir = Circuit(n_width)
        h_act_list = [x for x in range(1,self._register_C+1)]
        H|hhl_cir[h_act_list]
        consumed_copies = 0

        # exponentiation of controled e^(-iF*t)
        for i in reversed(range(self._register_C)):
            m = int(2**i)
            for j in range(m):
                copies_shift = 1+self._register_C+(1+consumed_copies)*self._register_I
                for qbit in range(self._register_I):
                    act_copies_qbit = qbit+copies_shift
                    CSwap &[1+i,1+self._register_C+qbit,act_copies_qbit]|hhl_cir
                consumed_copies += 1
                if consumed_copies>n_copies:
                    raise ValueError('n_copies now is not enough')
        #
        phase = list(range(1, 1 + self._register_C))
        IQFT(len(phase)) | hhl_cir(list(reversed(phase)))

        # Controlled-Rotation
        control_rotation = c_rotation(phase, 0)
        control_rotation | hhl_cir

        # Inversed-QPE
        QFT(len(phase)) | hhl_cir(list(reversed(phase)))

        # exponentiation of coutroled e^(-iF*t) hermite
        consumed_copies = 0
        for i in range(self._register_C):
            m = int(2**i)
            for j in range(m):
                copies_shift = 1+self._register_C+(1+n_copies+consumed_copies)*self._register_I
                for qbit in range(self._register_I):
                    act_copies_qbit = qbit+copies_shift
                    CSwap &[1+i,1+self._register_C+qbit,act_copies_qbit]|hhl_cir
                consumed_copies += 1
     
        H|hhl_cir[h_act_list]
        return hhl_cir
    def init_density_mat(self,n_copies):
        zero_size = int(2**(self._register_C+1))
        regeister_SC_dens = np.eye(zero_size,zero_size,dtype=complex)
        Y= np.zeros((1,self._expend_M),dtype=complex)
        Y[:]=self._Y[:]
        regeister_I_dens = np.kron(Y.T.conj(),Y)
        dens_mat = np.kron(regeister_SC_dens,regeister_I_dens)
        for i in range(n_copies):
            dens_mat = np.kron(dens_mat,self._F)
        for i in range(n_copies):
            dens_mat = np.kron(dens_mat,self._F.T.conj())
        return dens_mat
    
   
    def enact_K_hat(self,): # it shouldn't be using numerical calculate here!
        X = self._X
        N_= np.sum(np.sum(X*X,axis= 1))
        expend_M = int(2**math.ceil(np.log2(X.shape[0])))
        K_hat = np.zeros((expend_M,expend_M),dtype=complex)
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K_hat[i,j] = np.dot(self._quantum_X[i],self._quantum_X[j])*modulus(X[i])*modulus(X[j])
        K_hat = K_hat /N_
        for i in range(X.shape[0],expend_M):
            K_hat[i,i] = 1
        return K_hat,expend_M
    def kernel(self,x1,x2):  
        """
        this method use swap-test to calculate the inner product of two vector
        """
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
        size = self._expend_M
        F = np.zeros((size,size),dtype=complex)
        F[0,1:self._X.shape[0]+1] = 1
        F[1:self._X.shape[0]+1,0] = 1
        F[1:,1:] =self._K_hat[:-1,:-1]
        tr = np.trace(F)
        F= F/tr
        return F
    

    

if __name__ == '__main__':  

    data_x = np.random.random((3,10))
    data_y = np.random.random(3 )
    qsvm = QSVM([data_x,data_y])

    qsvm.solve()

