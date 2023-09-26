#from itertools import permutations
import numpy as np
from QuICT.core.gate import DiagonalGate
from QuICT.core.gate import CompositeGate, CX, Rz, U1, GPhase
#print the whole array
#np.set_printoptions(threshold=np.inf)

#from QuICT.core.gate import *
from QuICT.core import Circuit

def test_gray_code():
    for code in DiagonalGate.lucal_gray_code(0, 3):
        print(code)
    print(DiagonalGate.partitioned_gray_code(4, 2))

def test_Ainv():
    n = 3
    A = np.zeros((1 << n, 1 << n))
    for s in range(1, 1 << n):
        for x in range(1, 1 << n):
            A[s, x] = DiagonalGate.binary_inner_prod(s, x, width=n)
    A = A[1:, 1:]
    # A_inv = 2^(1-n) (2A - J)
    A_inv = (2 * A - 1) / (1 << (n - 1))
    print(A)
    print(A_inv)
    print(np.dot(A, A_inv))

def test_phase_shift_no_aux():
    n = 3
    theta = 2 * np.pi * np.random.random(1 << n)
    seq = np.random.permutation(np.arange(1, 1 << n))
    gates = DiagonalGate.phase_shift(theta, seq)
    assert np.allclose(theta, np.mod(np.angle(np.diagonal(gates.matrix())), 2 * np.pi))
    circuit = Circuit(n)
    gates | circuit
    circuit.draw(filename='test_phase_shift_no_aux_1.jpg',flatten=True)

def test_phase_shift_with_aux():
    n = 3
    theta = 2 * np.pi * np.random.random(1 << n)
    seq = np.random.permutation(np.arange(1, 1 << n))
    gates = DiagonalGate.phase_shift(theta, seq, aux=n)
    assert np.allclose(theta, np.mod(np.angle(np.diagonal(gates.matrix()))[::2], 2 * np.pi))
    circuit = Circuit(n*2)
    gates | circuit
    circuit.draw(filename='test_phase_shift_with_aux_2.jpg', flatten=True)

def test_partitioned_gray_code():
    n = 4
    m = 8
    t = int(np.floor(np.log2(m / 2)))
    s = DiagonalGate.partitioned_gray_code(n,t)
    print(s)
    #print(s[1][0]) #s(2,1)

def test_alpha_s():
    n = 4
    s = 6
    theta = 2 * np.pi * np.random.random(1 << n)
    #theta = 2 * np.pi * np.random.random(len(A_inv))

    print(DiagonalGate.alpha_s(theta, s, n))

def test_phase_shift_s():
    n = 4
    s = 6
    theta = 2 * np.pi * np.random.random(1 << n)
    alpha = DiagonalGate.alpha_s(theta, s, n)
    gates = DiagonalGate.phase_shift_s(s, n, alpha,aux=n)
    circuit = Circuit(n+1)
    gates | circuit
    circuit.draw(filename='test_phase_shift_s_3.jpg', flatten=True)
    matrix = gates.matrix()
    print(matrix)
    is_diagonal = np.all(matrix == np.diag(np.diagonal(matrix)))

    if is_diagonal:
        print("The circuit matrix is diagonal.")
    else:
        print("The circuit matrix is not diagonal.")

    exp_theta = np.exp(1j * theta)
    print("exp_theta:", exp_theta)

    diagonal_matrix = np.diag(matrix)

    print("alpha:",alpha)
    print("exp(i*alpha):",np.exp(1j*alpha))

    I_n = np.eye(2 ** n)
    R_alpha = [[1,0],[0,np.exp(1j*alpha)]]
    # Compute the tensor product of diagonal_theta with the identity matrix
    tensor_product = np.kron(I_n,R_alpha)
    print("tensor_product:", tensor_product)

    print("diagonal_matrix:", diagonal_matrix)
    diagonal_tensor = np.diag(tensor_product)
    print("diagonal_tensor:", diagonal_tensor)

    if np.allclose(tensor_product,matrix):
        print("Two matrices are equal.")
    else:
        print("The two matrices are not equal.")
    # print(exp_theta)

    circuit2 = Circuit(n+1)
    gates2 = CompositeGate()
    U1(alpha) & n | gates2
    matrix2 = gates2.matrix()
    diagonal_matrix2 = np.diag(matrix2)
    print("diagonal_matrix2:", diagonal_matrix2)




def test_linear_fjk():
    n = 4
    m = 8
    t = int(np.floor(np.log2(m / 2)))
    resu = DiagonalGate.linear_fjk(2,1,10,n,t)
    print(resu)

def test_with_aux_qubit():
    n = 2 #number of target qubit
    m = 4 #number of ancillary qubit
    nn=DiagonalGate(n,m)
    #nn.target = 4
    #nn.aux = 8
    #theta = 2 * np.pi * np.random.random(1 << n)
    theta = np.append([0], 2 * np.pi * np.random.random((1 << n) - 1))

    size = n+m+1
    circuit = Circuit(size)
    gates = nn.with_aux_qubit(theta)
    gates | circuit
    circuit.draw(filename='test_with_aux_qubit_n=3,m=6.jpg', flatten=True)
    matrix = gates.matrix()
    print(matrix)
    #new_mat = matrix[:2 ** n, :2 ** n] #the first 2**n row and coloum
    new_mat = matrix[-2 ** n:, -2 ** n:]

    #print(matrix)
    #print(new_mat)
    #shape = matrix.shape
    #print(shape) #2^(n+m)

    """
    #The part of circuit related to target qubit is diagonal! :)
    
    is_diagonal = np.all(new_mat == np.diag(np.diagonal(new_mat)))
    
    if is_diagonal:
        print("The circuit matrix is diagonal.")
    else:
        print("The circuit matrix is not diagonal.")
    """

    # The part of circuit of all qubits is diagonal! :)

    is_diagonal = np.all(matrix == np.diag(np.diagonal(matrix)))

    if is_diagonal:
        print("The circuit matrix is diagonal.")
    else:
        print("The circuit matrix is not diagonal.")


    """
    trace = np.trace(new_mat)
    print("trace:",trace)
    """

    exp_theta = np.exp(1j * theta)
    print("exp_theta:",exp_theta)
    diagonal_theta = np.diag(exp_theta)
    diagonal_elements = np.diag(matrix)
    # Create a 2^m x 2^m identity matrix
    I_m = np.eye(2 ** m)
    # Compute the tensor product of diagonal_theta with the identity matrix
    tensor_product = np.kron(diagonal_theta, I_m)
    print("tensor_product:",tensor_product)
    diagonal_tensor = np.diag(tensor_product)
    print("diagonal_tensor:",diagonal_tensor)

    # Remove duplicate elements
    #unique_diagonal_elements = list(set(diagonal_elements))

    print("Diagonal elements:", diagonal_elements)

    """
    if np.allclose(new_mat, diagonal_theta):
        print("Two matrices are equal.")
    else:
        print("The two matrices are not equal.")
    #print(exp_theta)
    """




if __name__ == '__main__':
    #test_gray_code()
    #test_Ainv()
    #test_phase_shift_no_aux() #need change the size of A_inv
    #test_phase_shift_with_aux() #need change the size of A_inv
    #test_partitioned_gray_code()
    #test_linear_fjk()
    #test_alpha_s()
    #test_phase_shift_s() #here dim(A_inv)=dim(theta)
    test_with_aux_qubit()

