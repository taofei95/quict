from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *


# QPE
def gen_pe(qubits, the_phase):
    cir = Circuit(qubits)
    for q in range(qubits):
        H | cir(q)
    for q in range(qubits - 1, -1, -1):
        CU1(np.pi * the_phase * 2 ** (qubits - q)) | cir([q, qubits])
    for q in range(qubits):
        H | cir(q)
        for tar in range(q + 1, qubits):
            CU1(-np.pi / (2 ** (tar - q))) | cir([q, tar])
    return cir.qasm()

# grover
def gen_grover(qubits, r):
    cir = Circuit(2 * qubits - 1)
    # add H
    for q in range(qubits):
        H | cir(q)
        H | cir(2 * qubits - 2)
    for k in range(r):
        # add tofolli
        CCX | cir([0, 1, qubits])
        for q in range(2, qubits):
            CCX | cir([q, q + qubits - 2, q + qubits - 1])
        for q in range(qubits - 2, 1, -1):
            CCX | cir([q, q + qubits - 2, q + qubits - 1])
        if qubits > 2:
            CCX | cir([0, 1, qubits])
        # add H
        for q in range(qubits):
            H | cir(q)
        # add X
        for q in range(qubits):
            X | cir(q)
        H | cir(qubits - 1)
        if qubits == 2:
            CX | cir([0, 1])
        elif qubits == 3:
            CCX | cir([0, 1 , 2])
        else:
            CCX | cir([0, 1, qubits])
            for q in range(2, qubits - 2):
                CCX | cir([q, q + qubits - 2, q + qubits - 1])
        CCX | cir([qubits - 2, 2 * qubits - 4, qubits - 1]) 
        for q in range(qubits - 3, 1, -1):
            CCX | cir([q, q + qubits - 2, q + qubits -1])
        if qubits > 2:
            CCX | cir([0, 1, qubits])
    H | cir(qubits - 1)
    # add X
    for q in range(qubits):
        X | cir(q)
    for q in range(qubits):
        H | cir(q)
    H | cir(2 * qubits - 2)

    return cir.qasm()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# cir = QuantumCircuit ( qubits +1 , qubits +1)
# for q in range( qubits ):
#     cir . h(q)
# for q in range( qubits -1 , -1 , -1):
#     cir . cu1 ( np . pi * the_phase *2**( qubits -q),q , qubits )
# for q in range( qubits ):
#     cir . h(q)
# for tar in range(q +1 , qubits ):
#     cir . cu1 (- np . pi /(2**( tar -q )) ,q , tar )
# return cir . qasm ()

# grover
# def gen_grover ( qubits , r ):
#     cir = QuantumCircuit (2 * qubits - 1)
#     # add H
#     for q in range( qubits ):
#         cir . h(q)
#         cir .h (2 * qubits - 2)
#     for k in range(r ):
#     # add t o f o l l i
#         cir . ccx (0 , 1, qubits )
#         for q in range(2 , qubits ):
#             cir . ccx (q , q + qubits - 2, q + qubits - 1)
#         for q in range( qubits - 2, 1, -1):
#             cir . ccx (q , q + qubits - 2, q + qubits - 1)
#         if qubits > 2:
#             cir . ccx (0 , 1, qubits )
#         # add H
#         for q in range( qubits ):
#             cir .h(q)
#         # add X
#         for q in range( qubits ):
#             cir .x(q)
#         cir . h( qubits - 1)
#         if qubits == 2:
#             cir . cx (0 , 1)
#         elif qubits == 3:
#             cir . ccx (0 , 1, 2)
#         else :
#             cir . ccx (0 , 1, qubits )
#             for q in range (2 , qubits - 2):
#             cir . ccx (q , q + qubits - 2, q + qubits - 1)
#         cir . ccx ( qubits - 2, 2 * qubits - 4, qubits - 1)
#             for q in range( qubits - 3, 1, -1):
#             cir . ccx (q , q + qubits - 2, q + qubits - 1)
#             if qubits > 2:
#             cir . ccx (0 , 1, qubits )
#     cir . h( qubits - 1)
#     # add X
#     for q in range( qubits ):
#         cir .x(q)
#     # add H
#     for q in range( qubits ):
#         cir .h(q)
#     cir .h (2 * qubits - 2)
#     return cir . qasm ()

# # clifford
# from qiskit import quantum_info
# def gen_rand_cliff (n ):
#     cliff = quantum_info . random_clifford ( n)
#     AG = quantum_info . decompose_clifford ( cliff , method = ’AG’ )
#     GD = quantum_info . decompose_clifford ( cliff , method = ’ g r e e d y ’ )
#     return AG . qasm () , GD . qasm ()