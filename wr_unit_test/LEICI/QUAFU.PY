import numpy as np
 
from QuICT.core import *
from QuICT.core.gate import *
# from QuICT.algorithm.synthetical_unitary import SyntheticalUnitary
 
 
def naive_prefix_cnot(n):
    # depth n prefix cnot
    circ = Circuit(n)
    for i in range(0, n-1):
        CX | circ([i,  i+1])
    return circ
 
 
def prefix_cnot(n):
    # depth 2*log(n)-1 prefix cnot
    circ = Circuit(n)
    lg = int(np.ceil(np.log2(n)))
    m = 1 << lg
 
    for j in range(1, lg + 1):
        for k in range(0, m // (1 << j)):
            x = (1 << j) * k + (1 << (j-1))
            y = (1 << j) * k + (1 << j)
            if x <= n and y <= n:
                CX | circ([x - 1, y - 1])
    for j in range(lg - 1, 0, -1):
        for k in range(1, m // (1 << j)):
            x = (1 << j) * k
            y = (1 << j) * k + (1 << (j-1))
            if x <= n and y <= n:
                CX | circ([x - 1, y - 1])
 
    return circ
 
 
def draw_partial_cnot(n):
    for i in range(2, n+1):
        naive_circ = naive_prefix_cnot(i)
        circ = prefix_cnot(i)
        circ.draw(filename=f"prefix_cnot_{i}.jpg")
        # naive_circ.draw(filename=f"naive_prefix_cnot_{i}.jpg")

        print(circ.qasm())
        # print(naive_circ.qasm())

        
        # file = open("wr_unit_test/LEICI/circ_qasm","w+")
        # file.write(circ.qasm())
        # file.close()
        # if n <= 10:
        #     circ_v = naive_prefix_cnot(n)
        #     mat_1 = SyntheticalUnitary.run(circ_v)
        #     mat_2 = SyntheticalUnitary.run(circ)
        #     assert np.allclose(mat_1, mat_2)
 
 
if __name__ == '__main__':
    draw_partial_cnot(20)