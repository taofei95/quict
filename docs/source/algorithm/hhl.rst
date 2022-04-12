HHL
==================================

The quantum algorithm for linear systems of equations, 
also called HHL algorithm, designed by Aram Harrow, Avinatan Hassidim, 
and Seth Lloyd, is a quantum algorithm formulated in 2009 for solving linear systems. 
The algorithm estimates the result of a scalar measurement on the solution vector 
to a given linear system of equations.

HHL algorithm in QuICT takes an :math:`N \times N` Hermitian matrix :math:`A` 
and an unit vector :math:`b` as input. At the end of the quantum computation,
the target qureg would be in a state representing vector :math:`x = A^{-1}b`,
that is, the solution for linear equation :math:`Ax = b`. 

When HHL is done in simulation way, QuICT can get the quantum state of the circuit, 
so can output :math:`x` directly, while on real quantum devices, the superposition
state cannot be known without multiple measurements. So HHL is often followed by
other procedures to get some feature of :math:`x`, rather than :math:`x` itself explicitly. 


.. code:: python

    def HHL_one_procedure(A, b):
        #trotter number exposed
        n = np.log2(len(A))
        N = 20    #trotter times
        m = 2*n
        l = 2*n
    
        eigenvalues, eigenvectors = np.linalg.eig(A)
        C = min(eigenvalues)
    
        circuit = Circuit(n + m + l + 1)
        ancilla = circuit(0)
        qreg_theta = circuit([i for i in range(1,l+1)])
        qreg_lambda = circuit([i for i in range(l+1,l+m+1)])
        qreg_x = circuit([i for i in range(l+m+1,l+n+m+1)])
        
        #fake initialization process
        qreg_x.force_assign_values(b)
        
        #phase estimation
        H | qreg_lambda
        (p_list,h) = calc_pauli_string_from_hamiltonian(n,A)
        construct_circuit(n, p_list, h, N)
        IQFT | qreg_lambda
    
        #rotation
        rotation(circuit,ancilla,qreg_theta,qreg_lambda,qreg_x,C)
    
        #inverse phase estimation
        QFT | qreg_lambda
        for i in range(len(h)):
            h = 1/h
        construct_circuit(n, p_list, h, N)
        H | qreg_lambda
    
        Measure | ancilla
        
        #move out
        amplitude = Amplitude.run(circuit)
        amplitude_x = amplitude[2**(l+m+n), 2**(l+m+n)+2**n]
        return (int(ancilla),amplitude_x)