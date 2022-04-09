import abc
from operator import matmul
from statistics import mean
from unittest.result import failfast
import numpy as np
import pandas
import math
import random
import QuICT.simulation.unitary_simulator
import QuICT.ops.linalg.cpu_calculator as CPUCalculator
import QuICT.simulation.unitary_simulator
import QuICT.core.utils

class DensityMatrixSimulation:

    def __init__(
        self,
        device: str = "CPU",
        precision: str = "double"
    ):
        self._device = device
        self._precision = precision

        if device == "CPU":
            self._computer = CPUCalculator
        else:
            import QuICT.ops.linalg.gpu_calculator as GPUCalculator
            self._computer = GPUCalculator


    def createInitMatrix(n):
        initialM = np.zeros(2 ** n, 2 ** n)
        initialM[0,0] = 1;
        return initialM;

    def checkMatrix(matrix):
        check = True;
        if(matrix.T.conjugate() != matrix): 
            check = False;
        eigenvalues = np.linalg.eig(matrix)[0];
        for ev in eigenvalues:
            if(ev < 0): check = False;
        if(matrix.trace() != 1):
            check = False;
        return False;
            
    def densityMatrixSimu(density_matrix):
        UnitaryMatrix = UnitarySimulator.circut();
        # ?
        density_matrix = UnitaryMatrix.dot(density_matrix).dot(UnitaryMatrix.conj().T);
        return density_matrix;

    
        
    def Measure(mea_circuit,matrix, n):
        P0 = np.mat([[1, 0], [0, 0]]);
        P1 = np.mat([[0, 0], [0, 1]]);
        mea_0 = mea_circuit.matrix_product_to_circuit(P0);
        prob_0 = np.matmul(mea_0, matrix).trace();
        if(random() < prob_0):
            U = np.matmul(mea_0, np.eye(2**n) / np.sqrt(prob_0));
            matrix = U.dot(matrix).dot(U.conj().T);
        else:
            mea_1 = mea_circuit.matrix_product_to_circuit(P1);
            U = np.matmul(mea_1, np.eye(2**n)/np.sqrt(1 - prob_0));
            matrix = U.dot(matrix).dot(U.conj().T);
            
        # self.circuit.qubits[gate.targ].measured = int(result);
    
        

        


