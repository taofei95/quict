import os
from math import asin, pi, sqrt, sin
import numpy as np
from QuICT.algorithm.quantum_algorithm.CNF.cnf import CNFSATOracle
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.algorithm.quantum_algorithm.grover import Grover

FULL_TEST = False  # test on CNF with 16+ variables. CAN BE VERY SLOW.


def read_CNF(cnf_file):
    # file analysis
    variable_number = 0
    clause_number = 0
    CNF_data = []
    f = open(cnf_file, "r")
    for line in f.readlines():
        new = line.strip().split()
        int_new = []
        if new[0] == "p":
            variable_number = int(new[2])
            clause_number = int(new[3])
        else:
            for x in new:
                if (x != "0") and (int(x) not in int_new):
                    int_new.append(int(x))
                    if (-int(x)) in int_new:
                        int_new = []
                        break
        CNF_data.append(int_new)
    f.close()
    return variable_number, clause_number, CNF_data


def check_solution(variable_data, variable_number, clause_number, CNF_data):
    cnf_result = 1
    for i in range(clause_number):
        clause_result = 0
        if CNF_data[i + 1] == []:
            clause_result = 1
        else:
            for j in range(len(CNF_data[i + 1])):
                if CNF_data[i + 1][j] > 0:
                    clause_result = (
                        clause_result + variable_data[CNF_data[i + 1][j] - 1]
                    )
                else:
                    if CNF_data[i + 1][j] < 0:
                        clause_result = clause_result + (
                            1 - variable_data[-CNF_data[i + 1][j] - 1]
                        )
            if clause_result == 0:
                cnf_result = 0
                break
    if cnf_result == 1:
        return True
    else:
        return False


def find_solution_count(filename_test):
    solutions = []
    variable_number, clause_number, CNF_data = read_CNF(filename_test)
    for i in range(1 << variable_number):
        variable_data = bin(i)[2:].rjust(variable_number, "0")[::-1]
        variable_data = [int(x) for x in variable_data]
        if check_solution(variable_data, variable_number, clause_number, CNF_data):
            solutions.append(variable_data)
    return len(solutions)


def one_test(filename_test, variable_number, clause_number, CNF_data, n_solution, runs):
    AuxQubitNumber = 5
    cnf = CNFSATOracle()
    cnf.run(filename_test, AuxQubitNumber, 1)
    n_hit = 0
    oracle = cnf.circuit()
    grover = Grover(ConstantStateVectorSimulator())

    circ = grover.circuit(
        variable_number,
        AuxQubitNumber + 1,
        oracle,
        n_solution,
        measure=False,
        is_bit_flip=True,
    )
    grover.simulator.run(circ)
    result_samples = grover.simulator.sample(runs)
    result_var_samples = (
        np.array(result_samples)
        .reshape((1 << variable_number, 1 << (AuxQubitNumber + 1)))
        .sum(axis=1)
    )
    for result in range(1 << variable_number):
        result_str = bin(result)[2:].rjust(variable_number, "0")
        if check_solution(
            [int(x) for x in result_str], variable_number, clause_number, CNF_data
        ):
            n_hit += result_var_samples[result]
    return n_hit


def test_cnf():
    file_dir = "./"
    assert os.getcwd().endswith("unit_test/algorithm/quantum_algorithm/CNF")
    filename_test_list = os.listdir(file_dir)
    filename_test_list.sort()
    i = 0
    l = len(filename_test_list)
    n_all = 500
    for filename_test in filename_test_list:
        i += 1
        file_path = file_dir + filename_test
        n_solution = find_solution_count(file_path)
        variable_number, clause_number, CNF_data = read_CNF(file_path)
        print(
            f"[{i:3}/{l:3}]{n_solution:4} solution in {1<<variable_number:4} possibility"
        )
        if variable_number >= 16 and not FULL_TEST:
            print("skipped")
            continue
        if n_solution == 0:
            n_hit = None
            print(f"{filename_test:10} with zero solution")
        else:
            n_hit = one_test(
                file_path, variable_number, clause_number, CNF_data, n_solution, n_all
            )
            print(
                f"{filename_test:10} success rate: {n_hit/n_all:5.3f}[{n_hit:3}/{n_all:3}]"
            )
            theta = asin(sqrt(n_solution / (1 << variable_number)))
            n_iter = round((pi / 2 - theta) / (2 * theta))
            assert abs(n_hit / n_all - sin((2 * n_iter + 1) * theta) ** 2) < 1 / sqrt(
                n_all
            )
