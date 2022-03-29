from QuICT.algorithm.quantum_algorithm import (
    ShorFactor
)
import logging
logging.root.setLevel(logging.INFO)

def _test_ShorFactor_run(mode: str):
    from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
    simulator = ConstantStateVectorSimulator()
    number_list = [
        4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
    ]
    for number in number_list:
        print('-------------------FACTORING %d-------------------------' % number)
        a = ShorFactor(mode=mode,N=number).run(simulator=simulator)
        assert number % a == 0

def test_ShorFactor_BEA():
    _test_ShorFactor_run("BEA")

def test_ShorFactor_BEA_zip():
    _test_ShorFactor_run("BEA_zip")

# def test_ShorFactor_HRS():
#     _test_ShorFactor_run("HRS")

# def test_ShorFactor_HRS_zip():
#     _test_ShorFactor_run("HRS_zip")