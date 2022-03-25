from QuICT.algorithm.quantum_algorithm import (
    ShorFactor
)

def test_ShorFactor_BEA():
    from QuICT.simulation.cpu_simulator import CircuitSimulator
    simulator = CircuitSimulator()
    number_list = [
        4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
    ]
    for number in number_list:
        print('-------------------FACTORING %d-------------------------' % number)
        a = ShorFactor(mode="BEA",N=number).run(simulator=simulator)
        assert number % a == 0

test_ShorFactor_BEA()
