import pytest

from QuICT.algorithm.quantum_algorithm import ShorFactor, BEA_order_finding, HRS_order_finding

def test_ShorFactor_on_ConstantStateVectorSimulator():
    from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
    simulator = ConstantStateVectorSimulator(
        precision="double",
        gpu_device_id=0,
        sync=True
    )
    number_list = [
        4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
    ]
    for number in number_list:
        print('-------------------FACTORING %d-------------------------' % number)
        a = BEAShorFactor.run(N=number, max_rd=10, simulator=simulator)
        assert number % a == 0


def test_HRSShorFactor_on_ConstantStateVectorSimulator():
    from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
    simulator = ConstantStateVectorSimulator(
        precision="double",
        gpu_device_id=0,
        sync=True
    )
    number_list = [
        4, 6, 8, 9, 10,
        12, 14, 15, 16, 18, 20,
        21, 22, 24, 25, 26, 27,
    ]
    for number in number_list:
        print('-------------------FACTORING %d-------------------------' % number)
        a = HRSShorFactor.run(N=number, max_rd=10, simulator=simulator)
        assert number % a == 0


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
