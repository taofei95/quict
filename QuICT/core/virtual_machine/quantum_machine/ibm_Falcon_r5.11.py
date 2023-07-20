""" Generate The Virtual Quantum Machine Model for IBM's ibmq_algiers. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 27
iset = InstructionSet(GateType.cx, [GateType.id, GateType.rz, GateType.sx, GateType.x])
layout = Layout(qubit_number)
layout.add_edge(0, 1, directional=False)
layout.add_edge(1, 4, directional=False)
layout.add_edge(1, 2, directional=False)
layout.add_edge(2, 3, directional=False)
layout.add_edge(3, 5, directional=False)
layout.add_edge(4, 7, directional=False)
layout.add_edge(5, 8, directional=False)
layout.add_edge(6, 7, directional=False)
layout.add_edge(7, 10, directional=False)
layout.add_edge(8, 11, directional=False)
layout.add_edge(8, 9, directional=False)
layout.add_edge(10, 12, directional=False)
layout.add_edge(11, 14, directional=False)
layout.add_edge(12, 15, directional=False)
layout.add_edge(12, 13, directional=False)
layout.add_edge(13, 14, directional=False)
layout.add_edge(15, 12, directional=False)
layout.add_edge(15, 18, directional=False)
layout.add_edge(19, 16, directional=False)
layout.add_edge(19, 20, directional=False)
layout.add_edge(19, 22, directional=False)
layout.add_edge(21, 18, directional=False)
layout.add_edge(21, 23, directional=False)
layout.add_edge(23, 27, directional=False)
layout.add_edge(24, 25, directional=False)
layout.add_edge(24, 23, directional=False)
layout.add_edge(25, 22, directional=False)
layout.add_edge(25, 26, directional=False)

qubit_fidelity = [
    (0.0138, 0.0064), (0.0074, 0.00319999999999998), (0.00919999999999998, 0.0056), (0.0103999999999999, 0.0132), (0.0054, 0.00480000000000002),
    (0.00839999999999996, 0.005), (0.00819999999999998, 0.0046), (0.0123999999999999, 0.0108), (0.0113999999999999, 0.0084),
    (0.014, 0.00639999999999996), (0.009, 0.0046), (0.0121999999999999, 0.0062), (0.0151999999999999, 0.0106), (0.009, 0.0142),
    (0.0266666666666666, 0.00666666666666671), (0.02, 0.0125999999999999), (0.0533333333333333, 0.0699999999999999),
    (0.0096, 0.00560000000000004), (0.0191999999999999, 0.0118), (0.1006, 0.0984), (0.032, 0.0284), (0.0092, 0.0128),
    (0.0138, 0.014), (0.0102, 0.007), (0.00760000000000005, 0.0036), (0.014, 0.0141999999999999), (0.0636, 0.0782)
    ]
gate_fidelity = [
    0.000267578094560045, 0.00022573194379589, 0.000265496566696267, 0.000235984351869834, 0.000209130730675593, 0.000219331530963963,
    0.000274385280660636, 0.000329202192739417, 0.000203632330675963, 0.000229628046124126, 0.000279204338613108, 0.00048389314738577,
    0.000587122353745529, 0.000600967926622458, 0.000493225704758826, 0.000960773963756658, 0.000354553420675743, 0.000170493371868683,
    0.00104856436718485, 0.000363098094319884, 0.00160142625773646, 0.000944772507200785, 0.000254450900679886, 0.000280427865208902,
    0.000183262735824819, 0.000275451717836697, 0.000897742086561505
    ]
coupling_strength = [
    (0, 1, 0.014000110151077216), (1, 4, 0.006048098871143381), (1, 2, 0.005393187607765054), (2, 3, 0.009289076969497323),
    (3, 5, 0.006754470295178533), (4, 7, 0.00592196942664619), (5, 8, 0.005061728708608992), (6, 7, 0.00722469893603106),
    (7, 10, 0.005838487686313898), (8, 11, 0.006683103383540168), (8, 9, 0.006116030108683668), (10, 12, 0.011942875499241806),
    (11, 14, 0.007477734629236593), (12, 15, 0.007925243264717469), (12, 13, 0.006321396655201733), (13, 14, 0.00584411844553423),
    (15, 12, 0.007925243264717469), (15, 18, 1), (19, 16, 1), (19, 20, 0.011625544083029543), (19, 22, 0.010725824436359555), 
    (21, 18, 0.011459715172174229), (21, 23, 0.01694112723331001), (24, 25, 0.008679386171843606), (24, 23, 0.007004780589654563),
    (25, 22, 0.007063948910990159), (25, 26, 0.021374335585729348)
    ]
readout_frequency = [
    4.94626484328382, 4.83604031465005, 5.05330896487853, 5.19663703711542, 4.95912408401899, 5.11765808778436, 4.98715943122157,
    4.88245590528334, 5.01864809132042, 4.90424583450549, 4.94892562500556, 5.09508061056882, 5.06416729056118, 5.11380460912087,
    4.98674180338145, 4.80713237903069, 4.88303839290579, 4.88819188047995, 5.00974859088973, 5.21591727321382, 4.9619046730355,
    5.05650560569351, 4.97840215193066, 5.21368295978203, 5.02383291327505, 4.89715648353441, 4.84024414498553
    ]
T1_times = [
    146.523793605932, 180.979640678976, 186.763610616574, 174.853495070592, 155.52470067039, 175.56771778664, 129.431547225744,
    222.273330617266, 178.676678258262, 220.778138854149, 138.297888997029, 165.342670634554, 104.945114045117, 96.6061803141594,
    161.300987607189, 150.738290717035, 206.513030211253, 214.611686917553, 154.162424962014, 79.3799650004999, 176.860770798179,
    97.0084693225578, 164.137520057589, 136.63810106023, 125.334771244675, 166.941187796772, 97.7986454265473
    ]
T2_times = [
    139.182436241025, 182.105313471114, 262.214098875516, 45.833135304283, 292.603793491474, 167.605050952788, 283.168285586175,
    81.0756418773887, 282.605920148731, 126.227328365139, 224.510340835167, 15.3070297608113, 76.2645242421076, 174.139811840501,
    99.9967072015217, 148.41945249711, 66.5544320534622, 54.3389814196333, 55.6468177820989, 37.7103026395134, 107.744081300505,
    157.118369703998, 174.748939459785, 49.4681017507677, 84.0792828838771, 68.7923839974839, 50.4384933190443
    ]

IBMFalconr511 = VirtualQuantumMachine(
    qubits=qubit_number,
    instruction_set=iset,
    qubit_fidelity=qubit_fidelity,
    gate_fidelity=gate_fidelity,
    coupling_strength=coupling_strength,
    layout=layout,
    readout_frequency=readout_frequency,
    t1_coherence_time=T1_times,
    t2_coherence_time=T2_times
)