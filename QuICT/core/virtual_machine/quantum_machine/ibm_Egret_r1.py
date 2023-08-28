""" Generate The Virtual Quantum Machine Model for IBM's ibmq_prague. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 33
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
layout.add_edge(27, 28, directional=False)
layout.add_edge(28, 29, directional=False)
layout.add_edge(30, 3, directional=False)
layout.add_edge(30, 31, directional=False)
layout.add_edge(31, 32, directional=False)

qubit_fidelity = [
    (0.013, 0.006), (0.021, 0.015), (0.0312, 0.01), (0.0121999999999999, 0.0042), (0.0106, 0.0114), (0.0136, 0.0272),
    (0.00739999999999996, 0.0092), (0.0143999999999999, 0.0038), (0.0132, 0.00839999999999996), (0.0138, 0.009),
    (0.0165999999999999, 0.0098), (0.009, 0.00480000000000002), (0.011, 0.0036), (0.0216, 0.007),
    (0.00980000000000003, 0.0032), (0.0166, 0.0116), (0.0182, 0.0325999999999999), (0.0635999999999999, 0.0142),
    (0.1128, 0.8704), (0.3692, 0.0114), (0.0058, 0.00980000000000003), (0.1068, 0.0252), (0.0096, 0.00580000000000002),
    (0.629, 0.257399999999999), (0.03, 0.0228), (0.0124, 0.00960000000000005), (0.018, 0.026), (0.4022, 0.279),
    (0.0412, 0.00860000000000005), (0.0424, 0.0447999999999999), (0.0118, 0.0108), (0.0134, 0.0135999999999999),
    (0.0156, 0.01)
]
gate_fidelity = [
    0.000117615220032362, 0.000163938126664874, 0.000582308335214044, 0.00013610005126178, 0.000148385066748241,
    0.000118899193710876, 0.000163858955015098, 0.000150549247214256, 0.000171040483008005, 9.20704588751013e-05,
    0.000141651010228793, 0.000151551240596235, 0.000131365999054446, 0.000461983906774566, 0.000128708290300634,
    0.000122210152934165, 8.90713513492833e-05, 0.000671200360371153, 0.000698733539730475, 0.0199882377517596,
    0.000135023937344572, 0.00175172990222703, 0.000163092646490909, 0.00281375157758785, 0.000249709254806724,
    0.000164410495330842, 0.000128484437611922, 0.000728434012065255, 0.00138122903892391, 0.000105024020460099,
    0.000135181185743771, 0.000113948906404817, 0.000129845611409249
]
coupling_strength = [
    (0, 1, 0.002492700783726348), (1, 4, 0.0035499412425835097), (1, 2, 0.005168082642493549),
    (2, 3, 0.005316493425177121), (3, 5, 0.0027938115876474934), (4, 7, 0.0018550961598121918),
    (5, 8, 0.00252661768269144), (6, 7, 0.002086339649547242), (7, 10, 0.0036195979792001676),
    (8, 11, 0.0015780809424507103), (8, 9, 0.0015131333028158833), (10, 12, 0.004265820378532459),
    (11, 14, 0.21339411275387787), (12, 15, 0.003666288951674257), (12, 13, 0.006321396655201733),
    (13, 14, 0.004578157921410919), (15, 12, 0.003666288951674257), (15, 18, 0.00458956761769061),
    (19, 16, 0.299144274521494), (19, 20, 0.06750446970203283), (19, 22, 0.0742886092159433),
    (21, 18, 0.008616248954974842), (21, 23, 0.1492726370949424), (23, 27, 1), (24, 25, 0.00201280297225439),
    (24, 23, 0.030664697191251145), (25, 22, 0.002177851362360844), (25, 26, 0.0026157481258743676),
    (27, 28, 1), (28, 29, 0.008622434637630738), (30, 3, 0.002117485225874266), (30, 31, 0.0020673164591699933),
    (31, 32, 0.002873202357238397)
]
readout_frequency = [
    5.00737650881582, 4.56918116956246, 4.96382081514398, 4.58667358271842, 5.0008973119641, 4.99025938731041,
    4.99331161019961, 4.60350180525456, 4.6424892122049, 5.00224369650376, 5.02658118634019, 4.97853524783689,
    4.63087971567596, 5.03031249518789, 4.57254274089231, 4.96950984066092, 4.97570868353899, 4.99977844033356,
    4.63637397148425, 4.79218705480374, 4.93446239482393, 5.04224190063609, 5.09037597608893, 4.77279057128703,
    5.02703191021625, 4.61856399658404, 5.0270039553176, 5.01407702468029, 4.59777505093564, 4.99149557007489,
    4.97604016134876, 4.5920644197041, 4.91713000113084
]
T1_times = [
    188.121373274505, 119.002301818354, 12.2367127971166, 128.071824081076, 222.631094871234, 232.103951953318,
    119.489261679664, 109.365056022738, 171.195497984803, 238.538902400409, 134.513898337671, 104.250394953884,
    179.329035531507, 37.1735105905585, 188.531781651317, 272.254047956763, 169.502955855349, 235.762859860486,
    122.544906796714, 7.74994332068377, 136.230087671366, 114.600115730306, 169.641412254019, 72.1506977577127,
    79.4290328336923, 202.045618708458, 83.6239998777859, 229.770878757992, 82.0600323109682, 217.160462725394,
    168.297391848848, 107.357301224188, 224.732942853759
]
T2_times = [
    102.129839791333, 96.8354816659688, 15.7445164983114, 154.83530008283, 213.181450084991, 275.32863203226,
    134.480621742919, 175.697679896182, 167.704207304451, 212.212321531882, 136.90776547073, 128.225524954737,
    154.427937881602, 61.9738804668736, 198.368957532915, 96.3130725859842, 276.699599354475, 257.137111418754,
    149.921844499389, 0.0932330444461839, 80.3986336764786, 1.08165618249239, 138.476481703219, 0.993980782687985,
    12.9118804197078, 163.193593371227, 11.6706628566269, 23.0250919576256, 1.28518376189089, 238.532153083322,
    148.238778411612, 155.39173031258, 165.83464729914
]

IBMEgretR1 = VirtualQuantumMachine(
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
