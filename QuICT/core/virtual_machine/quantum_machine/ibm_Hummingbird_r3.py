""" Generate The Virtual Quantum Machine Model for IBM's ibm_ithaca. """

from QuICT.core.utils import GateType
from QuICT.core.layout import Layout
from QuICT.core.virtual_machine import VirtualQuantumMachine, InstructionSet


qubit_number = 65
iset = InstructionSet(GateType.cx, [GateType.id, GateType.rz, GateType.sx, GateType.x])
layout = Layout(qubit_number)
layout.add_edge(0, 10, directional=False)
layout.add_edge(0, 1, directional=False)
layout.add_edge(1, 2, directional=False)
layout.add_edge(2, 3, directional=False)
layout.add_edge(3, 4, directional=False)
layout.add_edge(4, 11, directional=False)
layout.add_edge(4, 5, directional=False)
layout.add_edge(5, 6, directional=False)
layout.add_edge(7, 6, directional=False)
layout.add_edge(8, 7, directional=False)
layout.add_edge(8, 9, directional=False)
layout.add_edge(8, 12, directional=False)
layout.add_edge(10, 13, directional=False)
layout.add_edge(10, 0, directional=False)
layout.add_edge(11, 4, directional=False)
layout.add_edge(11, 17, directional=False)
layout.add_edge(12, 21, directional=False)
layout.add_edge(14, 13, directional=False)
layout.add_edge(14, 15, directional=False)
layout.add_edge(15, 24, directional=False)
layout.add_edge(16, 17, directional=False)
layout.add_edge(16, 15, directional=False)
layout.add_edge(17, 18, directional=False)
layout.add_edge(19, 25, directional=False)
layout.add_edge(19, 20, directional=False)
layout.add_edge(19, 18, directional=False)
layout.add_edge(20, 21, directional=False)
layout.add_edge(22, 21, directional=False)
layout.add_edge(22, 23, directional=False)
layout.add_edge(26, 37, directional=False)
layout.add_edge(26, 23, directional=False)
layout.add_edge(43, 44, directional=False)
layout.add_edge(28, 27, directional=False)
layout.add_edge(29, 24, directional=False)
layout.add_edge(29, 28, directional=False)
layout.add_edge(30, 29, directional=False)
layout.add_edge(31, 30, directional=False)
layout.add_edge(31, 32, directional=False)
layout.add_edge(33, 34, directional=False)
layout.add_edge(33, 32, directional=False)
layout.add_edge(33, 25, directional=False)
layout.add_edge(34, 35, directional=False)
layout.add_edge(35, 40, directional=False)
layout.add_edge(36, 37, directional=False)
layout.add_edge(36, 35, directional=False)
layout.add_edge(38, 41, directional=False)
layout.add_edge(38, 27, directional=False)
layout.add_edge(39, 31, directional=False)
layout.add_edge(42, 43, directional=False)
layout.add_edge(42, 41, directional=False)
layout.add_edge(45, 44, directional=False)
layout.add_edge(45, 39, directional=False)
layout.add_edge(46, 45, directional=False)
layout.add_edge(47, 46, directional=False)
layout.add_edge(47, 53, directional=False)
layout.add_edge(48, 47, directional=False)
layout.add_edge(49, 40, directional=False)
layout.add_edge(49, 48, directional=False)
layout.add_edge(50, 49, directional=False)
layout.add_edge(51, 50, directional=False)
layout.add_edge(52, 56, directional=False)
layout.add_edge(52, 43, directional=False)
layout.add_edge(54, 64, directional=False)
layout.add_edge(54, 51, directional=False)
layout.add_edge(57, 56, directional=False)
layout.add_edge(56, 55, directional=False)
layout.add_edge(57, 58, directional=False)
layout.add_edge(59, 58, directional=False)
layout.add_edge(60, 59, directional=False)
layout.add_edge(60, 61, directional=False)
layout.add_edge(60, 53, directional=False)
layout.add_edge(61, 62, directional=False)
layout.add_edge(63, 64, directional=False)
layout.add_edge(63, 62, directional=False)

qubit_fidelity = [
    (0.0132, 0.0016), (0.0226, 0.006), (0.0156, 0.0026), (0.0184, 0.0012), (0.013, 0.0108), (0.0378, 0.0096),
    (0.0926, 0.0748), (0.0146, 0.0038), (0.0258, 0.0018), (0.0138, 0.0012), (0.134, 0.0428), (0.0128, 0.0102),
    (0.0436, 0.0058), (0.0156, 0.004), (0.0904, 0.0312), (0.0214, 0.0026), (0.0304, 0.0292), (0.015, 0.003),
    (0.0146, 0.0034), (0.0152, 0.0024), (0.0382, 0.0234), (0.0262, 0.0118), (0.017, 0.002), (0.0196, 0.0056),
    (0.1176, 0.0526), (0.0418, 0.0066), (0.256666667, 0.39), (0.0646, 0.0312), (0.0302, 0.0162), (0.0418, 0.0388),
    (0.0222, 0.0036), (0.0198, 0.012), (0.0164, 0.0026), (0.0158, 0.002), (0.0634, 0.0156), (0.0216, 0.0018),
    (0.1234, 0.0864), (0.013333333, 0.0), (0.0338, 0.005), (0.0244, 0.007), (0.0564, 0.0368), (0.2138, 0.0152),
    (0.082, 0.0166), (0.0346, 0.0098), (0.044, 0.0284), (0.05, 0.0056), (0.025, 0.0074), (0.0124, 0.0018),
    (0.026, 0.013), (0.015, 0.0012), (0.0288, 0.0054), (0.0286, 0.0366), (0.0318, 0.0178), (0.0284, 0.0026),
    (0.0284, 0.0064), (0.0438, 0.0414), (0.0834, 0.0318), (0.0702, 0.0564), (0.0294, 0.009), (0.0306, 0.0094),
    (0.0562, 0.0232), (0.062, 0.0484), (0.0757999999999999, 0.0236), (0.0133999999999999, 0.0218),
    (0.0265999999999999, 0.0016)
]
gate_fidelity = [
    0.000139698, 0.001057873, 0.00016001, 0.000258517, 0.000831734, 0.000402865, 0.000210451, 0.000118894, 0.00016498,
    0.000111386, 0.000223298, 0.000405815, 0.001648386, 0.000217532, 0.00039112, 0.000230048, 0.000215923, 0.000432101,
    0.000146722, 0.000309772, 0.000456124, 0.000342341, 0.000166974, 0.000287958, 0.00013731, 0.000245059, 0.000222355,
    0.000289297, 0.000655687, 0.000312014, 0.004381745, 0.000187747, 0.000267362, 0.000244369, 0.000775934, 0.000205962,
    0.000395927, 0.000297865, 0.001069265, 0.000158094, 0.000247922, 0.006147284, 0.000444197, 0.000450061, 0.000315913,
    0.001314309, 0.0001617, 0.000305174, 0.002503764, 0.000336551, 0.000343072, 0.000503556, 0.000275182, 0.000323902,
    0.000440045, 0.000327502, 0.012383079, 0.000270761, 0.00030092, 0.000264166, 0.00050374, 0.000359798694193482,
    0.000616108049924521, 0.000335279721430018, 0.000329814102040841
]
coupling_strength = [
    (0, 10, 0.0048787038336782496), (0, 1, 0.018492584194803596), (1, 2, 0.011607980434997861), (2, 3, 1),
    (3, 4, 0.008605470267615045), (4, 11, 0.009581421812349389), (4, 5, 0.0168443034968892),
    (5, 6, 0.013103716320157444), (7, 6, 0.007512093972512374), (8, 7, 0.0042120317548426656),
    (8, 9, 0.003062757246794112), (8, 12, 0.00565598711930182), (10, 13, 0.008337727807575301),
    (10, 0, 0.0048787038336782496), (11, 4, 0.009581421812349389), (11, 17, 0.006829492577564905),
    (12, 21, 0.004576885537053793), (14, 13, 0.0055117398078868285), (14, 15, 0.008207686520307622),
    (15, 24, 0.004530881074445325), (16, 17, 0.005261689584252133), (16, 15, 0.005796091273299303),
    (17, 18, 0.009211766252545828), (19, 25, 0.00844729064333774), (19, 20, 0.006106136688024427),
    (19, 18, 0.006842239872192957), (20, 21, 0.010863325642364141), (22, 21, 0.007903758813974066),
    (22, 23, 0.004833855038401003), (26, 37, 0.00631711716874328), (26, 23, 0.010061334774982139),
    (43, 44, 0.0063763271269401955), (28, 27, 0.008053881584905914), (29, 24, 0.003810324514066349),
    (29, 28, 0.010730819775694206), (30, 29, 0.008415167534116474), (31, 30, 0.008456258219165264),
    (31, 32, 0.008948168996772171), (33, 34, 0.013530576242814146), (33, 32, 0.006391836576033483),
    (33, 25, 0.005644135787405663), (34, 35, 0.010254931095457054), (35, 40, 0.015235560757204503),
    (36, 37, 0.008142104350602669), (36, 35, 0.009739741784730416), (38, 41, 0.0304230536063691),
    (38, 27, 0.018152987396058085), (39, 31, 0.00584345950546622), (42, 43, 0.006057354623360406),
    (42, 41, 0.025501088003277994), (45, 44, 0.03768940710034524), (45, 39, 0.02209526372999146),
    (46, 45, 0.010957774179229662), (47, 46, 0.003645603189377178), (47, 53, 0.008188722161684037),
    (48, 47, 1), (49, 40, 0.008396688716518136), (49, 48, 1), (50, 49, 0.005540391966444291),
    (51, 50, 0.009604540880230661), (52, 56, 0.014466516692194842), (52, 43, 0.010431564768317747),
    (54, 64, 0.007789212524240807), (54, 51, 0.008622916911107531), (57, 56, 0.024418980022700237),
    (56, 55, 0.012259521954376268), (57, 58, 0.00931804171589043), (59, 58, 0.007940941129541912),
    (60, 59, 0.00451681818685068), (60, 61, 0.005472918417241623), (60, 53, 0.008560828894919936),
    (61, 62, 0.009392419559424714), (63, 64, 0.0073663601845198046), (63, 62, 0.016553147436820875)
]

readout_frequency = [
    4.868947305, 4.714127244, 4.608429081, 4.839537728, 4.712769483, 4.828676784, 4.638582254, 4.903298669, 4.725716308,
    4.780190649, 4.76703431, 4.655860726, 4.849249218, 4.810274083, 4.714291914, 4.754810332, 4.631440434, 4.717279064,
    4.813445785, 4.583104969, 4.716849311, 4.752266734, 4.901806139, 4.859146304, 4.850316926, 4.657882167, 4.789123537,
    4.699599816, 4.655778398, 4.732017581, 4.928071723, 4.657239653, 4.930799694, 4.764715641, 4.874081359, 4.620836681,
    4.762798337, 4.844396671, 4.582619618, 4.729109816, 4.794241949, 4.66956218, 4.764707979, 4.632101465, 4.879991075,
    4.817622416, 4.869000959, 4.738818761, 4.593337224, 4.731883145, 4.800766311, 4.677733699, 4.874296328, 4.682066381,
    4.635796413, 4.669517911, 4.761500283, 4.594712326, 4.631319405, 4.711682033, 4.541607737, 4.59164051290135,
    4.65673644572878, 4.69912015889331, 4.73433915328094
]
T1_times = [
    231.0101544, 130.3987788, 201.0812797, 205.9311089, 209.3818586, 60.31513376, 135.4114378, 247.0231792, 141.3545296,
    262.8748364, 106.9552121, 318.1338159, 111.4886489, 194.7420277, 130.8811979, 165.6339508, 148.7017199, 272.9171183,
    193.8473289, 185.0271803, 183.0633475, 232.3397512, 183.5620411, 193.2209089, 256.5290565, 214.2613999, 242.358536,
    244.2311758, 229.1559006, 227.4740159, 103.3905368, 198.4717457, 80.90234401, 222.402233, 154.31068, 159.335708,
    76.98908429, 149.7953732, 254.9287569, 134.1346075, 72.19787687, 210.6219257, 167.9943169, 136.708966, 173.7066567,
    167.344389, 159.3695664, 214.2347096, 196.7781729, 315.0557425, 216.1591824, 17.49876062, 177.3693471, 143.5701065,
    119.3264912, 240.4407178, 270.2294031, 97.97196756, 211.3602781, 120.3328949, 233.5026792, 215.720361999382,
    122.313230196504, 188.85067500285, 138.962585096789
]
T2_times = [
    414.7816978, 107.2531545, 200.5131059, 120.4268471, 112.6348881, 76.02882422, 233.3597419, 266.2810302, 56.32076206,
    160.4532189, 190.2735883, 75.1355942, 193.8343326, 271.3017685, 67.84016136, 40.6400007, 155.9297645, 148.8134484,
    208.0808345, 161.0025473, 191.9683885, 73.35364878, 293.0317679, 255.1347208, 446.2235245, 431.6162493, 260.277513,
    20.75884121, 220.7911567, 189.9767221, 139.2379526, 178.5019861, 138.4134657, 18.21946615, 22.26352066, 147.8212378,
    24.13681508, 96.85574113, 233.3051423, 125.0597708, 368.3512972, 194.2942662, 326.2649672, 197.1237974, 282.4850213,
    86.14104502, 271.2413917, 305.6583741, 114.0266838, 317.0884356, 227.3262035, 98.89966087, 181.7102917, 150.2750507,
    87.53071792, 162.1462228, 30.91366412, 143.628762, 95.49621116, 38.4152686, 324.9796048, 160.519603791669,
    125.775419811538, 59.8915965988326, 203.183009653005
]

IBMHummingbirdr3 = VirtualQuantumMachine(
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
