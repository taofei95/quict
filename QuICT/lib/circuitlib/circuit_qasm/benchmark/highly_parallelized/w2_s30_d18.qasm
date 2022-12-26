OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
cx q[1], q[0];
cx q[0], q[1];
cx q[0], q[1];
rz(3.1896302884370833) q[0];
rz(0.8184189576576961) q[1];
rz(3.137674901026067) q[0];
rz(5.410198527614596) q[1];
rz(0.11044583573270389) q[1];
rz(5.3174744943414565) q[0];
rz(2.376951430887648) q[1];
rz(5.928006281672268) q[0];
rz(1.5283215736566502) q[1];
rz(3.2327654068811182) q[0];
cx q[1], q[0];
rz(1.6832896103363886) q[0];
rz(4.378242997151948) q[1];
cx q[1], q[0];
rz(3.895845528553957) q[0];
rz(5.591347446587169) q[1];
rz(5.971417738629283) q[0];
rz(4.865347230536571) q[1];
rz(3.3178420294725424) q[1];
rz(0.6469625004001514) q[0];
rz(5.590690451960048) q[1];
rz(4.89992916090757) q[0];
rz(4.40990368046911) q[0];
rz(3.797765308876202) q[1];
rz(3.4633896013679393) q[0];
measure q[0] -> c[0];
measure q[1] -> c[1];
