OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
rz(1.4211748288213564) q[1];
rz(6.224989282758129) q[0];
cx q[0], q[1];
rz(1.7712791427660899) q[1];
rz(2.23875550793006) q[0];
cx q[0], q[1];
rz(4.443117787013043) q[1];
rz(2.4953315561978977) q[0];
rz(1.8452520195786661) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
rz(1.8702561700419782) q[0];
rz(4.004784906031563) q[1];
rz(3.712829403670115) q[0];
cx q[0], q[1];
rz(2.2812453271732314) q[0];
rz(3.030027575667931) q[1];
rz(3.5984685639863763) q[1];