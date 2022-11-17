OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
rz(6.045895024636931) q[5];
rz(3.4724665825270287) q[3];
cx q[2], q[0];
rz(0.00011954889819942002) q[4];
rz(0.3664359518919668) q[1];
rz(0.5416362900869849) q[2];
rz(3.1521048184023432) q[0];
rz(1.6968885344467912) q[1];
cx q[3], q[5];
rz(5.95025759075401) q[4];
rz(3.8693906835414063) q[4];
rz(4.007636723142915) q[2];
rz(0.34852324363749243) q[5];
rz(3.8297528970481687) q[3];
rz(3.689385166191857) q[0];
rz(3.649922696361309) q[1];
rz(5.507021335596823) q[5];
cx q[4], q[1];
rz(5.894127374646758) q[3];
rz(3.8501863158617327) q[2];
rz(4.5428212866103745) q[0];
rz(4.067875619248001) q[5];
rz(4.829959213308155) q[3];
cx q[1], q[4];
rz(5.744662308691468) q[2];
rz(2.0732233256082258) q[0];
cx q[0], q[4];
rz(4.44723824472718) q[5];
rz(1.91332807798732) q[3];
rz(3.899082431376714) q[2];
rz(4.834273221283764) q[1];
rz(2.7771394966920453) q[4];
rz(4.404174320985571) q[0];
rz(2.4636933536938854) q[3];
rz(2.7891038943199753) q[2];
rz(5.622496726423105) q[1];
rz(6.243991505375417) q[5];
rz(4.649396457577145) q[5];
rz(5.8949304705396415) q[3];
rz(1.2915499973433449) q[2];
cx q[0], q[4];
rz(2.1426576144174545) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];