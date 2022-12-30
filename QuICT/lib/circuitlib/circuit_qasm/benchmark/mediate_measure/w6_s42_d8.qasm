OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
rz(4.117772653604396) q[2];
rz(3.036289695024901) q[3];
rz(1.9832350461612773) q[1];
rz(1.3195743179041501) q[4];
rz(2.532991136147564) q[5];
rz(1.8497579063275738) q[0];
rz(3.435947017631531) q[2];
cx q[0], q[5];
rz(4.618560092880875) q[3];
rz(6.274838718855028) q[4];
rz(5.015820459980561) q[1];
rz(0.32043817865967433) q[4];
rz(2.314552834701394) q[0];
rz(5.15962497377366) q[5];
rz(5.692521403549144) q[2];
rz(4.688741947608832) q[1];
rz(2.1026628747859624) q[3];
rz(0.2123850781008236) q[2];
cx q[5], q[0];
cx q[1], q[4];
rz(5.971279244709425) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
rz(4.365678293903665) q[3];
rz(3.140933657075599) q[2];
rz(2.7288366546360434) q[1];
rz(3.6833225530787073) q[5];
rz(5.405118575309654) q[4];
rz(5.745677194178415) q[0];
rz(3.902042235635491) q[3];
rz(2.6997828440279945) q[2];
rz(3.4254544645549685) q[4];
rz(3.8673533600424594) q[0];
cx q[5], q[1];
rz(3.7813286873966296) q[4];
rz(0.8605988823479326) q[3];
rz(4.492773054732604) q[2];
rz(3.415203205876368) q[0];