OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
t q[2];
ch q[3], q[7];
u1(5.455098103349925) q[1];
ry(3.8595803931493857) q[1];
ry(4.536801531116241) q[7];
sdg q[6];
p(3.300074502962655) q[6];
p(4.053874153027735) q[7];
tdg q[1];
sdg q[5];
u3(1.4914971954790288, 3.635755558102988, 0.22480342985257526) q[6];
rz(2.6475399906123003) q[4];
u1(2.3330056725420154) q[1];
cz q[1], q[0];
s q[5];
x q[5];
u1(5.166347377906474) q[4];
ry(5.181617988789142) q[1];
ry(5.847098143458239) q[2];
ry(4.371164443705895) q[2];
x q[7];
cz q[4], q[1];
ryy(5.480079232718744) q[1], q[3];
t q[6];
u1(3.288456977166092) q[4];
u2(6.181733458790849, 2.2523475472983296) q[2];
rz(0.2332519861313162) q[7];
h q[2];
u2(1.7627455984402898, 1.3273336650996144) q[6];
s q[1];
x q[7];
p(2.102232455885018) q[2];
rx(5.696046685093279) q[2];
t q[7];
rz(4.281211944212298) q[7];
cx q[7], q[3];
h q[2];
cu1(1.807009729353199) q[4], q[5];
s q[1];
s q[3];
h q[0];
id q[1];
cu3(0.07217215817208625, 1.2603854133688188, 3.364910531461479) q[6], q[0];
u1(4.903715588978007) q[7];
tdg q[7];
cx q[7], q[2];
t q[6];
sdg q[3];
t q[2];
rz(3.449083469061378) q[2];
tdg q[0];
s q[4];
rxx(6.1346354057773365) q[4], q[0];
id q[2];
rx(3.6283228949708204) q[0];
u3(6.067592108226052, 4.7359940038730395, 3.7005512248586956) q[4];