OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
t q[0];
id q[4];
p(0.12901284107641856) q[9];
x q[9];
rx(5.680925277024297) q[8];
tdg q[3];
u1(3.683709289530533) q[8];
x q[7];
u3(3.129781729100748, 3.589641692018958, 1.8460626391965378) q[1];
rz(1.9744598945857224) q[8];
rz(5.24713772956667) q[9];
x q[2];
rxx(0.265050883532855) q[6], q[1];
p(2.4771437057082952) q[6];
u3(2.7105711855311823, 1.9844564607660662, 6.011765704168473) q[1];
swap q[9], q[3];
s q[9];
s q[3];
cx q[2], q[0];
u2(0.6002187599837724, 0.34814004381243663) q[9];
cx q[2], q[8];
swap q[5], q[1];
ch q[5], q[2];
t q[3];
ryy(6.122041816565889) q[3], q[0];
rxx(6.221019779390913) q[4], q[1];
cy q[9], q[0];
h q[9];
u1(3.1980130926451626) q[6];
u2(4.748824592097363, 4.192914184405127) q[9];
p(0.3458569759075806) q[7];
x q[5];
cx q[6], q[1];
sdg q[3];
h q[0];
ry(5.712890852387893) q[9];
u2(0.11754032904535239, 3.4888460866551547) q[9];
u3(4.693046060953512, 5.033959387135618, 5.40768055016923) q[6];
tdg q[1];
rz(3.8390150932532108) q[1];
u1(4.970948204478114) q[6];
crz(1.109457436725274) q[9], q[3];
p(1.7981319804904154) q[9];
h q[2];
s q[6];
x q[9];
sdg q[7];
u3(2.4876067174915963, 5.623384817262638, 2.9598202741884623) q[7];
cu3(1.0311825992644366, 1.4781476543664291, 0.31246944217671485) q[9], q[8];
rx(4.7037025888398105) q[5];