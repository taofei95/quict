OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
id q[2];
ry(1.3174104231422936) q[3];
rz(2.1702933531183617) q[1];
s q[4];
s q[2];
t q[2];
p(2.3417559698820085) q[3];
id q[1];
id q[0];
sdg q[0];
id q[1];
swap q[0], q[2];
cx q[1], q[2];
s q[4];
s q[2];
id q[3];
sdg q[4];
cu1(0.9551599896544716) q[0], q[2];
x q[2];
u1(1.987454051298124) q[4];
id q[1];
cu1(6.226199442176848) q[4], q[3];
u1(3.989652108755415) q[1];
tdg q[1];
cy q[2], q[3];
id q[4];
x q[2];
u2(1.2806065850905182, 2.075146106531348) q[1];
rx(0.5470799626977465) q[1];
t q[1];
p(4.329698049421696) q[0];
id q[3];
tdg q[3];
p(0.6528629742819758) q[1];
cy q[0], q[3];
t q[4];
ch q[1], q[2];
sdg q[3];
p(3.756932430523978) q[4];
p(4.182550472536911) q[2];
p(0.15362541714368483) q[2];
tdg q[4];
rz(6.135750406812494) q[1];
u1(2.7764430561595206) q[3];
rx(4.102676251625331) q[4];
rx(3.2383659780335416) q[4];
s q[2];
cx q[0], q[4];
crz(2.333928183068682) q[1], q[2];
u3(4.297946225992294, 5.949823567001795, 3.523433005100037) q[3];
s q[1];
cx q[2], q[3];
sdg q[4];
x q[2];
rz(4.4092677843684625) q[1];