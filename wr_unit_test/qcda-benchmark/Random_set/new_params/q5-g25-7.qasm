OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
cz q[0], q[1];
cz q[4], q[2];
x q[1];
ch q[4], q[1];
sdg q[0];
u2(5.066204911303787, 5.0993159302340345) q[0];
swap q[1], q[2];
x q[2];
t q[3];
ry(4.402002563292558) q[2];
ry(3.1666091085524157) q[2];
u1(5.574495741353951) q[2];
s q[3];
p(0.3873367974887948) q[4];
u2(5.474606884884709, 0.27610091336800285) q[0];
rz(5.572409454159864) q[3];
tdg q[1];
s q[4];
rz(2.083107329866601) q[0];
s q[4];
h q[1];
s q[2];
h q[2];
s q[3];
h q[0];