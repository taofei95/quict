OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
p(0) q[4];
t q[4];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[4];
t q[6];
rx(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[2];
t q[8];
ry(1.5707963267948966) q[0];
p(0) q[3];
u1(1.5707963267948966) q[10];
rx(1.5707963267948966) q[9];
cu3(1.5707963267948966, 0, 0) q[10], q[8];
u1(1.5707963267948966) q[11];
p(0) q[8];
s q[6];
t q[1];
cu1(1.5707963267948966) q[5], q[1];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[4];
swap q[2], q[11];
p(0) q[1];
s q[2];
s q[9];
rz(1.5707963267948966) q[6];
cz q[1], q[9];
u1(1.5707963267948966) q[2];
p(0) q[9];
rz(1.5707963267948966) q[7];
rz(1.5707963267948966) q[3];
cu3(1.5707963267948966, 0, 0) q[9], q[11];
u3(0, 0, 1.5707963267948966) q[11];
tdg q[0];
cu1(1.5707963267948966) q[4], q[5];
rx(1.5707963267948966) q[6];
t q[7];
s q[2];
swap q[11], q[2];
sdg q[3];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[3];
s q[0];
u2(1.5707963267948966, 1.5707963267948966) q[10];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[8];
u1(1.5707963267948966) q[3];
h q[9];
id q[10];
u3(0, 0, 1.5707963267948966) q[6];
cu3(1.5707963267948966, 0, 0) q[6], q[4];
cu3(1.5707963267948966, 0, 0) q[4], q[3];
ry(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[0];
p(0) q[2];