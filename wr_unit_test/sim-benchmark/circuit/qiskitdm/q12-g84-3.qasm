OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
cy q[0], q[11];
id q[3];
cx q[7], q[8];
t q[4];
sdg q[3];
t q[8];
s q[5];
p(0) q[8];
u1(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[9];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[6];
rzz(1.5707963267948966) q[9], q[11];
rz(1.5707963267948966) q[11];
p(0) q[0];
sdg q[10];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[1];
s q[8];
ry(1.5707963267948966) q[7];
tdg q[6];
cu1(1.5707963267948966) q[11], q[6];
u1(1.5707963267948966) q[7];
tdg q[4];
rx(1.5707963267948966) q[11];
h q[10];
rx(1.5707963267948966) q[10];
s q[11];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rzz(1.5707963267948966) q[1], q[3];
h q[8];
rzz(1.5707963267948966) q[3], q[9];
t q[10];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[11];
p(0) q[1];
id q[4];
ry(1.5707963267948966) q[0];
id q[8];
h q[2];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[4];
t q[1];
cx q[1], q[5];
u3(0, 0, 1.5707963267948966) q[11];
rx(1.5707963267948966) q[6];
t q[1];
rzz(1.5707963267948966) q[10], q[11];
swap q[8], q[11];
u1(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[4];
t q[8];
u3(0, 0, 1.5707963267948966) q[0];
h q[11];
u1(1.5707963267948966) q[10];
s q[0];
rx(1.5707963267948966) q[9];
sdg q[1];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
s q[11];
ry(1.5707963267948966) q[7];
t q[8];
ry(1.5707963267948966) q[1];
tdg q[4];
ry(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[10];
s q[0];
s q[1];
rzz(1.5707963267948966) q[10], q[2];
u3(0, 0, 1.5707963267948966) q[9];
rx(1.5707963267948966) q[5];
p(0) q[2];
rx(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[10];
cu1(1.5707963267948966) q[3], q[7];
rz(1.5707963267948966) q[10];