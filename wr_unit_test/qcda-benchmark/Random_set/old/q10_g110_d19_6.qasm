OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
tdg q[0];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[4];
tdg q[1];
tdg q[6];
cz q[4], q[1];
id q[9];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[4];
t q[0];
rx(1.5707963267948966) q[9];
sdg q[9];
h q[5];
s q[7];
cu1(1.5707963267948966) q[6], q[2];
ch q[0], q[6];
u3(0, 0, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
rzz(1.5707963267948966) q[6], q[8];
cy q[8], q[2];
ry(1.5707963267948966) q[0];
id q[7];
u1(1.5707963267948966) q[2];
tdg q[3];
rz(1.5707963267948966) q[6];
x q[4];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
h q[1];
x q[3];
ry(1.5707963267948966) q[7];
ch q[5], q[7];
h q[1];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[5];
h q[8];
u1(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
p(0) q[5];
t q[2];
u1(1.5707963267948966) q[7];
s q[7];
tdg q[2];
id q[3];
cu1(1.5707963267948966) q[8], q[2];
t q[4];
p(0) q[9];
tdg q[5];
x q[5];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[5];
t q[7];
tdg q[2];
tdg q[1];
rzz(1.5707963267948966) q[0], q[6];
u1(1.5707963267948966) q[8];
crz(1.5707963267948966) q[6], q[8];
s q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[1];
tdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
s q[8];
rx(1.5707963267948966) q[1];
id q[5];
tdg q[4];
rx(1.5707963267948966) q[9];
id q[9];
s q[0];
t q[1];
x q[7];
tdg q[3];
ry(1.5707963267948966) q[4];
h q[4];
crz(1.5707963267948966) q[0], q[5];
cx q[0], q[2];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rx(1.5707963267948966) q[7];
id q[9];
tdg q[8];
tdg q[0];
cy q[2], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[2];
swap q[3], q[2];
rxx(0) q[6], q[4];
s q[8];
s q[6];
rzz(1.5707963267948966) q[6], q[0];
id q[7];
rz(1.5707963267948966) q[1];
sdg q[2];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[4];
rx(1.5707963267948966) q[6];
sdg q[9];
id q[1];
cu3(1.5707963267948966, 0, 0) q[8], q[5];
tdg q[8];
p(0) q[7];