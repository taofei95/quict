OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
sdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
p(0) q[6];
t q[1];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[3];
id q[5];
ch q[1], q[2];
x q[4];
h q[5];
sdg q[3];
rzz(1.5707963267948966) q[2], q[4];
x q[2];
u3(0, 0, 1.5707963267948966) q[3];
cy q[3], q[2];
h q[5];
rzz(1.5707963267948966) q[4], q[6];
ch q[4], q[3];
tdg q[2];
ch q[0], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[1];
rzz(1.5707963267948966) q[6], q[4];
sdg q[2];
h q[0];
x q[5];
h q[6];
crz(1.5707963267948966) q[2], q[4];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[2], q[3];
id q[6];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[5];
cx q[0], q[2];
tdg q[0];
ry(1.5707963267948966) q[1];
t q[1];
s q[5];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[1];
ch q[2], q[0];
rz(1.5707963267948966) q[6];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[4];
swap q[2], q[0];
rx(1.5707963267948966) q[5];
s q[3];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
p(0) q[6];
h q[0];
tdg q[0];
t q[0];
cu1(1.5707963267948966) q[6], q[1];
x q[6];
t q[4];
s q[0];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[4];
x q[0];
u3(0, 0, 1.5707963267948966) q[0];
p(0) q[6];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[3];
x q[0];
p(0) q[1];
u1(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
p(0) q[0];
x q[2];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[6];
cy q[4], q[1];
ch q[4], q[2];
rz(1.5707963267948966) q[1];
s q[4];
ry(1.5707963267948966) q[1];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[6];
h q[4];
rx(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[3], q[4];
sdg q[4];
p(0) q[2];
rz(1.5707963267948966) q[6];
s q[4];
x q[5];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[0];
tdg q[6];
s q[6];