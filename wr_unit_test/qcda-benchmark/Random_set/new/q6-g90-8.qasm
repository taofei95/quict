OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
rz(1.5707963267948966) q[0];
x q[1];
ry(1.5707963267948966) q[4];
t q[4];
cu1(1.5707963267948966) q[3], q[1];
t q[0];
ch q[3], q[5];
id q[4];
sdg q[1];
rz(1.5707963267948966) q[2];
rxx(0) q[4], q[0];
cz q[4], q[3];
cu1(1.5707963267948966) q[5], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rzz(1.5707963267948966) q[2], q[3];
t q[1];
s q[2];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[3];
cy q[4], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[5];
s q[0];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[4];
cu3(1.5707963267948966, 0, 0) q[1], q[4];
cz q[1], q[0];
cx q[1], q[0];
sdg q[4];
cu3(1.5707963267948966, 0, 0) q[1], q[4];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
ryy(1.5707963267948966) q[0], q[1];
u3(0, 0, 1.5707963267948966) q[0];
s q[0];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
ch q[5], q[0];
rxx(0) q[5], q[0];
sdg q[3];
ch q[1], q[4];
p(0) q[3];
rxx(0) q[5], q[0];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[3];
swap q[3], q[0];
ry(1.5707963267948966) q[0];
cy q[3], q[2];
sdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[1];
x q[2];
id q[3];
tdg q[2];
rx(1.5707963267948966) q[2];
rxx(0) q[1], q[4];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
t q[3];
swap q[1], q[3];
swap q[3], q[1];
rxx(0) q[2], q[3];
t q[4];
crz(1.5707963267948966) q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[3], q[4];
ry(1.5707963267948966) q[1];
cu3(1.5707963267948966, 0, 0) q[3], q[0];
u1(1.5707963267948966) q[2];
id q[2];
x q[4];
ry(1.5707963267948966) q[0];
cz q[0], q[2];
ry(1.5707963267948966) q[4];
cu1(1.5707963267948966) q[5], q[1];
rxx(0) q[4], q[5];
x q[5];
tdg q[5];
tdg q[2];
u1(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
h q[2];
u3(0, 0, 1.5707963267948966) q[2];
swap q[0], q[4];
p(0) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cu1(1.5707963267948966) q[2], q[1];
tdg q[3];