OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
t q[1];
rx(1.5707963267948966) q[4];
p(0) q[4];
crz(1.5707963267948966) q[3], q[0];
rx(1.5707963267948966) q[4];
sdg q[4];
t q[0];
h q[4];
cx q[2], q[1];
cy q[3], q[2];
rz(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[0], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[0];
t q[4];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[4];
tdg q[0];
id q[0];
swap q[3], q[4];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[0];
ch q[1], q[2];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[1];
s q[3];
u1(1.5707963267948966) q[4];
h q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[2];
cy q[1], q[0];
p(0) q[1];
rxx(0) q[0], q[4];
t q[4];
ry(1.5707963267948966) q[2];
cz q[4], q[2];
p(0) q[2];
u3(0, 0, 1.5707963267948966) q[4];
rzz(1.5707963267948966) q[3], q[4];
cu1(1.5707963267948966) q[3], q[4];
crz(1.5707963267948966) q[0], q[3];
rxx(0) q[3], q[4];
ch q[0], q[3];
sdg q[4];
p(0) q[2];
ry(1.5707963267948966) q[0];
s q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
id q[4];
t q[1];
s q[0];
u3(0, 0, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
p(0) q[0];
s q[2];
rx(1.5707963267948966) q[0];
s q[3];
id q[4];
rz(1.5707963267948966) q[4];
tdg q[4];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
p(0) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[1];
swap q[1], q[3];
s q[3];