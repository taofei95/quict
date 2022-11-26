OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[0];
swap q[2], q[1];
p(0) q[4];
t q[3];
u3(0, 0, 1.5707963267948966) q[1];
p(0) q[4];
cu1(1.5707963267948966) q[0], q[2];
u3(0, 0, 1.5707963267948966) q[4];
p(0) q[3];
ch q[3], q[1];
rz(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
sdg q[3];
ch q[0], q[4];
x q[2];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
ch q[2], q[4];
cu1(1.5707963267948966) q[1], q[4];
id q[3];
t q[0];
u1(1.5707963267948966) q[3];
sdg q[0];
ch q[2], q[0];
swap q[1], q[2];
ch q[0], q[4];
sdg q[2];
id q[0];
rx(1.5707963267948966) q[3];
t q[3];
rx(1.5707963267948966) q[0];
u1(1.5707963267948966) q[1];
t q[2];
u3(0, 0, 1.5707963267948966) q[1];
t q[2];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
sdg q[3];
tdg q[1];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[3];
p(0) q[4];
cx q[3], q[2];
t q[4];
id q[2];
ry(1.5707963267948966) q[4];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
swap q[4], q[3];
p(0) q[3];
swap q[0], q[3];
sdg q[2];
ry(1.5707963267948966) q[0];
h q[2];
id q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[0];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
x q[2];