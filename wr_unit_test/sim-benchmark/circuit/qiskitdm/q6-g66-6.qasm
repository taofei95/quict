OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[5];
cy q[5], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[0];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[4];
ry(1.5707963267948966) q[5];
u1(1.5707963267948966) q[2];
sdg q[2];
tdg q[4];
swap q[5], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cx q[3], q[2];
p(0) q[4];
rx(1.5707963267948966) q[2];
sdg q[4];
rx(1.5707963267948966) q[3];
s q[5];
t q[1];
t q[5];
p(0) q[2];
swap q[0], q[1];
p(0) q[1];
rzz(1.5707963267948966) q[4], q[3];
cy q[4], q[5];
rxx(0) q[3], q[1];
rz(1.5707963267948966) q[5];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[5];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[3];
s q[3];
sdg q[2];
u1(1.5707963267948966) q[1];
cy q[3], q[5];
ry(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[5], q[4];
cx q[4], q[1];
tdg q[4];
tdg q[2];
rz(1.5707963267948966) q[5];
cz q[2], q[0];
rxx(0) q[3], q[5];
cy q[3], q[5];
u3(0, 0, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
cu1(1.5707963267948966) q[2], q[0];
tdg q[5];
t q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[0];
rz(1.5707963267948966) q[3];
cy q[2], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[5];
tdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];