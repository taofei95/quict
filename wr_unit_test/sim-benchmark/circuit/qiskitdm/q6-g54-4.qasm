OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
sdg q[3];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
u1(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[2];
cy q[1], q[2];
ry(1.5707963267948966) q[1];
rzz(1.5707963267948966) q[1], q[4];
ry(1.5707963267948966) q[2];
cy q[3], q[4];
swap q[1], q[3];
p(0) q[2];
sdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rzz(1.5707963267948966) q[0], q[2];
id q[2];
cx q[3], q[0];
s q[1];
tdg q[3];
id q[3];
rzz(1.5707963267948966) q[4], q[2];
u1(1.5707963267948966) q[2];
p(0) q[2];
h q[1];
u1(1.5707963267948966) q[0];
t q[4];
rxx(0) q[0], q[2];
ry(1.5707963267948966) q[3];
id q[1];
h q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cz q[4], q[0];
rzz(1.5707963267948966) q[5], q[3];
cz q[3], q[4];
s q[1];
sdg q[1];
rxx(0) q[3], q[2];
p(0) q[2];
rz(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[3];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[2];
cy q[2], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
p(0) q[4];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
sdg q[3];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[3];
cx q[0], q[3];
ry(1.5707963267948966) q[3];