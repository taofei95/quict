OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
id q[1];
t q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rzz(1.5707963267948966) q[1], q[0];
u3(0, 0, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[2];
rzz(1.5707963267948966) q[1], q[2];
rxx(0) q[1], q[0];
s q[3];
id q[0];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
cy q[0], q[1];
rx(1.5707963267948966) q[1];
h q[2];
sdg q[2];
p(0) q[0];
sdg q[3];
t q[0];
t q[3];
p(0) q[3];
sdg q[2];
rxx(0) q[0], q[3];
sdg q[3];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
h q[0];
ry(1.5707963267948966) q[2];
tdg q[3];
u3(0, 0, 1.5707963267948966) q[2];
cu3(1.5707963267948966, 0, 0) q[0], q[1];
ry(1.5707963267948966) q[3];
cz q[3], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[0];
u1(1.5707963267948966) q[1];
p(0) q[3];
swap q[3], q[0];
rz(1.5707963267948966) q[1];
swap q[0], q[3];
cz q[2], q[1];
id q[2];
u3(0, 0, 1.5707963267948966) q[2];
h q[3];
u1(1.5707963267948966) q[0];
s q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
p(0) q[3];
u1(1.5707963267948966) q[1];
cx q[1], q[2];
ry(1.5707963267948966) q[2];
t q[0];
u1(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
id q[3];
t q[0];
cy q[1], q[3];
rz(1.5707963267948966) q[1];