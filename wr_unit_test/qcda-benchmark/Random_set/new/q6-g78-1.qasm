OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
cy q[2], q[0];
u3(0, 0, 1.5707963267948966) q[2];
t q[1];
t q[0];
rz(1.5707963267948966) q[0];
rxx(0) q[2], q[1];
x q[3];
u3(0, 0, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
sdg q[4];
h q[5];
p(0) q[3];
t q[4];
h q[1];
h q[0];
cu1(1.5707963267948966) q[0], q[1];
s q[2];
s q[1];
s q[2];
u3(0, 0, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
cz q[5], q[4];
u3(0, 0, 1.5707963267948966) q[3];
t q[3];
ry(1.5707963267948966) q[1];
s q[5];
sdg q[1];
t q[2];
tdg q[0];
ry(1.5707963267948966) q[2];
s q[1];
u3(0, 0, 1.5707963267948966) q[4];
s q[3];
x q[5];
rz(1.5707963267948966) q[0];
t q[5];
cu3(1.5707963267948966, 0, 0) q[3], q[1];
u1(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
cu3(1.5707963267948966, 0, 0) q[2], q[4];
h q[3];
u3(0, 0, 1.5707963267948966) q[1];
swap q[4], q[0];
u3(0, 0, 1.5707963267948966) q[3];
tdg q[1];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
p(0) q[2];
s q[2];
rz(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[4], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cy q[5], q[1];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[5];
cz q[1], q[4];
rz(1.5707963267948966) q[5];
cx q[4], q[1];
p(0) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[4];
rx(1.5707963267948966) q[4];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[2];
cu1(1.5707963267948966) q[0], q[5];
id q[3];
cz q[0], q[5];
id q[4];
rx(1.5707963267948966) q[3];
cu3(1.5707963267948966, 0, 0) q[1], q[0];
x q[0];
t q[2];
t q[2];
rz(1.5707963267948966) q[0];
sdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];