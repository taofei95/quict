OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
id q[5];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
id q[3];
tdg q[4];
sdg q[2];
rz(1.5707963267948966) q[2];
cu3(1.5707963267948966, 0, 0) q[4], q[1];
tdg q[0];
sdg q[5];
rxx(0) q[5], q[0];
x q[3];
u1(1.5707963267948966) q[0];
p(0) q[3];
p(0) q[4];
cz q[2], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[3];
x q[5];
ry(1.5707963267948966) q[5];
t q[0];
ryy(1.5707963267948966) q[1], q[0];
t q[4];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[4];
u1(1.5707963267948966) q[5];
id q[5];
sdg q[2];
id q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rxx(0) q[3], q[5];
t q[1];
sdg q[5];
ch q[5], q[3];
u1(1.5707963267948966) q[0];
x q[5];
rz(1.5707963267948966) q[3];
rxx(0) q[2], q[5];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[5];
tdg q[4];
x q[2];
t q[3];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
t q[0];
cu3(1.5707963267948966, 0, 0) q[1], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
h q[2];
id q[2];
tdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[1];
p(0) q[2];
cy q[2], q[0];
rx(1.5707963267948966) q[0];
tdg q[0];
s q[4];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
sdg q[3];
s q[1];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[3];
rx(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
id q[5];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
ch q[2], q[3];
cu1(1.5707963267948966) q[4], q[2];
sdg q[3];
sdg q[5];
u1(1.5707963267948966) q[4];