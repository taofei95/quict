OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
t q[2];
crz(1.5707963267948966) q[4], q[1];
tdg q[3];
ry(1.5707963267948966) q[5];
ch q[2], q[5];
s q[4];
x q[2];
s q[2];
cz q[1], q[4];
x q[5];
cu3(1.5707963267948966, 0, 0) q[3], q[2];
h q[1];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[2];
x q[1];
rz(1.5707963267948966) q[3];
tdg q[0];
u3(0, 0, 1.5707963267948966) q[1];
p(0) q[1];
u3(0, 0, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
tdg q[5];
sdg q[2];
s q[3];
t q[5];
sdg q[0];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[2];
p(0) q[0];
ryy(1.5707963267948966) q[2], q[1];
cz q[4], q[5];
u3(0, 0, 1.5707963267948966) q[0];
h q[5];
h q[5];
t q[0];
ch q[4], q[3];
u1(1.5707963267948966) q[2];
ch q[3], q[2];
ryy(1.5707963267948966) q[3], q[1];
ryy(1.5707963267948966) q[0], q[4];
u1(1.5707963267948966) q[0];
id q[1];
h q[3];
rx(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[1], q[3];
t q[2];
id q[0];
ch q[5], q[2];
x q[1];
x q[2];
rz(1.5707963267948966) q[1];
cu3(1.5707963267948966, 0, 0) q[5], q[3];
rxx(0) q[4], q[5];
cy q[3], q[1];
cu1(1.5707963267948966) q[5], q[4];
t q[0];
rzz(1.5707963267948966) q[1], q[2];
crz(1.5707963267948966) q[3], q[5];
rx(1.5707963267948966) q[5];
x q[4];
rz(1.5707963267948966) q[5];
x q[0];
sdg q[4];
ch q[2], q[4];
rz(1.5707963267948966) q[3];
sdg q[2];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[3];
t q[2];
rz(1.5707963267948966) q[5];
s q[4];
p(0) q[5];
rz(1.5707963267948966) q[1];
x q[5];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cu3(1.5707963267948966, 0, 0) q[5], q[3];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
ryy(1.5707963267948966) q[3], q[5];
h q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
x q[3];