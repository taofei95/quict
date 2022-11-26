OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
x q[4];
x q[0];
p(0) q[2];
cu1(1.5707963267948966) q[1], q[2];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
h q[5];
rx(1.5707963267948966) q[1];
rzz(1.5707963267948966) q[3], q[0];
x q[3];
t q[1];
id q[2];
rzz(1.5707963267948966) q[5], q[2];
rx(1.5707963267948966) q[7];
id q[6];
u3(0, 0, 1.5707963267948966) q[7];
cu3(1.5707963267948966, 0, 0) q[1], q[4];
ry(1.5707963267948966) q[1];
h q[7];
x q[1];
h q[7];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
id q[6];
rx(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[4];
id q[3];
x q[6];
h q[7];
cz q[4], q[0];
s q[3];
rxx(0) q[3], q[7];
crz(1.5707963267948966) q[6], q[3];
rxx(0) q[2], q[5];
t q[3];
h q[7];
tdg q[1];
rzz(1.5707963267948966) q[0], q[7];
u1(1.5707963267948966) q[2];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
cu1(1.5707963267948966) q[6], q[4];
s q[5];
u1(1.5707963267948966) q[4];
cz q[4], q[7];
p(0) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cx q[6], q[5];
t q[3];
p(0) q[4];
p(0) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
swap q[1], q[7];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
h q[0];
x q[4];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[2];
cy q[5], q[1];
s q[6];
t q[1];
ryy(1.5707963267948966) q[5], q[6];
crz(1.5707963267948966) q[5], q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
h q[3];
rz(1.5707963267948966) q[1];
id q[0];
u1(1.5707963267948966) q[0];
t q[3];
tdg q[2];
h q[6];
rx(1.5707963267948966) q[6];
cz q[7], q[4];
cu1(1.5707963267948966) q[4], q[1];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[6];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
crz(1.5707963267948966) q[3], q[5];
cx q[1], q[4];
rzz(1.5707963267948966) q[2], q[6];
p(0) q[4];
h q[0];
ryy(1.5707963267948966) q[7], q[2];
rx(1.5707963267948966) q[6];
crz(1.5707963267948966) q[3], q[5];
x q[2];
rxx(0) q[4], q[1];
u3(0, 0, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
id q[4];