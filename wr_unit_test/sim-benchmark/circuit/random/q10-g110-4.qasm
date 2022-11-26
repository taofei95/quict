OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
t q[9];
rzz(1.5707963267948966) q[3], q[5];
s q[7];
rz(1.5707963267948966) q[6];
t q[5];
s q[2];
rz(1.5707963267948966) q[5];
cy q[9], q[2];
t q[4];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[0];
x q[6];
id q[0];
p(0) q[1];
cu3(1.5707963267948966, 0, 0) q[3], q[0];
p(0) q[3];
u3(0, 0, 1.5707963267948966) q[2];
cu1(1.5707963267948966) q[3], q[0];
cy q[2], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
sdg q[4];
x q[7];
ryy(1.5707963267948966) q[2], q[8];
cy q[9], q[3];
x q[9];
u2(1.5707963267948966, 1.5707963267948966) q[2];
p(0) q[3];
u1(1.5707963267948966) q[0];
rz(1.5707963267948966) q[3];
id q[4];
rxx(0) q[1], q[4];
t q[3];
u1(1.5707963267948966) q[3];
p(0) q[9];
sdg q[3];
s q[4];
tdg q[1];
s q[3];
id q[9];
cy q[1], q[7];
p(0) q[1];
p(0) q[7];
id q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
id q[2];
x q[4];
h q[4];
x q[0];
tdg q[5];
cz q[2], q[6];
rz(1.5707963267948966) q[6];
id q[4];
p(0) q[2];
ch q[0], q[7];
cz q[0], q[4];
t q[6];
s q[5];
id q[7];
s q[6];
p(0) q[8];
x q[3];
rz(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
tdg q[5];
id q[0];
ch q[5], q[8];
rx(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[0];
tdg q[2];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[7];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[4];
s q[4];
cz q[6], q[9];
crz(1.5707963267948966) q[0], q[2];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rzz(1.5707963267948966) q[6], q[2];
ryy(1.5707963267948966) q[7], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ch q[3], q[6];
p(0) q[7];
tdg q[1];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[7];
ry(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ryy(1.5707963267948966) q[1], q[0];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[2];
cz q[2], q[6];
h q[9];
cz q[3], q[0];
id q[8];
sdg q[9];
x q[1];
tdg q[9];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[7];
tdg q[1];
rx(1.5707963267948966) q[3];
cu1(1.5707963267948966) q[2], q[5];
rxx(0) q[9], q[5];
t q[7];