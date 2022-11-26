OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u1(1.5707963267948966) q[7];
id q[5];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[8];
t q[2];
h q[8];
h q[1];
id q[9];
h q[6];
crz(1.5707963267948966) q[0], q[7];
crz(1.5707963267948966) q[2], q[4];
cy q[3], q[7];
id q[5];
id q[9];
ch q[0], q[9];
h q[3];
ry(1.5707963267948966) q[9];
sdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[9];
crz(1.5707963267948966) q[8], q[4];
cu1(1.5707963267948966) q[7], q[8];
s q[7];
h q[6];
s q[6];
ch q[7], q[0];
rz(1.5707963267948966) q[8];
cz q[1], q[7];
id q[6];
sdg q[1];
u1(1.5707963267948966) q[1];
sdg q[9];
p(0) q[5];
u3(0, 0, 1.5707963267948966) q[9];
tdg q[7];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[4];
s q[1];
rz(1.5707963267948966) q[6];
cy q[3], q[7];
id q[2];
t q[0];
h q[7];
t q[7];
ch q[3], q[4];
x q[9];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[4];
cu3(1.5707963267948966, 0, 0) q[3], q[0];
h q[6];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[9];
t q[2];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[7];
h q[2];
id q[7];
rz(1.5707963267948966) q[8];
t q[8];
cu3(1.5707963267948966, 0, 0) q[5], q[2];
p(0) q[3];
tdg q[6];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[2];
cu1(1.5707963267948966) q[5], q[0];
u1(1.5707963267948966) q[7];
p(0) q[7];
rxx(0) q[7], q[8];
tdg q[1];
rxx(0) q[8], q[5];
cz q[3], q[9];
rz(1.5707963267948966) q[8];
swap q[5], q[9];
ry(1.5707963267948966) q[1];
t q[1];
h q[1];
t q[9];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
swap q[6], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[9];
id q[9];
x q[4];
u1(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[5];
t q[7];
ry(1.5707963267948966) q[3];
tdg q[9];
rxx(0) q[2], q[7];
cu3(1.5707963267948966, 0, 0) q[2], q[8];
h q[0];
ch q[8], q[0];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[9];
id q[0];
cx q[4], q[0];
s q[3];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
tdg q[4];
t q[4];
u1(1.5707963267948966) q[0];
p(0) q[0];
id q[0];
rxx(0) q[1], q[7];
rxx(0) q[2], q[6];
u1(1.5707963267948966) q[7];
rzz(1.5707963267948966) q[0], q[5];
t q[2];
u1(1.5707963267948966) q[3];
sdg q[2];
t q[2];
s q[0];
rxx(0) q[5], q[4];
x q[6];
t q[3];
s q[4];
h q[7];
h q[5];
s q[6];
ry(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[9];
x q[0];
h q[2];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[3];
tdg q[2];
ry(1.5707963267948966) q[5];
h q[1];
u1(1.5707963267948966) q[3];
x q[7];
ry(1.5707963267948966) q[5];
h q[9];
p(0) q[2];
x q[2];
swap q[4], q[1];
t q[6];
cu1(1.5707963267948966) q[1], q[7];
cu1(1.5707963267948966) q[6], q[5];
rz(1.5707963267948966) q[0];