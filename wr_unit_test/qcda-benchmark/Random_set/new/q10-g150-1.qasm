OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[0];
id q[3];
h q[7];
cy q[6], q[8];
u3(0, 0, 1.5707963267948966) q[4];
x q[9];
s q[1];
cu1(1.5707963267948966) q[7], q[2];
s q[2];
u1(1.5707963267948966) q[7];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
t q[6];
s q[6];
u3(0, 0, 1.5707963267948966) q[6];
t q[6];
rxx(0) q[8], q[1];
cz q[3], q[5];
t q[9];
ry(1.5707963267948966) q[7];
h q[3];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
h q[3];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
cz q[3], q[7];
ry(1.5707963267948966) q[3];
x q[5];
t q[6];
id q[7];
x q[4];
h q[3];
rx(1.5707963267948966) q[0];
t q[4];
t q[7];
tdg q[8];
p(0) q[0];
rx(1.5707963267948966) q[8];
u1(1.5707963267948966) q[2];
s q[3];
u3(0, 0, 1.5707963267948966) q[6];
id q[9];
s q[0];
t q[8];
h q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[2];
t q[4];
rz(1.5707963267948966) q[8];
t q[4];
x q[3];
cx q[3], q[5];
t q[0];
ry(1.5707963267948966) q[1];
cz q[7], q[4];
u3(0, 0, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[6];
x q[2];
sdg q[6];
cy q[7], q[1];
h q[1];
crz(1.5707963267948966) q[0], q[5];
id q[8];
t q[0];
rz(1.5707963267948966) q[3];
cx q[8], q[4];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[9];
id q[8];
h q[9];
rzz(1.5707963267948966) q[1], q[0];
t q[4];
cz q[2], q[3];
x q[0];
s q[4];
u1(1.5707963267948966) q[3];
swap q[5], q[2];
h q[6];
x q[1];
cu1(1.5707963267948966) q[2], q[1];
h q[9];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[3];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[1];
p(0) q[6];
crz(1.5707963267948966) q[0], q[4];
t q[0];
u2(1.5707963267948966, 1.5707963267948966) q[8];
sdg q[1];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[8];
p(0) q[6];
cu3(1.5707963267948966, 0, 0) q[9], q[5];
t q[6];
rx(1.5707963267948966) q[8];
t q[3];
ry(1.5707963267948966) q[8];
p(0) q[3];
cu3(1.5707963267948966, 0, 0) q[7], q[6];
cu1(1.5707963267948966) q[4], q[6];
u1(1.5707963267948966) q[7];
h q[2];
rz(1.5707963267948966) q[2];
id q[4];
cz q[7], q[5];
ry(1.5707963267948966) q[1];
sdg q[8];
cu3(1.5707963267948966, 0, 0) q[3], q[0];
cy q[0], q[5];
x q[3];
p(0) q[2];
h q[6];
sdg q[6];
t q[9];
cu1(1.5707963267948966) q[7], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[8];
h q[3];
t q[2];
u3(0, 0, 1.5707963267948966) q[4];
t q[1];
ry(1.5707963267948966) q[6];
cy q[8], q[0];
t q[9];
id q[1];
x q[0];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[6];
swap q[9], q[7];
rx(1.5707963267948966) q[3];
x q[0];
sdg q[3];
p(0) q[2];
rz(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[8];
t q[4];
id q[8];
rz(1.5707963267948966) q[6];