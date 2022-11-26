OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
p(0) q[0];
t q[8];
rz(1.5707963267948966) q[9];
cu1(1.5707963267948966) q[6], q[9];
ry(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[9];
sdg q[0];
tdg q[9];
cz q[11], q[0];
cz q[2], q[3];
t q[1];
ry(1.5707963267948966) q[2];
t q[10];
cu1(1.5707963267948966) q[9], q[6];
sdg q[8];
s q[0];
sdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[7];
t q[6];
sdg q[1];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[10];
h q[9];
s q[9];
cx q[0], q[2];
ry(1.5707963267948966) q[5];
rxx(0) q[6], q[4];
t q[8];
rxx(0) q[2], q[9];
t q[2];
rx(1.5707963267948966) q[3];
cx q[5], q[8];
rx(1.5707963267948966) q[6];
rxx(0) q[10], q[3];
p(0) q[7];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[7];
cy q[5], q[1];
rxx(0) q[5], q[11];
id q[6];
p(0) q[9];
cz q[8], q[4];
cz q[11], q[2];
t q[7];
rz(1.5707963267948966) q[6];
tdg q[3];
swap q[1], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[10];
p(0) q[7];
tdg q[9];
rz(1.5707963267948966) q[6];
swap q[2], q[0];
p(0) q[9];
sdg q[1];
cx q[3], q[5];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[4];
t q[6];
u3(0, 0, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[11];
s q[0];
cu1(1.5707963267948966) q[9], q[8];
ry(1.5707963267948966) q[1];
cz q[10], q[1];
rz(1.5707963267948966) q[5];
h q[7];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[11];
cu1(1.5707963267948966) q[7], q[6];
u3(0, 0, 1.5707963267948966) q[2];
cy q[10], q[1];
ry(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[10];
t q[2];
sdg q[0];
cy q[4], q[11];
s q[0];
p(0) q[4];
u3(0, 0, 1.5707963267948966) q[2];
s q[5];
cz q[8], q[1];
rx(1.5707963267948966) q[11];
t q[5];
p(0) q[0];
ry(1.5707963267948966) q[10];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cz q[4], q[7];
ry(1.5707963267948966) q[9];
cx q[0], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[10];
sdg q[11];
u1(1.5707963267948966) q[8];
rxx(0) q[1], q[5];
t q[11];
h q[11];
rz(1.5707963267948966) q[2];
swap q[3], q[7];
s q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
s q[4];
ry(1.5707963267948966) q[10];
u1(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[5];
cy q[10], q[7];
sdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
p(0) q[5];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[4];
swap q[7], q[9];
u1(1.5707963267948966) q[8];
cu1(1.5707963267948966) q[0], q[4];
tdg q[3];
sdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[0];
id q[8];
rx(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[11];
cz q[1], q[10];
t q[11];
rxx(0) q[9], q[7];
cu1(1.5707963267948966) q[5], q[7];
ry(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[6];
tdg q[11];
cz q[4], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[8];
rx(1.5707963267948966) q[9];
rzz(1.5707963267948966) q[3], q[9];
t q[9];
rzz(1.5707963267948966) q[7], q[2];
u3(0, 0, 1.5707963267948966) q[0];
s q[4];
u1(1.5707963267948966) q[11];
p(0) q[1];
swap q[8], q[0];
rz(1.5707963267948966) q[5];
u1(1.5707963267948966) q[6];
cz q[3], q[11];
cy q[2], q[10];
rz(1.5707963267948966) q[11];
h q[3];
cx q[3], q[1];
u3(0, 0, 1.5707963267948966) q[9];
t q[2];
id q[1];
u1(1.5707963267948966) q[8];
cz q[7], q[6];
ry(1.5707963267948966) q[8];
p(0) q[5];
rxx(0) q[2], q[10];
sdg q[7];
u1(1.5707963267948966) q[10];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[0];
sdg q[10];
sdg q[5];
p(0) q[6];
u1(1.5707963267948966) q[0];
cz q[5], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[10];
sdg q[8];
rx(1.5707963267948966) q[10];
h q[5];
rxx(0) q[2], q[8];
tdg q[4];
rz(1.5707963267948966) q[11];
tdg q[10];
sdg q[6];
p(0) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
cy q[2], q[5];