OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
ry(1.5707963267948966) q[7];
p(0) q[8];
u3(0, 0, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
p(0) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
cx q[1], q[3];
u3(0, 0, 1.5707963267948966) q[8];
cx q[11], q[6];
h q[9];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[10];
h q[0];
rzz(1.5707963267948966) q[1], q[8];
sdg q[11];
cu1(1.5707963267948966) q[6], q[0];
tdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[6];
t q[10];
id q[7];
cz q[5], q[3];
id q[10];
h q[11];
u3(0, 0, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[9];
swap q[0], q[1];
ry(1.5707963267948966) q[1];
p(0) q[1];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[5];
tdg q[1];
rx(1.5707963267948966) q[11];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[1];
p(0) q[2];
t q[4];
p(0) q[9];
swap q[9], q[3];
s q[2];
t q[3];
h q[3];
tdg q[10];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[11];
rx(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[1];
sdg q[8];
t q[7];
u1(1.5707963267948966) q[5];
sdg q[11];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[7];
t q[7];
ry(1.5707963267948966) q[2];
h q[11];
t q[6];
rx(1.5707963267948966) q[8];
p(0) q[4];
rx(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[6];
u1(1.5707963267948966) q[7];
s q[11];
p(0) q[8];
u1(1.5707963267948966) q[9];
h q[4];
rx(1.5707963267948966) q[10];
tdg q[2];
ry(1.5707963267948966) q[11];
h q[8];
t q[10];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[11];
rzz(1.5707963267948966) q[0], q[5];
cz q[2], q[11];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[9];
rzz(1.5707963267948966) q[10], q[11];
rx(1.5707963267948966) q[9];
u1(1.5707963267948966) q[3];
u1(1.5707963267948966) q[0];
rzz(1.5707963267948966) q[4], q[7];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
tdg q[7];
u3(0, 0, 1.5707963267948966) q[6];
swap q[8], q[0];
rx(1.5707963267948966) q[5];
cu1(1.5707963267948966) q[11], q[5];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
t q[5];
u3(0, 0, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[10];
cy q[10], q[11];
ry(1.5707963267948966) q[4];
s q[1];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[7];
id q[9];
rx(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[7];
id q[8];
s q[0];
rx(1.5707963267948966) q[6];
u1(1.5707963267948966) q[6];
sdg q[0];
s q[4];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[7];
id q[7];
t q[2];
s q[11];
ry(1.5707963267948966) q[10];
p(0) q[5];
rxx(0) q[0], q[9];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
cu1(1.5707963267948966) q[8], q[3];
rz(1.5707963267948966) q[8];
s q[0];
u3(0, 0, 1.5707963267948966) q[5];
id q[2];
cu1(1.5707963267948966) q[5], q[6];
s q[4];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[5];
sdg q[2];
rxx(0) q[0], q[5];
cy q[2], q[10];
rzz(1.5707963267948966) q[0], q[8];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
sdg q[3];
id q[4];
cu1(1.5707963267948966) q[5], q[1];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[8];
cu1(1.5707963267948966) q[4], q[0];
cz q[1], q[6];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[11];
p(0) q[10];
rz(1.5707963267948966) q[3];
id q[0];
p(0) q[3];
cu1(1.5707963267948966) q[5], q[9];
rxx(0) q[3], q[7];
ry(1.5707963267948966) q[11];
s q[2];
s q[8];
s q[11];
cy q[11], q[6];
ry(1.5707963267948966) q[9];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[10];
h q[4];
rzz(1.5707963267948966) q[4], q[7];