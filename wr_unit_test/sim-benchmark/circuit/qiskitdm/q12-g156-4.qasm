OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
cu1(1.5707963267948966) q[8], q[6];
sdg q[8];
ry(1.5707963267948966) q[7];
t q[2];
tdg q[5];
s q[5];
ry(1.5707963267948966) q[11];
rxx(0) q[11], q[10];
ry(1.5707963267948966) q[6];
rz(1.5707963267948966) q[4];
swap q[2], q[3];
rz(1.5707963267948966) q[2];
p(0) q[2];
rz(1.5707963267948966) q[4];
t q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[2];
cu1(1.5707963267948966) q[3], q[11];
p(0) q[9];
cu1(1.5707963267948966) q[9], q[1];
ry(1.5707963267948966) q[10];
p(0) q[5];
swap q[9], q[1];
ry(1.5707963267948966) q[3];
rzz(1.5707963267948966) q[9], q[11];
ry(1.5707963267948966) q[0];
h q[8];
h q[9];
cy q[8], q[1];
p(0) q[9];
cx q[5], q[4];
p(0) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[9];
s q[4];
cz q[2], q[0];
t q[3];
rx(1.5707963267948966) q[6];
cu3(1.5707963267948966, 0, 0) q[2], q[10];
t q[11];
sdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[9];
cx q[4], q[9];
h q[9];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[7];
cu1(1.5707963267948966) q[11], q[9];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
p(0) q[7];
cy q[8], q[6];
h q[0];
cu3(1.5707963267948966, 0, 0) q[7], q[9];
cu3(1.5707963267948966, 0, 0) q[3], q[8];
rz(1.5707963267948966) q[10];
ry(1.5707963267948966) q[4];
p(0) q[4];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[5];
u1(1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[10];
sdg q[1];
u1(1.5707963267948966) q[6];
tdg q[7];
rx(1.5707963267948966) q[9];
cu3(1.5707963267948966, 0, 0) q[0], q[10];
rxx(0) q[1], q[3];
tdg q[0];
cu1(1.5707963267948966) q[8], q[7];
p(0) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[8];
sdg q[2];
s q[10];
rxx(0) q[9], q[7];
cu3(1.5707963267948966, 0, 0) q[8], q[1];
rz(1.5707963267948966) q[4];
cx q[5], q[9];
s q[1];
id q[1];
u1(1.5707963267948966) q[8];
cu1(1.5707963267948966) q[11], q[9];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rzz(1.5707963267948966) q[10], q[4];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[3];
cy q[5], q[1];
t q[6];
s q[9];
s q[0];
t q[6];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[11];
sdg q[0];
rx(1.5707963267948966) q[0];
rzz(1.5707963267948966) q[8], q[9];
s q[9];
rz(1.5707963267948966) q[4];
s q[8];
swap q[8], q[6];
tdg q[10];
rz(1.5707963267948966) q[7];
cz q[6], q[3];
rz(1.5707963267948966) q[10];
id q[9];
id q[11];
u1(1.5707963267948966) q[8];
s q[2];
ry(1.5707963267948966) q[6];
tdg q[9];
cu1(1.5707963267948966) q[4], q[3];
t q[2];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[0];
s q[8];
sdg q[1];
u3(0, 0, 1.5707963267948966) q[8];
cz q[4], q[11];
p(0) q[11];
rz(1.5707963267948966) q[8];
cu1(1.5707963267948966) q[0], q[7];
cu3(1.5707963267948966, 0, 0) q[9], q[10];
rzz(1.5707963267948966) q[3], q[10];
p(0) q[11];
t q[6];
rx(1.5707963267948966) q[5];
t q[1];
t q[7];
p(0) q[9];
cu3(1.5707963267948966, 0, 0) q[2], q[1];
u3(0, 0, 1.5707963267948966) q[10];
swap q[5], q[3];
ry(1.5707963267948966) q[3];
p(0) q[3];
cx q[11], q[7];
s q[5];
rz(1.5707963267948966) q[2];
sdg q[9];
rxx(0) q[5], q[10];
s q[3];
rxx(0) q[2], q[10];
rx(1.5707963267948966) q[8];
p(0) q[2];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[6];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[3];
tdg q[10];