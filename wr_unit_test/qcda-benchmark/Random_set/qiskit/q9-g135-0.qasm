OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c[9];
cu1(1.5707963267948966) q[4], q[1];
rz(1.5707963267948966) q[6];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
sdg q[2];
h q[0];
rzz(1.5707963267948966) q[4], q[8];
tdg q[4];
ry(1.5707963267948966) q[7];
t q[0];
rx(1.5707963267948966) q[2];
sdg q[7];
rx(1.5707963267948966) q[4];
rzz(1.5707963267948966) q[2], q[1];
cy q[7], q[5];
id q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[4];
s q[4];
id q[3];
rz(1.5707963267948966) q[6];
cy q[3], q[5];
p(0) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rxx(0) q[0], q[5];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[5];
ry(1.5707963267948966) q[4];
t q[5];
p(0) q[5];
p(0) q[2];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[4];
swap q[4], q[6];
id q[8];
tdg q[7];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
t q[4];
rz(1.5707963267948966) q[7];
rxx(0) q[5], q[4];
ry(1.5707963267948966) q[4];
tdg q[4];
id q[0];
cu1(1.5707963267948966) q[6], q[5];
ry(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[5], q[1];
cu3(1.5707963267948966, 0, 0) q[5], q[4];
rx(1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
cu3(1.5707963267948966, 0, 0) q[2], q[8];
p(0) q[4];
sdg q[0];
swap q[5], q[6];
s q[6];
t q[8];
p(0) q[8];
rzz(1.5707963267948966) q[2], q[7];
id q[8];
h q[0];
h q[5];
ch q[8], q[2];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[2];
sdg q[0];
swap q[8], q[0];
sdg q[5];
t q[3];
rz(1.5707963267948966) q[8];
cu3(1.5707963267948966, 0, 0) q[7], q[4];
ry(1.5707963267948966) q[4];
tdg q[2];
t q[4];
id q[6];
u3(0, 0, 1.5707963267948966) q[3];
s q[6];
p(0) q[0];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[8];
cz q[1], q[4];
u3(0, 0, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[5];
cu3(1.5707963267948966, 0, 0) q[6], q[3];
rx(1.5707963267948966) q[2];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[0];
p(0) q[3];
rxx(0) q[0], q[1];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
sdg q[3];
tdg q[7];
rz(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[0], q[1];
rz(1.5707963267948966) q[6];
ch q[1], q[4];
cu1(1.5707963267948966) q[2], q[0];
cx q[1], q[3];
t q[6];
id q[8];
u3(0, 0, 1.5707963267948966) q[8];
rzz(1.5707963267948966) q[1], q[0];
p(0) q[0];
ch q[8], q[1];
s q[0];
p(0) q[4];
sdg q[6];
tdg q[2];
swap q[6], q[1];
h q[4];
rz(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[7];
t q[1];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[1];
p(0) q[7];
rzz(1.5707963267948966) q[5], q[3];
u3(0, 0, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[1];
id q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
s q[4];
rx(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[1];
id q[7];
rz(1.5707963267948966) q[5];
cu3(1.5707963267948966, 0, 0) q[5], q[8];
u1(1.5707963267948966) q[6];