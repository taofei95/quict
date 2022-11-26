OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rx(1.5707963267948966) q[4];
tdg q[6];
p(0) q[2];
tdg q[0];
rz(1.5707963267948966) q[6];
rzz(1.5707963267948966) q[6], q[9];
t q[1];
cu1(1.5707963267948966) q[2], q[7];
rzz(1.5707963267948966) q[7], q[8];
cx q[0], q[7];
u1(1.5707963267948966) q[0];
cz q[6], q[1];
cy q[0], q[4];
p(0) q[8];
u1(1.5707963267948966) q[3];
h q[2];
h q[9];
rz(1.5707963267948966) q[1];
h q[1];
sdg q[7];
ry(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
s q[8];
rx(1.5707963267948966) q[6];
p(0) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
sdg q[5];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
t q[0];
ry(1.5707963267948966) q[0];
sdg q[4];
p(0) q[8];
rzz(1.5707963267948966) q[7], q[4];
cx q[5], q[7];
sdg q[0];
sdg q[5];
ry(1.5707963267948966) q[9];
h q[9];
s q[4];
u3(0, 0, 1.5707963267948966) q[7];
s q[2];
s q[5];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cu3(1.5707963267948966, 0, 0) q[5], q[8];
cu3(1.5707963267948966, 0, 0) q[5], q[2];
h q[1];
id q[7];
rx(1.5707963267948966) q[0];
cz q[0], q[5];
cy q[4], q[7];
cu1(1.5707963267948966) q[0], q[1];
id q[9];
rz(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
cy q[6], q[2];
p(0) q[1];
swap q[4], q[5];
s q[2];
p(0) q[7];
t q[6];
h q[0];
cu3(1.5707963267948966, 0, 0) q[9], q[2];
u3(0, 0, 1.5707963267948966) q[8];
cy q[0], q[2];
cu3(1.5707963267948966, 0, 0) q[7], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[0];
sdg q[3];
cu1(1.5707963267948966) q[6], q[8];
cx q[3], q[2];
u2(1.5707963267948966, 1.5707963267948966) q[8];
cu3(1.5707963267948966, 0, 0) q[4], q[7];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[3];
t q[2];
rx(1.5707963267948966) q[0];
p(0) q[4];
u3(0, 0, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[8];
rzz(1.5707963267948966) q[3], q[4];
u1(1.5707963267948966) q[8];
u1(1.5707963267948966) q[2];
sdg q[1];
p(0) q[9];
swap q[8], q[7];
cu1(1.5707963267948966) q[1], q[7];
tdg q[7];
t q[9];
id q[0];
cu3(1.5707963267948966, 0, 0) q[7], q[5];
rx(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
p(0) q[7];
tdg q[4];
u1(1.5707963267948966) q[8];
swap q[5], q[8];
u1(1.5707963267948966) q[9];
s q[0];
cu1(1.5707963267948966) q[9], q[6];
tdg q[7];
t q[9];
t q[2];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[7];
t q[0];
t q[1];
rz(1.5707963267948966) q[5];
t q[4];
cz q[5], q[6];
sdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[3];
sdg q[7];
rx(1.5707963267948966) q[4];
s q[5];
cy q[1], q[0];
u1(1.5707963267948966) q[2];
p(0) q[5];
h q[1];
ry(1.5707963267948966) q[5];