OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u3(0, 0, 1.5707963267948966) q[5];
s q[2];
cx q[7], q[9];
rx(1.5707963267948966) q[6];
s q[6];
cx q[7], q[1];
u3(0, 0, 1.5707963267948966) q[1];
cy q[4], q[6];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[1];
p(0) q[1];
tdg q[5];
p(0) q[6];
p(0) q[4];
rx(1.5707963267948966) q[8];
u1(1.5707963267948966) q[3];
cu3(1.5707963267948966, 0, 0) q[6], q[9];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
id q[7];
u1(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[2];
t q[5];
u3(0, 0, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[2];
s q[6];
tdg q[2];
ch q[5], q[2];
ch q[0], q[7];
rx(1.5707963267948966) q[7];
sdg q[5];
tdg q[1];
sdg q[3];
rz(1.5707963267948966) q[9];
rzz(1.5707963267948966) q[8], q[5];
rxx(0) q[1], q[6];
cx q[5], q[0];
p(0) q[4];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[9];
id q[9];
t q[3];
tdg q[3];
p(0) q[9];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[5];
sdg q[1];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
tdg q[3];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[1];
u1(1.5707963267948966) q[8];
p(0) q[0];
swap q[9], q[0];
p(0) q[8];
sdg q[8];
s q[0];
t q[1];
u1(1.5707963267948966) q[8];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
ch q[8], q[2];
h q[2];
p(0) q[4];
s q[9];
swap q[1], q[6];
tdg q[8];
ry(1.5707963267948966) q[7];
u1(1.5707963267948966) q[3];
ch q[4], q[9];
p(0) q[3];
cy q[0], q[2];
p(0) q[2];
swap q[7], q[3];
rz(1.5707963267948966) q[9];
p(0) q[1];
p(0) q[5];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
s q[4];
s q[6];
rx(1.5707963267948966) q[8];
t q[1];
u1(1.5707963267948966) q[5];
t q[0];
t q[1];
u3(0, 0, 1.5707963267948966) q[4];
t q[5];
tdg q[4];
u3(0, 0, 1.5707963267948966) q[9];
sdg q[9];
cz q[1], q[8];
t q[9];
swap q[3], q[9];
rzz(1.5707963267948966) q[0], q[1];
rzz(1.5707963267948966) q[8], q[4];
cz q[5], q[0];
p(0) q[0];
s q[5];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[2];
cy q[0], q[8];
cu1(1.5707963267948966) q[2], q[6];
u3(0, 0, 1.5707963267948966) q[9];
t q[7];
u1(1.5707963267948966) q[4];
t q[5];
rzz(1.5707963267948966) q[5], q[0];
p(0) q[2];
s q[6];
tdg q[5];
p(0) q[1];
u1(1.5707963267948966) q[1];
id q[8];
ry(1.5707963267948966) q[3];
crz(1.5707963267948966) q[2], q[3];
t q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
s q[5];
crz(1.5707963267948966) q[6], q[1];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[3];