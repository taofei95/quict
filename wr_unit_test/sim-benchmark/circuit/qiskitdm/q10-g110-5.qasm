OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u3(0, 0, 1.5707963267948966) q[5];
p(0) q[6];
cy q[9], q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[9];
ry(1.5707963267948966) q[0];
u1(1.5707963267948966) q[7];
p(0) q[5];
h q[1];
sdg q[1];
id q[4];
u2(1.5707963267948966, 1.5707963267948966) q[7];
h q[2];
sdg q[2];
cu1(1.5707963267948966) q[5], q[7];
ry(1.5707963267948966) q[7];
tdg q[8];
rzz(1.5707963267948966) q[1], q[0];
swap q[2], q[5];
rz(1.5707963267948966) q[4];
sdg q[3];
ry(1.5707963267948966) q[9];
t q[4];
sdg q[2];
p(0) q[8];
cy q[7], q[6];
u2(1.5707963267948966, 1.5707963267948966) q[8];
sdg q[9];
rzz(1.5707963267948966) q[6], q[2];
rx(1.5707963267948966) q[9];
rzz(1.5707963267948966) q[3], q[7];
t q[4];
id q[5];
cy q[1], q[8];
h q[2];
cu3(1.5707963267948966, 0, 0) q[2], q[5];
sdg q[7];
swap q[9], q[1];
s q[1];
rx(1.5707963267948966) q[3];
u1(1.5707963267948966) q[7];
t q[1];
rzz(1.5707963267948966) q[7], q[6];
id q[5];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[7];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
cy q[9], q[3];
u1(1.5707963267948966) q[4];
s q[7];
swap q[0], q[6];
u2(1.5707963267948966, 1.5707963267948966) q[4];
s q[4];
u2(1.5707963267948966, 1.5707963267948966) q[9];
cu3(1.5707963267948966, 0, 0) q[9], q[4];
sdg q[0];
s q[9];
rz(1.5707963267948966) q[8];
tdg q[2];
s q[1];
tdg q[9];
sdg q[3];
cu3(1.5707963267948966, 0, 0) q[7], q[8];
u3(0, 0, 1.5707963267948966) q[4];
p(0) q[5];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[9];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
cu3(1.5707963267948966, 0, 0) q[6], q[8];
s q[1];
sdg q[2];
cx q[1], q[0];
t q[1];
h q[3];
id q[5];
cu3(1.5707963267948966, 0, 0) q[7], q[0];
s q[9];
p(0) q[4];
id q[5];
rx(1.5707963267948966) q[2];
cx q[7], q[8];
id q[9];
p(0) q[3];
sdg q[1];
rz(1.5707963267948966) q[2];
cx q[0], q[7];
ry(1.5707963267948966) q[7];
rxx(0) q[7], q[1];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[2];
t q[7];
rx(1.5707963267948966) q[6];
rz(1.5707963267948966) q[8];
rxx(0) q[1], q[5];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
cx q[7], q[2];
rz(1.5707963267948966) q[4];
cx q[1], q[5];
t q[1];
cz q[0], q[8];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
p(0) q[8];
p(0) q[2];
ry(1.5707963267948966) q[3];
id q[5];