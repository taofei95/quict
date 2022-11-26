OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
sdg q[0];
ry(1.5707963267948966) q[4];
tdg q[5];
cu1(1.5707963267948966) q[9], q[3];
u3(0, 0, 1.5707963267948966) q[9];
sdg q[0];
tdg q[8];
u2(1.5707963267948966, 1.5707963267948966) q[6];
p(0) q[9];
rz(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[8];
id q[1];
rzz(1.5707963267948966) q[4], q[0];
rz(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
sdg q[7];
tdg q[2];
u1(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[6];
p(0) q[8];
rx(1.5707963267948966) q[0];
h q[7];
t q[1];
rz(1.5707963267948966) q[8];
h q[0];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[2];
tdg q[7];
tdg q[6];
rz(1.5707963267948966) q[6];
rzz(1.5707963267948966) q[6], q[7];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[5];
sdg q[8];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
s q[2];
p(0) q[6];
rx(1.5707963267948966) q[0];
u1(1.5707963267948966) q[5];
t q[5];
swap q[0], q[5];
cu3(1.5707963267948966, 0, 0) q[5], q[1];
s q[7];
h q[2];
id q[4];
p(0) q[0];
t q[4];
u3(0, 0, 1.5707963267948966) q[4];
cz q[9], q[6];
rxx(0) q[3], q[0];
crz(1.5707963267948966) q[8], q[6];
rxx(0) q[1], q[7];
rz(1.5707963267948966) q[0];
cx q[8], q[3];
tdg q[5];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[9];
swap q[8], q[2];
p(0) q[4];
id q[4];
cu1(1.5707963267948966) q[6], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[4];
sdg q[8];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[2];
id q[0];
cy q[6], q[3];
p(0) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u1(1.5707963267948966) q[8];
rx(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[4];
t q[4];
t q[2];
id q[6];
u1(1.5707963267948966) q[0];
id q[0];
tdg q[1];
crz(1.5707963267948966) q[3], q[6];
tdg q[0];
cu3(1.5707963267948966, 0, 0) q[6], q[2];
h q[7];
ry(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[2];
rxx(0) q[2], q[4];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
tdg q[8];
cu1(1.5707963267948966) q[7], q[0];
cx q[2], q[9];
h q[4];
s q[9];
ry(1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[6];
ry(1.5707963267948966) q[1];
p(0) q[9];
h q[2];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[9];
rz(1.5707963267948966) q[8];
id q[6];
h q[1];
rx(1.5707963267948966) q[1];
s q[8];
cx q[6], q[8];
id q[0];
rz(1.5707963267948966) q[4];
rxx(0) q[1], q[6];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[7];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[5];
id q[4];
id q[8];
u3(0, 0, 1.5707963267948966) q[9];
swap q[2], q[9];
p(0) q[0];
u1(1.5707963267948966) q[6];