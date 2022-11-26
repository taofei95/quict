OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
t q[3];
h q[3];
tdg q[2];
u1(1.5707963267948966) q[4];
t q[8];
rz(1.5707963267948966) q[7];
id q[2];
p(0) q[5];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[9];
ry(1.5707963267948966) q[6];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
p(0) q[2];
ry(1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
h q[7];
rx(1.5707963267948966) q[8];
t q[0];
cz q[2], q[6];
id q[5];
swap q[0], q[4];
t q[8];
u3(0, 0, 1.5707963267948966) q[3];
p(0) q[1];
rz(1.5707963267948966) q[2];
id q[6];
h q[6];
sdg q[5];
sdg q[4];
s q[2];
cu1(1.5707963267948966) q[2], q[5];
cu1(1.5707963267948966) q[5], q[2];
p(0) q[4];
rz(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
s q[0];
ry(1.5707963267948966) q[5];
t q[5];
sdg q[8];
u1(1.5707963267948966) q[8];
rz(1.5707963267948966) q[8];
tdg q[9];
rz(1.5707963267948966) q[1];
t q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[9];
cu1(1.5707963267948966) q[6], q[7];
sdg q[7];
u1(1.5707963267948966) q[6];
cz q[2], q[8];
rx(1.5707963267948966) q[1];
s q[0];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[8];
p(0) q[3];
t q[3];
ry(1.5707963267948966) q[1];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
s q[2];
u1(1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
id q[0];
h q[2];
cx q[3], q[4];
cz q[9], q[1];
id q[4];
cz q[1], q[4];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
t q[7];
cu3(1.5707963267948966, 0, 0) q[0], q[7];
ry(1.5707963267948966) q[4];
u1(1.5707963267948966) q[0];
u1(1.5707963267948966) q[4];
rx(1.5707963267948966) q[6];
rxx(0) q[9], q[7];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[4];
id q[8];
cy q[2], q[1];
t q[7];
h q[2];
p(0) q[6];
rz(1.5707963267948966) q[9];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[4];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[3];
rxx(0) q[2], q[3];
tdg q[7];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[6];
h q[1];
cu1(1.5707963267948966) q[2], q[6];
u1(1.5707963267948966) q[6];
tdg q[9];
cu1(1.5707963267948966) q[3], q[4];
rxx(0) q[5], q[3];
t q[5];
cu1(1.5707963267948966) q[4], q[7];
tdg q[3];
p(0) q[0];
h q[8];
ry(1.5707963267948966) q[6];
h q[6];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
t q[0];
id q[1];
cu3(1.5707963267948966, 0, 0) q[1], q[7];
h q[8];
u3(0, 0, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
h q[4];
ry(1.5707963267948966) q[4];
sdg q[7];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
t q[0];
rxx(0) q[9], q[5];
t q[9];
id q[3];
u1(1.5707963267948966) q[4];
cz q[4], q[7];
sdg q[7];
cu3(1.5707963267948966, 0, 0) q[7], q[6];
rz(1.5707963267948966) q[7];
p(0) q[5];
h q[1];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[5];
tdg q[8];
cx q[8], q[0];
u3(0, 0, 1.5707963267948966) q[5];
rzz(1.5707963267948966) q[7], q[6];
cu1(1.5707963267948966) q[1], q[0];
s q[2];
rz(1.5707963267948966) q[0];
rzz(1.5707963267948966) q[8], q[7];
t q[1];
u1(1.5707963267948966) q[3];