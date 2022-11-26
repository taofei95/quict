OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(1.5707963267948966) q[1];
t q[2];
tdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[5];
cu1(1.5707963267948966) q[1], q[3];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[7];
cu3(1.5707963267948966, 0, 0) q[6], q[3];
h q[6];
id q[0];
cu3(1.5707963267948966, 0, 0) q[6], q[0];
u1(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[3];
cz q[2], q[3];
tdg q[4];
p(0) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
cy q[0], q[5];
ry(1.5707963267948966) q[4];
s q[5];
tdg q[1];
ry(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[0];
tdg q[4];
s q[7];
ry(1.5707963267948966) q[1];
t q[0];
p(0) q[6];
id q[1];
tdg q[5];
ry(1.5707963267948966) q[6];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[2];
id q[9];
id q[8];
u1(1.5707963267948966) q[0];
rxx(0) q[3], q[4];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[2];
cy q[1], q[6];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[0];
tdg q[0];
rzz(1.5707963267948966) q[4], q[1];
cz q[5], q[6];
s q[5];
u3(0, 0, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[7];
rx(1.5707963267948966) q[8];
u1(1.5707963267948966) q[4];
cy q[3], q[5];
rxx(0) q[3], q[2];
sdg q[6];
s q[7];
u3(0, 0, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
rzz(1.5707963267948966) q[0], q[4];
id q[8];
u3(0, 0, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[9];
swap q[0], q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
cu1(1.5707963267948966) q[7], q[9];
t q[3];
sdg q[4];
rxx(0) q[6], q[3];
cz q[5], q[9];
rx(1.5707963267948966) q[4];
u1(1.5707963267948966) q[1];
sdg q[2];
id q[0];
u2(1.5707963267948966, 1.5707963267948966) q[9];
id q[5];
cz q[6], q[2];
u3(0, 0, 1.5707963267948966) q[9];
s q[7];
ry(1.5707963267948966) q[5];
id q[2];
h q[7];
cy q[9], q[6];
s q[6];
id q[8];
h q[8];
u2(1.5707963267948966, 1.5707963267948966) q[2];
p(0) q[0];
p(0) q[7];
cu3(1.5707963267948966, 0, 0) q[2], q[5];
rz(1.5707963267948966) q[6];
tdg q[8];
cu1(1.5707963267948966) q[7], q[3];
s q[1];
u3(0, 0, 1.5707963267948966) q[3];
cy q[4], q[3];
tdg q[9];
u3(0, 0, 1.5707963267948966) q[4];
rzz(1.5707963267948966) q[9], q[3];
u3(0, 0, 1.5707963267948966) q[9];
id q[6];
rzz(1.5707963267948966) q[6], q[8];
t q[7];
id q[1];
u3(0, 0, 1.5707963267948966) q[9];
s q[4];