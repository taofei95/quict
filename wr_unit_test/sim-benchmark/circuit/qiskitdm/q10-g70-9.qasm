OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u1(1.5707963267948966) q[3];
id q[0];
rx(1.5707963267948966) q[7];
rz(1.5707963267948966) q[0];
sdg q[0];
rzz(1.5707963267948966) q[1], q[0];
rx(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[2];
swap q[7], q[9];
id q[9];
rz(1.5707963267948966) q[6];
p(0) q[4];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
cu1(1.5707963267948966) q[1], q[9];
tdg q[1];
sdg q[9];
h q[4];
tdg q[1];
cz q[0], q[9];
sdg q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
s q[1];
t q[8];
sdg q[7];
cx q[0], q[3];
cu3(1.5707963267948966, 0, 0) q[2], q[0];
cz q[0], q[9];
rxx(0) q[1], q[0];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
id q[5];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[9];
cu3(1.5707963267948966, 0, 0) q[1], q[9];
rx(1.5707963267948966) q[5];
rxx(0) q[6], q[7];
rz(1.5707963267948966) q[2];
rxx(0) q[6], q[3];
cu1(1.5707963267948966) q[5], q[2];
rzz(1.5707963267948966) q[4], q[5];
cz q[5], q[0];
rx(1.5707963267948966) q[9];
h q[5];
cy q[4], q[0];
u3(0, 0, 1.5707963267948966) q[5];
h q[5];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[3];
id q[0];
sdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[7];
h q[7];
rz(1.5707963267948966) q[9];
ry(1.5707963267948966) q[0];
id q[8];
sdg q[2];
rzz(1.5707963267948966) q[9], q[4];
ry(1.5707963267948966) q[8];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
cu3(1.5707963267948966, 0, 0) q[3], q[6];
u3(0, 0, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];