OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[1];
id q[3];
cu1(1.5707963267948966) q[3], q[8];
cu3(1.5707963267948966, 0, 0) q[9], q[5];
rxx(0) q[8], q[1];
sdg q[7];
s q[3];
t q[9];
u3(0, 0, 1.5707963267948966) q[1];
rxx(0) q[2], q[0];
t q[3];
p(0) q[1];
u1(1.5707963267948966) q[7];
swap q[1], q[6];
ry(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rxx(0) q[9], q[7];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[7];
p(0) q[2];
u3(0, 0, 1.5707963267948966) q[0];
h q[3];
cy q[7], q[2];
id q[9];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[9];
swap q[4], q[8];
s q[2];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[5];
h q[5];
cz q[4], q[7];
cz q[3], q[9];
cx q[3], q[7];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rzz(1.5707963267948966) q[1], q[5];
rx(1.5707963267948966) q[6];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[4];
rxx(0) q[1], q[9];
h q[0];
ry(1.5707963267948966) q[8];
s q[4];
swap q[3], q[1];
u1(1.5707963267948966) q[9];
t q[2];
u1(1.5707963267948966) q[4];
t q[9];
rz(1.5707963267948966) q[5];
t q[2];
u1(1.5707963267948966) q[8];
rx(1.5707963267948966) q[0];
h q[5];
u3(0, 0, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[5];
tdg q[5];
ry(1.5707963267948966) q[9];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[5];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[6];
s q[2];
rzz(1.5707963267948966) q[4], q[5];
t q[9];
cy q[9], q[2];
id q[3];
u3(0, 0, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[9];
cx q[1], q[4];
sdg q[1];
cy q[5], q[9];
h q[8];
h q[2];
u3(0, 0, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[3];
t q[2];
h q[7];
cz q[4], q[2];
id q[2];
t q[4];