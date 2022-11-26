OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(1.5707963267948966) q[0];
cu3(1.5707963267948966, 0, 0) q[4], q[8];
id q[0];
rx(1.5707963267948966) q[2];
tdg q[7];
t q[2];
h q[2];
rz(1.5707963267948966) q[0];
id q[6];
u1(1.5707963267948966) q[9];
rxx(0) q[7], q[1];
rz(1.5707963267948966) q[5];
tdg q[0];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[9];
cu3(1.5707963267948966, 0, 0) q[6], q[1];
cu3(1.5707963267948966, 0, 0) q[7], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
ry(1.5707963267948966) q[7];
cy q[3], q[4];
sdg q[1];
t q[1];
u3(0, 0, 1.5707963267948966) q[6];
id q[8];
id q[0];
sdg q[3];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[7];
rzz(1.5707963267948966) q[8], q[4];
p(0) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[2];
cy q[3], q[6];
u3(0, 0, 1.5707963267948966) q[3];
rxx(0) q[1], q[7];
rz(1.5707963267948966) q[1];
id q[3];
sdg q[1];
p(0) q[2];
rx(1.5707963267948966) q[0];
tdg q[8];
id q[3];
sdg q[8];
tdg q[7];
tdg q[3];
t q[1];
s q[1];
cy q[0], q[9];
tdg q[9];
s q[6];
p(0) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[9];
ry(1.5707963267948966) q[4];
cy q[2], q[7];
ry(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
s q[0];
id q[5];
t q[1];
u3(0, 0, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[0];
u1(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[9];
sdg q[0];
t q[4];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[0];
h q[1];
ry(1.5707963267948966) q[5];
tdg q[8];
cu1(1.5707963267948966) q[3], q[0];
tdg q[0];
rxx(0) q[7], q[9];
cu3(1.5707963267948966, 0, 0) q[6], q[2];
swap q[1], q[4];
s q[5];
u2(1.5707963267948966, 1.5707963267948966) q[4];
tdg q[8];
tdg q[6];
s q[6];
rx(1.5707963267948966) q[5];
cy q[9], q[5];
cx q[0], q[5];
h q[8];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
s q[4];
u1(1.5707963267948966) q[8];
s q[0];