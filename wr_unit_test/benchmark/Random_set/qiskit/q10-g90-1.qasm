OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
u1(1.5707963267948966) q[9];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[9];
cu1(1.5707963267948966) q[8], q[3];
tdg q[8];
t q[9];
ch q[8], q[6];
u1(1.5707963267948966) q[3];
tdg q[2];
ry(1.5707963267948966) q[6];
t q[8];
s q[4];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[2];
sdg q[2];
sdg q[6];
sdg q[8];
rxx(0) q[6], q[1];
id q[0];
rx(1.5707963267948966) q[0];
p(0) q[5];
swap q[5], q[8];
s q[4];
s q[2];
u2(1.5707963267948966, 1.5707963267948966) q[9];
u1(1.5707963267948966) q[6];
u1(1.5707963267948966) q[8];
ch q[6], q[8];
sdg q[8];
ry(1.5707963267948966) q[7];
u1(1.5707963267948966) q[7];
t q[7];
u3(0, 0, 1.5707963267948966) q[6];
tdg q[0];
sdg q[5];
p(0) q[2];
rzz(1.5707963267948966) q[1], q[3];
h q[6];
u2(1.5707963267948966, 1.5707963267948966) q[4];
sdg q[4];
h q[4];
id q[1];
p(0) q[8];
id q[5];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[0];
p(0) q[1];
cy q[7], q[3];
id q[3];
cu3(1.5707963267948966, 0, 0) q[1], q[4];
s q[0];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[6];
h q[8];
rx(1.5707963267948966) q[2];
cx q[0], q[2];
p(0) q[7];
h q[1];
s q[3];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[8];
s q[6];
h q[2];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[3];
u3(0, 0, 1.5707963267948966) q[1];
t q[0];
u3(0, 0, 1.5707963267948966) q[1];
ch q[6], q[1];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[2];
ch q[8], q[7];
h q[5];
u1(1.5707963267948966) q[0];
id q[5];
id q[4];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[4];
ry(1.5707963267948966) q[5];
t q[7];
h q[0];
u3(0, 0, 1.5707963267948966) q[1];
cy q[4], q[8];
id q[0];
cu3(1.5707963267948966, 0, 0) q[7], q[5];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];