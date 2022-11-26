OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
cu1(1.5707963267948966) q[3], q[5];
sdg q[6];
u1(1.5707963267948966) q[1];
sdg q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[3];
h q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
h q[1];
rxx(0) q[2], q[4];
p(0) q[4];
rxx(0) q[5], q[3];
ry(1.5707963267948966) q[3];
ch q[5], q[1];
rz(1.5707963267948966) q[2];
p(0) q[1];
id q[5];
u3(0, 0, 1.5707963267948966) q[4];
tdg q[6];
cx q[1], q[3];
sdg q[2];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
rxx(0) q[1], q[3];
cu3(1.5707963267948966, 0, 0) q[0], q[6];
id q[2];
ch q[2], q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
cu1(1.5707963267948966) q[2], q[4];
rz(1.5707963267948966) q[1];
cu3(1.5707963267948966, 0, 0) q[0], q[4];
ry(1.5707963267948966) q[6];
cu3(1.5707963267948966, 0, 0) q[4], q[2];
u1(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
id q[3];
u3(0, 0, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[1];
s q[1];
h q[6];
sdg q[2];
s q[3];
h q[6];
cy q[3], q[6];
swap q[1], q[2];
swap q[4], q[0];
rzz(1.5707963267948966) q[5], q[4];
u1(1.5707963267948966) q[3];
t q[6];
tdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
cy q[6], q[1];
sdg q[0];
u1(1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
t q[1];
sdg q[0];
u3(0, 0, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[6];
tdg q[3];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[5], q[6];
u1(1.5707963267948966) q[6];
s q[3];
rxx(0) q[5], q[3];
t q[4];
u3(0, 0, 1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
s q[3];
id q[0];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[0];
swap q[0], q[2];
t q[4];
id q[4];
s q[6];
ch q[5], q[3];
ry(1.5707963267948966) q[0];
sdg q[6];
ry(1.5707963267948966) q[1];
p(0) q[6];
t q[6];