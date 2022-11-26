OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
u2(1.5707963267948966, 1.5707963267948966) q[4];
u1(1.5707963267948966) q[5];
rz(1.5707963267948966) q[6];
rx(1.5707963267948966) q[5];
s q[2];
cu3(1.5707963267948966, 0, 0) q[6], q[3];
rx(1.5707963267948966) q[3];
id q[2];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[6];
id q[0];
s q[5];
sdg q[4];
u3(0, 0, 1.5707963267948966) q[3];
swap q[3], q[5];
s q[4];
s q[1];
h q[0];
cx q[3], q[4];
id q[4];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[3];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[7];
crz(1.5707963267948966) q[1], q[6];
ry(1.5707963267948966) q[2];
s q[6];
sdg q[0];
id q[5];
rz(1.5707963267948966) q[3];
sdg q[7];
sdg q[3];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
rxx(0) q[0], q[6];
t q[5];
id q[2];
rx(1.5707963267948966) q[5];
cu3(1.5707963267948966, 0, 0) q[3], q[6];
id q[5];
rx(1.5707963267948966) q[7];
p(0) q[0];
h q[1];
p(0) q[6];
id q[3];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
sdg q[3];
s q[1];
cu3(1.5707963267948966, 0, 0) q[3], q[6];
u3(0, 0, 1.5707963267948966) q[0];
p(0) q[1];
tdg q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[7];
cy q[2], q[7];
rx(1.5707963267948966) q[4];
h q[0];
u3(0, 0, 1.5707963267948966) q[3];
cu3(1.5707963267948966, 0, 0) q[4], q[6];
id q[0];
u3(0, 0, 1.5707963267948966) q[7];
t q[4];
h q[6];
u1(1.5707963267948966) q[7];
rz(1.5707963267948966) q[4];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[0];
rz(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[4];
tdg q[1];
u1(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
h q[5];
p(0) q[2];
p(0) q[3];
h q[6];
id q[3];
ch q[7], q[5];
sdg q[2];