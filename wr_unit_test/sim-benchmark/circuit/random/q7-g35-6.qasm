OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
cy q[0], q[3];
ch q[3], q[2];
rz(1.5707963267948966) q[0];
rxx(0) q[5], q[4];
p(0) q[3];
p(0) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[4];
p(0) q[5];
x q[6];
sdg q[1];
p(0) q[3];
u3(0, 0, 1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[3];
u1(1.5707963267948966) q[4];
t q[1];
ry(1.5707963267948966) q[2];
rz(1.5707963267948966) q[4];
p(0) q[6];
p(0) q[0];
rx(1.5707963267948966) q[2];
crz(1.5707963267948966) q[2], q[4];
u1(1.5707963267948966) q[0];
id q[2];
rz(1.5707963267948966) q[1];
cu1(1.5707963267948966) q[3], q[0];
t q[0];
ch q[0], q[4];
crz(1.5707963267948966) q[4], q[3];
t q[2];
id q[6];
u2(1.5707963267948966, 1.5707963267948966) q[6];