OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
sdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[0];
cy q[3], q[1];
u3(0, 0, 1.5707963267948966) q[0];
cx q[2], q[3];
sdg q[3];
t q[2];
rz(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[0], q[3];
t q[2];
p(0) q[3];
rz(1.5707963267948966) q[2];
t q[1];
ry(1.5707963267948966) q[2];
rzz(1.5707963267948966) q[2], q[1];
id q[1];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
s q[1];
s q[0];
s q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
p(0) q[1];
u1(1.5707963267948966) q[3];
sdg q[2];
u3(0, 0, 1.5707963267948966) q[1];