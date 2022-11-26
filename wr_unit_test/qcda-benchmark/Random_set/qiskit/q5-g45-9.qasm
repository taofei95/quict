OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(1.5707963267948966) q[3];
u1(1.5707963267948966) q[2];
s q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
cx q[4], q[0];
rxx(0) q[0], q[4];
rx(1.5707963267948966) q[1];
rz(1.5707963267948966) q[1];
id q[2];
cu1(1.5707963267948966) q[2], q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
id q[0];
swap q[2], q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
ch q[0], q[2];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[3];
tdg q[4];
cu1(1.5707963267948966) q[1], q[4];
rx(1.5707963267948966) q[0];
swap q[1], q[0];
ry(1.5707963267948966) q[4];
rz(1.5707963267948966) q[1];
s q[0];
rzz(1.5707963267948966) q[2], q[3];
tdg q[2];
cy q[1], q[3];
swap q[0], q[2];
tdg q[0];
u1(1.5707963267948966) q[0];
tdg q[4];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[1];
tdg q[0];
rz(1.5707963267948966) q[3];
p(0) q[3];
tdg q[1];
t q[1];
t q[2];
s q[4];
tdg q[3];
rx(1.5707963267948966) q[4];
id q[0];